"""
Análisis de un micro-espectro local de SPHEREx.

Dado un pixel (x, y) en un detector, extrae un cutout de 100×100 px y
calcula propiedades espectrales:

- Micro-espectro: flujo promedio por fila (cada fila ≈ una λ).
- Pendiente log-log (spectral_slope): color espectral.
  - slope < 0 → más flujo a λ cortas → objeto azul/caliente.
  - slope > 0 → más flujo a λ largas → objeto rojo/frío.
- Flujo total → magnitud relativa.
- Clasificación heurística en STAR / ASTEROID / UNKNOWN.

NO pretende ser ciencia rigurosa: es una interpretación visual con
disclaimer. Con un solo detector, el rango espectral es ~0.017 μm, apenas
suficiente para medir "color", no para temperatura ni composición.
"""

from __future__ import annotations

import logging

import numpy as np
from astropy.io import fits

from app.services.fits_loader import (
    get_clean_image,
    get_wavelength_map,
)

logger = logging.getLogger(__name__)

CUTOUT_SIZE = 100  # px de alto × ancho
MIN_VALID_FRACTION = 0.10  # si <10% de píxeles válidos → unknown


# --------------------------------------------------------------------------- #
# Extracción del micro-espectro
# --------------------------------------------------------------------------- #

def extract_local_spectrum(
    hdul: fits.HDUList,
    pixel_x: int,
    pixel_y: int,
    cutout_size: int = CUTOUT_SIZE,
) -> dict:
    """
    Extrae un micro-espectro local centrado en (pixel_x, pixel_y).

    Devuelve:
        {
            "wavelengths_um": list[float],   # λ por fila del cutout
            "flux_per_row": list[float],     # flujo promedio por fila
            "wavelength_range_um": (λ_min, λ_max),
            "total_flux": float,             # suma de flujo
            "peak_flux": float,              # max de flux_per_row
            "valid_fraction": float,         # fracción de píxeles no-enmascarados
            "shape": (ny, nx),               # forma del cutout
        }
    """
    clean = get_clean_image(hdul, subtract_zodi=True)
    wave_map = get_wavelength_map(hdul, downsample=4)

    ny, nx = clean.shape
    half = cutout_size // 2
    x0 = max(0, pixel_x - half)
    x1 = min(nx, pixel_x + half)
    y0 = max(0, pixel_y - half)
    y1 = min(ny, pixel_y + half)

    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(
            f"Cutout demasiado pequeño en pixel ({pixel_x},{pixel_y}). "
            f"Shape cutout: ({y1-y0}, {x1-x0})"
        )

    cut_clean = clean[y0:y1, x0:x1]
    cut_wave = wave_map[y0:y1, x0:x1]

    # Fracción de píxeles válidos (no enmascarados)
    valid_mask = cut_clean != 0
    valid_fraction = float(valid_mask.mean())

    if valid_fraction < MIN_VALID_FRACTION:
        logger.warning(
            "Valid fraction %.2f%% < %.2f%% en (%d,%d); retornando vacío",
            valid_fraction * 100, MIN_VALID_FRACTION * 100, pixel_x, pixel_y,
        )
        return {
            "wavelengths_um": [],
            "flux_per_row": [],
            "wavelength_range_um": (0.0, 0.0),
            "total_flux": 0.0,
            "peak_flux": 0.0,
            "valid_fraction": valid_fraction,
            "shape": cut_clean.shape,
        }

    # Flujo promedio por fila (cada fila = una λ aproximada)
    flux_per_row = []
    wave_per_row = []
    for row_idx in range(cut_clean.shape[0]):
        row_flux = cut_clean[row_idx]
        row_wave = cut_wave[row_idx]
        row_valid = valid_mask[row_idx]

        if not row_valid.any():
            continue

        flux_avg = float(np.mean(row_flux[row_valid]))
        wave_avg = float(np.mean(row_wave[row_valid]))
        flux_per_row.append(flux_avg)
        wave_per_row.append(wave_avg)

    if not flux_per_row:
        return {
            "wavelengths_um": [],
            "flux_per_row": [],
            "wavelength_range_um": (0.0, 0.0),
            "total_flux": 0.0,
            "peak_flux": 0.0,
            "valid_fraction": valid_fraction,
            "shape": cut_clean.shape,
        }

    wavelengths_um = np.array(wave_per_row, dtype=np.float32)
    flux_per_row_arr = np.array(flux_per_row, dtype=np.float32)

    # Asegurar orden creciente en λ para el cálculo de pendiente
    sort_idx = np.argsort(wavelengths_um)
    wavelengths_um = wavelengths_um[sort_idx]
    flux_per_row_arr = flux_per_row_arr[sort_idx]

    return {
        "wavelengths_um": wavelengths_um.tolist(),
        "flux_per_row": flux_per_row_arr.tolist(),
        "wavelength_range_um": (
            float(wavelengths_um[0]),
            float(wavelengths_um[-1]),
        ),
        "total_flux": float(flux_per_row_arr.sum()),
        "peak_flux": float(flux_per_row_arr.max()),
        "valid_fraction": valid_fraction,
        "shape": list(cut_clean.shape),
    }


# --------------------------------------------------------------------------- #
# Análisis espectral
# --------------------------------------------------------------------------- #

def compute_spectral_slope(
    wavelengths_um: list[float],
    flux: list[float],
) -> float:
    """
    Calcula d log F / d log λ (pendiente en espacio log-log).
    - Retorna 0.0 si hay <3 puntos o todo el flujo es <=0.
    """
    if len(wavelengths_um) < 3 or len(flux) < 3:
        return 0.0

    w = np.array(wavelengths_um, dtype=np.float64)
    f = np.array(flux, dtype=np.float64)

    # Filtrar λ y F positivos
    mask = (w > 0) & (f > 0) & np.isfinite(w) & np.isfinite(f)
    if mask.sum() < 3:
        return 0.0

    log_w = np.log10(w[mask])
    log_f = np.log10(f[mask])

    # Regresión lineal simple
    if log_w.std() < 1e-9:
        return 0.0

    slope = float(np.polyfit(log_w, log_f, 1)[0])
    return slope


def classify_from_spectrum(
    slope: float,
    total_flux: float,
    flux_percentile: float,
    known_objects: list[dict] | None = None,
) -> str:
    """
    Clasificación heurística:
    - Si hay objetos JPL conocidos → ASTEROID (o STAR si el brillo es enorme).
    - Si slope < -0.3 → STAR (azul/caliente).
    - Si slope > 0.2 → ASTEROID (rojizo, típico de asteroides del cinturón).
    - Otro → UNKNOWN.

    flux_percentile: percentil de total_flux del objeto en el campo
                     (0=todo negro, 100=la fuente más brillante).
    """
    if known_objects:
        # Si JPL ya lo tiene catalogado, priorizar eso
        if flux_percentile > 98:
            return "STAR"  # fuente muy brillante y catalogada
        return "ASTEROID"

    if flux_percentile < 30:
        return "UNKNOWN"

    if slope < -0.3:
        return "STAR"
    if slope > 0.2:
        return "ASTEROID"
    return "UNKNOWN"


def estimate_visual_hints(
    classification: str,
    slope: float,
    flux_percentile: float,
) -> dict:
    """
    Devuelve sugerencias visuales para el visor 3D del frontend.
    """
    # Color base: derivado del slope (azul → rojo pasando por blanco)
    # slope < -0.5 → muy azul
    # slope  0    → blanco/gris
    # slope >  0.5 → muy rojo
    if slope < -0.5:
        base_color = "#a8c8ff"
    elif slope < -0.2:
        base_color = "#d8e5ff"
    elif slope < 0.2:
        base_color = "#e0e0e0"
    elif slope < 0.5:
        base_color = "#ffd8b8"
    else:
        base_color = "#e8a070"

    if classification == "STAR":
        return {
            "base_color_hex": base_color,
            "roughness": 0.25,
            "metalness": 0.05,
            "emissive_intensity": 0.85,
            "scale_hint": "large",
            "suggested_3d_model": "star",
        }
    if classification == "ASTEROID":
        return {
            "base_color_hex": base_color,
            "roughness": 0.92,
            "metalness": 0.15,
            "emissive_intensity": 0.0,
            "scale_hint": "small",
            "suggested_3d_model": "rocky",
        }
    # UNKNOWN
    return {
        "base_color_hex": base_color,
        "roughness": 0.6,
        "metalness": 0.3,
        "emissive_intensity": 0.2,
        "scale_hint": "medium",
        "suggested_3d_model": "unknown",
    }


# --------------------------------------------------------------------------- #
# Función pública
# --------------------------------------------------------------------------- #

def analyze_pixel(
    hdul: fits.HDUList,
    pixel_x: int,
    pixel_y: int,
    known_objects: list[dict] | None = None,
    field_flux_percentiles: tuple[float, float, float] | None = None,
) -> dict:
    """
    Analiza el píxel y devuelve un perfil espectral completo.

    field_flux_percentiles: (p50, p90, p99) del flujo del campo, para
        clasificar el píxel como "brillante" o "débil". Si es None, se
        estima rápidamente con el cutout.

    Retorna dict compatible con SpherexProfile (Pydantic).
    """
    spectrum = extract_local_spectrum(hdul, pixel_x, pixel_y)

    if not spectrum["flux_per_row"]:
        return {
            "classification": "UNKNOWN",
            "temperature_k": None,
            "dominant_composition": None,
            "spectral_slope": 0.0,
            "color_index": 0.0,
            "wavelength_range_um": [0.0, 0.0],
            "total_flux": 0.0,
            "valid_fraction": spectrum["valid_fraction"],
            "visual_hints": estimate_visual_hints("UNKNOWN", 0.0, 0.0),
        }

    slope = compute_spectral_slope(
        spectrum["wavelengths_um"], spectrum["flux_per_row"]
    )

    # Percentil aproximado del flujo del píxel dentro del campo.
    # Sin percentiles del campo, usamos el peak del cutout como proxy.
    total_flux = spectrum["total_flux"]
    peak_flux = max(spectrum["peak_flux"], 1e-9)
    flux_percentile = min(100.0, max(0.0, (total_flux / (peak_flux * 100)) * 100))

    classification = classify_from_spectrum(
        slope=slope,
        total_flux=total_flux,
        flux_percentile=flux_percentile,
        known_objects=known_objects,
    )

    # Color index: mapeo del slope a [-1, 1]
    color_index = float(np.clip(-slope, -1.0, 1.0))

    return {
        "classification": classification,
        "temperature_k": None,  # no estimable con un solo detector
        "dominant_composition": None,  # idem
        "spectral_slope": round(slope, 4),
        "color_index": round(color_index, 4),
        "wavelength_range_um": list(spectrum["wavelength_range_um"]),
        "total_flux": round(total_flux, 6),
        "valid_fraction": round(spectrum["valid_fraction"], 3),
        "visual_hints": estimate_visual_hints(
            classification, slope, flux_percentile
        ),
    }
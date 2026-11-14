"""
Análisis espectral local de un píxel de SPHEREx (Level 2, un solo detector).

IMPORTANTE — limitaciones físicas:
  - Un exposure contiene UNA λ por píxel. Con un solo detector, una fuente
    puntual ocupa solo ~5-10 filas → 1-3 nm de rango espectral efectivo.
  - Esto es insuficiente para medir temperatura, composición o espectro real.
  - Solo se puede medir "color": F(λ_larga) / F(λ_corta).
  - La clasificación es un CANDIDATO estadístico, no una identificación.
    STAR_LIKE  → color_ratio < 0.85 (más flujo a λ cortas = más azul)
    ROCKY_LIKE → color_ratio > 1.15 (más flujo a λ largas = más rojo)
    UNKNOWN    → plano, ruido, o rango espectral insuficiente

Pipeline:
  1. Recorte 60×60 alrededor del click.
  2. Fotometría de apertura: flux en radio 5 px, fondo en anillo 8-15 px.
  3. Micro-espectro: apertura deslizante en cada fila con flujo > umbral.
  4. Color ratio: F_long / F_short en las filas extremas de λ.
  5. Significancia del ratio propagando errores (VARIANCE).
  6. Solo clasificar si |ratio - 1| > 1.5σ_ratio.
"""

from __future__ import annotations

import logging

import numpy as np
from astropy.io import fits

from app.services.fits_loader import (
    get_clean_image,
    get_variance,
    get_wavelength_map,
)

logger = logging.getLogger(__name__)

# --- Configuración ---
CUTOUT_SIZE = 60
APERTURE_RADIUS = 5
ANNULUS_INNER = 8
ANNULUS_OUTER = 15
MIN_SNR = 3.0
MIN_SIGMA_RATIO = 1.5
MIN_WAVELENGTH_RANGE_NM = 3.0


# --------------------------------------------------------------------------- #
# 1. Recorte
# --------------------------------------------------------------------------- #

def _extract_cutout(hdul: fits.HDUList, pixel_x: int, pixel_y: int) -> dict:
    """Extrae el recorte de análisis alrededor de un píxel seleccionado.

    Carga la imagen limpia, el mapa de longitudes de onda y la varianza, y los
    recorta al área ``CUTOUT_SIZE`` centrada en la posición indicada. Ajusta el
    recorte cuando está cerca del borde y registra si toca el límite del
    detector.

    Args:
        hdul: Archivo FITS abierto de un detector SPHEREx.
        pixel_x: Coordenada X del píxel seleccionado.
        pixel_y: Coordenada Y del píxel seleccionado.

    Returns:
        Diccionario con ``clean``, ``wave``, ``var``, centro local, forma y
        bandera de borde.

    Raises:
        ValueError: Si el recorte resultante es demasiado pequeño para la
            apertura fotométrica.
    """
    clean = get_clean_image(hdul, subtract_zodi=True)
    wave = get_wavelength_map(hdul, downsample=4)
    var = get_variance(hdul)

    ny, nx = clean.shape
    half = CUTOUT_SIZE // 2
    x0 = max(0, pixel_x - half)
    x1 = min(nx, pixel_x + half)
    y0 = max(0, pixel_y - half)
    y1 = min(ny, pixel_y + half)

    if x1 - x0 < APERTURE_RADIUS * 4 or y1 - y0 < APERTURE_RADIUS * 4:
        raise ValueError(
            f"Cutout demasiado pequeño en ({pixel_x},{pixel_y}). "
            f"Muy cerca del borde del detector ({nx}×{ny})."
        )

    return {
        "clean": clean[y0:y1, x0:x1],
        "wave": wave[y0:y1, x0:x1],
        "var": var[y0:y1, x0:x1] if var is not None else None,
        "center_x": pixel_x - x0,
        "center_y": pixel_y - y0,
        "shape": (y1 - y0, x1 - x0),
        "edge": (x0 == 0) or (y0 == 0) or (x1 == nx) or (y1 == ny),
    }


# --------------------------------------------------------------------------- #
# 2. Fotometría de apertura (centrada en el click)
# --------------------------------------------------------------------------- #

def _aperture_photometry(cut: dict) -> dict:
    """Calcula fotometría de apertura y relación señal/ruido.

    Suma el flujo dentro de una apertura circular, estima el fondo con un
    anillo concéntrico y lo sustrae. El error se obtiene de ``VARIANCE`` cuando
    existe; si no, se estima con la desviación del anillo.

    Args:
        cut: Diccionario producido por ``_extract_cutout``.

    Returns:
        Diccionario con flujo, error, SNR y fracciones válidas de apertura y
        anillo. Si hay menos de tres píxeles válidos en la apertura, las
        medidas fotométricas se devuelven como ``None``.
    """
    clean = cut["clean"]
    var = cut["var"]
    cy, cx = cut["center_y"], cut["center_x"]
    h, w = clean.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    aperture_mask = r2 <= APERTURE_RADIUS ** 2
    annulus_mask = (r2 >= ANNULUS_INNER ** 2) & (r2 <= ANNULUS_OUTER ** 2)

    valid = clean != 0
    ap_valid = aperture_mask & valid
    ann_valid = annulus_mask & valid

    frac_ap = float(ap_valid.sum() / max(aperture_mask.sum(), 1))
    frac_ann = float(ann_valid.sum() / max(annulus_mask.sum(), 1))

    if ap_valid.sum() < 3:
        return {"flux": None, "flux_err": None, "snr": None,
                "valid_fraction_aperture": frac_ap,
                "valid_fraction_annulus": frac_ann}

    bkg = float(np.median(clean[ann_valid])) if ann_valid.sum() >= 3 else 0.0
    flux = float(clean[ap_valid].sum() - bkg * ap_valid.sum())

    if var is not None and np.isfinite(var[ap_valid]).any():
        var_v = np.where(np.isfinite(var[ap_valid]), var[ap_valid], 0.0)
        flux_err = float(np.sqrt(var_v.sum()))
    elif ann_valid.sum() >= 3:
        sigma = float(np.std(clean[ann_valid]))
        flux_err = sigma * np.sqrt(ap_valid.sum())
    else:
        flux_err = None

    snr = float(flux / flux_err) if flux_err and flux_err > 0 else None

    return {"flux": flux, "flux_err": flux_err, "snr": snr,
            "valid_fraction_aperture": frac_ap,
            "valid_fraction_annulus": frac_ann}


# --------------------------------------------------------------------------- #
# 3. Micro-espectro con apertura deslizante
# --------------------------------------------------------------------------- #

def _spectral_cutout(cut: dict) -> dict:
    """
    Construye un micro-espectro recorriendo todas las filas del recorte.

    En cada fila mide la media del flujo en una apertura horizontal centrada en
    X del clic. Calcula la longitud de onda media y, si existe, el error a partir
    de ``VARIANCE``. Omite filas con menos de dos píxeles válidos y ordena los
    puntos finales por longitud de onda.

    Args:
        cut: Diccionario producido por ``_extract_cutout``.

    Returns:
        Diccionario con listas ordenadas de ``wavelengths``, ``fluxes`` y
        ``flux_errs``. Si no hay filas válidas, las tres listas quedan vacías.
    """
    clean = cut["clean"]
    wave = cut["wave"]
    var = cut["var"]
    cy, cx = cut["center_y"], cut["center_x"]
    h, w = clean.shape

    # Perfil del fondo: mediana por fila (fuera de la apertura)
    wavelengths, fluxes, flux_errs = [], [], []

    for row in range(h):
        # Apertura horizontal centrada en X del click
        x_lo = max(0, cx - APERTURE_RADIUS)
        x_hi = min(w, cx + APERTURE_RADIUS + 1)
        ap_slice = slice(x_lo, x_hi)

        row_clean = clean[row, ap_slice]
        row_wave = wave[row, ap_slice]
        row_valid = (row_clean != 0) & np.isfinite(row_clean)
        if row_valid.sum() < 2:
            continue

        f = float(np.mean(row_clean[row_valid]))
        wl = float(np.mean(row_wave[row_valid]))

        # Error
        if var is not None:
            v = var[row, ap_slice]
            v = np.where(np.isfinite(v), v, 0.0)
            ferr = float(np.sqrt(np.mean(v[row_valid])))
        else:
            ferr = 0.0

        wavelengths.append(wl)
        fluxes.append(f)
        flux_errs.append(ferr)

    if not wavelengths:
        return {"wavelengths": [], "fluxes": [], "flux_errs": []}

    order = np.argsort(wavelengths)
    return {
        "wavelengths": [wavelengths[i] for i in order],
        "fluxes": [fluxes[i] for i in order],
        "flux_errs": [flux_errs[i] for i in order],
    }


# --------------------------------------------------------------------------- #
# 4. Color ratio (reemplaza el slope log-log)
# --------------------------------------------------------------------------- #

def _compute_color_ratio(
    wavelengths: list[float],
    fluxes: list[float],
    flux_errs: list[float],
    n_extreme: int = 3,
) -> dict:
    """
    Calcula el índice de color como ``median(F_long) / median(F_short)``.

    ``F_short`` usa las filas con menor longitud de onda y ``F_long`` las de
    mayor longitud de onda. También propaga los errores medianos, informa el
    rango espectral en nanómetros y conserva el número de puntos disponibles.

    Args:
        wavelengths: Longitudes de onda de los puntos del micro-espectro, en μm.
        fluxes: Flujos correspondientes a cada longitud de onda.
        flux_errs: Errores de flujo correspondientes.
        n_extreme: Cantidad de puntos extremos usados por cada lado.

    Returns:
        Diccionario con ratio, error, flujos corto/largo, rango en nm y número
        de puntos. El ratio es ``None`` si faltan puntos o el flujo corto no es
        positivo.
    """
    if len(wavelengths) < 2 * n_extreme:
        return {
            "color_ratio": None, "color_ratio_err": None,
            "f_short": None, "f_long": None,
            "wavelength_range_nm": 0.0, "n_points": 0,
        }

    w = np.array(wavelengths)
    f = np.array(fluxes)
    e = np.array(flux_errs)

    # Ordenar por λ
    order = np.argsort(w)
    w, f, e = w[order], f[order], e[order]

    f_short = float(np.median(f[:n_extreme]))
    f_long = float(np.median(f[-n_extreme:]))
    e_short = float(np.median(e[:n_extreme]))
    e_long = float(np.median(e[-n_extreme:]))

    wavelength_range_nm = (w[-1] - w[0]) * 1000.0
    reliable = wavelength_range_nm >= MIN_WAVELENGTH_RANGE_NM

    if f_short <= 0 or not np.isfinite(f_short):
        return {
            "color_ratio": None, "color_ratio_err": None,
            "f_short": f_short, "f_long": f_long,
            "wavelength_range_nm": wavelength_range_nm,
            "n_points": len(w),
        }

    ratio = f_long / f_short
    # Error relativo propagado: sqrt((e_long/f_long)^2 + (e_short/f_short)^2)
    if f_long > 0:
        rel_err = float(np.sqrt(
            (e_long / f_long) ** 2 + (e_short / f_short) ** 2
        ))
        ratio_err = ratio * rel_err
    else:
        ratio_err = None

    return {
        "color_ratio": float(ratio),
        "color_ratio_err": ratio_err,
        "f_short": f_short,
        "f_long": f_long,
        "wavelength_range_nm": wavelength_range_nm,
        "n_points": len(w),
    }


# --------------------------------------------------------------------------- #
# 5. Clasificación con color_ratio
# --------------------------------------------------------------------------- #

def _classify_color(
    ratio: float | None,
    ratio_err: float | None,
    snr: float | None,
    reliable: bool,
) -> tuple[str, float, list[str]]:
    """Clasifica estadísticamente el candidato usando color y calidad de señal.

    Rechaza resultados con rango espectral insuficiente, SNR bajo o datos
    incompletos. Para resultados válidos compara la desviación del ratio
    respecto a 1 con ``ratio_err`` y devuelve ``STAR_LIKE`` para color azul o
    ``ROCKY_LIKE`` para color rojo.

    Args:
        ratio: Ratio de flujo largo/corto.
        ratio_err: Error estimado del ratio.
        snr: Relación señal/ruido fotométrica.
        reliable: Indica si el rango espectral mínimo está cubierto.

    Returns:
        Tupla con clase, confianza entre 0 y 1 y lista de razones.
    """
    if not reliable:
        return "UNKNOWN", 0.0, ["wavelength_range_too_small"]
    if snr is None or snr < MIN_SNR:
        return "UNKNOWN", 0.0, ["low_snr"]
    if ratio is None:
        return "UNKNOWN", 0.0, ["insufficient_data"]

    # Significancia de |ratio - 1|
    delta = abs(ratio - 1.0)
    if ratio_err is not None and ratio_err > 0:
        significance = delta / ratio_err
    else:
        significance = delta / 0.15  # fallback

    if significance < MIN_SIGMA_RATIO:
        return "UNKNOWN", 0.0, ["color_consistent_with_flat"]

    confidence = float(np.clip((significance - MIN_SIGMA_RATIO) / 3.0, 0.0, 1.0))

    if ratio > 1.0:
        return "ROCKY_LIKE", confidence, ["red_color"]
    return "STAR_LIKE", confidence, ["blue_color"]


# --------------------------------------------------------------------------- #
# 6. Visual hints
# --------------------------------------------------------------------------- #

def _visual_hints(cls: str, ratio: float | None, confidence: float) -> dict:
    """Genera parámetros visuales orientativos para representar el candidato.

    Elige un color base según el ratio y reduce la confianza visual cuando la
    evidencia es débil. Después asigna rugosidad, metalicidad, emisión, escala
    y modelo sugerido según la clase estadística.

    Args:
        cls: Clase ``STAR_LIKE``, ``ROCKY_LIKE`` o ``UNKNOWN``.
        ratio: Ratio de color, si está disponible.
        confidence: Confianza de la clasificación.

    Returns:
        Diccionario de sugerencias de material y modelo 3D.
    """
    if ratio is None or confidence < 0.2:
        base_color = "#c8c8c8"
    elif ratio < 0.7:
        base_color = "#a8c8ff"
    elif ratio < 0.9:
        base_color = "#d8e5ff"
    elif ratio < 1.1:
        base_color = "#e0e0e0"
    elif ratio < 1.4:
        base_color = "#ffd8b8"
    else:
        base_color = "#e8a070"

    if cls == "STAR_LIKE":
        return {"base_color_hex": base_color, "roughness": 0.25,
                "metalness": 0.05, "emissive_intensity": 0.85,
                "scale_hint": "large", "suggested_3d_model": "star"}
    if cls == "ROCKY_LIKE":
        return {"base_color_hex": base_color, "roughness": 0.92,
                "metalness": 0.15, "emissive_intensity": 0.0,
                "scale_hint": "small", "suggested_3d_model": "rocky"}
    return {"base_color_hex": base_color, "roughness": 0.6,
            "metalness": 0.3, "emissive_intensity": 0.15,
            "scale_hint": "medium", "suggested_3d_model": "unknown"}


# --------------------------------------------------------------------------- #
# 7. API pública
# --------------------------------------------------------------------------- #

def analyze_pixel(hdul: fits.HDUList, pixel_x: int, pixel_y: int) -> dict:
    """Ejecuta el análisis fotométrico y espectral completo de un píxel.

    Combina recorte, fotometría de apertura, micro-espectro, ratio de color,
    clasificación estadística e indicaciones visuales. Si el píxel está
    demasiado cerca del borde, devuelve una respuesta vacía con la razón.

    Args:
        hdul: Archivo FITS abierto.
        pixel_x: Coordenada X del píxel seleccionado.
        pixel_y: Coordenada Y del píxel seleccionado.

    Returns:
        Diccionario con clase candidata, confianza, color, flujo, SNR, rango
        espectral, puntos del espectro y sugerencias visuales.
    """
    try:
        cut = _extract_cutout(hdul, pixel_x, pixel_y)
    except ValueError as e:
        logger.info("analyze_pixel cutout failed: %s", e)
        return _empty_result(str(e))

    photo = _aperture_photometry(cut)
    spec = _spectral_cutout(cut)

    color = _compute_color_ratio(
        spec["wavelengths"], spec["fluxes"], spec["flux_errs"]
    )

    cls, confidence, reasons = _classify_color(
        color["color_ratio"],
        color["color_ratio_err"],
        photo["snr"],
        color["wavelength_range_nm"] >= MIN_WAVELENGTH_RANGE_NM,
    )

    # Color index: mapeo lineal de ratio a [-1, +1] centrado en 1.0
    # ratio=0.5 → +1 (azul), ratio=1.0 → 0 (neutro), ratio=2.0 → -1 (rojo)
    color_index = None
    if color["color_ratio"] is not None:
        ci = (1.0 - color["color_ratio"]) * 2.0
        color_index = float(np.clip(ci, -1.0, 1.0))

    if spec["wavelengths"]:
        wmin = float(min(spec["wavelengths"]))
        wmax = float(max(spec["wavelengths"]))
    else:
        wmin = wmax = 0.0

    return {
        "candidate_class": cls,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "color_ratio": round(color["color_ratio"], 4) if color["color_ratio"] is not None else None,
        "color_ratio_error": round(color["color_ratio_err"], 4) if color["color_ratio_err"] is not None else None,
        "color_index": round(color_index, 4) if color_index is not None else None,
        "f_short": round(color["f_short"], 6) if color["f_short"] is not None else None,
        "f_long": round(color["f_long"], 6) if color["f_long"] is not None else None,
        "wavelength_range_um": [wmin, wmax],
        "wavelength_range_nm": round(color["wavelength_range_nm"], 2),
        "n_spectral_points": color["n_points"],
        "reliable": color["wavelength_range_nm"] >= MIN_WAVELENGTH_RANGE_NM,
        "total_flux": round(photo["flux"], 6) if photo["flux"] is not None else None,
        "flux_error": round(photo["flux_err"], 6) if photo["flux_err"] is not None else None,
        "snr": round(photo["snr"], 2) if photo["snr"] is not None else None,
        "valid_fraction_aperture": round(photo["valid_fraction_aperture"], 3),
        "valid_fraction_annulus": round(photo["valid_fraction_annulus"], 3),
        "cutout_edge": cut["edge"],
        "visual_hints": _visual_hints(cls, color["color_ratio"], confidence),
        "spectrum_wavelengths": spec["wavelengths"],
        "spectrum_fluxes": spec["fluxes"],
    }


def _empty_result(reason: str) -> dict:
    """Construye una respuesta estándar cuando no puede completarse el análisis.

    Args:
        reason: Motivo que se incluirá en la lista de razones del resultado.

    Returns:
        Diccionario con la misma estructura que ``analyze_pixel``, pero con
        valores nulos, listas vacías y clase ``UNKNOWN``.
    """
    return {
        "candidate_class": "UNKNOWN",
        "confidence": 0.0,
        "reasons": [reason],
        "color_ratio": None,
        "color_ratio_error": None,
        "color_index": None,
        "f_short": None,
        "f_long": None,
        "wavelength_range_um": [0.0, 0.0],
        "wavelength_range_nm": 0.0,
        "n_spectral_points": 0,
        "reliable": False,
        "total_flux": None,
        "flux_error": None,
        "snr": None,
        "valid_fraction_aperture": 0.0,
        "valid_fraction_annulus": 0.0,
        "cutout_edge": False,
        "visual_hints": _visual_hints("UNKNOWN", None, 0.0),
        "spectrum_wavelengths": [],
        "spectrum_fluxes": [],
    }

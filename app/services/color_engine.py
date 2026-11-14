"""
Motor de color para visualización de imágenes espectrales SPHEREx.

Estrategia "spectral paint":
- Brillo = flujo (IMAGE - ZODI), con stretch arcsinh
- Color = λ del píxel (WCS-WAVE)

Modos de normalización:
- "per_row": cada fila (≈ una λ) se normaliza con sus propios percentiles.
            Elimina el gradiente de fondo instrumental/zodiacal y permite
            ver estructura a TODAS las λ. Recomendado.
- "global":  percentiles globales (versión original). Solo útil si el
            gradiente de fondo es despreciable.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from astropy.io import fits
from PIL import Image
from scipy.ndimage import median_filter

from app.services.fits_loader import (
    DETECTOR_WAVELENGTH_RANGES,
    get_clean_image,
    get_wavelength_map
)

logger = logging.getLogger(__name__)

_COLORMAP_ANCHORS = [
    (0.00, (0.20, 0.00, 0.55)),  # violeta
    (0.20, (0.00, 0.25, 1.00)),  # azul
    (0.40, (0.00, 0.90, 1.00)),  # cian
    (0.60, (0.30, 1.00, 0.00)),  # verde
    (0.80, (1.00, 0.90, 0.00)),  # amarillo
    (1.00, (1.00, 0.15, 0.00)),  # rojo
]
_POSITIONS = np.array([p for p, _ in _COLORMAP_ANCHORS], dtype=np.float32)
_COLORS = np.array([c for _, c in _COLORMAP_ANCHORS], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Stretch y normalización
# --------------------------------------------------------------------------- #

def normalize_percentile(data, p_low=1.0, p_high=99.5):
    """Normaliza una matriz al intervalo ``[0, 1]`` mediante percentiles.

    Ignora valores no finitos y ceros al calcular los límites inferior y
    superior, pero conserva la forma original de la matriz en el resultado.
    Los valores fuera del rango calculado se recortan a ``[0, 1]``. Si no hay
    datos válidos, devuelve una matriz de ceros de tipo ``float32``.

    Args:
        data: Matriz numérica cuyos valores se desean normalizar.
        p_low: Percentil usado como límite inferior.
        p_high: Percentil usado como límite superior.

    Returns:
        Matriz normalizada de tipo ``float32``.
    """
    flat = data[np.isfinite(data) & (data != 0)]
    if flat.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = float(np.percentile(flat, p_low))
    hi = float(np.percentile(flat, p_high))
    span = max(hi - lo, 1e-9)
    return np.clip((data - lo) / span, 0.0, 1.0).astype(np.float32)


def subtract_row_background(data):
    """Resta a cada fila su mediana válida para eliminar el fondo por longitud de onda.

    Los ceros y valores no finitos se excluyen del cálculo de la mediana. Las
    filas sin valores válidos usan una mediana de cero y la salida conserva la
    forma de entrada con tipo ``float32``.

    Args:
        data: Matriz de flujo organizada por filas espectrales.

    Returns:
        Matriz con el fondo estimado por fila sustraído.
    """
    valid = (data != 0) & np.isfinite(data)
    masked = np.where(valid, data, np.nan)
    med = np.nanmedian(masked, axis=1, keepdims=True)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    return (data - med).astype(np.float32)


def row_wise_normalize(data, p_low=1.0, p_high=99.5):
    """Normaliza cada fila independientemente al intervalo ``[0, 1]``.

    Para cada fila calcula sus propios percentiles inferior y superior,
    ignorando ceros y valores no finitos. Los píxeles inválidos se convierten
    en cero y los valores válidos se recortan al intervalo de salida.

    Args:
        data: Matriz de flujo organizada por filas espectrales.
        p_low: Percentil inferior aplicado a cada fila.
        p_high: Percentil superior aplicado a cada fila.

    Returns:
        Matriz normalizada por fila de tipo ``float32``.
    """
    valid = (data != 0) & np.isfinite(data)
    masked = np.where(valid, data, np.nan)
    lo = np.nanpercentile(masked, p_low, axis=1, keepdims=True)
    hi = np.nanpercentile(masked, p_high, axis=1, keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    out = (masked - lo) / span
    out = np.where(valid, np.clip(out, 0.0, 1.0), 0.0)
    return out.astype(np.float32)


def arcsinh_stretch(data_01, scale=10.0):
    """Aplica un estiramiento ``arcsinh`` a datos previamente normalizados.

    El estiramiento realza los valores débiles sin saturar tan rápidamente como
    una transformación lineal. El parámetro de escala se fuerza a ser positivo
    y la salida se limita al intervalo ``[0, 1]``.

    Args:
        data_01: Datos, normalmente en el intervalo ``[0, 1]``.
        scale: Intensidad del estiramiento no lineal.

    Returns:
        Matriz transformada de tipo ``float32``.
    """
    s = max(float(scale), 1e-3)
    out = np.arcsinh(data_01 * s) / np.arcsinh(s)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Colormap λ → RGB
# --------------------------------------------------------------------------- #

def wavelength_colormap(wave_um, wave_min=None, wave_max=None):
    """Convierte longitudes de onda en colores RGB mediante interpolación.

    Normaliza las longitudes de onda al rango indicado y las interpola entre
    anclas de color que van de violeta a rojo. Si el rango no se proporciona,
    lo estima usando los percentiles 1 y 99 de los valores finitos plausibles
    entre 0.1 y 50 micrómetros.

    Args:
        wave_um: Matriz de longitudes de onda expresadas en micrómetros.
        wave_min: Límite inferior opcional del rango cromático.
        wave_max: Límite superior opcional del rango cromático.

    Returns:
        Matriz con una dimensión final de tamaño 3 que contiene canales RGB
        en el intervalo ``[0, 1]``.

    Raises:
        RuntimeError: Si no se puede inferir un rango a partir de valores
            válidos de longitud de onda.
    """
    if wave_min is None or wave_max is None:
        valid = wave_um[np.isfinite(wave_um) & (wave_um > 0.1) & (wave_um < 50.0)]
        if valid.size == 0:
            raise RuntimeError("No valid wavelength values for colormap")
        wave_min = float(np.percentile(valid, 1))
        wave_max = float(np.percentile(valid, 99))

    span = max(wave_max - wave_min, 1e-9)
    t = np.clip((wave_um - wave_min) / span, 0.0, 1.0).astype(np.float32)

    flat = t.ravel()
    r = np.interp(flat, _POSITIONS, _COLORS[:, 0]).astype(np.float32)
    g = np.interp(flat, _POSITIONS, _COLORS[:, 1]).astype(np.float32)
    b = np.interp(flat, _POSITIONS, _COLORS[:, 2]).astype(np.float32)

    return np.stack([r, g, b], axis=-1).reshape(t.shape + (3,))


# --------------------------------------------------------------------------- #
# Fill masked pixels (opcional, cosmético)
# --------------------------------------------------------------------------- #

def fill_masked_pixels(data, size=5):
    """
    Reemplaza píxeles con flujo cero por la mediana de su vecindario.

    La operación está pensada únicamente como corrección cosmética para
    ocultar regiones enmascaradas por FLAGS que, de otro modo, dominarían la
    visualización como círculos negros. Los píxeles distintos de cero se
    conservan sin cambios.

    Args:
        data: Matriz de flujo que puede contener píxeles enmascarados a cero.
        size: Tamaño de la ventana usada por el filtro de mediana.

    Returns:
        Matriz con los ceros reemplazados, de tipo ``float32`` cuando se aplica
        el filtrado; si no hay ceros, devuelve la entrada sin modificar.
    """
    zero_mask = (data == 0)
    if not zero_mask.any():
        return data
    filtered = median_filter(data, size=size)
    return np.where(zero_mask, filtered, data).astype(np.float32)


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #

def spectral_paint(
    hdul: fits.HDUList,
    stretch_scale: float = 10.0,
    p_low: float = 1.0,
    p_high: float = 99.5,
    downsample: int = 8,
    wave_min: float | None = None,
    wave_max: float | None = None,
    subtract_zodi: bool = True,
    mode: str = "per_row",
    fill_masked: bool = True,
) -> Image.Image:
    """
    Genera una imagen RGB coloreando cada píxel según su longitud de onda.

    Obtiene el flujo limpio y el mapa de longitudes de onda desde el HDUList,
    opcionalmente rellena píxeles enmascarados, calcula el brillo mediante
    normalización global o por fila, aplica un estiramiento ``arcsinh`` y
    mezcla el brillo con el color espectral. Finalmente invierte verticalmente
    la imagen para adaptarla a la orientación de visualización y devuelve un
    objeto de imagen PIL.

    Args:
        hdul: Archivo FITS abierto con los datos de imagen y calibración.
        stretch_scale: Escala del estiramiento ``arcsinh`` del brillo.
        p_low: Percentil inferior para la normalización.
        p_high: Percentil superior para la normalización.
        downsample: Factor usado al obtener el mapa de longitudes de onda.
        wave_min: Límite inferior opcional del rango espectral mostrado.
        wave_max: Límite superior opcional del rango espectral mostrado.
        subtract_zodi: Indica si se sustrae la contribución zodiacal.
        mode: ``"per_row"`` para normalización por fila o ``"global"`` para
            normalización sobre toda la imagen.
        fill_masked: Indica si se rellenan cosméticamente los píxeles FLAGS.

    Returns:
        Imagen PIL en modo RGB.

    Raises:
        RuntimeError: Si las dimensiones del mapa de longitudes de onda y de
            la imagen limpia no coinciden.
        ValueError: Si ``mode`` no es ``"per_row"`` ni ``"global"``.
    """
    logger.info("spectral_paint mode=%s fill_masked=%s", mode, fill_masked)

    clean = get_clean_image(hdul, subtract_zodi=subtract_zodi)
    logger.info("  image shape=%s max=%.3g mean=%.3g",
                clean.shape, float(clean.max()), float(clean.mean()))

    wave = get_wavelength_map(hdul, downsample=downsample)
    if wave.shape != clean.shape:
        raise RuntimeError(f"Shape mismatch: {wave.shape} vs {clean.shape}")

    if fill_masked:
        clean = fill_masked_pixels(clean, size=5)

    if mode == "per_row":
        clean = subtract_row_background(clean)
        brightness = row_wise_normalize(clean, p_low=p_low, p_high=p_high)
    elif mode == "global":
        brightness = normalize_percentile(clean, p_low=p_low, p_high=p_high)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'per_row' or 'global'.")

    brightness = arcsinh_stretch(brightness, scale=stretch_scale)
    # Inferir rango λ del detector si no se especificó (consistencia entre épocas)
    if wave_min is None or wave_max is None:
        det = int(hdul["IMAGE"].header.get("DETECTOR", 0))
        if det in DETECTOR_WAVELENGTH_RANGES:
            wave_min, wave_max = DETECTOR_WAVELENGTH_RANGES[det]
    color = wavelength_colormap(wave, wave_min=wave_min, wave_max=wave_max)

    rgb = color * brightness[..., np.newaxis]
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    rgb_u8 = np.flipud(rgb_u8)
    logger.info("spectral_paint done")
    return Image.fromarray(rgb_u8, mode="RGB")


def to_webp_bytes(img: Image.Image, quality: int = 85, method: int = 6) -> bytes:
    """Serializa una imagen PIL al formato WEBP en memoria.

    Args:
        img: Imagen PIL que se desea codificar.
        quality: Calidad de compresión WEBP, según las opciones de PIL.
        method: Nivel de esfuerzo del codificador WEBP.

    Returns:
        Bytes que contienen la imagen codificada en WEBP.
    """
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=method)
    return buf.getvalue()

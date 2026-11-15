"""
Color engine for visualizing SPHEREx spectral images.

"spectral paint" strategy:
- Brightness = flux (IMAGE - ZODI), with arcsinh stretch
- Color = pixel λ (WCS-WAVE)

Normalization modes:
- "per_row": each row (≈ one λ) is normalized using its own percentiles.
            Removes the instrumental/zodiacal background gradient and allows
            structure to be seen at ALL λ. Recommended.
- "global":  global percentiles (original version). Useful only if the
            background gradient is negligible.
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
    (0.00, (0.20, 0.00, 0.55)),  # violet
    (0.20, (0.00, 0.25, 1.00)),  # blue
    (0.40, (0.00, 0.90, 1.00)),  # cyan
    (0.60, (0.30, 1.00, 0.00)),  # green
    (0.80, (1.00, 0.90, 0.00)),  # yellow
    (1.00, (1.00, 0.15, 0.00)),  # red
]
_POSITIONS = np.array([p for p, _ in _COLORMAP_ANCHORS], dtype=np.float32)
_COLORS = np.array([c for _, c in _COLORMAP_ANCHORS], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Stretch and normalization
# --------------------------------------------------------------------------- #

def normalize_percentile(data, p_low=1.0, p_high=99.5):
    """Normalize an array to the ``[0, 1]`` interval using percentiles.

    Ignore non-finite values and zeros when calculating the lower and upper
    bounds, while preserving the original array shape in the result.
    Values outside the calculated range are clipped to ``[0, 1]``. If there
    is no valid data, return an array of zeros with type ``float32``.

    Args:
        data: Numeric array whose values are to be normalized.
        p_low: Percentile used as the lower bound.
        p_high: Percentile used as the upper bound.

    Returns:
        Normalized array of type ``float32``.
    """
    flat = data[np.isfinite(data) & (data != 0)]
    if flat.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = float(np.percentile(flat, p_low))
    hi = float(np.percentile(flat, p_high))
    span = max(hi - lo, 1e-9)
    return np.clip((data - lo) / span, 0.0, 1.0).astype(np.float32)


def subtract_row_background(data):
    """Subtract each row's valid median to remove the wavelength-dependent background.

    Zeros and non-finite values are excluded from the median calculation. Rows
    without valid values use a median of zero, and the output preserves the
    input shape with type ``float32``.

    Args:
        data: Flux array organized by spectral rows.

    Returns:
        Array with the estimated row background subtracted.
    """
    valid = (data != 0) & np.isfinite(data)
    masked = np.where(valid, data, np.nan)
    med = np.nanmedian(masked, axis=1, keepdims=True)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    return (data - med).astype(np.float32)


def row_wise_normalize(data, p_low=1.0, p_high=99.5):
    """Normalize each row independently to the ``[0, 1]`` interval.

    For each row, calculate its own lower and upper percentiles while ignoring
    zeros and non-finite values. Invalid pixels become zero, and valid values
    are clipped to the output interval.

    Args:
        data: Flux array organized by spectral rows.
        p_low: Lower percentile applied to each row.
        p_high: Upper percentile applied to each row.

    Returns:
        Row-normalized array of type ``float32``.
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
    """Apply an ``arcsinh`` stretch to previously normalized data.

    The stretch enhances faint values without saturating as quickly as a
    linear transformation. The scale parameter is forced to be positive, and
    the output is clipped to the ``[0, 1]`` interval.

    Args:
        data_01: Data, typically in the ``[0, 1]`` interval.
        scale: Intensity of the nonlinear stretch.

    Returns:
        Transformed array of type ``float32``.
    """
    s = max(float(scale), 1e-3)
    out = np.arcsinh(data_01 * s) / np.arcsinh(s)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Colormap λ → RGB
# --------------------------------------------------------------------------- #

def wavelength_colormap(wave_um, wave_min=None, wave_max=None):
    """Convert wavelengths to RGB colors by interpolation.

    Normalize wavelengths to the specified range and interpolate them between
    color anchors ranging from violet to red. If the range is not provided,
    estimate it using the 1st and 99th percentiles of plausible finite values
    between 0.1 and 50 micrometers.

    Args:
        wave_um: Array of wavelengths expressed in micrometers.
        wave_min: Optional lower bound of the color range.
        wave_max: Optional upper bound of the color range.

    Returns:
        Array with a final dimension of size 3 containing RGB channels
        in the ``[0, 1]`` interval.

    Raises:
        RuntimeError: If a range cannot be inferred from valid wavelength
            values.
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
# Fill masked pixels (optional, cosmetic)
# --------------------------------------------------------------------------- #

def fill_masked_pixels(data, size=5):
    """
    Replace zero-flux pixels with the median of their neighborhood.

    This operation is intended solely as a cosmetic correction to hide regions
    masked by FLAGS that would otherwise dominate the visualization as black
    circles. Nonzero pixels are preserved unchanged.

    Args:
        data: Flux array that may contain pixels masked to zero.
        size: Size of the window used by the median filter.

    Returns:
        Array with zeros replaced, of type ``float32`` when filtering is
        applied; if there are no zeros, return the input unchanged.
    """
    zero_mask = (data == 0)
    if not zero_mask.any():
        return data
    filtered = median_filter(data, size=size)
    return np.where(zero_mask, filtered, data).astype(np.float32)


# --------------------------------------------------------------------------- #
# Main pipeline
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
    Generate an RGB image coloring each pixel according to its wavelength.

    Obtain the clean flux and wavelength map from the HDUList, optionally fill
    masked pixels, calculate brightness using global or row-wise normalization,
    apply an ``arcsinh`` stretch, and blend brightness with spectral color.
    Finally, flip the image vertically to match the display orientation and
    return a PIL image object.

    Args:
        hdul: Open FITS file with image and calibration data.
        stretch_scale: Scale of the brightness ``arcsinh`` stretch.
        p_low: Lower percentile for normalization.
        p_high: Upper percentile for normalization.
        downsample: Factor used when obtaining the wavelength map.
        wave_min: Optional lower bound of the displayed spectral range.
        wave_max: Optional upper bound of the displayed spectral range.
        subtract_zodi: Whether to subtract the zodiacal contribution.
        mode: ``"per_row"`` for row-wise normalization or ``"global"`` for
            normalization over the entire image.
        fill_masked: Whether to cosmetically fill FLAGS pixels.

    Returns:
        PIL image in RGB mode.

    Raises:
        RuntimeError: If the dimensions of the wavelength map and clean image
            do not match.
        ValueError: If ``mode`` is neither ``"per_row"`` nor ``"global"``.
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
    # Infer detector λ range if not specified (consistency across epochs)
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
    """Serialize a PIL image to WEBP format in memory.

    Args:
        img: PIL image to encode.
        quality: WEBP compression quality, according to PIL options.
        method: WEBP encoder effort level.

    Returns:
        Bytes containing the encoded WEBP image.
    """
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=method)
    return buf.getvalue()

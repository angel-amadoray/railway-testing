"""
Difference engine — Phase 4 (v3, consolidated).

Fixes compared to v2:
- Fast 2D background (downsampling + small median + Gaussian + upsampling)
- Background is applied AFTER reprojection → row residuals cancel out
- Local noise via blocks (SExtractor-style), O(N)
- Defaults = v1 (absolute + row_background + percentile 99) to preserve
  sensitivity; v2 features available as opt-in
- mask_stars affects threshold calculation (not rendering by default)
- New: mask_stars_display to hide donuts in the final image
"""

from __future__ import annotations

import io
import logging

import numpy as np
from astropy.io import fits
from PIL import Image
from reproject import reproject_interp
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    median_filter,
    zoom,
)

from app.services.color_engine import subtract_row_background
from app.services.fits_loader import get_clean_image, get_spatial_wcs

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Fast 2D background
# --------------------------------------------------------------------------- #

def estimate_background_2d_fast(
    data: np.ndarray,
    downsample: int = 8,
    kernel: int = 7,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Efficiently estimate the smooth two-dimensional background of an image.

    The algorithm first reduces the resolution, applies a median filter to
    remove point sources, smooths the result with a Gaussian filter, and
    finally interpolates it back to the original size. Zero or non-finite
    pixels are excluded from the estimate and replaced with the global median
    only during background calculation.

    Args:
        data: Two-dimensional data array whose shape will be preserved.
        downsample: Downsampling step used on both axes.
        kernel: Median filter window size in the reduced image.
        sigma: Standard deviation of the Gaussian smoothing.

    Returns:
        ``float32`` array with the estimated background and the same size as
        ``data``. If there is no valid data, return an array of zeros.
    """
    h, w = data.shape
    valid = np.isfinite(data) & (data != 0)

    small = data[::downsample, ::downsample]
    small_valid = valid[::downsample, ::downsample]

    if not small_valid.any():
        return np.zeros_like(data, dtype=np.float32)

    fill = float(np.median(data[valid])) if valid.any() else 0.0
    small_filled = np.where(small_valid, small, fill)

    bg_small = median_filter(small_filled, size=kernel)
    bg_small = gaussian_filter(bg_small, sigma=sigma)

    bg = zoom(bg_small, zoom=downsample, order=1)
    bg = bg[:h, :w]
    if bg.shape != data.shape:
        padded = np.zeros_like(data, dtype=np.float32)
        padded[: bg.shape[0], : bg.shape[1]] = bg
        bg = padded

    return bg.astype(np.float32)


def flatten_2d(data: np.ndarray, downsample: int = 8) -> np.ndarray:
    """Remove the estimated 2D background from an image.

    Args:
        data: Two-dimensional signal array.
        downsample: Reduction factor used when estimating the background.

    Returns:
        ``data - background`` residual as a ``float32`` array.
    """
    return (data - estimate_background_2d_fast(data, downsample=downsample)).astype(np.float32)


# --------------------------------------------------------------------------- #
# WCS alignment
# --------------------------------------------------------------------------- #

def align_to_reference(
    src_data: np.ndarray,
    src_wcs,
    ref_wcs,
    shape_out: tuple[int, int],
    order: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    """Align an image by reprojecting it to the reference WCS system.

    Args:
        src_data: Source image to reproject.
        src_wcs: WCS associated with ``src_data``.
        ref_wcs: WCS of the output grid.
        shape_out: Height and width of the resulting image.
        order: Interpolation used by ``reproject_interp``.

    Returns:
        ``(aligned, footprint)`` tuple with the aligned image and the coverage
        fraction for each pixel. Non-finite image values are set to zero.
    """
    aligned, footprint = reproject_interp(
        (src_data, src_wcs),
        ref_wcs,
        shape_out=shape_out,
        order=order,
        return_footprint=True,
    )
    aligned = np.asarray(aligned, dtype=np.float32)
    footprint = np.asarray(footprint, dtype=np.float32)
    aligned[~np.isfinite(aligned)] = 0.0
    return aligned, footprint


# --------------------------------------------------------------------------- #
# Fast local noise (SExtractor-style blocks)
# --------------------------------------------------------------------------- #

def estimate_local_noise_blocks(diff: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Estimate local noise in a difference using blocks and MAD.

    Divide the image into blocks, calculate the median absolute deviation (MAD)
    in each, and convert it to a sigma estimate using the 1.4826 factor. Then
    interpolate the block map to the original size and ensure a positive
    minimum value. For images smaller than one block, use a global estimate.

    Args:
        diff: Difference array.
        block_size: Size of each square block.

    Returns:
        ``float32`` local-noise map with the same shape as ``diff``.
    """
    h, w = diff.shape
    ny, nx = h // block_size, w // block_size
    if ny == 0 or nx == 0:
        # Image smaller than one block: use global estimate
        valid = diff[np.isfinite(diff) & (diff != 0)]
        sigma = 1.4826 * float(np.median(np.abs(valid - np.median(valid)))) if valid.size else 1.0
        return np.full_like(diff, max(sigma, 1e-6), dtype=np.float32)

    noise_map = np.full((ny, nx), np.nan, dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            block = diff[j * block_size:(j + 1) * block_size, i * block_size:(i + 1) * block_size]
            v = block[np.isfinite(block) & (block != 0)]
            if v.size < 20:
                continue
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            noise_map[j, i] = 1.4826 * mad

    if np.all(np.isnan(noise_map)):
        return np.full_like(diff, 1e-6, dtype=np.float32)
    # Fill NaN blocks with the global median
    fill_val = float(np.nanmedian(noise_map))
    noise_map = np.where(np.isnan(noise_map), fill_val, noise_map)

    noise_full = zoom(noise_map, zoom=block_size, order=1)
    noise_full = noise_full[:h, :w]
    if noise_full.shape != diff.shape:
        padded = np.zeros_like(diff)
        padded[: noise_full.shape[0], : noise_full.shape[1]] = noise_full
        noise_full = padded

    return np.maximum(noise_full, 1e-6).astype(np.float32)


# --------------------------------------------------------------------------- #
# Bright-star mask
# --------------------------------------------------------------------------- #

def bright_star_mask(
    t1: np.ndarray,
    t2_aligned: np.ndarray,
    percentile: float = 99.9,
    dilate_iter: int = 2,
) -> np.ndarray:
    """Detect and dilate a mask of the brightest sources.

    Combine the absolute values of both images, obtain a percentile-based
    threshold, and include pixels exceeding it. Optional dilation covers the
    immediate surroundings of each star.

    Args:
        t1: Already-corrected reference image.
        t2_aligned: Second image aligned with ``t1``.
        percentile: Percentile used to define extreme brightness.
        dilate_iter: Number of binary dilation iterations.

    Returns:
        Boolean mask with the shape of the images. If there is no positive
        signal, return an all-False mask.
    """
    valid = (t1 != 0) | (t2_aligned != 0)
    combined = np.maximum(np.abs(t1), np.abs(t2_aligned))
    combined = np.where(valid, combined, 0.0)

    if not (combined > 0).any():
        return np.zeros_like(t1, dtype=bool)

    thresh = float(np.percentile(combined[combined > 0], percentile))
    mask = combined >= thresh
    if dilate_iter > 0 and mask.any():
        mask = binary_dilation(mask, iterations=dilate_iter)
    return mask


# --------------------------------------------------------------------------- #
# Difference and threshold
# --------------------------------------------------------------------------- #

def compute_signed_difference(
    t1: np.ndarray,
    t2_aligned: np.ndarray,
    footprint: np.ndarray,
    min_coverage: float = 0.5,
    ratio: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """Calculate the difference between two images and discard low coverage.

    With ``ratio=False``, calculate ``t2_aligned - t1``. With ``ratio=True``,
    calculate the relative change ``(t2_aligned + eps) / (t1 + eps) - 1``.
    Keep only pixels whose ``footprint`` reaches ``min_coverage``.

    Args:
        t1: Reference image.
        t2_aligned: Second image on the same grid.
        footprint: Per-pixel reprojection coverage.
        min_coverage: Minimum accepted coverage.
        ratio: Select relative rather than absolute difference.
        eps: Constant to avoid division by zero.

    Returns:
        ``float32`` difference with out-of-coverage pixels set to zero.
    """
    if ratio:
        diff = (t2_aligned + eps) / (t1 + eps) - 1.0
    else:
        diff = t2_aligned - t1
    valid = footprint >= min_coverage
    return np.where(valid, diff, 0.0).astype(np.float32)


def compute_threshold(
    signal: np.ndarray,
    exclude_mask: np.ndarray | None = None,
    method: str = "percentile",
    value: float | None = None,
    percentile: float = 99.0,
    sigma_k: float = 5.0,
) -> float:
    """Calculate the threshold used to separate significant changes from noise.

    The ``absolute`` method returns ``value``; ``percentile`` uses the percentile
    of the signal's absolute value; and ``sigma`` uses the median plus ``sigma_k``
    times a robust sigma derived from MAD. ``exclude_mask`` omits regions such
    as bright stars from the calculation.

    Args:
        signal: Absolute or noise-normalized signal.
        exclude_mask: Optional mask of pixels to exclude.
        method: ``absolute``, ``percentile``, or ``sigma``.
        value: Direct threshold required by ``absolute``.
        percentile: Percentile used by ``percentile``.
        sigma_k: Sigma multiplier used by ``sigma``.

    Returns:
        Numeric threshold as ``float``; return zero if there is no valid data.

    Raises:
        ValueError: If ``value`` is missing for ``absolute`` or the method is unknown.
    """
    if method == "absolute":
        if value is None:
            raise ValueError("method='absolute' requires value")
        return float(value)

    mask = np.isfinite(signal) & (signal != 0)
    if exclude_mask is not None:
        mask &= ~exclude_mask
    flat = signal[mask]
    if flat.size == 0:
        return 0.0

    if method == "percentile":
        return float(np.percentile(np.abs(flat), percentile))

    if method == "sigma":
        median = float(np.median(flat))
        mad = float(np.median(np.abs(flat - median)))
        sigma = 1.4826 * mad
        return float(median + sigma_k * sigma)

    raise ValueError(f"Unknown method: {method!r}")


def apply_threshold(diff: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a threshold to a difference array.

    Preserve the sign and original value of pixels whose absolute value is
    greater than or equal to ``threshold``; replace the rest with zero.

    Args:
        diff: Difference array.
        threshold: Minimum magnitude a difference must exceed.

    Returns:
        ``float32`` array with insignificant differences removed.
    """
    return np.where(np.abs(diff) >= threshold, diff, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

def render_heatmap(
    diff_masked: np.ndarray,
    style: str = "signed",
    p_high: float = 99.9,
) -> Image.Image:
    """Render a difference as an RGB image oriented for display.

    The ``signed`` style represents positive changes in red and negative changes
    in green. The ``absolute`` style represents magnitude using an intensity
    scale. The scale is obtained from the upper percentile of nonzero values,
    and the image is flipped vertically before creating the PIL object.

    Args:
        diff_masked: Difference with discarded pixels set to zero.
        style: ``signed`` or ``absolute``.
        p_high: Percentile used to calculate the maximum scale.

    Returns:
        PIL image in RGB mode. If there is no valid signal, return a black image.

    Raises:
        ValueError: If ``style`` is neither ``signed`` nor ``absolute``.
    """
    flat = diff_masked[np.isfinite(diff_masked) & (diff_masked != 0)]
    if flat.size == 0:
        return Image.fromarray(np.zeros((*diff_masked.shape, 3), dtype=np.uint8), mode="RGB")

    scale = max(float(np.percentile(np.abs(flat), p_high)), 1e-9)

    if style == "signed":
        norm = diff_masked / scale
        r = np.clip(norm, 0.0, 1.0)
        g = np.clip(-norm, 0.0, 1.0)
        b = np.zeros_like(r)
    elif style == "absolute":
        norm = np.clip(np.abs(diff_masked) / scale, 0.0, 1.0)
        r = norm
        g = norm
        b = np.clip(norm - 0.5, 0.0, 1.0) * 2.0
    else:
        raise ValueError(f"Unknown style: {style!r}")

    rgb = np.stack([r, g, b], axis=-1)
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    rgb_u8 = np.flipud(rgb_u8)
    return Image.fromarray(rgb_u8, mode="RGB")


# --------------------------------------------------------------------------- #
# Complete pipeline (v3)
# --------------------------------------------------------------------------- #

def compute_difference(
    hdul_t1: fits.HDUList,
    hdul_t2: fits.HDUList,
    # Background
    background_mode: str = "row",       # "row" (v1, fast) | "2d" (v3, clean) | "none"
    background_downsample: int = 8,
    # Difference
    ratio: bool = False,
    # Masks
    mask_stars_for_threshold: bool = True,
    mask_stars_for_display: bool = False,
    star_percentile: float = 99.9,
    # Normalization
    normalize: str = "absolute",        # "absolute" (v1) | "snr" (v2)
    noise_block_size: int = 64,
    # Threshold
    threshold_method: str = "percentile",
    threshold_value: float | None = None,
    threshold_percentile: float = 99.0,
    sigma_k: float = 5.0,
    # Miscellaneous
    min_coverage: float = 0.5,
    style: str = "signed",
    downsample: int = 1,
) -> dict:
    """Compare two observations and generate a visualization of their differences.

    Clean both images, reproject the second onto the first, correct the selected
    background, calculate the difference, optionally estimate local noise,
    exclude stars from threshold calculation, and render changes exceeding the
    threshold. Also return coverage metrics, counts, and configuration used.

    Args:
        hdul_t1: FITS of the reference observation.
        hdul_t2: FITS of the observation to align with the reference.
        background_mode: ``row``, ``2d``, or ``none``.
        background_downsample: Downsampling for the 2D background.
        ratio: Use relative differences when ``True``.
        mask_stars_for_threshold: Exclude stars from threshold calculation.
        mask_stars_for_display: Also hide stars in the final image.
        star_percentile: Percentile for detecting bright stars.
        normalize: ``absolute`` uses direct difference; ``snr`` divides it
            by local noise.
        noise_block_size: Block size for estimating local noise.
        threshold_method: Threshold calculation method.
        threshold_value: Explicit value for the ``absolute`` method.
        threshold_percentile: Percentile for the ``percentile`` method.
        sigma_k: Sigma factor for the ``sigma`` method.
        min_coverage: Minimum coverage required after reprojection.
        style: Heatmap rendering style.
        downsample: Final reduction applied before rendering.

    Returns:
        Dictionary with ``image``, threshold, counts, and analysis metadata.

    Raises:
        ValueError: If coverage is insufficient or an invalid option is used.
    """
    logger.info(
        "diff_engine v3: bg=%s normalize=%s ratio=%s mask_thr=%s mask_disp=%s",
        background_mode, normalize, ratio, mask_stars_for_threshold, mask_stars_for_display,
    )

    clean_t1 = get_clean_image(hdul_t1, subtract_zodi=True)
    clean_t2 = get_clean_image(hdul_t2, subtract_zodi=True)

    wcs_t1 = get_spatial_wcs(hdul_t1)
    wcs_t2 = get_spatial_wcs(hdul_t2)

    # 1) Reproject FIRST (raw, without background)
    logger.info("  reprojecting T2 → T1 (%.0fx%.0f)", *clean_t1.shape)
    aligned_t2, footprint = align_to_reference(
        clean_t2, wcs_t2, wcs_t1, shape_out=clean_t1.shape
    )
    coverage = float((footprint >= min_coverage).mean())
    logger.info("  coverage=%.1f%%", coverage * 100)

    if coverage < 0.05:
        raise ValueError(
            f"T2 cubre solo {coverage*100:.1f}% del campo de T1. "
            "¿Estás comparando la misma región del cielo?"
        )

    # 2) Background NOW, on the same pixel grid
    if background_mode == "none":
        flat_t1, flat_t2 = clean_t1, aligned_t2
    elif background_mode == "row":
        logger.info("  subtract_row_background (fast)")
        flat_t1 = subtract_row_background(clean_t1)
        flat_t2 = subtract_row_background(aligned_t2)
    elif background_mode == "2d":
        logger.info("  background 2D (downsample=%d)", background_downsample)
        flat_t1 = flatten_2d(clean_t1, downsample=background_downsample)
        flat_t2 = flatten_2d(aligned_t2, downsample=background_downsample)
    else:
        raise ValueError(f"background_mode must be row|2d|none, got {background_mode!r}")

    # 3) Difference
    signed_diff = compute_signed_difference(
        flat_t1, flat_t2, footprint, min_coverage=min_coverage, ratio=ratio,
    )

    # 4) Star mask
    star_mask = None
    if mask_stars_for_threshold or mask_stars_for_display:
        star_mask = bright_star_mask(flat_t1, flat_t2, percentile=star_percentile)
        logger.info("  star mask: %.2f%% pixels", float(star_mask.mean()) * 100)

    # 5) Signal for the threshold
    if normalize == "snr":
        noise = estimate_local_noise_blocks(signed_diff, block_size=noise_block_size)
        signal = (signed_diff / noise).astype(np.float32)
    elif normalize == "absolute":
        signal = signed_diff
    else:
        raise ValueError(f"normalize must be absolute|snr, got {normalize!r}")

    # 6) Threshold
    threshold = compute_threshold(
        signal,
        exclude_mask=star_mask if mask_stars_for_threshold else None,
        method=threshold_method,
        value=threshold_value,
        percentile=threshold_percentile,
        sigma_k=sigma_k,
    )
    logger.info("  threshold = %.5g (normalize=%s)", threshold, normalize)

    # 7) Apply threshold in `signal` space, display the actual difference
    keep = np.abs(signal) >= threshold
    display = np.where(keep, signed_diff, 0.0).astype(np.float32)

    # 8) Optional: hide residual stars from the render
    if mask_stars_for_display and star_mask is not None:
        display = np.where(star_mask, 0.0, display).astype(np.float32)

    n_above = int((display != 0).sum())
    n_total = int((footprint >= min_coverage).sum())

    if downsample > 1:
        display = display[::downsample, ::downsample]

    img = render_heatmap(display, style=style)

    return {
        "image": img,
        "threshold": threshold,
        "normalize": normalize,
        "n_pixels_above": n_above,
        "n_pixels_total": n_total,
        "metadata": {
            "coverage_fraction": coverage,
            "shape_t1": list(clean_t1.shape),
            "shape_t2": list(clean_t2.shape),
            "diff_min": float(signed_diff.min()),
            "diff_max": float(signed_diff.max()),
            "background_mode": background_mode,
            "ratio": ratio,
            "normalize": normalize,
            "mask_stars_threshold": mask_stars_for_threshold,
            "mask_stars_display": mask_stars_for_display,
            "star_pixel_fraction": float(star_mask.mean()) if star_mask is not None else 0.0,
        },
    }


def to_webp_bytes(img: Image.Image, quality: int = 85, method: int = 6) -> bytes:
    """Encode a PIL image as WEBP without creating a temporary file.

    Args:
        img: Image to serialize.
        quality: Requested compression quality.
        method: WEBP encoder effort level.

    Returns:
        Byte sequence containing the encoded WEBP image.
    """
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=method)
    return buf.getvalue()

"""
Motor de diferencias — Fase 4 (v3, consolidada).

Fixes respecto a v2:
- Background 2D rápido (submuestreo + mediana pequeña + gaussian + upsample)
- Background se aplica DESPUÉS de reproject → residuos por fila se cancelan
- Ruido local vía bloques (SExtractor-style), O(N)
- Defaults = v1 (absolute + row_background + percentile 99) para no perder
  sensibilidad; features de v2 disponibles como opt-in
- mask_stars afecta al cálculo del umbral (no al render por defecto)
- Nuevo: mask_stars_display para ocultar los donuts en la imagen final
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
# Background 2D rápido
# --------------------------------------------------------------------------- #

def estimate_background_2d_fast(
    data: np.ndarray,
    downsample: int = 8,
    kernel: int = 7,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Background 2D rápido:
      1. Submuestrear por `downsample`
      2. Mediana de `kernel×kernel` (equivalente a ~(kernel*downsample)² en original)
      3. Suavizado gaussiano
      4. Upsample bilineal

    O(N) en la práctica. ~150× más rápido que median_filter(51) a full res.
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
    return (data - estimate_background_2d_fast(data, downsample=downsample)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Alineación WCS
# --------------------------------------------------------------------------- #

def align_to_reference(
    src_data: np.ndarray,
    src_wcs,
    ref_wcs,
    shape_out: tuple[int, int],
    order: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray]:
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
# Ruido local rápido (bloques estilo SExtractor)
# --------------------------------------------------------------------------- #

def estimate_local_noise_blocks(diff: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Estima σ del ruido por bloques de `block_size` usando MAD.
    O(N) con un loop pequeño sobre bloques. ~50× más rápido que median_filter.
    """
    h, w = diff.shape
    ny, nx = h // block_size, w // block_size
    if ny == 0 or nx == 0:
        # Imagen más pequeña que un bloque: usar estimación global
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

    # Rellenar bloques NaN con la mediana global
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
# Máscara de estrellas brillantes
# --------------------------------------------------------------------------- #

def bright_star_mask(
    t1: np.ndarray,
    t2_aligned: np.ndarray,
    percentile: float = 99.9,
    dilate_iter: int = 2,
) -> np.ndarray:
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
# Diferencia y threshold
# --------------------------------------------------------------------------- #

def compute_signed_difference(
    t1: np.ndarray,
    t2_aligned: np.ndarray,
    footprint: np.ndarray,
    min_coverage: float = 0.5,
    ratio: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
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
    return np.where(np.abs(diff) >= threshold, diff, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

def render_heatmap(
    diff_masked: np.ndarray,
    style: str = "signed",
    p_high: float = 99.9,
) -> Image.Image:
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
    return Image.fromarray(rgb_u8, mode="RGB")


# --------------------------------------------------------------------------- #
# Pipeline completo (v3)
# --------------------------------------------------------------------------- #

def compute_difference(
    hdul_t1: fits.HDUList,
    hdul_t2: fits.HDUList,
    # Background
    background_mode: str = "row",       # "row" (v1, rápido) | "2d" (v3, limpio) | "none"
    background_downsample: int = 8,
    # Diferencia
    ratio: bool = False,
    # Máscaras
    mask_stars_for_threshold: bool = True,
    mask_stars_for_display: bool = False,
    star_percentile: float = 99.9,
    # Normalización
    normalize: str = "absolute",        # "absolute" (v1) | "snr" (v2)
    noise_block_size: int = 64,
    # Threshold
    threshold_method: str = "percentile",
    threshold_value: float | None = None,
    threshold_percentile: float = 99.0,
    sigma_k: float = 5.0,
    # Resto
    min_coverage: float = 0.5,
    style: str = "signed",
    downsample: int = 1,
) -> dict:
    logger.info(
        "diff_engine v3: bg=%s normalize=%s ratio=%s mask_thr=%s mask_disp=%s",
        background_mode, normalize, ratio, mask_stars_for_threshold, mask_stars_for_display,
    )

    clean_t1 = get_clean_image(hdul_t1, subtract_zodi=True)
    clean_t2 = get_clean_image(hdul_t2, subtract_zodi=True)

    wcs_t1 = get_spatial_wcs(hdul_t1)
    wcs_t2 = get_spatial_wcs(hdul_t2)

    # 1) Reproyectar PRIMERO (en bruto, sin background)
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

    # 2) Background AHORA, en el mismo grid de píxeles
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

    # 3) Diferencia
    signed_diff = compute_signed_difference(
        flat_t1, flat_t2, footprint, min_coverage=min_coverage, ratio=ratio,
    )

    # 4) Máscara de estrellas
    star_mask = None
    if mask_stars_for_threshold or mask_stars_for_display:
        star_mask = bright_star_mask(flat_t1, flat_t2, percentile=star_percentile)
        logger.info("  star mask: %.2f%% pixels", float(star_mask.mean()) * 100)

    # 5) Señal para el umbral
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

    # 7) Aplicar threshold en el espacio de `signal`, mostrar en diff real
    keep = np.abs(signal) >= threshold
    display = np.where(keep, signed_diff, 0.0).astype(np.float32)

    # 8) Opcional: ocultar estrellas residuales del render
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
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=method)
    return buf.getvalue()
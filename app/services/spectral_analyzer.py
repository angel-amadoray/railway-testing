"""
Local spectral analysis of a SPHEREx pixel (Level 2, single detector).

IMPORTANT — physical limitations:
  - An exposure contains ONE λ per pixel. With a single detector, a point
    source occupies only ~5-10 rows → 1-3 nm of effective spectral range.
  - This is insufficient to measure temperature, composition, or a true spectrum.
  - Only "color" can be measured: F(λ_long) / F(λ_short).
  - The classification is a statistical CANDIDATE, not an identification.
    STAR_LIKE  → color_ratio < 0.85 (more flux at short λ = bluer)
    ROCKY_LIKE → color_ratio > 1.15 (more flux at long λ = redder)
    UNKNOWN    → flat, noisy, or insufficient spectral range

Pipeline:
  1. 60×60 cutout around the click.
  2. Aperture photometry: flux within a 5 px radius, background in an 8-15 px annulus.
  3. Micro-spectrum: sliding aperture on each row with flux above the threshold.
  4. Color ratio: F_long / F_short on the extreme λ rows.
  5. Ratio significance with propagated errors (VARIANCE).
  6. Classify only if |ratio - 1| > 1.5σ_ratio.
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

# --- Configuration ---
CUTOUT_SIZE = 60
APERTURE_RADIUS = 5
ANNULUS_INNER = 8
ANNULUS_OUTER = 15
MIN_SNR = 3.0
MIN_SIGMA_RATIO = 1.5
MIN_WAVELENGTH_RANGE_NM = 3.0


# --------------------------------------------------------------------------- #
# 1. Cutout
# --------------------------------------------------------------------------- #

def _extract_cutout(hdul: fits.HDUList, pixel_x: int, pixel_y: int) -> dict:
    """Extract the analysis cutout around a selected pixel.

    Load the clean image, wavelength map, and variance, and crop them to the
    ``CUTOUT_SIZE`` area centered on the indicated position. Adjust the cutout
    near the edge and record whether it touches the detector boundary.

    Args:
        hdul: Open FITS file from a SPHEREx detector.
        pixel_x: X coordinate of the selected pixel.
        pixel_y: Y coordinate of the selected pixel.

    Returns:
        Dictionary with ``clean``, ``wave``, ``var``, local center, shape, and
        edge flag.

    Raises:
        ValueError: If the resulting cutout is too small for aperture photometry.
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
# 2. Aperture photometry (centered on the click)
# --------------------------------------------------------------------------- #

def _aperture_photometry(cut: dict) -> dict:
    """Calculate aperture photometry and signal-to-noise ratio.

    Sum the flux within a circular aperture, estimate the background with a
    concentric annulus, and subtract it. Obtain the error from ``VARIANCE``
    when available; otherwise estimate it from the annulus deviation.

    Args:
        cut: Dictionary produced by ``_extract_cutout``.

    Returns:
        Dictionary with flux, error, SNR, and valid aperture and annulus
        fractions. If fewer than three valid pixels exist in the aperture,
        photometric measurements are returned as ``None``.
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
# 3. Micro-spectrum with sliding aperture
# --------------------------------------------------------------------------- #

def _spectral_cutout(cut: dict) -> dict:
    """
    Build a micro-spectrum by traversing all rows of the cutout.

    On each row, measure the mean flux in a horizontal aperture centered on the
    click's X coordinate. Calculate the mean wavelength and, when available,
    the error from ``VARIANCE``. Skip rows with fewer than two valid pixels and
    sort the final points by wavelength.

    Args:
        cut: Dictionary produced by ``_extract_cutout``.

    Returns:
        Dictionary with sorted lists of ``wavelengths``, ``fluxes``, and
        ``flux_errs``. If no valid rows exist, all three lists are empty.
    """
    clean = cut["clean"]
    wave = cut["wave"]
    var = cut["var"]
    cy, cx = cut["center_y"], cut["center_x"]
    h, w = clean.shape

    # Background profile: median per row (outside the aperture)
    wavelengths, fluxes, flux_errs = [], [], []

    for row in range(h):
        # Horizontal aperture centered on the click's X coordinate
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
    # 4. Color ratio (replaces the log-log slope)
# --------------------------------------------------------------------------- #

def _compute_color_ratio(
    wavelengths: list[float],
    fluxes: list[float],
    flux_errs: list[float],
    n_extreme: int = 3,
) -> dict:
    """
    Calculate the color index as ``median(F_long) / median(F_short)``.

    ``F_short`` uses the rows with the shortest wavelengths and ``F_long`` those
    with the longest wavelengths. Also propagate median errors, report the
    spectral range in nanometers, and preserve the number of available points.

    Args:
        wavelengths: Wavelengths of the micro-spectrum points, in μm.
        fluxes: Fluxes corresponding to each wavelength.
        flux_errs: Corresponding flux errors.
        n_extreme: Number of extreme points used on each side.

    Returns:
        Dictionary with ratio, error, short/long fluxes, range in nm, and point
        count. The ratio is ``None`` if points are missing or short flux is not
        positive.
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

    # Sort by λ
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
    # Propagated relative error: sqrt((e_long/f_long)^2 + (e_short/f_short)^2)
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
# 5. Classification with color_ratio
# --------------------------------------------------------------------------- #

def _classify_color(
    ratio: float | None,
    ratio_err: float | None,
    snr: float | None,
    reliable: bool,
) -> tuple[str, float, list[str]]:
    """Statistically classify the candidate using color and signal quality.

    Reject results with insufficient spectral range, low SNR, or incomplete
    data. For valid results, compare the ratio's deviation from 1 with
    ``ratio_err`` and return ``STAR_LIKE`` for blue color or ``ROCKY_LIKE`` for
    red color.

    Args:
        ratio: Long-to-short flux ratio.
        ratio_err: Estimated ratio error.
        snr: Photometric signal-to-noise ratio.
        reliable: Whether the minimum spectral range is covered.

    Returns:
        Tuple with class, confidence between 0 and 1, and a list of reasons.
    """
    if not reliable:
        return "UNKNOWN", 0.0, ["wavelength_range_too_small"]
    if snr is None or snr < MIN_SNR:
        return "UNKNOWN", 0.0, ["low_snr"]
    if ratio is None:
        return "UNKNOWN", 0.0, ["insufficient_data"]

    # Significance of |ratio - 1|
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
    """Generate indicative visual parameters for representing the candidate.

    Choose a base color according to the ratio and reduce visual confidence
    when evidence is weak. Then assign roughness, metalness, emission, scale,
    and a suggested model according to the statistical class.

    Args:
        cls: ``STAR_LIKE``, ``ROCKY_LIKE``, or ``UNKNOWN`` class.
        ratio: Color ratio, if available.
        confidence: Classification confidence.

    Returns:
        Dictionary of material and 3D model suggestions.
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
# 7. Public API
# --------------------------------------------------------------------------- #

def analyze_pixel(hdul: fits.HDUList, pixel_x: int, pixel_y: int) -> dict:
    """Run the complete photometric and spectral analysis of a pixel.

    Combine cutout extraction, aperture photometry, micro-spectrum, color ratio,
    statistical classification, and visual hints. If the pixel is too close to
    the edge, return an empty response with the reason.

    Args:
        hdul: Open FITS file.
        pixel_x: X coordinate of the selected pixel.
        pixel_y: Y coordinate of the selected pixel.

    Returns:
        Dictionary with candidate class, confidence, color, flux, SNR, spectral
        range, spectrum points, and visual suggestions.
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

    # Color index: linear mapping of ratio to [-1, +1] centered at 1.0
    # ratio=0.5 → +1 (blue), ratio=1.0 → 0 (neutral), ratio=2.0 → -1 (red)
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
    """Build a standard response when the analysis cannot be completed.

    Args:
        reason: Reason to include in the result's reasons list.

    Returns:
        Dictionary with the same structure as ``analyze_pixel``, but with null
        values, empty lists, and the ``UNKNOWN`` class.
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

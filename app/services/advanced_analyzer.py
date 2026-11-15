"""
SPHEREx multi-detector analysis — telescope pairs.

SPHEREx has 3 telescopes physically separated by ~3.9° in RA.
Each telescope carries 2 LVF detectors:

    Telescope 1: D1 (0.75-1.09 μm) + D4 (2.42-3.82 μm)
    Telescope 2: D2 (1.10-1.62 μm) + D5 (3.83-4.41 μm)
    Telescope 3: D3 (1.63-2.41 μm) + D6 (4.42-5.00 μm)

An object in the sky is seen by only ONE telescope per visit.
Therefore, in a single visit, the maximum spectral information
per object is 2 points (the 2 detectors of the telescope covering it).

Pipeline:
  1. Detect which pair (D1+D4, D2+D5, D3+D6) covers the (RA, Dec).
  2. Aperture photometry in both detectors → 2 fluxes.
  3. Ratio = F(long) / F(short) with propagated error.
  4. Classify:
     - ratio < 0.7  → strong ABSORPTION (ice if pair=D1+D4)
     - ratio < 0.9  → weak ABSORPTION
     - 0.9..1.1     → CONTINUUM (likely star)
     - > 1.1        → EMISSION (hot dust)
     - snr < 3      → UNDETERMINED
"""

from __future__ import annotations

import logging

import numpy as np
from astropy.io import fits
from astropy import units as u
from scipy.optimize import brentq

from app.services.fits_loader import (
    DETECTOR_WAVELENGTH_RANGES,
    get_spatial_wcs,
    load_local_fits,
    get_spectral_wcs
)
from app.services.spectral_analyzer import (
    _aperture_photometry,
    _extract_cutout,
)

logger = logging.getLogger(__name__)

# Telescope pairs: (detector_short, detector_long)
TELESCOPE_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
]

# Classifications based on SED shape (2 points), not absorption:
STEEP_RED = 0.7       # ratio < 0.7 → steep drop toward long λ
MODERATE_RED = 0.9    # 0.7-0.9 → moderate drop
FLAT_MAX = 1.1        # 0.9-1.1 → flat
MIN_SNR = 3.0
MIN_RATIO_SIGNIFICANCE = 2.0  # |ratio - 1| / sigma_ratio
_HC_K = 14387.77  # μm·K

# --------------------------------------------------------------------------- #
# Ratio utilities
# --------------------------------------------------------------------------- #

def _bb_fnu_ratio(T: float, lam_s: float, lam_l: float) -> float:
    """
    F_ν(long)/F_ν(short) ratio for a blackbody at temperature T.

    B_ν(λ, T) ∝ (1/λ³) / (e^(hc/λkT) - 1)
    """
    if T <= 0 or lam_s <= 0 or lam_l <= 0:
        return 0.0
    xs = _HC_K / (lam_s * T)
    xl = _HC_K / (lam_l * T)
    if xs > 700 or xl > 700:  # avoid overflow
        return 0.0
    return float((lam_s / lam_l) ** 3 * (np.expm1(xs) / np.expm1(xl)))


def _color_temperature(
    ratio: float | None,
    lam_s: float | None,
    lam_l: float | None,
) -> float | None:
    """Invert the observed ratio assuming a blackbody. None if it does not converge."""
    if ratio is None or lam_s is None or lam_l is None:
        return None
    if ratio <= 0 or lam_s <= 0 or lam_l <= 0 or lam_l <= lam_s:
        return None

    def f(T: float) -> float:
        return _bb_fnu_ratio(T, lam_s, lam_l) - ratio

    try:
        # f(500K) > 0 (cold = high ratio), f(30000K) < 0 (hot = low ratio)
        f_lo = f(500.0)
        f_hi = f(30000.0)
        if f_lo * f_hi > 0:
            return None  # out of range
        return float(brentq(f, 500.0, 30000.0, maxiter=100))
    except (ValueError, RuntimeError):
        return None

# --------------------------------------------------------------------------- #
# Geometric utilities
# --------------------------------------------------------------------------- #

def _pixel_for_world(hdul: fits.HDUList, ra_deg: float, dec_deg: float):
    """
    Convert RA/Dec → (x, y) using the spatial WCS.

    Strategy:
      1. Try `all_world2pix` (iterative SIP, accurate near the center).
      2. If it fails due to divergence, use `wcs_world2pix` (one pass, does not diverge).
      3. ALWAYS normalize the output to float (varies by Astropy version).
      4. Return None if it falls outside the frame.

    Return integer (x, y) or None.
    """
    wcs = get_spatial_wcs(hdul)

    x = y = None

    def _to_float(value) -> float | None:
        """Convert a scalar, array([v]), or array([[v]]) to float. None on failure."""
        try:
            arr = np.asarray(value, dtype=np.float64).ravel()
            if arr.size < 1 or not np.isfinite(arr[0]):
                return None
            return float(arr[0])
        except Exception:
            return None

    # Attempt 1: iterative (better precision with SIP)
    try:
        raw = wcs.all_world2pix(ra_deg, dec_deg, 0)
        # raw may be a tuple, ndarray (2,), or ndarray (1,2)
        raw_arr = np.asarray(raw, dtype=np.float64).ravel()
        if raw_arr.size >= 2:
            x = float(raw_arr[0])
            y = float(raw_arr[1])
    except Exception as e:
        logger.info("all_world2pix failed (%s); falling back", e)

    # Attempt 2: direct (does not diverge)
    if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
        try:
            raw = wcs.wcs_world2pix([[ra_deg, dec_deg]], 0)
            raw_arr = np.asarray(raw, dtype=np.float64).ravel()
            if raw_arr.size >= 2:
                x = float(raw_arr[0])
                y = float(raw_arr[1])
        except Exception as e:
            logger.warning("world2pix also failed: %s", e)
            return None

    if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
        return None

    ny, nx = hdul["IMAGE"].data.shape
    if not (0 <= x < nx and 0 <= y < ny):
        logger.debug(
            "RA=%.4f Dec=%.4f → (%.1f, %.1f) fuera de frame %dx%d",
            ra_deg, dec_deg, x, y, nx, ny,
        )
        return None

    return int(round(x)), int(round(y))


def _effective_wavelength_range(detector: int) -> tuple[float, float]:
    """Official detector λ range (μm)."""
    return DETECTOR_WAVELENGTH_RANGES.get(detector, (0.0, 0.0))


def _center_wavelength(detector: int) -> float:
    lo, hi = _effective_wavelength_range(detector)
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# Detect the pair covering the RA/Dec
# --------------------------------------------------------------------------- #

def detect_covering_pair(
    sector_id: str,
    epoch: str,
    ra_deg: float,
    dec_deg: float,
    obs_id: str | None = None,
    visit: int | None = None,
) -> tuple[int, int] | None:
    """
    Return the pair (det_short, det_long) covering the RA/Dec, or None
    if neither covers it. Requires at least one of the pair's two detectors
    to have the point inside the frame.

    Does not check whether BOTH are inside; extract_pair_flux does that.
    """
    for pair in TELESCOPE_PAIRS:
        for det in pair:
            try:
                hdul = load_local_fits(
                    sector_id=sector_id, epoch=epoch, detector=det,
                    obs_id=obs_id, visit=visit,
                )
            except Exception:
                continue
            try:
                pixel = _pixel_for_world(hdul, ra_deg, dec_deg)
            finally:
                hdul.close()
            if pixel is not None:
                return pair
    return None


# --------------------------------------------------------------------------- #
# Photometry on a detector
# --------------------------------------------------------------------------- #

def _pixel_wavelength(hdul, x: int, y: int) -> tuple[float | None, float | None]:
    """Return (λ_μm, Δλ_μm) for pixel (x, y) using the spectral WCS."""
    try:
        wcs = get_spectral_wcs(hdul)
        world = wcs.pixel_to_world(float(x), float(y))
        lam = float(np.asarray(world[0].to(u.um).value).ravel()[0])
        dlam = float(np.asarray(world[1].to(u.um).value).ravel()[0])
        if not (np.isfinite(lam) and lam > 0):
            return None, None
        return lam, dlam
    except Exception as e:
        logger.warning("Spectral WCS evaluation failed: %s", e)
        return None, None

def _photometry_detector(
    sector_id: str,
    epoch: str,
    detector: int,
    ra_deg: float,
    dec_deg: float,
    obs_id: str | None,
    visit: int | None,
) -> dict:
    """
    Extract flux, error, S/N, and position for a specific detector.

    Return a dict with:
        available: bool
        detector, pixel_x, pixel_y
        flux, flux_err, snr
        valid_fraction_aperture
        wavelength_center_um
    """
    lo_um, hi_um = _effective_wavelength_range(detector)
    result = {
        "available": False,
        "detector": detector,
        "pixel_x": None,
        "pixel_y": None,
        "flux": None,
        "flux_err": None,
        "snr": None,
        "valid_fraction_aperture": 0.0,
        "wavelength_center_um": (lo_um + hi_um) / 2.0,
        "wavelength_range_um": [lo_um, hi_um],
    }

    try:
        hdul = load_local_fits(
            sector_id=sector_id, epoch=epoch, detector=detector,
            obs_id=obs_id, visit=visit,
        )
    except Exception as e:
        logger.warning("Could not load D%d: %s", detector, e)
        return result

    try:
        pixel = _pixel_for_world(hdul, ra_deg, dec_deg)
        if pixel is None:
            return result

        lam_pix, dlam_pix = _pixel_wavelength(hdul, pixel[0], pixel[1])

        cut = _extract_cutout(hdul, pixel[0], pixel[1])
        photo = _aperture_photometry(cut)

        result["available"] = True
        result["pixel_x"] = pixel[0]
        result["pixel_y"] = pixel[1]
        result["flux"] = photo["flux"]
        result["flux_err"] = photo["flux_err"]
        result["snr"] = photo["snr"]
        result["valid_fraction_aperture"] = photo["valid_fraction_aperture"]
        result["wavelength_center_um"] = lam_pix or ((lo_um + hi_um) / 2.0)
        result["wavelength_bandwidth_um"] = dlam_pix

    except ValueError as e:
        logger.info("Cutout failed for D%d: %s", detector, e)
    finally:
        hdul.close()

    return result


# --------------------------------------------------------------------------- #
# Ratio calculation and classification
# --------------------------------------------------------------------------- #

def _compute_ratio(short: dict, long: dict) -> tuple[float | None, float | None]:
    """Return (ratio, ratio_err). None if any flux is missing."""
    fs = short.get("flux")
    fl = long.get("flux")
    es = short.get("flux_err")
    el = long.get("flux_err")

    if fs is None or fl is None or fs <= 0:
        return None, None

    ratio = fl / fs

    if es is not None and el is not None and es > 0 and el > 0:
        rel_err = float(np.sqrt((es / fs) ** 2 + (el / fl) ** 2))
        ratio_err = ratio * rel_err
    else:
        ratio_err = None

    return float(ratio), ratio_err


def classify_pair(
    ratio: float | None,
    ratio_err: float | None,
    snr_short: float | None,
    snr_long: float | None,
    lam_short: float | None,
    lam_long: float | None,
    pair: tuple[int, int],
    flux_short: float | None = None,
    flux_long: float | None = None,
    valid_short: float = 1.0,
    valid_long: float = 1.0,
) -> dict:
    """Classify based on the color temperature derived from the ratio."""

    # --- Guards ---
    if valid_short < 0.1:
        return _undet(f"D{pair[0]} aperture is masked.")
    if valid_long < 0.1:
        return _undet(f"D{pair[1]} aperture is masked.")
    if flux_short is not None and flux_short <= 0:
        return _undet(f"No source: flux in D{pair[0]} = {flux_short:.4g}")
    if flux_long is not None and flux_long <= 0:
        return _undet(f"No source: flux in D{pair[1]} = {flux_long:.4g}")
    if snr_short is None or snr_short < MIN_SNR:
        return _undet(f"Low S/N in D{pair[0]} ({snr_short})")
    if snr_long is None or snr_long < MIN_SNR:
        return _undet(f"Low S/N in D{pair[1]} ({snr_long})")
    if ratio is None:
        return _undet("Ratio could not be computed.")

    # --- Color temperature ---
    T = _color_temperature(ratio, lam_short, lam_long)

    # Propagated T error
    T_err = None
    if T is not None and ratio_err is not None and ratio_err > 0 and lam_short and lam_long:
        # dT/dratio via finite differences
        eps = max(ratio * 0.01, 0.005)
        T_hi = _color_temperature(ratio + eps, lam_short, lam_long)
        T_lo = _color_temperature(max(ratio - eps, 0.01), lam_short, lam_long)
        if T_hi is not None and T_lo is not None:
            T_err = abs((T_hi - T_lo) / (2 * eps)) * ratio_err

    if T is None:
        return {
            "classification": "UNDETERMINED",
            "confidence": 0.0,
            "diagnostic_band": None,
            "reason": (
                f"Ratio {ratio:.2f} is outside the blackbody range for "
                f"λ={lam_short:.2f}-{lam_long:.2f} μm. "
                f"Likely a non-stellar source or measurement issue."
            ),
            "color_temperature_k": None,
            "color_temperature_err_k": None,
        }

    # --- Classification by T ---
    if T > 10000:
        cls, label, conf = "HOT_STAR", "Estrella caliente (O/B/A)", 0.7
    elif T > 5500:
        cls, label, conf = "SUN_LIKE_STAR", "Estrella tipo solar (F/G)", 0.6
    elif T > 3500:
        cls, label, conf = "COOL_STAR", "Estrella fría (K/M)", 0.5
    elif T > 1500:
        cls, label, conf = "VERY_COOL", "Enana M tardía / enana marrón / polvo", 0.4
    else:
        cls, label, conf = "COOL_DUSTY", "Polvo frío o medición dudosa", 0.3

    T_str = f"T_color ≈ {T:.0f} K"
    T_err_str = f" ± {T_err:.0f} K" if T_err is not None else ""

    reason = (
        f"{label}. {T_str}{T_err_str} "
        f"(from Fν ratio F(D{pair[1]})/F(D{pair[0]}) = {ratio:.2f}). "
        f"Color temperature is degenerate with dust reddening; "
        f"interpretation requires additional data."
    )

    return {
        "classification": cls,
        "confidence": conf,
        "diagnostic_band": None,
        "reason": reason,
        "color_temperature_k": T,
        "color_temperature_err_k": T_err,
    }


def _undet(reason: str) -> dict:
    return {
        "classification": "UNDETERMINED",
        "confidence": 0.0,
        "diagnostic_band": None,
        "reason": reason,
        "color_temperature_k": None,
        "color_temperature_err_k": None,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def analyze_pair(
    sector_id: str,
    epoch: str,
    ra_deg: float,
    dec_deg: float,
    obs_id: str | None = None,
    visit: int | None = None,
) -> dict:
    """
    Complete pipeline for an (RA, Dec).

    Return a dict compatible with PairSpectrumResponse.
    """
    pair = detect_covering_pair(
        sector_id=sector_id,
        epoch=epoch,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        obs_id=obs_id,
        visit=visit,
    )

    if pair is None:
        return {
            "sector_id": sector_id,
            "epoch": epoch,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "pair": None,
            "short": None,
            "long": None,
            "ratio": None,
            "ratio_error": None,
            "classification": {
                "classification": "UNDETERMINED",
                "confidence": 0.0,
                "diagnostic_band": None,
                "reason": "No telescope pair covers this RA/Dec in this visit.",
            },
        }

    det_short, det_long = pair

    short = _photometry_detector(
        sector_id, epoch, det_short, ra_deg, dec_deg, obs_id, visit
    )
    long = _photometry_detector(
        sector_id, epoch, det_long, ra_deg, dec_deg, obs_id, visit
    )

    ratio, ratio_err = _compute_ratio(short, long)

    cls = classify_pair(
        ratio=ratio,
        ratio_err=ratio_err,
        snr_short=short["snr"],
        snr_long=long["snr"],
        lam_short=short["wavelength_center_um"],
        lam_long=long["wavelength_center_um"],
        pair=pair,
        flux_short=short["flux"],
        flux_long=long["flux"],
        valid_short=short["valid_fraction_aperture"],
        valid_long=long["valid_fraction_aperture"]
    )

    return {
        "sector_id": sector_id,
        "epoch": epoch,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "pair": f"D{det_short}+D{det_long}",
        "short": short,
        "long": long,
        "ratio": ratio,
        "ratio_error": ratio_err,
        "classification": cls,
    }

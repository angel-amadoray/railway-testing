"""
SPHEREx FITS loading and processing (Level 2).

A SPHEREx Spectral Image is an MEF with extensions:
    PRIMARY, IMAGE, FLAGS, VARIANCE, ZODI, PSF, WCS-WAVE

Spectral information is not stored on a cube axis, but in the
WCS-WAVE BinTable extension, referenced by the IMAGE header with suffix 'W':
    CTYPE1W = 'WAVE-TAB'
    PS1_0W  = 'WCS-WAVE'      -> table extension name
    PS1_2W  = 'X'             -> independent column for λ (varies with X)
    PV1_3W  = 1               -> linear interpolation

References:
- FITS WCS Paper III (Greisen et al. 2006), Wave-Tab section
- IRSA tutorial: caltech-ipac.github.io/irsa-tutorials/spherex-intro/
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

FILENAME_PATTERN = re.compile(
    r"level2_(?P<planning_period>.+?)_(?P<obs_id>\d+)_(?P<visit>\d+)D(?P<detector>\d+)"
    r"_spx_l2b-v(?P<version>\d+)-(?P<processing_date>\d{4}-\d{3})\.fits$"
)

# Defensive validation to prevent path traversal in sector_id/epoch
_ID_RE = re.compile(r"^[a-z0-9_\-]+$")

# FLAGS bits considered "unreliable" according to the SPHEREx Explanatory Supplement.
# Other bits are informational (e.g., corrected pixel, out of range but valid).
BAD_BITS = (
    (1 << 0)   # bad data / no signal
    | (1 << 1)   # saturated
    | (1 << 2)   # cosmic ray hit
    | (1 << 6)   # dead pixel
    | (1 << 7)   # hot pixel
    | (1 << 9)   # high noise
    | (1 << 10)  # calibration issue
    | (1 << 11)  # non-linear response
    | (1 << 14)  # source extraction failed
    | (1 << 15)  # invalid flux
    | (1 << 17)  # zodiacal subtraction failed
    | (1 << 19)  # overflow
)

RA_KEYS = ("RA_TARG", "RA_OBJ", "RA_CEN", "CRVAL1", "OBJ_RA")
DEC_KEYS = ("DEC_TARG", "DEC_OBJ", "DEC_CEN", "CRVAL2", "OBJ_DEC")

# Spectral ranges by detector (μm). Fixed by SPHEREx design.
DETECTOR_WAVELENGTH_RANGES = {
    1: (0.75, 1.09),
    2: (1.10, 1.62),
    3: (1.63, 2.41),
    4: (2.42, 3.82),
    5: (3.83, 4.41),
    6: (4.42, 5.00),
}


# --------------------------------------------------------------------------- #
# HDUList helpers
# --------------------------------------------------------------------------- #

def open_mef(path: str | Path) -> fits.HDUList:
    """Open a SPHEREx MEF file using deferred memory access.

    Args:
        path: Path to the FITS file to open.

    Returns:
        ``HDUList`` opened with ``memmap=True``; the caller must close it.

    Raises:
        FileNotFoundError: If the path does not point to an existing file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FITS not found: {path}")
    return fits.open(path, memmap=True)


def _get_ext(hdul: fits.HDUList, name: str) -> fits.hdu.base.ExtensionHDU:
    """Get a FITS extension by name and validate that it exists.

    Args:
        hdul: Open FITS file.
        name: Name of the requested extension.

    Returns:
        The extension whose name matches ``name``.

    Raises:
        KeyError: If it does not exist; the message includes available extensions.
    """
    try:
        return hdul[name]
    except KeyError:
        available = [hdu.name for hdu in hdul]
        raise KeyError(
            f"Extension '{name}' not found in FITS. Available: {available}"
        )

def _validate_id(name: str, kind: str) -> str:
    """Validate a path identifier to prevent traversal and invalid names.

    Accept only lowercase letters, digits, underscores, and hyphens.
    """
    if not _ID_RE.match(name or ""):
        raise ValueError(
            f"Invalid {kind}: {name!r}. "
            f"Only lowercase letters, digits, '_' and '-' are allowed."
        )
    return name


# --------------------------------------------------------------------------- #
# WCS: spatial and spectral
# --------------------------------------------------------------------------- #

def get_spatial_wcs(hdul: fits.HDUList) -> WCS:
    """Build the spatial RA/Dec WCS from the ``IMAGE`` header.

    The WCS preserves the SIP distortions defined in the file and is used to
    convert between sky coordinates and detector pixels.
    """
    return WCS(_get_ext(hdul, "IMAGE").header)


def get_spectral_wcs(hdul: fits.HDUList) -> WCS:
    """
    Build the wavelength spectral WCS using ``key='W'``.

    First remove SIP keys because they belong to the spatial WCS and Astropy
    could incorrectly interpret them as part of the spectral WCS.

    Args:
        hdul: Open FITS file containing ``IMAGE`` and ``WCS-WAVE``.

    Returns:
        WCS capable of converting pixel coordinates to wavelength and bandwidth.
    """
    header = _strip_sip_keys(_get_ext(hdul, "IMAGE").header)
    w = WCS(header, fobj=hdul, key="W")
    w.sip = None
    return w


def detector_for_wavelength(wavelength_um: float) -> int | None:
    """Identify the detector whose range contains a wavelength.

    Args:
        wavelength_um: Wavelength in micrometers.

    Returns:
        Detector number from 1 to 6, or ``None`` if outside all known ranges.
    """
    for det, (lo, hi) in DETECTOR_WAVELENGTH_RANGES.items():
        if lo <= wavelength_um <= hi:
            return det
    return None


# --------------------------------------------------------------------------- #
# λ map (2D)
# --------------------------------------------------------------------------- #

def _quantity_to_array(q, target_unit=u.um) -> np.ndarray:
    """
    Convert an Astropy value to a unit-bearing ``float32`` ``ndarray``.

    If the object has a ``to`` method, convert it to ``target_unit``; if it has
    a ``value`` attribute, extract the numeric value. Unitless objects are
    treated as already expressed in the target unit.

    Args:
        q: Quantity, scalar, or array convertible to NumPy.
        target_unit: Output unit, micrometers by default.

    Returns:
        NumPy array of type ``float32``.
    """
    if hasattr(q, "to"):
        q = q.to(target_unit)
    if hasattr(q, "value"):
        q = q.value
    return np.asarray(q, dtype=np.float32)


def get_wavelength_map(hdul: fits.HDUList, downsample: int = 4) -> np.ndarray:
    """
    Generate a 2D wavelength map in micrometers.

    Evaluate the spectral WCS on a reduced grid and interpolate the map to full
    size using bilinear interpolation. This leverages the fact that in SPHEREx,
    wavelength changes primarily with LVF position.

    Args:
        hdul: Open FITS file with ``IMAGE`` and spectral WCS.
        downsample: Sampling step; 1 preserves full resolution.

    Returns:
        ``float32`` array of shape ``(height, width)`` with λ in micrometers.

    Raises:
        ValueError: If ``IMAGE`` contains no data.
        RuntimeError: If the spectral WCS cannot be evaluated.
    """
    image = _get_ext(hdul, "IMAGE").data
    if image is None:
        raise ValueError("IMAGE extension has no data")
    ny, nx = image.shape

    ds = max(1, int(downsample))
    ys = np.arange(0, ny, ds)
    xs = np.arange(0, nx, ds)
    xx, yy = np.meshgrid(xs, ys)

    wcs_spec = get_spectral_wcs(hdul)
    try:
        world = wcs_spec.pixel_to_world(xx, yy)
    except Exception as e:
        logger.exception("pixel_to_world failed on spectral WCS")
        raise RuntimeError(
            f"Spectral WCS evaluation failed. Is the WCS-WAVE extension present? ({e})"
        )

    wave_low = _quantity_to_array(world[0])

    if ds == 1:
        return wave_low

    wave_full = zoom(wave_low, zoom=ds, order=1)
    wave_full = wave_full[:ny, :nx]
    if wave_full.shape != (ny, nx):
        # Resize defensively if zoom did not produce exact dimensions
        padded = np.zeros((ny, nx), dtype=np.float32)
        padded[: wave_full.shape[0], : wave_full.shape[1]] = wave_full
        wave_full = padded
    return wave_full


# --------------------------------------------------------------------------- #
# Clean image (IMAGE - ZODI, with FLAGS masked)
# --------------------------------------------------------------------------- #

def get_clean_image(hdul: fits.HDUList, subtract_zodi: bool = True, bad_bits: int = BAD_BITS) -> np.ndarray:
    """
    Obtain the clean science image and mask unreliable pixels.

    Start with ``IMAGE``, optionally subtract ``ZODI``, and use ``FLAGS`` to
    set pixels whose bits intersect ``bad_bits`` to zero. Bits not included are
    preserved. NaNs and infinities are also converted to zero.

    Args:
        hdul: Open FITS file.
        subtract_zodi: Whether to subtract the zodiacal extension.
        bad_bits: Mask of bits considered invalid.

    Returns:
        ``float32`` image of shape ``(height, width)``.

    Raises:
        ValueError: If the ``IMAGE`` extension contains no data.
    """
    image_hdu = _get_ext(hdul, "IMAGE")
    if image_hdu.data is None:
        raise ValueError("IMAGE extension has no data")

    image = np.array(image_hdu.data, dtype=np.float32, copy=True)

    if subtract_zodi:
        try:
            zodi = np.asarray(_get_ext(hdul, "ZODI").data, dtype=np.float32)
            image -= zodi
        except KeyError:
            logger.warning("ZODI extension missing; skipping zodiacal subtraction")

    try:
        flags = _get_ext(hdul, "FLAGS").data
        if flags is not None:
            bad = (flags & bad_bits) != 0
            image[bad] = np.nan
    except KeyError:
        logger.warning("FLAGS extension missing; not masking bad pixels")

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image


# --------------------------------------------------------------------------- #
# Extract a "slice" at a given λ
# --------------------------------------------------------------------------- #

def extract_2d_at_wavelength(
    hdul: fits.HDUList,
    target_um: float,
    tol_um: float = 0.02,
    downsample: int = 4,
) -> np.ndarray:
    """
    Extract a 2D image corresponding to a target spectral band.

    Keep the flux of pixels whose wavelength is within ``target_um ± tol_um``
    and set the rest to zero. In SPHEREx this typically produces a strip
    associated with the LVF pattern.

    Args:
        hdul: Open FITS file.
        target_um: Central wavelength in micrometers.
        tol_um: Spectral tolerance around the target wavelength.
        downsample: Step used to build the spectral map.

    Returns:
        ``float32`` image with the same shape as ``IMAGE``.

    Raises:
        RuntimeError: If the spectral map and image have different shapes.
    """
    clean = get_clean_image(hdul)
    wave_map = get_wavelength_map(hdul, downsample=downsample)

    if wave_map.shape != clean.shape:
        raise RuntimeError(
            f"Shape mismatch: wave_map={wave_map.shape}, image={clean.shape}"
        )

    mask = np.abs(wave_map - float(target_um)) <= float(tol_um)
    n_kept = int(mask.sum())
    if n_kept == 0:
        logger.warning(
            "extract_2d_at_wavelength: no pixels match λ=%s μm ± %s μm. "
            "Check wavelength range with /api/debug/wavelength/...",
            target_um,
            tol_um,
        )

    return np.where(mask, clean, 0.0).astype(np.float32)



def wavelength_range(
    hdul: fits.HDUList,
    downsample: int = 8,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[float, float]:
    """
    Calculate the robust wavelength range covered by the image.

    Discard non-finite values and values outside the physical 0.1–50 μm range,
    and use percentiles to prevent WCS extrapolations at the corners from
    distorting the result.

    Args:
        hdul: Open FITS file.
        downsample: Step used to obtain the spectral map.
        p_low: Lower percentile of the range.
        p_high: Upper percentile of the range.

    Returns:
        ``(minimum, maximum)`` tuple in micrometers.

    Raises:
        RuntimeError: If no valid wavelengths are found.
    """
    wave = get_wavelength_map(hdul, downsample=downsample)
    valid = wave[np.isfinite(wave) & (wave > 0.1) & (wave < 50.0)]
    if valid.size == 0:
        raise RuntimeError("No valid wavelength values found in WCS-WAVE map")
    return (
        float(np.percentile(valid, p_low)),
        float(np.percentile(valid, p_high)),
    )


def wavelength_profiles(
    hdul: fits.HDUList, n_samples: int = 20
) -> dict:
    """
    Sample spectral profiles along the X and Y axes.

    Calculate wavelength and bandwidth along a central row across X and a
    central column across Y. The result helps diagnose detector LVF geometry.

    Args:
        hdul: Open FITS file.
        n_samples: Number of positions sampled on each axis.

    Returns:
        Dictionary with the image shape and ``along_x`` and ``along_y`` lists.

    Raises:
        ValueError: If ``IMAGE`` contains no data.
    """
    image = _get_ext(hdul, "IMAGE").data
    if image is None:
        raise ValueError("IMAGE extension has no data")
    ny, nx = image.shape
    wcs_spec = get_spectral_wcs(hdul)

    xs = np.linspace(0, nx - 1, n_samples).astype(int)
    y_mid = ny // 2
    world_x = wcs_spec.pixel_to_world(xs, np.full_like(xs, y_mid))
    wave_x = _quantity_to_array(world_x[0])
    bw_x = _quantity_to_array(world_x[1])

    ys = np.linspace(0, ny - 1, n_samples).astype(int)
    x_mid = nx // 2
    world_y = wcs_spec.pixel_to_world(np.full_like(ys, x_mid), ys)
    wave_y = _quantity_to_array(world_y[0])
    bw_y = _quantity_to_array(world_y[1])

    return {
        "image_shape": [int(ny), int(nx)],
        "along_x": [
            {"x": int(x), "wavelength_um": float(w), "bandwidth_um": float(b)}
            for x, w, b in zip(xs, wave_x, bw_x)
        ],
        "along_y": [
            {"y": int(y), "wavelength_um": float(w), "bandwidth_um": float(b)}
            for y, w, b in zip(ys, wave_y, bw_y)
        ],
    }


# --------------------------------------------------------------------------- #
# File discovery on disk
# --------------------------------------------------------------------------- #

def parse_filename_metadata(filename: str) -> dict:
    """Extract structured metadata from a standard SPHEREx FITS filename.

    Args:
        filename: Filename, not necessarily a full path.

    Returns:
        Dictionary with planning period, ``obs_id``, visit, detector, version,
        and processing date. Return an empty dictionary if the name does not
        match the expected pattern.
    """
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return {}
    g = m.groupdict()
    g["visit"] = int(g["visit"])
    g["detector"] = int(g["detector"])
    g["version"] = int(g["version"])
    return g


def find_sector_files(sector_id: str, epoch: str, data_dir: Path | None = None) -> list[Path]:
    """List local FITS files for a sector and epoch.

    Validate both identifiers before forming the path. If the directory does
    not exist, return an empty list instead of raising an exception.

    Args:
        sector_id: Sector identifier.
        epoch: Observation epoch identifier.
        data_dir: Optional data root; if omitted, use global configuration.

    Returns:
        Sorted list of ``.fits`` paths.

    Raises:
        ValueError: If an identifier contains disallowed characters.
    """
    _validate_id(sector_id, "sector_id")
    _validate_id(epoch, "epoch")

    if data_dir is None:
        from app.config import settings
        data_dir = Path(settings.data_dir)
    sector_dir = Path(data_dir) / sector_id / epoch
    if not sector_dir.is_dir():
        return []
    return sorted(sector_dir.glob("*.fits"))


def find_fits_file(
    sector_id: str,
    epoch: str,
    detector: int,
    obs_id: str | None = None,
    visit: int | None = None,
    data_dir: Path | None = None,
) -> Path:
    """
    Locate a single FITS file by sector, epoch, and detector.

    Restrict the search by ``obs_id`` and ``visit`` to resolve ambiguities among
    multiple matching files.

    Raises:
        FileNotFoundError: If no matching file exists.
        ValueError: If multiple candidates remain without sufficient filters or
            if path identifiers are invalid.
    """
    _validate_id(sector_id, "sector_id")
    _validate_id(epoch, "epoch")

    files = find_sector_files(sector_id, epoch, data_dir)
    matches: list[Path] = []
    for f in files:
        meta = parse_filename_metadata(f.name)
        if not meta:
            continue
        if meta["detector"] != detector:
            continue
        if obs_id is not None and meta.get("obs_id") != obs_id:
            continue
        if visit is not None and meta.get("visit") != visit:
            continue
        matches.append(f)

    where = f"data/sample_fits/{sector_id}/{epoch}"
    if not matches:
        raise FileNotFoundError(
            f"No FITS for sector={sector_id} epoch={epoch} detector={detector} "
            f"(obs_id={obs_id}, visit={visit}) in {where}"
        )
    if len(matches) > 1:
        names = [m.name for m in matches[:5]]
        raise ValueError(
            f"Ambiguous FITS match ({len(matches)} files). "
            f"Specify obs_id and/or visit. Candidates: {names}"
        )
    return matches[0]


def load_local_fits(
    sector_id: str,
    epoch: str,
    detector: int,
    obs_id: str | None = None,
    visit: int | None = None,
    data_dir: Path | None = None,
) -> fits.HDUList:
    """
    Locate and open the corresponding FITS file.

    If ``data_dir`` is provided, the search is exclusively local. If it is
    ``None``, use the storage service to download the file if it is not locally
    available.

    Args:
        sector_id: Sector identifier.
        epoch: Observation epoch.
        detector: Detector number.
        obs_id: Optional observation identifier.
        visit: Optional visit number.
        data_dir: Local root that forces no-download mode.

    Returns:
        Open ``HDUList``; the caller must close it.
    """
    from app.services.storage import download_fits_if_missing

    if data_dir is not None:
        path = find_fits_file(
            sector_id, epoch, detector, obs_id, visit, data_dir
        )
    else:
        path = download_fits_if_missing(
            sector_id=sector_id,
            epoch=epoch,
            detector=detector,
            obs_id=obs_id,
            visit=visit,
        )
    logger.info("Loading FITS: %s", path)
    return open_mef(path)


# --------------------------------------------------------------------------- #
# Header information (used by the /debug router)
# --------------------------------------------------------------------------- #

def _extract_coordinates(header: fits.Header) -> dict:
    """Extract RA and Dec using the first valid key from each known list.

    Non-numeric values are ignored and the search continues with an alternative
    key. The result also preserves the name of the key used.
    """
    ra = dec = ra_key = dec_key = None
    for k in RA_KEYS:
        if k in header:
            try:
                ra, ra_key = float(header[k]), k
                break
            except (TypeError, ValueError):
                continue
    for k in DEC_KEYS:
        if k in header:
            try:
                dec, dec_key = float(header[k]), k
                break
            except (TypeError, ValueError):
                continue
    return {"ra": ra, "dec": dec, "ra_key": ra_key, "dec_key": dec_key}


def _header_to_dict(header: fits.Header) -> dict:
    """Convert a FITS header into a JSON-compatible dictionary.

    Omit ``COMMENT`` and ``HISTORY``, convert NumPy scalars to native types,
    decode bytes, and convert any remaining object to text.
    """
    out = {}
    for key, value in header.items():
        if key in ("COMMENT", "HISTORY", ""):
            continue
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            value = value.item()
        elif isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if not isinstance(value, (str, int, float, bool, type(None))):
            value = str(value)
        out[key] = value
    return out


def list_extensions(file_path: Path) -> list[str]:
    """Read and return the names of all extensions in a FITS file.

    Args:
        file_path: FITS file path.

    Returns:
        List of names in the order in which they appear in the file.
    """
    with fits.open(file_path, memmap=True) as hdul:
        return [hdu.name for hdu in hdul]


def read_header_info(file_path: Path) -> dict:
    """Collect headers, coordinates, and extensions from a FITS file.

    Read the primary header and, when present, the ``IMAGE`` header. Search for
    coordinates in the primary first, then in ``IMAGE`` if necessary.

    Returns:
        Dictionary with ``primary_header``, ``image_header``, ``extensions``,
        and ``coordinates``.
    """
    logger.info("Reading FITS headers: %s", file_path)
    with fits.open(file_path, memmap=True) as hdul:
        primary = _header_to_dict(hdul[0].header)
        coordinates = _extract_coordinates(hdul[0].header)

        image_header = None
        if len(hdul) > 1 and hdul[1].name.upper() == "IMAGE":
            image_header = _header_to_dict(hdul[1].header)
            if coordinates["ra"] is None:
                c = _extract_coordinates(hdul[1].header)
                if c["ra"] is not None or c["dec"] is not None:
                    coordinates = c

        extensions = [hdu.name for hdu in hdul]

    return {
        "primary_header": primary,
        "image_header": image_header,
        "extensions": extensions,
        "coordinates": coordinates,
    }

def _strip_sip_keys(header: fits.Header) -> fits.Header:
    """
    Return a copy without spatial-distortion SIP keys.

    Remove keys with prefixes ``A_``, ``B_``, ``AP_``, and ``BP_`` to prevent
    Astropy from applying spatial distortion to the ``WAVE-TAB`` spectral WCS.
    """
    h = header.copy()
    sip_prefixes = ("A_", "B_", "AP_", "BP_")
    for key in list(h.keys()):
        if not isinstance(key, str):
            continue
        if key.upper().startswith(sip_prefixes):
            del h[key]
    return h

# --------------------------------------------------------------------------- #
# WCS serialization and pixel ↔ world conversions
# --------------------------------------------------------------------------- #

def serialize_wcs(hdul: fits.HDUList) -> dict:
    """
    Serialize the main parameters of the spatial WCS.

    The result contains reference values, reference pixels, coordinate types,
    scale matrix, and whether SIP distortion exists; it is intended for
    diagnostics or reconstruction in a frontend.

    Returns:
        Dictionary with JSON-serializable values from the ``IMAGE`` WCS.
    """
    wcs = get_spatial_wcs(hdul)
    cd = wcs.pixel_scale_matrix.tolist()
    return {
        "crval1": float(wcs.wcs.crval[0]),
        "crval2": float(wcs.wcs.crval[1]),
        "crpix1": float(wcs.wcs.crpix[0]),
        "crpix2": float(wcs.wcs.crpix[1]),
        "ctype1": str(wcs.wcs.ctype[0]),
        "ctype2": str(wcs.wcs.ctype[1]),
        "cd": cd,
        "has_sip": wcs.sip is not None,
    }


def world_to_pixel_batch(
    hdul: fits.HDUList,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert batches of RA/Dec coordinates to pixel coordinates.

    Use ``all_world2pix`` and include SIP distortion. Non-finite inputs and
    points that fail in the WCS are represented as ``NaN``.

    Args:
        hdul: Open FITS file.
        ra_deg: Array of right ascensions in degrees.
        dec_deg: Array of declinations in degrees.

    Returns:
        Two ``float64`` arrays ``(x, y)`` with the same shape as the inputs.
    """
    wcs = get_spatial_wcs(hdul)
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)

    x = np.full(ra.shape, np.nan, dtype=np.float64)
    y = np.full(ra.shape, np.nan, dtype=np.float64)

    for i in range(ra.size):
        if not (np.isfinite(ra.flat[i]) and np.isfinite(dec.flat[i])):
            continue
        try:
            xi, yi = wcs.all_world2pix(ra.flat[i], dec.flat[i], 0)
            x.flat[i] = xi
            y.flat[i] = yi
        except Exception:
            # Point outside the valid WCS: remains NaN
            continue

    return x, y


def pixel_to_world(
    hdul: fits.HDUList,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float]:
    """
    Convert a pixel position to RA and Dec in degrees.

    Use the complete spatial WCS from ``IMAGE``, including SIP distortion.

    Args:
        hdul: Open FITS file.
        pixel_x: Pixel X coordinate.
        pixel_y: Pixel Y coordinate.

    Returns:
        ``(ra_deg, dec_deg)`` tuple as ``float`` values.
    """
    wcs = get_spatial_wcs(hdul)
    ra, dec = wcs.all_pix2world(pixel_x, pixel_y, 0)
    return float(ra), float(dec)

def get_variance(hdul: fits.HDUList) -> np.ndarray | None:
    """
    Get the ``VARIANCE`` extension as a ``float32`` copy.

    Non-finite values are converted to ``NaN``. If the extension is missing or
    contains no data, return ``None``; the ``FLAGS`` mask is not applied.

    Args:
        hdul: Open FITS file.

    Returns:
        Variance array or ``None`` when unavailable.
    """
    try:
        var_hdu = _get_ext(hdul, "VARIANCE")
        if var_hdu.data is None:
            return None
        var = np.array(var_hdu.data, dtype=np.float32, copy=True)
        var[~np.isfinite(var)] = np.nan
        return var
    except KeyError:
        return None

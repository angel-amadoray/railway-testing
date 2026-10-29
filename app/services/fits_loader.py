"""
Carga y procesamiento de FITS de SPHEREx (Level 2).

Un Spectral Image de SPHEREx es un MEF con extensiones:
    PRIMARY, IMAGE, FLAGS, VARIANCE, ZODI, PSF, WCS-WAVE

La información espectral no está en un eje del cubo, sino en la extensión
BinTable WCS-WAVE, referenciada por el header de IMAGE con sufijo 'W':
    CTYPE1W = 'WAVE-TAB'
    PS1_0W  = 'WCS-WAVE'      -> nombre de la extensión tabla
    PS1_2W  = 'X'             -> columna independiente para λ (varía con X)
    PV1_3W  = 1               -> interpolación lineal

Referencias:
- FITS WCS Paper III (Greisen et al. 2006), sección Wave-Tab
- Tutorial IRSA: caltech-ipac.github.io/irsa-tutorials/spherex-intro/
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

FILENAME_PATTERN = re.compile(
    r"level2_(?P<planning_period>.+?)_(?P<obs_id>\d+)_(?P<visit>\d+)D(?P<detector>\d+)"
    r"_spx_l2b-v(?P<version>\d+)-(?P<processing_date>\d{4}-\d{3})\.fits$"
)

RA_KEYS = ("RA_TARG", "RA_OBJ", "RA_CEN", "CRVAL1", "OBJ_RA")
DEC_KEYS = ("DEC_TARG", "DEC_OBJ", "DEC_CEN", "CRVAL2", "OBJ_DEC")

# Rangos espectrales por detector (μm). Fijos por diseño de SPHEREx.
DETECTOR_WAVELENGTH_RANGES = {
    1: (0.75, 1.09),
    2: (1.10, 1.62),
    3: (1.63, 2.41),
    4: (2.42, 3.82),
    5: (3.83, 4.41),
    6: (4.42, 5.00),
}


# --------------------------------------------------------------------------- #
# Helpers de HDUList
# --------------------------------------------------------------------------- #

def open_mef(path: str | Path) -> fits.HDUList:
    """Abre un MEF de SPHEREx con memmap (no carga la imagen hasta que se usa)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FITS not found: {path}")
    return fits.open(path, memmap=True)


def _get_ext(hdul: fits.HDUList, name: str) -> fits.hdu.base.ExtensionHDU:
    try:
        return hdul[name]
    except KeyError:
        available = [hdu.name for hdu in hdul]
        raise KeyError(
            f"Extension '{name}' not found in FITS. Available: {available}"
        )


# --------------------------------------------------------------------------- #
# WCS: espacial y espectral
# --------------------------------------------------------------------------- #

def get_spatial_wcs(hdul: fits.HDUList) -> WCS:
    """WCS espacial (RA/Dec) del header de IMAGE. Incluye SIP."""
    return WCS(_get_ext(hdul, "IMAGE").header)


def get_spectral_wcs(hdul: fits.HDUList) -> WCS:
    """
    WCS espectral (λ, Δλ) construido con key='W'.
    Limpiamos SIP primero porque el header de SPHEREx los incluye para el
    WCS espacial y astropy los interpretaría erróneamente como SIP espectral.
    """
    header = _strip_sip_keys(_get_ext(hdul, "IMAGE").header)
    w = WCS(header, fobj=hdul, key="W")
    w.sip = None
    return w


def detector_for_wavelength(wavelength_um: float) -> int | None:
    """Devuelve el detector que cubre esa λ, o None si está fuera de rango."""
    for det, (lo, hi) in DETECTOR_WAVELENGTH_RANGES.items():
        if lo <= wavelength_um <= hi:
            return det
    return None


# --------------------------------------------------------------------------- #
# Mapa de λ (2D)
# --------------------------------------------------------------------------- #

def _quantity_to_array(q) -> np.ndarray:
    """Convierte Quantity de astropy a ndarray float32 (usa .value si existe)."""
    if hasattr(q, "value"):
        q = q.value
    return np.asarray(q, dtype=np.float32)


def get_wavelength_map(hdul: fits.HDUList, downsample: int = 4) -> np.ndarray:
    """
    Mapa 2D de λ (μm) a resolución (downsample×downsample), upsampled de vuelta
    a tamaño completo con interpolación bilineal.

    Para SPHEREx, λ depende principalmente de X (patrón LVF), por lo que
    downsample=4 preserva la estructura con enorme ahorro de cómputo.
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
        # Redimensionar defensivamente si zoom no dio dimensiones exactas
        padded = np.zeros((ny, nx), dtype=np.float32)
        padded[: wave_full.shape[0], : wave_full.shape[1]] = wave_full
        wave_full = padded
    return wave_full


# --------------------------------------------------------------------------- #
# Imagen limpia (IMAGE - ZODI, con FLAGS enmascarados)
# --------------------------------------------------------------------------- #

def get_clean_image(hdul: fits.HDUList, subtract_zodi: bool = True) -> np.ndarray:
    """
    Devuelve IMAGE - ZODI con píxeles flaggeados puestos a 0.
    Salida: float32 (H, W).
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
            bad = flags != 0
            image[bad] = np.nan
    except KeyError:
        logger.warning("FLAGS extension missing; not masking bad pixels")

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image


# --------------------------------------------------------------------------- #
# Extracción de un "corte" a una λ dada
# --------------------------------------------------------------------------- #

def extract_2d_at_wavelength(
    hdul: fits.HDUList,
    target_um: float,
    tol_um: float = 0.02,
    downsample: int = 4,
) -> np.ndarray:
    """
    Devuelve una imagen 2D donde solo los píxeles cuya λ cae en
    [target_um - tol_um, target_um + tol_um] conservan su flujo.
    El resto se pone a 0.

    Para SPHEREx (LVF), esto produce una "franja vertical" de ~constante X.
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
    Rango (min, max) de λ en μm sobre la imagen, ignorando outliers.
    Con p_low/p_high por defecto descartamos el 1% extremo de cada lado
    (esquinas con extrapolación del WCS suelen dar valores espurios).
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
    Perfiles de λ a lo largo de X (fila central) y de Y (columna central).
    Útil para entender la geometría LVF del detector.
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
# Descubrimiento de archivos en disco
# --------------------------------------------------------------------------- #

def parse_filename_metadata(filename: str) -> dict:
    """Parsea el nombre de un FITS de SPHEREx. Devuelve {} si no matchea."""
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return {}
    g = m.groupdict()
    g["visit"] = int(g["visit"])
    g["detector"] = int(g["detector"])
    g["version"] = int(g["version"])
    return g


def find_sector_files(sector_id: str, epoch: str, data_dir: Path | None = None) -> list[Path]:
    """Lista los .fits en data/sample_fits/{sector}/{epoch}/."""
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
    Localiza el FITS para un sector/epoch/detector.
    Opcionalmente filtra por obs_id y visit si hay ambigüedad.

    Lanza FileNotFoundError si no hay match; ValueError si hay más de uno
    y no se especificó obs_id/visit.
    """
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
    Abre el FITS correspondiente. Si no existe localmente, intenta descargarlo
    desde OCI (ver app.services.storage). Caller debe cerrar el HDUList.
    """
    # Import local para evitar ciclo: storage importa fits_loader
    from app.services.storage import download_fits_if_missing

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
# Info de headers (usada por el router /debug)
# --------------------------------------------------------------------------- #

def _extract_coordinates(header: fits.Header) -> dict:
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
    with fits.open(file_path, memmap=True) as hdul:
        return [hdu.name for hdu in hdul]


def read_header_info(file_path: Path) -> dict:
    """Devuelve header primario + IMAGE + coordenadas + extensiones."""
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
    Devuelve una copia del header sin claves SIP (A_*, B_*, AP_*, BP_*).
    Necesario para construir el WCS espectral con key='W' sin que astropy
    intente aplicar la distorsión espacial a los ejes WAVE-TAB.
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
# Serialización WCS y conversiones pixel ↔ world
# --------------------------------------------------------------------------- #

def serialize_wcs(hdul: fits.HDUList) -> dict:
    """
    Extrae los parámetros WCS espaciales en formato JSON-serializable.
    Suficiente para que un frontend reconstruya la transformación world→pixel
    usando una librería tipo wcs-js, o para mostrar diagnóstico.
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
    Convierte arrays de (RA, Dec) en grados a (x, y) en píxeles usando
    all_world2pix (incluye SIP). Los puntos fuera del WCS válido se devuelven
    como NaN en lugar de lanzar excepción.
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
            # Punto fuera del WCS válido: queda NaN
            continue

    return x, y


def pixel_to_world(
    hdul: fits.HDUList,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float]:
    """
    Convierte un pixel (x, y) a (RA, Dec) en grados usando all_pix2world
    (incluye SIP).
    """
    wcs = get_spatial_wcs(hdul)
    ra, dec = wcs.all_pix2world(pixel_x, pixel_y, 0)
    return float(ra), float(dec)
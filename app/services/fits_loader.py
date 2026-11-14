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
import astropy.units as u
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

FILENAME_PATTERN = re.compile(
    r"level2_(?P<planning_period>.+?)_(?P<obs_id>\d+)_(?P<visit>\d+)D(?P<detector>\d+)"
    r"_spx_l2b-v(?P<version>\d+)-(?P<processing_date>\d{4}-\d{3})\.fits$"
)

# Validación defensiva para evitar path traversal en sector_id/epoch
_ID_RE = re.compile(r"^[a-z0-9_\-]+$")

# Bits de FLAGS considerados "no confiables" según SPHEREx Explanatory Supplement.
# Otros bits son informativos (ej: pixel corregido, fuera de rango pero válido).
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
    """Abre un archivo MEF de SPHEREx usando acceso de memoria diferido.

    Args:
        path: Ruta del archivo FITS que se desea abrir.

    Returns:
        ``HDUList`` abierto con ``memmap=True``; el llamador debe cerrarlo.

    Raises:
        FileNotFoundError: Si la ruta no corresponde a un archivo existente.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FITS not found: {path}")
    return fits.open(path, memmap=True)


def _get_ext(hdul: fits.HDUList, name: str) -> fits.hdu.base.ExtensionHDU:
    """Obtiene una extensión FITS por nombre y valida su existencia.

    Args:
        hdul: Archivo FITS abierto.
        name: Nombre de la extensión solicitada.

    Returns:
        La extensión cuyo nombre coincide con ``name``.

    Raises:
        KeyError: Si no existe; el mensaje incluye las extensiones disponibles.
    """
    try:
        return hdul[name]
    except KeyError:
        available = [hdu.name for hdu in hdul]
        raise KeyError(
            f"Extension '{name}' not found in FITS. Available: {available}"
        )

def _validate_id(name: str, kind: str) -> str:
    """Valida un identificador de ruta para prevenir traversal y nombres inválidos.

    Solo acepta letras minúsculas, dígitos, guiones bajos y guiones.
    """
    if not _ID_RE.match(name or ""):
        raise ValueError(
            f"Invalid {kind}: {name!r}. "
            f"Only lowercase letters, digits, '_' and '-' are allowed."
        )
    return name


# --------------------------------------------------------------------------- #
# WCS: espacial y espectral
# --------------------------------------------------------------------------- #

def get_spatial_wcs(hdul: fits.HDUList) -> WCS:
    """Construye el WCS espacial RA/Dec a partir del header de ``IMAGE``.

    El WCS conserva las distorsiones SIP definidas en el archivo y se usa para
    convertir entre coordenadas celestes y píxeles del detector.
    """
    return WCS(_get_ext(hdul, "IMAGE").header)


def get_spectral_wcs(hdul: fits.HDUList) -> WCS:
    """
    Construye el WCS espectral de longitud de onda usando ``key='W'``.

    Elimina primero las claves SIP, porque pertenecen al WCS espacial y
    Astropy podría interpretarlas incorrectamente como parte del WCS espectral.

    Args:
        hdul: Archivo FITS abierto que contiene ``IMAGE`` y ``WCS-WAVE``.

    Returns:
        WCS capaz de convertir coordenadas de píxel a longitud de onda y ancho
        de banda.
    """
    header = _strip_sip_keys(_get_ext(hdul, "IMAGE").header)
    w = WCS(header, fobj=hdul, key="W")
    w.sip = None
    return w


def detector_for_wavelength(wavelength_um: float) -> int | None:
    """Identifica el detector cuyo rango contiene una longitud de onda.

    Args:
        wavelength_um: Longitud de onda en micrómetros.

    Returns:
        Número de detector entre 1 y 6, o ``None`` si está fuera de todos los
        rangos conocidos.
    """
    for det, (lo, hi) in DETECTOR_WAVELENGTH_RANGES.items():
        if lo <= wavelength_um <= hi:
            return det
    return None


# --------------------------------------------------------------------------- #
# Mapa de λ (2D)
# --------------------------------------------------------------------------- #

def _quantity_to_array(q, target_unit=u.um) -> np.ndarray:
    """
    Convierte un valor de Astropy a un ``ndarray`` ``float32`` con una unidad.

    Si el objeto tiene método ``to``, se convierte a ``target_unit``; si tiene
    atributo ``value``, se extrae el valor numérico. Los objetos sin unidades
    se interpretan como si ya estuvieran expresados en la unidad objetivo.

    Args:
        q: Quantity, valor escalar o arreglo convertible a NumPy.
        target_unit: Unidad de salida, por defecto micrómetros.

    Returns:
        Arreglo NumPy de tipo ``float32``.
    """
    if hasattr(q, "to"):
        q = q.to(target_unit)
    if hasattr(q, "value"):
        q = q.value
    return np.asarray(q, dtype=np.float32)


def get_wavelength_map(hdul: fits.HDUList, downsample: int = 4) -> np.ndarray:
    """
    Genera un mapa 2D de longitud de onda en micrómetros.

    Evalúa el WCS espectral en una cuadrícula reducida y reinterpola el mapa al
    tamaño completo mediante interpolación bilineal. Esto aprovecha que, en
    SPHEREx, la longitud de onda cambia principalmente con la posición del LVF.

    Args:
        hdul: Archivo FITS abierto con extensión ``IMAGE`` y WCS espectral.
        downsample: Paso de muestreo; 1 conserva resolución completa.

    Returns:
        Matriz ``float32`` de forma ``(alto, ancho)`` con λ en micrómetros.

    Raises:
        ValueError: Si ``IMAGE`` no contiene datos.
        RuntimeError: Si no se puede evaluar el WCS espectral.
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

def get_clean_image(hdul: fits.HDUList, subtract_zodi: bool = True, bad_bits: int = BAD_BITS) -> np.ndarray:
    """
    Obtiene la imagen científica limpia y enmascara píxeles no confiables.

    Parte de ``IMAGE``, sustrae ``ZODI`` opcionalmente y usa ``FLAGS`` para
    convertir a cero los píxeles cuyos bits intersectan ``bad_bits``. Los bits
    no incluidos se conservan. NaN e infinitos también se convierten a cero.

    Args:
        hdul: Archivo FITS abierto.
        subtract_zodi: Indica si debe sustraerse la extensión zodiacal.
        bad_bits: Máscara de bits considerados inválidos.

    Returns:
        Imagen ``float32`` de forma ``(alto, ancho)``.

    Raises:
        ValueError: Si la extensión ``IMAGE`` no contiene datos.
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
# Extracción de un "corte" a una λ dada
# --------------------------------------------------------------------------- #

def extract_2d_at_wavelength(
    hdul: fits.HDUList,
    target_um: float,
    tol_um: float = 0.02,
    downsample: int = 4,
) -> np.ndarray:
    """
    Extrae una imagen 2D correspondiente a una banda espectral objetivo.

    Conserva el flujo de píxeles cuya longitud de onda está dentro de
    ``target_um ± tol_um`` y pone el resto a cero. En SPHEREx esto suele
    producir una franja asociada al patrón LVF.

    Args:
        hdul: Archivo FITS abierto.
        target_um: Longitud de onda central en micrómetros.
        tol_um: Tolerancia espectral alrededor de la longitud objetivo.
        downsample: Paso usado al construir el mapa espectral.

    Returns:
        Imagen ``float32`` con la misma forma que ``IMAGE``.

    Raises:
        RuntimeError: Si el mapa espectral y la imagen tienen formas distintas.
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
    Calcula el rango robusto de longitudes de onda cubierto por la imagen.

    Descarta valores no finitos y fuera del intervalo físico 0.1–50 μm, y usa
    percentiles para evitar que extrapolaciones del WCS en las esquinas
    distorsionen el resultado.

    Args:
        hdul: Archivo FITS abierto.
        downsample: Paso usado al obtener el mapa espectral.
        p_low: Percentil inferior del rango.
        p_high: Percentil superior del rango.

    Returns:
        Tupla ``(mínimo, máximo)`` en micrómetros.

    Raises:
        RuntimeError: Si no se encuentran longitudes de onda válidas.
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
    Muestrea perfiles espectrales a lo largo de los ejes X e Y.

    Calcula longitud de onda y ancho de banda en una fila central recorriendo X
    y en una columna central recorriendo Y. El resultado sirve para diagnosticar
    la geometría LVF del detector.

    Args:
        hdul: Archivo FITS abierto.
        n_samples: Número de posiciones que se muestrean en cada eje.

    Returns:
        Diccionario con la forma de la imagen y listas ``along_x`` y ``along_y``.

    Raises:
        ValueError: Si ``IMAGE`` no contiene datos.
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
    """Extrae metadatos estructurados del nombre estándar de un FITS SPHEREx.

    Args:
        filename: Nombre de archivo, no necesariamente una ruta completa.

    Returns:
        Diccionario con periodo de planificación, ``obs_id``, visita, detector,
        versión y fecha de procesamiento. Devuelve un diccionario vacío si el
        nombre no cumple el patrón esperado.
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
    """Lista los archivos FITS locales de un sector y época.

    Valida ambos identificadores antes de formar la ruta. Si el directorio no
    existe, devuelve una lista vacía en lugar de lanzar una excepción.

    Args:
        sector_id: Identificador del sector.
        epoch: Identificador de la época de observación.
        data_dir: Raíz de datos opcional; si falta, usa la configuración global.

    Returns:
        Lista ordenada de rutas ``.fits``.

    Raises:
        ValueError: Si algún identificador contiene caracteres no permitidos.
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
    Localiza un único FITS según sector, época y detector.

    Puede restringir la búsqueda por ``obs_id`` y ``visit`` para resolver
    ambigüedades entre varios archivos compatibles.

    Raises:
        FileNotFoundError: Si no hay ningún archivo coincidente.
        ValueError: Si hay varios candidatos sin filtros suficientes o si los
            identificadores de ruta no son válidos.
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
    Localiza y abre el FITS correspondiente.

    Si ``data_dir`` se proporciona, la búsqueda es exclusivamente local. Si es
    ``None``, usa el servicio de almacenamiento para descargar el archivo si no
    está disponible localmente.

    Args:
        sector_id: Identificador del sector.
        epoch: Época de observación.
        detector: Número de detector.
        obs_id: Identificador de observación opcional.
        visit: Número de visita opcional.
        data_dir: Raíz local que fuerza el modo sin descarga.

    Returns:
        ``HDUList`` abierto; el llamador debe cerrarlo.
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
# Info de headers (usada por el router /debug)
# --------------------------------------------------------------------------- #

def _extract_coordinates(header: fits.Header) -> dict:
    """Extrae RA y Dec usando la primera clave válida de cada lista conocida.

    Los valores no numéricos se ignoran y se continúa buscando una clave
    alternativa. El resultado conserva también el nombre de la clave utilizada.
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
    """Convierte un header FITS en un diccionario compatible con JSON.

    Omite ``COMMENT`` y ``HISTORY``, convierte escalares NumPy a tipos nativos,
    decodifica bytes y transforma cualquier objeto restante a texto.
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
    """Lee y devuelve los nombres de todas las extensiones de un FITS.

    Args:
        file_path: Ruta del archivo FITS.

    Returns:
        Lista de nombres en el orden en que aparecen en el archivo.
    """
    with fits.open(file_path, memmap=True) as hdul:
        return [hdu.name for hdu in hdul]


def read_header_info(file_path: Path) -> dict:
    """Recopila headers, coordenadas y extensiones de un archivo FITS.

    Lee el header primario y, cuando existe, el de ``IMAGE``. Las coordenadas
    se buscan primero en el primario y luego en ``IMAGE`` si son necesarias.

    Returns:
        Diccionario con ``primary_header``, ``image_header``, ``extensions`` y
        ``coordinates``.
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
    Devuelve una copia sin claves SIP de distorsión espacial.

    Elimina claves con prefijos ``A_``, ``B_``, ``AP_`` y ``BP_`` para evitar
    que Astropy aplique la distorsión espacial al WCS espectral ``WAVE-TAB``.
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
    Serializa los parámetros principales del WCS espacial.

    El resultado contiene valores de referencia, píxeles de referencia, tipos
    de coordenada, matriz de escala y si existe una distorsión SIP; está pensado
    para diagnóstico o reconstrucción en un frontend.

    Returns:
        Diccionario con valores JSON-serializables del WCS de ``IMAGE``.
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
    Convierte lotes de coordenadas RA/Dec a coordenadas de píxel.

    Usa ``all_world2pix`` e incluye la distorsión SIP. Las entradas no finitas
    y los puntos que fallen en el WCS se representan como ``NaN``.

    Args:
        hdul: Archivo FITS abierto.
        ra_deg: Arreglo de ascensiones rectas en grados.
        dec_deg: Arreglo de declinaciones en grados.

    Returns:
        Dos arreglos ``float64`` ``(x, y)`` con la misma forma que las entradas.
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
    Convierte una posición de píxel a RA y Dec en grados.

    Usa el WCS espacial completo de ``IMAGE``, incluida la distorsión SIP.

    Args:
        hdul: Archivo FITS abierto.
        pixel_x: Coordenada X del píxel.
        pixel_y: Coordenada Y del píxel.

    Returns:
        Tupla ``(ra_deg, dec_deg)`` como valores ``float``.
    """
    wcs = get_spatial_wcs(hdul)
    ra, dec = wcs.all_pix2world(pixel_x, pixel_y, 0)
    return float(ra), float(dec)

def get_variance(hdul: fits.HDUList) -> np.ndarray | None:
    """
    Obtiene la extensión ``VARIANCE`` como una copia ``float32``.

    Los valores no finitos se convierten en ``NaN``. Si la extensión falta o no
    contiene datos, devuelve ``None``; no aplica la máscara de ``FLAGS``.

    Args:
        hdul: Archivo FITS abierto.

    Returns:
        Matriz de varianza o ``None`` cuando no está disponible.
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

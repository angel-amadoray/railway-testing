import logging
from pathlib import Path

import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services import fits_loader as fl

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug", tags=["debug"])

class FileSummary(BaseModel):
    filename: str
    size_mb: float
    detector: int | None = None
    visit: int | None = None
    obs_id: str | None = None
    processing_date: str | None = None


class SectorFilesResponse(BaseModel):
    sector_id: str
    epoch: str
    directory: str
    exists: bool
    file_count: int
    files: list[FileSummary]


class HeaderResponse(BaseModel):
    filename: str
    size_mb: float
    parsed: dict
    coordinates: dict
    extensions: list[str]
    primary_header: dict
    image_header: dict | None

class ProfilePoint(BaseModel):
    x: int | None = None
    y: int | None = None
    wavelength_um: float
    bandwidth_um: float

class WavelengthResponse(BaseModel):
    filename: str
    detector_from_name: int | None
    detector_from_header: int | None
    wavelength_range_um: tuple[float, float]
    center_pixel_wavelength_um: float
    profile_along_x: list[ProfilePoint]
    profile_along_y: list[ProfilePoint]
    image_shape: list[int]

class WcsWaveTableResponse(BaseModel):
    extension: str
    n_rows: int
    columns: list[str]
    column_details: dict
    wavelength_min: float
    wavelength_max: float
    wavelength_count: int


class WavelengthGridResponse(BaseModel):
    n: int
    xs: list[int]
    ys: list[int]
    wavelengths_um: list[list[float]]
    bandwidths_um: list[list[float]]


# --------------------------- helpers --------------------------------------- #

def _resolve_file(sector_id: str, epoch: str, filename: str) -> Path:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    p = Path(settings.data_dir) / sector_id / epoch / filename
    if not p.is_file():
        raise HTTPException(404, f"File not found: {p}")
    return p

def _summarize_column(col_data) -> dict:
    """Resume una columna (escalar o vectorial) sin asumir scalaridad."""
    arr = np.asarray(col_data)
    if arr.size == 0:
        return {"kind": "empty"}
    if arr.size == 1:
        val = arr.ravel()[0]
        return {"kind": "scalar", "value": val.item() if hasattr(val, "item") else val}
    flat = arr.ravel()
    n = flat.size
    idx = sorted({0, 1, n // 4, n // 2, 3 * n // 4, n - 2, n - 1})
    samples = [float(flat[i]) for i in idx if i < n]
    return {
        "kind": "vector",
        "shape": list(arr.shape),
        "n_total": int(n),
        "samples_at_indices": idx[: len(samples)],
        "samples": samples,
        "min": float(flat.min()),
        "max": float(flat.max()),
    }

# --------------------------- endpoints ------------------------------------- #

@router.get("/sector/{sector_id}/{epoch}", response_model=SectorFilesResponse)
def debug_sector(sector_id: str, epoch: str):
    sector_dir = Path(settings.data_dir) / sector_id / epoch
    files = fl.find_sector_files(sector_id, epoch)
    summaries = [
        FileSummary(
            filename=f.name,
            size_mb=round(f.stat().st_size / 1e6, 2),
            **{
                k: v
                for k, v in fl.parse_filename_metadata(f.name).items()
                if k in ("detector", "visit", "obs_id", "processing_date")
            },
        )
        for f in files
    ]
    return SectorFilesResponse(
        sector_id=sector_id,
        epoch=epoch,
        directory=str(sector_dir.resolve()),
        exists=sector_dir.is_dir(),
        file_count=len(files),
        files=summaries,
    )


@router.get("/header/{sector_id}/{epoch}/{filename}", response_model=HeaderResponse)
def debug_header(sector_id: str, epoch: str, filename: str):
    p = _resolve_file(sector_id, epoch, filename)
    try:
        info = fl.read_header_info(p)
    except Exception as e:
        logger.exception("Error reading FITS header")
        raise HTTPException(500, f"Error reading FITS: {e}")

    return HeaderResponse(
        filename=filename,
        size_mb=round(p.stat().st_size / 1e6, 2),
        parsed=fl.parse_filename_metadata(filename),
        coordinates=info["coordinates"],
        extensions=info["extensions"],
        primary_header=info["primary_header"],
        image_header=info["image_header"],
    )


@router.get("/wavelength/{sector_id}/{epoch}/{filename}", response_model=WavelengthResponse)
def debug_wavelength(sector_id: str, epoch: str, filename: str):
    """
    Devuelve el rango de λ de la imagen, el valor en el píxel central y dos
    perfiles (a lo largo de X y de Y) para inspeccionar la geometría LVF.
    """
    p = _resolve_file(sector_id, epoch, filename)
    try:
        with fl.open_mef(p) as hdul:
            wmin, wmax = fl.wavelength_range(hdul)
            ny, nx = hdul["IMAGE"].data.shape
            wcs_spec = fl.get_spectral_wcs(hdul)
            center = wcs_spec.pixel_to_world(nx // 2, ny // 2)
            center_w = fl._quantity_to_array(center[0]).item()
            profiles = fl.wavelength_profiles(hdul, n_samples=20)
            det_hdr = int(hdul["IMAGE"].header.get("DETECTOR", 0)) or None
    except Exception as e:
        logger.exception("Error computing wavelength info")
        raise HTTPException(500, f"Error computing wavelength info: {e}")

    return WavelengthResponse(
        filename=filename,
        detector_from_name=fl.parse_filename_metadata(filename).get("detector"),
        detector_from_header=det_hdr,
        wavelength_range_um=(wmin, wmax),
        center_pixel_wavelength_um=float(center_w),
        profile_along_x=[
            ProfilePoint(x=p["x"], wavelength_um=p["wavelength_um"],
                         bandwidth_um=p["bandwidth_um"])
            for p in profiles["along_x"]
        ],
        profile_along_y=[
            ProfilePoint(y=p["y"], wavelength_um=p["wavelength_um"],
                         bandwidth_um=p["bandwidth_um"])
            for p in profiles["along_y"]
        ],
        image_shape=profiles["image_shape"],
    )

@router.get("/wcs_wave/{sector_id}/{epoch}/{filename}", response_model=WcsWaveTableResponse)
def debug_wcs_wave(sector_id: str, epoch: str, filename: str):
    p = _resolve_file(sector_id, epoch, filename)
    try:
        with fl.open_mef(p) as hdul:
            table_hdu = fl._get_ext(hdul, "WCS-WAVE")
            data = table_hdu.data
            if data is None or len(data) == 0:
                raise HTTPException(500, "WCS-WAVE table is empty")

            cols = list(data.names)
            n_rows = len(data)

            column_details = {}
            wave_values = None
            for c in cols:
                column_details[c] = _summarize_column(data[c])
                if c.upper() in ("WAVELENGTH", "VALUES", "WAVE"):
                    wave_values = np.asarray(data[c], dtype=float).ravel()

            if wave_values is None:
                raise HTTPException(500, f"No wavelength column in {cols}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error inspecting WCS-WAVE")
        raise HTTPException(500, f"Error reading WCS-WAVE: {e}")

    return WcsWaveTableResponse(
        extension="WCS-WAVE",
        n_rows=n_rows,
        columns=cols,
        column_details=column_details,
        wavelength_min=float(np.nanmin(wave_values)),
        wavelength_max=float(np.nanmax(wave_values)),
        wavelength_count=int(wave_values.size),
    )

@router.get("/wavelength_grid/{sector_id}/{epoch}/{filename}", response_model=WavelengthGridResponse)
def debug_wavelength_grid(
    sector_id: str, epoch: str, filename: str, n: int = 10
):
    """
    Devuelve una matriz n×n de λ (μm) muestreada uniformemente sobre la imagen.
    Con n=10 tendrás una 'foto' de baja resolución de cómo varía λ en 2D.
    """
    if not (2 <= n <= 50):
        raise HTTPException(400, "n debe estar entre 2 y 50")

    p = _resolve_file(sector_id, epoch, filename)
    try:
        with fl.open_mef(p) as hdul:
            image = hdul["IMAGE"].data
            if image is None:
                raise HTTPException(500, "IMAGE extension has no data")
            ny, nx = image.shape
            xs = np.linspace(0, nx - 1, n).astype(int)
            ys = np.linspace(0, ny - 1, n).astype(int)
            xx, yy = np.meshgrid(xs, ys)
            wcs_spec = fl.get_spectral_wcs(hdul)
            world = wcs_spec.pixel_to_world(xx, yy)
            wave = fl._quantity_to_array(world[0])
            bw = fl._quantity_to_array(world[1])
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing wavelength grid")
        raise HTTPException(500, f"Error computing grid: {e}")

    return WavelengthGridResponse(
        n=n,
        xs=xs.tolist(),
        ys=ys.tolist(),
        wavelengths_um=np.round(wave, 5).tolist(),
        bandwidths_um=np.round(bw, 5).tolist(),
    )
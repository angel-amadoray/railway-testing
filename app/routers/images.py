import logging

from fastapi import APIRouter, HTTPException, Query, Request

from app.core import cache
from app.models.image import CompositeResponse
from app.services import color_engine as ce
from app.services import fits_loader as fl

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])

@router.get("/composite", response_model=CompositeResponse)
def composite(
    request: Request,
    sector_id: str,
    epoch: str,
    detector: int = Query(1, ge=1, le=6),
    stretch_scale: float = Query(10.0, gt=0, le=100),
    p_low: float = Query(1.0, ge=0.0, lt=50.0),
    p_high: float = Query(99.5, gt=50.0, le=100.0),
    quality: int = Query(85, ge=10, le=100),
    downsample: int = Query(8, ge=1, le=32),
    mode: str = Query("per_row", pattern="^(global|per_row)$"),
    fill_masked: bool = Query(True),
    obs_id: str | None = None,
    visit: int | None = None,
):
    hotspot = request.app.state.hotspots.get(sector_id)
    if hotspot is None:
        raise HTTPException(404, f"Sector '{sector_id}' not found")
    if epoch not in hotspot.epochs:
        raise HTTPException(
            400,
            f"Epoch '{epoch}' not available for sector '{sector_id}'. "
            f"Available: {hotspot.epochs}",
        )

    params = {
        "sector_id": sector_id,
        "epoch": epoch,
        "detector": detector,
        "stretch_scale": stretch_scale,
        "p_low": p_low,
        "p_high": p_high,
        "quality": quality,
        "downsample": downsample,
        "mode": mode,                 # ← nuevo
        "fill_masked": fill_masked,   # ← nuevo
    }
    if obs_id:
        params["obs_id"] = obs_id
    if visit is not None:
        params["visit"] = visit

    hit = cache.get("composite", params)
    if hit:
        return CompositeResponse(url=cache.url_for(hit), cached=True, parameters=params)

    try:
        hdul = fl.load_local_fits(
            sector_id=sector_id,
            epoch=epoch,
            detector=detector,
            obs_id=obs_id,
            visit=visit,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))

    try:
        img = ce.spectral_paint(
            hdul,
            stretch_scale=stretch_scale,
            p_low=p_low,
            p_high=p_high,
            downsample=downsample,
            mode=mode,
            fill_masked=fill_masked,
        )
    except Exception as e:
        logger.exception("spectral_paint failed")
        raise HTTPException(500, f"Image processing failed: {e}")
    finally:
        hdul.close()

    buf = ce.to_webp_bytes(img, quality=quality)
    path = cache.put("composite", params, buf)

    return CompositeResponse(url=cache.url_for(path), cached=False, parameters=params)
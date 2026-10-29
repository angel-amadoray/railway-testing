import logging

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.models.astrometry import AstrometryResponse, AstrometryWCS
from app.services import fits_loader as fl

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/astrometry", tags=["astrometry"])


@router.get("/{sector_id}/{epoch}/{detector}", response_model=AstrometryResponse)
def get_astrometry(
    request: Request,
    sector_id: str,
    epoch: str,
    detector: int,
    obs_id: str | None = None,
    visit: int | None = None,
):
    """
    Devuelve el WCS espacial serializable del FITS correspondiente al sector.

    Útil para que el frontend convierta RA/Dec ↔ píxeles si lo necesita, o
    para diagnóstico y debugging.
    """
    hotspot = request.app.state.hotspots.get(sector_id)
    if hotspot is None:
        raise HTTPException(404, f"Sector '{sector_id}' not found")
    if epoch not in hotspot.epochs:
        raise HTTPException(
            400,
            f"Epoch '{epoch}' not available for sector '{sector_id}'. "
            f"Available: {hotspot.epochs}",
        )
    if not (1 <= detector <= 6):
        raise HTTPException(400, f"detector must be 1..6, got {detector}")

    try:
        hdul = fl.load_local_fits(
            sector_id=sector_id, epoch=epoch, detector=detector,
            obs_id=obs_id, visit=visit,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, f"No FITS found: {e}")
    except ValueError as e:
        raise HTTPException(409, str(e))

    try:
        shape = list(hdul["IMAGE"].data.shape)
        wcs_dict = fl.serialize_wcs(hdul)
    except Exception as e:
        logger.exception("Astrometry serialization failed")
        raise HTTPException(500, f"WCS serialization failed: {e}")
    finally:
        hdul.close()

    return AstrometryResponse(
        sector_id=sector_id,
        epoch=epoch,
        detector=detector,
        filename=f"det{detector}",
        shape=shape,
        wcs=AstrometryWCS(**wcs_dict),
    )
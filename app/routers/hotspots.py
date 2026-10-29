from fastapi import APIRouter, HTTPException, Request

from app.models.hotspot import Hotspot

router = APIRouter(prefix="/hotspots", tags=["hotspots"])


@router.get("", response_model=list[Hotspot])
def list_hotspots(request: Request):
    return request.app.state.hotspots.all()


@router.get("/{sector_id}", response_model=Hotspot)
def get_hotspot(sector_id: str, request: Request):
    h = request.app.state.hotspots.get(sector_id)
    if h is None:
        raise HTTPException(status_code=404, detail=f"Sector '{sector_id}' not found")
    return h
from pydantic import BaseModel


class AstrometryWCS(BaseModel):
    crval1: float
    crval2: float
    crpix1: float
    crpix2: float
    ctype1: str
    ctype2: str
    cd: list[list[float]]           # matrix [[cd11, cd12], [cd21, cd22]]
    has_sip: bool


class AstrometryResponse(BaseModel):
    sector_id: str
    epoch: str
    detector: int
    filename: str
    shape: list[int]                # [ny, nx]
    wcs: AstrometryWCS
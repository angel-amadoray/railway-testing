from pydantic import BaseModel


class DetectorSample(BaseModel):
    detector: int
    available: bool
    pixel_x: int | None = None
    pixel_y: int | None = None
    flux: float | None = None
    flux_err: float | None = None
    snr: float | None = None
    valid_fraction_aperture: float = 0.0
    wavelength_center_um: float
    wavelength_range_um: list[float]


class PairClassification(BaseModel):
    classification: str  # ABSORPTION | EMISSION | CONTINUUM | NO_SOURCE | MASKED | UNDETERMINED
    confidence: float
    diagnostic_band: str | None = None
    reason: str


class PairSpectrumResponse(BaseModel):
    sector_id: str
    epoch: str
    ra_deg: float
    dec_deg: float
    pair: str | None = None
    short: DetectorSample | None = None
    long: DetectorSample | None = None
    ratio: float | None = None
    ratio_error: float | None = None
    classification: PairClassification
    cached: bool = False

class BrightPixel(BaseModel):
    pixel_x: int
    pixel_y: int
    flux: float
    ra_deg: float
    dec_deg: float


class BrightestPixelsResponse(BaseModel):
    sector_id: str
    epoch: str
    detector: int
    n_pixels: int
    pixels: list[BrightPixel]

class SuggestedTarget(BaseModel):
    ra_deg: float
    dec_deg: float
    flux_short: float
    flux_long: float
    snr_short: float
    snr_long: float
    pixel_short: tuple[int, int]
    pixel_long: tuple[int, int]


class SuggestTargetResponse(BaseModel):
    sector_id: str
    epoch: str
    pair: str
    target: SuggestedTarget | None = None
    candidates_examined: int = 0
    reason: str = ""

class BrightPixelItem(BaseModel):
    rank: int
    ra_deg: float
    dec_deg: float
    flux_short: float
    flux_long: float
    snr_short: float
    snr_long: float
    ratio: float
    pixel_short_x: int
    pixel_short_y: int
    pixel_long_x: int
    pixel_long_y: int


class BrightestValidResponse(BaseModel):
    sector_id: str
    epoch: str
    pair: str
    n_candidates_scanned: int
    n_valid: int
    pixels: list[BrightPixelItem]

class PairClassification(BaseModel):
    classification: str  # HOT_STAR | SUN_LIKE_STAR | COOL_STAR | VERY_COOL | COOL_DUSTY | UNDETERMINED
    confidence: float
    diagnostic_band: str | None = None
    reason: str
    color_temperature_k: float | None = None
    color_temperature_err_k: float | None = None
from pydantic import BaseModel
from typing import Literal

# for spectral analyzer
class VisualHints(BaseModel):
    base_color_hex: str
    roughness: float
    metalness: float
    emissive_intensity: float
    scale_hint: Literal["small", "medium", "large"]
    suggested_3d_model: Literal["rocky", "star", "unknown"]

class SpherexProfile(BaseModel):
    candidate_class: Literal["STAR_LIKE", "ROCKY_LIKE", "UNKNOWN"]
    confidence: float = 0.0
    reasons: list[str] = []

    # Color (adimensional, interpretable)
    color_ratio: float | None = None
    color_ratio_error: float | None = None
    color_index: float | None = None
    f_short: float | None = None
    f_long: float | None = None

    # Cobertura espectral
    wavelength_range_um: list[float]
    wavelength_range_nm: float = 0.0
    n_spectral_points: int = 0
    reliable: bool = False

    # Fotometría de apertura
    total_flux: float | None = None
    flux_error: float | None = None
    snr: float | None = None

    # Calidad
    valid_fraction_aperture: float = 0.0
    valid_fraction_annulus: float = 0.0
    cutout_edge: bool = False

    visual_hints: VisualHints


class SpectrumResponse(BaseModel):
    sector_id: str
    epoch: str
    detector: int
    pixel_x: int
    pixel_y: int
    wavelengths_um: list[float]
    flux_per_row: list[float]
    profile: SpherexProfile

# for jpl integration

class KnownObject(BaseModel):
    name: str
    ra: str | None = None
    dec: str | None = None
    vmag: str | None = None
    ra_offset_arcsec: str | None = None
    dec_offset_arcsec: str | None = None
    total_offset_arcsec: str | None = None
    rate_arcsec_min: str | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    pixel_x: float | None = None
    pixel_y: float | None = None
    in_frame: bool | None = None


class KnownObjectsResponse(BaseModel):
    sector_id: str
    epoch: str
    ra_center_deg: float
    dec_center_deg: float
    obs_time_iso: str
    two_pass: bool
    mag_required: bool
    mode: str
    # Pipeline de filtrado
    n_objects_raw: int
    n_objects_after_offset: int
    n_objects_after_vmag: int
    n_objects: int
    filtered_out_offset: int
    filtered_out_vmag: int
    truncated: int
    # Límites aplicados
    max_offset_arcsec_allowed: float | None = None
    max_vmag: float | None = None
    max_objects: int | None = None
    # Contadores JPL
    n_first_pass: int
    n_second_pass: int | None = None
    objects: list[KnownObject]
    cached: bool

class DifferenceResponse(BaseModel):
    url: str
    cached: bool
    parameters: dict
    threshold: float
    n_pixels_above: int
    n_pixels_total: int
    metadata: dict

class IdentifyResponse(BaseModel):
    sector_id: str
    epoch: str
    detector: int
    pixel_x: int
    pixel_y: int
    ra_deg: float
    dec_deg: float
    tolerance_arcsec: float
    n_known: int
    known_objects: list[KnownObject]
    is_known: bool
    is_candidate: bool
    spherex_profile: SpherexProfile | None = None
    cached: bool
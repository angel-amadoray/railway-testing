import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Hotspot(BaseModel):
    id: str = Field(..., min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(..., min_length=1)
    description: str = ""
    ra: float = Field(..., ge=0, le=360)
    dec: float = Field(..., ge=-90, le=90)
    size_deg: float = Field(..., gt=0, le=10)
    epochs: List[str] = Field(default_factory=list)
    thumbnail_url: Optional[str] = None

    @field_validator("epochs")
    @classmethod
    def _unique_epochs(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("epochs must not contain duplicates")
        return v


class HotspotsCatalog:
    """Carga y valida data/hotspots.json una sola vez al arranque."""

    def __init__(self, path: Path):
        self.path = path
        self._by_id: dict[str, Hotspot] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Hotspots catalog not found at {self.path.resolve()}. "
                f"Run from the backend/ directory."
            )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{self.path} must contain a JSON array")
        for item in raw:
            h = Hotspot(**item)
            if h.id in self._by_id:
                raise ValueError(f"Duplicate hotspot id: {h.id}")
            self._by_id[h.id] = h

    def all(self) -> List[Hotspot]:
        return list(self._by_id.values())

    def get(self, sector_id: str) -> Optional[Hotspot]:
        return self._by_id.get(sector_id)
"""Cache en disco para imágenes procesadas. Clave = hash MD5 estable."""

import hashlib
import json
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def make_key(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _path_for(namespace: str, key: str) -> Path:
    return Path(settings.cache_dir) / namespace / f"{key}.webp"


def get(namespace: str, params: dict) -> Path | None:
    p = _path_for(namespace, make_key(params))
    if p.is_file():
        logger.info("cache_hit namespace=%s key=%s", namespace, p.stem)
        return p
    logger.info("cache_miss namespace=%s key=%s", namespace, _path_for(namespace, make_key(params)).stem)
    return None


def put(namespace: str, params: dict, content: bytes) -> Path:
    p = _path_for(namespace, make_key(params))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    logger.info("cache_put namespace=%s path=%s size=%d", namespace, p, len(content))
    return p


def url_for(path: Path) -> str:
    """Convierte un Path absoluto dentro de static/ a URL pública."""
    rel = path.relative_to(Path(settings.static_dir))
    return f"/static/{rel.as_posix()}"
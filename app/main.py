import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.logging import setup_logging
from app.models.hotspot import HotspotsCatalog
from app.routers import debug, hotspots, images, analytics, astrometry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger.info("Starting SPHEREx backend (env=%s)", settings.environment)

    for d in (settings.static_dir, settings.cache_dir, settings.data_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    catalog = HotspotsCatalog(Path(settings.hotspots_path))
    app.state.hotspots = catalog
    logger.info("Loaded %d hotspots from %s", len(catalog.all()), settings.hotspots_path)

    yield

    logger.info("Shutting down SPHEREx backend")


app = FastAPI(
    title="SPHEREx Backend",
    version="0.1.0",
    description="Backend para procesar imágenes espectrales de SPHEREx (NASA).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static = Path(settings.static_dir)
_static.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/health", tags=["system"])
def health():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": app.version,
    }


app.include_router(hotspots.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(astrometry.router, prefix="/api")
app.include_router(debug.router, prefix="/api")
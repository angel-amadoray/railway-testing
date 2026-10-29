from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: str = "development"
    port: int = 8000
    log_level: str = "INFO"

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])  

    s3_bucket: str = "nasa-irsa-spherex"
    s3_region: str = "us-east-1"

    # 🆕 OCI Object Storage (S3-compatible, para persistir tus FITS)
    oci_endpoint: str = os.getenv("OCI_ENDPOINT")
    oci_access_key: str = os.getenv("OCI_ACCESS_KEY")
    oci_secret_key: str = os.getenv("OCI_SECRET_KEY")
    oci_bucket: str = os.getenv("OCI_BUCKET")
    oci_region: str = os.getenv("OCI_REGION")
    oci_prefix: str = os.getenv("OCI_PREFIX")
    oci_download_enabled: bool = os.getenv("OCI_DOWNLOAD_ENABLED")

    static_dir: str = "app/static"
    cache_dir: str = "app/static/cache"
    data_dir: str = "data/sample_fits"
    hotspots_path: str = "data/hotspots.json"


settings = Settings()
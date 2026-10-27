"""Runtime configuration from environment variables (optionally loaded from a local .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_STAC_API_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_STAC_COLLECTION = "sentinel-2-l2a"

# Public Sentinel-2 COGs on AWS need no credentials; these make GDAL fast for HTTP range reads.
GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


@dataclass(frozen=True)
class Settings:
    stac_api_url: str
    stac_collection: str
    cache_dir: Path
    output_dir: Path


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        stac_api_url=os.getenv("STAC_API_URL", DEFAULT_STAC_API_URL),
        stac_collection=os.getenv("STAC_COLLECTION", DEFAULT_STAC_COLLECTION),
        cache_dir=Path(os.getenv("ASK_SAT_CACHE_DIR", "data/cache")),
        output_dir=Path(os.getenv("ASK_SAT_OUTPUT_DIR", "outputs")),
    )

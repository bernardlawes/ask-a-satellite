"""Synthetic Sentinel-2-like AOI rasters with known ground truth."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from ask_a_satellite.models import BandAsset, SceneInfo
from ask_a_satellite.tools.raster import AOIRaster

UTM17N = CRS.from_epsg(32617)
ORIGIN = (588000.0, 2886000.0)  # near Port Everglades
SHAPE = (200, 200)
LAND_COLS = 60

# (row0, col0, height, width) of synthetic ships in open water
SHIPS = [(40, 100, 3, 12), (90, 120, 4, 22), (130, 80, 2, 6)]
TINY_BOAT = (20, 170, 1, 1)
CLOUD = (150, 140, 35, 50)  # rows, cols, h, w; contains a hidden ship
HIDDEN_SHIP = (160, 160, 3, 10)


def make_scene(scene_id: str = "S2X_TEST_20260101_0_L2A") -> SceneInfo:
    assets = {k: BandAsset(href=f"memory://{k}") for k in ("blue", "green", "red", "nir", "scl")}
    return SceneInfo(id=scene_id, datetime=datetime(2026, 1, 1, 16, 0, tzinfo=UTC),
                     cloud_cover=5.0, assets=assets)


def synthetic_bands(seed: int = 0, ships=SHIPS, cloud: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = SHAPE
    green = np.full(SHAPE, 0.06, np.float32) + rng.normal(0, 0.002, SHAPE).astype(np.float32)
    nir = np.full(SHAPE, 0.02, np.float32) + rng.normal(0, 0.002, SHAPE).astype(np.float32)
    red = np.full(SHAPE, 0.04, np.float32)
    blue = np.full(SHAPE, 0.07, np.float32)
    scl = np.full(SHAPE, 6, np.uint8)  # water

    # land on the left
    green[:, :LAND_COLS], nir[:, :LAND_COLS], red[:, :LAND_COLS] = 0.08, 0.30, 0.09
    scl[:, :LAND_COLS] = 5

    def paint(r, c, hh, ww, nir_v=0.15, green_v=0.07):
        green[r:r + hh, c:c + ww] = green_v
        nir[r:r + hh, c:c + ww] = nir_v
        red[r:r + hh, c:c + ww] = 0.12

    for s in ships:
        paint(*s)
    paint(*TINY_BOAT)
    if cloud:
        r, c, hh, ww = CLOUD
        paint(r, c, hh, ww, nir_v=0.45, green_v=0.40)
        scl[r:r + hh, c:c + ww] = 9
        paint(*HIDDEN_SHIP, nir_v=0.5, green_v=0.42)
    return {"blue": blue, "green": green, "red": red, "nir": nir, "scl": scl}


@pytest.fixture
def synthetic_aoi() -> AOIRaster:
    return AOIRaster(scene=make_scene(), bands=synthetic_bands(),
                     transform=from_origin(*ORIGIN, 10, 10), crs=UTM17N)

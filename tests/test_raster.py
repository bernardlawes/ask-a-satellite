from datetime import UTC, datetime

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from ask_a_satellite.models import BandAsset, BBox, SceneInfo
from ask_a_satellite.tools.raster import read_aoi

from .conftest import UTM17N

X0, Y0 = 580000.0, 2890000.0
N10 = 400  # 4 km tile at 10 m


def _write(path, data, res):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype=data.dtype, crs=UTM17N, transform=from_origin(X0, Y0, res, res),
                       nodata=0) as dst:
        dst.write(data, 1)


@pytest.fixture
def local_scene(tmp_path):
    assets = {}
    rows, cols = np.mgrid[0:N10, 0:N10]
    for i, name in enumerate(("blue", "green", "red", "nir")):
        dn = (1000 + 100 * i + cols).astype(np.uint16)  # DN increases with column
        dn[:5, :5] = 0  # nodata corner
        _write(tmp_path / f"{name}.tif", dn, 10)
        assets[name] = BandAsset(href=str(tmp_path / f"{name}.tif"), scale=0.0001, offset=-0.1)
    r20, c20 = np.mgrid[0:N10 // 2, 0:N10 // 2]
    scl = np.where(c20 < 100, 6, 9).astype(np.uint8)  # water left half, cloud right half
    _write(tmp_path / "scl.tif", scl, 20)
    assets["scl"] = BandAsset(href=str(tmp_path / "scl.tif"))
    return SceneInfo(id="LOCAL", datetime=datetime(2026, 1, 1, tzinfo=UTC), assets=assets)


def _bbox_for(x0, y0, x1, y1):
    w, s, e, n = transform_bounds(UTM17N, "EPSG:4326", x0, y0, x1, y1)
    return BBox(west=w, south=s, east=e, north=n)


def test_read_aoi_window_scaling_and_resampling(local_scene):
    # 1 km x 1 km box centred in the tile, straddling the SCL water/cloud boundary at x = X0 + 2000
    bbox = _bbox_for(X0 + 1500, Y0 - 2500, X0 + 2500, Y0 - 1500)
    aoi = read_aoi(local_scene, bbox)

    h, w = aoi.shape
    assert 100 <= h <= 110 and 100 <= w <= 110  # lat/lon box slightly larger than the UTM box
    assert aoi.pixel_size_m == 10
    assert aoi.crs == UTM17N

    # reflectance = DN * 1e-4 - 0.1, DN = 1200 + col for red
    col0 = round((aoi.transform.c - X0) / 10)
    expected = (1200 + col0) * 1e-4 - 0.1
    assert aoi.bands["red"][0, 0] == pytest.approx(expected, abs=1e-6)
    assert aoi.bands["red"].dtype == np.float32

    scl = aoi.bands["scl"]
    assert scl.shape == aoi.shape and scl.dtype == np.uint8
    boundary_col = round((X0 + 2000 - aoi.transform.c) / 10)
    assert set(np.unique(scl[:, : boundary_col - 1])) == {6}
    assert set(np.unique(scl[:, boundary_col + 1 :])) == {9}


def test_read_aoi_marks_nodata_as_nan(local_scene):
    bbox = _bbox_for(X0, Y0 - 300, X0 + 300, Y0)
    aoi = read_aoi(local_scene, bbox)
    assert np.isnan(aoi.bands["nir"][0, 0])
    assert np.isfinite(aoi.bands["nir"][10, 10])


def test_read_aoi_outside_footprint_raises(local_scene):
    bbox = _bbox_for(X0 + 10000, Y0 - 10000, X0 + 11000, Y0 - 9000)
    with pytest.raises(ValueError, match="overlap"):
        read_aoi(local_scene, bbox)

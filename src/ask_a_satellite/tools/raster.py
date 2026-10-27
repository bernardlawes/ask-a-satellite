"""Windowed reads of an AOI from Sentinel-2 Cloud-Optimized GeoTIFFs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

from ..config import GDAL_ENV
from ..models import BBox, SceneInfo

REFLECTANCE_BANDS = ("blue", "green", "red", "nir")
OPTIONAL_BANDS = ("swir16",)  # 20 m; read when the scene has it (cloud-puff test)
REFERENCE_BAND = "red"  # a 10 m band defines the output grid


def default_bands(scene: SceneInfo) -> tuple[str, ...]:
    return (*REFLECTANCE_BANDS, *(b for b in OPTIONAL_BANDS if b in scene.assets), "scl")


@dataclass
class AOIRaster:
    """Bands for one AOI on a common 10 m grid. Reflectance bands are float32; ``scl`` is uint8."""

    scene: SceneInfo
    bands: dict[str, np.ndarray]
    transform: Affine
    crs: rasterio.crs.CRS

    @property
    def shape(self) -> tuple[int, int]:
        return self.bands[REFERENCE_BAND].shape

    @property
    def pixel_size_m(self) -> float:
        return abs(self.transform.a)


def _snap_window(win: Window, width: int, height: int) -> Window:
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    col0, row0 = max(int(win.col_off), 0), max(int(win.row_off), 0)
    col1 = min(int(win.col_off + win.width), width)
    row1 = min(int(win.row_off + win.height), height)
    if col1 <= col0 or row1 <= row0:
        raise ValueError("AOI does not overlap the scene footprint")
    return Window(col0, row0, col1 - col0, row1 - row0)


def read_aoi(scene: SceneInfo, bbox: BBox, bands: tuple[str, ...] | None = None) -> AOIRaster:
    """Read ``bands`` (default: all the pipeline uses that the scene has) for ``bbox`` from ``scene``.
    Only the AOI window is fetched over HTTP.

    Reflectance = DN * scale + offset (from the asset's ``raster:bands``); nodata (DN 0) becomes NaN.
    Coarser bands (SCL, SWIR, 20 m) are resampled with nearest neighbour onto the 10 m reference grid.
    """
    bands = bands or default_bands(scene)
    out: dict[str, np.ndarray] = {}
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(scene.assets[REFERENCE_BAND].href) as ref:
            crs = ref.crs
            native_bounds = transform_bounds("EPSG:4326", crs, *bbox.as_tuple(), densify_pts=21)
            win = _snap_window(from_bounds(*native_bounds, transform=ref.transform), ref.width, ref.height)
            transform = ref.window_transform(win)
            shape = (int(win.height), int(win.width))
            aoi_bounds = window_bounds(win, ref.transform)

        for name in bands:
            asset = scene.assets[name]
            with rasterio.open(asset.href) as src:
                if src.crs != crs:
                    raise ValueError(f"band {name} CRS {src.crs} differs from reference {crs}")
                bwin = from_bounds(*aoi_bounds, transform=src.transform)
                data = src.read(1, window=bwin, out_shape=shape, resampling=Resampling.nearest,
                                boundless=True, fill_value=0)
            if name == "scl":
                out[name] = data.astype(np.uint8)
            else:
                refl = data.astype(np.float32) * asset.scale + asset.offset
                refl[data == 0] = np.nan
                out[name] = refl

    return AOIRaster(scene=scene, bands=out, transform=transform, crs=crs)


def _cache_path(cache_dir: Path, scene: SceneInfo, bbox: BBox, bands: tuple[str, ...]) -> Path:
    key = json.dumps([scene.id, [round(v, 6) for v in bbox.as_tuple()], sorted(bands)])
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{scene.id}_{digest}.npz"


def read_aoi_cached(scene: SceneInfo, bbox: BBox, cache_dir: Path | None,
                    bands: tuple[str, ...] | None = None) -> AOIRaster:
    """``read_aoi`` with an on-disk .npz cache keyed by scene, bbox, and bands (``None`` disables it)."""
    bands = bands or default_bands(scene)
    if cache_dir is None:
        return read_aoi(scene, bbox, bands)
    path = _cache_path(cache_dir, scene, bbox, bands)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return AOIRaster(scene=scene, bands={b: z[b] for b in bands},
                             transform=Affine(*z["_transform"]), crs=CRS.from_wkt(str(z["_crs"])))
    aoi = read_aoi(scene, bbox, bands)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, _transform=np.array(aoi.transform[:6]), _crs=np.array(aoi.crs.to_wkt()),
                        **aoi.bands)
    tmp.replace(path)
    return aoi

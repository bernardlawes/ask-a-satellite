"""Masks and visualisation helpers derived from AOI bands."""

from __future__ import annotations

import numpy as np

# Sentinel-2 L2A Scene Classification (SCL) classes treated as "not observable".
SCL_NODATA = 0
SCL_SATURATED = 1
SCL_CLOUD_SHADOW = 3
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_CIRRUS = 10
SCL_CLOUDY = (SCL_CLOUD_SHADOW, SCL_CLOUD_MEDIUM, SCL_CLOUD_HIGH, SCL_CIRRUS)


def cloud_mask(scl: np.ndarray) -> np.ndarray:
    return np.isin(scl, SCL_CLOUDY)


def valid_mask(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Pixels with data in every reflectance band and not nodata/saturated in SCL."""
    valid = np.ones(bands["green"].shape, dtype=bool)
    for name in ("blue", "green", "red", "nir"):
        if name in bands:
            valid &= np.isfinite(bands[name])
    if "scl" in bands:
        valid &= ~np.isin(bands["scl"], (SCL_NODATA, SCL_SATURATED))
    return valid


def cloud_fraction(scl: np.ndarray) -> float:
    """Fraction of data pixels in the AOI flagged as cloud, cirrus, or cloud shadow."""
    data = scl != SCL_NODATA
    n = int(data.sum())
    return float(cloud_mask(scl)[data].sum() / n) if n else 1.0


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (green - nir) / (green + nir). Water is positive; ships and land are negative."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (green - nir) / (green + nir)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def true_color(bands: dict[str, np.ndarray], low_pct: float = 2, high_pct: float = 98,
               gamma: float = 1.0) -> np.ndarray:
    """RGB uint8 image using a shared percentile stretch (keeps colour balance across channels)."""
    rgb = np.stack([bands["red"], bands["green"], bands["blue"]], axis=-1).astype(np.float32)
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros(rgb.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    scaled = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        scaled = scaled ** (1 / gamma)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return (scaled * 255).round().astype(np.uint8)

"""Classical large-vessel detector for 10 m Sentinel-2 imagery.

Ships are bright in NIR against dark water:
  1. water = NDWI > water_ndwi
  2. sea   = large connected water bodies (drops pools/shadows on land), small closing, then fill only
             ship-sized holes (dark-hulled ships punch low-NDWI holes; big holes are islands/land)
  3. cand  = sea & ~cloud & nir above a robust water background (median + k * MAD) and a floor.
             Not "& ~water": white hulls reflect green ~= NIR, so their NDWI sits near 0 and many of
             their pixels pass the water test.
  4. connected components, filtered by size, then rejected if they touch land or sit < 60 m from it
     (surf, jetties, roofs -- and, unavoidably at 10 m, ships berthed flush against a quay), are thin
     strips (surf lines, wakes), vegetated, or very bright shallow water; measured with image moments.

Known misses/false alarms: ships berthed flush against a quay (they merge with land), piers, breaking
wakes, sun-glint, adjacent ships merging into one blob. See docs/architecture.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform
from scipy import ndimage

from ..models import Detection, DetectionResult
from .preprocess import cloud_fraction, cloud_mask, ndwi, valid_mask
from .raster import AOIRaster

METHOD = "classical-ndwi-nir-v1"


@dataclass(frozen=True)
class ClassicalParams:
    water_ndwi: float = 0.0
    min_water_body_px: int = 2000  # 0.2 km^2: open water / port basins, not pools or shadows
    close_radius_px: int = 2
    max_hole_px: int = 600  # larger enclosed non-water areas are islands, not ships
    # Blob rejection rules, derived from feature measurements on the 2026-10-21 scene
    # (docs/dev-log.md). Provisional until validated against hand labels.
    min_shore_dist_px: float = 6.0  # median distance to land; surf and mixed shore pixels sit closer
    thin_width_px: float = 1.6  # ...and thin_elongation: surf lines, wakes, causeways
    thin_elongation: float = 8.0
    max_ndvi: float = 0.5  # vegetated islets, seagrass
    max_ndwi: float = 0.45  # bright shallow water / surf over sand (white hulls measure 0.1-0.4)
    k_mad: float = 6.0
    nir_floor: float = 0.04
    min_area_px: int = 4
    max_area_px: int = 1500


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def _remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[labels]


def sea_mask(water: np.ndarray, radius: int, min_body_px: int = 0, max_hole_px: int | None = None,
             ) -> np.ndarray:
    body = _remove_small(water, min_body_px) if min_body_px else water
    if radius > 0:
        # Pad so closing does not erode water that touches the AOI edge.
        pad = radius + 1
        padded = np.pad(body, pad, mode="edge")
        body = ndimage.binary_closing(padded, structure=_disk(radius))[pad:-pad, pad:-pad] | body
    holes = ndimage.binary_fill_holes(body) & ~body
    if max_hole_px is not None:
        holes &= ~_remove_small(holes, max_hole_px + 1)
    return body | holes


def water_background(nir: np.ndarray, water: np.ndarray) -> tuple[float, float]:
    """Robust (median, sigma) of NIR over water; sigma = 1.4826 * MAD."""
    vals = nir[water & np.isfinite(nir)]
    if vals.size < 50:
        return float("nan"), float("nan")
    med = float(np.median(vals))
    sigma = float(1.4826 * np.median(np.abs(vals - med)))
    return med, max(sigma, 1e-4)


@dataclass
class Blob:
    label: int
    rows: np.ndarray
    cols: np.ndarray
    peak_z: float


def _shape(rows: np.ndarray, cols: np.ndarray) -> tuple[float, float]:
    """(length, width) in pixels from the coordinate covariance (uniform bar: var = L^2 / 12)."""
    if rows.size < 2:
        return 1.0, 1.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([rows, cols]).astype(float)))
    return max(float(np.sqrt(12 * ev[1])), 1.0), float(np.sqrt(12 * max(ev[0], 0.0)))


def reject_reason(rows: np.ndarray, cols: np.ndarray, dist_to_land: np.ndarray, nd: np.ndarray,
                  ndvi: np.ndarray, p: ClassicalParams) -> str | None:
    d = dist_to_land[rows, cols]
    if d.min() <= 1.0:
        return "touches_land"
    if np.median(d) < p.min_shore_dist_px:
        return "near_shore"
    length, width = _shape(rows, cols)
    if width < p.thin_width_px and length / max(width, 0.5) > p.thin_elongation:
        return "thin_strip"
    if np.nanmean(ndvi[rows, cols]) > p.max_ndvi:
        return "vegetation"
    if np.nanmean(nd[rows, cols]) > p.max_ndwi:
        return "shallow_bright_water"
    return None


def find_blobs(bands: dict[str, np.ndarray], cloud: np.ndarray, valid: np.ndarray,
               p: ClassicalParams) -> tuple[list[Blob], dict[str, float]]:
    """Array-level detector. Returns candidate blobs and diagnostics."""
    green, nir = bands["green"], bands["nir"]
    nd = ndwi(green, nir)
    water = (nd > p.water_ndwi) & valid & ~cloud
    sea = sea_mask(water, p.close_radius_px, p.min_water_body_px, p.max_hole_px)
    med, sigma = water_background(nir, water)
    diag = {
        "water_fraction": float(water.sum() / max(valid.sum(), 1)),
        "water_nir_median": med,
        "water_nir_sigma": sigma,
    }
    if not np.isfinite(med):
        return [], diag

    thresh = max(med + p.k_mad * sigma, p.nir_floor)
    with np.errstate(invalid="ignore"):
        cand = sea & ~cloud & valid & (nir > thresh)
    land = valid & ~sea & ~cloud
    dist_to_land = ndimage.distance_transform_edt(~land)
    ndvi = -nd if "red" not in bands else _ndvi(bands["red"], nir)
    labels, n = ndimage.label(cand, structure=np.ones((3, 3), dtype=bool))
    blobs: list[Blob] = []
    rejected = dict.fromkeys(REJECT_REASONS, 0)
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        rr, cc = np.nonzero(labels[sl] == i)
        if not (p.min_area_px <= rr.size <= p.max_area_px):
            continue
        rr = rr + sl[0].start
        cc = cc + sl[1].start
        reason = reject_reason(rr, cc, dist_to_land, nd, ndvi, p)
        if reason:
            rejected[reason] += 1
            continue
        peak = float(np.nanmax(nir[rr, cc]))
        blobs.append(Blob(label=i, rows=rr, cols=cc, peak_z=(peak - med) / sigma))
    diag["nir_threshold"] = thresh
    diag.update({f"rejected_{k}": v for k, v in rejected.items()})
    return blobs, diag


REJECT_REASONS = ("touches_land", "near_shore", "thin_strip", "vegetation", "shallow_bright_water")


def _ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir - red) / (nir + red)


def _score(peak_z: float, area_px: int, k: float) -> float:
    contrast = np.clip((peak_z - k) / (3 * k), 0, 1)
    size = np.clip(area_px / 20, 0, 1)
    return round(float(0.7 * contrast + 0.3 * size), 3)


def detect_vessels(aoi: AOIRaster, aoi_id: str, params: ClassicalParams | None = None) -> DetectionResult:
    p = params or ClassicalParams()
    b = aoi.bands
    cloud = cloud_mask(b["scl"])
    valid = valid_mask(b)
    blobs, diag = find_blobs(b, cloud, valid, p)

    px = aoi.pixel_size_m
    rows_c = [float(bl.rows.mean()) for bl in blobs]
    cols_c = [float(bl.cols.mean()) for bl in blobs]
    if blobs:
        xs, ys = xy(aoi.transform, rows_c, cols_c, offset="center")
        lons, lats = warp_transform(aoi.crs, "EPSG:4326", list(xs), list(ys))
    else:
        lons, lats = [], []

    detections = []
    for i, bl in enumerate(sorted(zip(blobs, rows_c, cols_c, lons, lats, strict=True),
                                  key=lambda t: (t[1], t[2])), start=1):
        blob, r, c, lon, lat = bl
        detections.append(Detection(
            id=i, row=round(r, 2), col=round(c, 2),
            bbox_px=(int(blob.rows.min()), int(blob.cols.min()),
                     int(blob.rows.max()) + 1, int(blob.cols.max()) + 1),
            lon=round(float(lon), 6), lat=round(float(lat), 6),
            area_px=int(blob.rows.size), area_m2=float(blob.rows.size * px * px),
            length_m=round(_shape(blob.rows, blob.cols)[0] * px, 1),
            score=_score(blob.peak_z, blob.rows.size, p.k_mad),
        ))

    cf = cloud_fraction(b["scl"])
    notes = []
    if cf > 0.2:
        notes.append(f"{cf:.0%} of the AOI is cloud/shadow; vessels under cloud are not counted.")
    if diag["water_fraction"] < 0.05:
        notes.append("Very little clear water detected; count is unreliable.")
    if not np.isfinite(diag["water_nir_median"]):
        notes.append("Not enough clear water to estimate background; no detections attempted.")

    return DetectionResult(
        aoi_id=aoi_id, scene_id=aoi.scene.id, datetime=aoi.scene.datetime, method=METHOD,
        count=len(detections), detections=detections, aoi_cloud_fraction=round(cf, 4),
        water_fraction=round(diag["water_fraction"], 4), pixel_size_m=px,
        params={**asdict(p), **{k: round(v, 5) for k, v in diag.items()
                                if k != "water_fraction" and np.isfinite(v)}},
        notes=notes,
    )

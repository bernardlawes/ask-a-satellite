"""Deterministic end-to-end pipeline (no LLM): pick a clear scene for an AOI and count vessels.

The agent (Milestone 3) calls the same tools individually; this module is the scripted path used by
the CLI and for building the evaluation set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .models import AOI, DetectionResult, SceneInfo
from .tools.annotate import save_annotated
from .tools.imagery import search_scenes
from .tools.preprocess import cloud_fraction, true_color
from .tools.raster import AOIRaster, read_aoi_cached
from .tools.vessels import ClassicalParams, detect_vessels

log = logging.getLogger(__name__)


@dataclass
class Observation:
    raster: AOIRaster
    result: DetectionResult


@dataclass
class Rejected:
    scene: SceneInfo
    reason: str


def clear_observations(aoi: AOI, start: date | str, end: date | str, max_scene_cloud: float = 30,
                       max_aoi_cloud: float = 0.1, params: ClassicalParams | None = None,
                       cache_dir: Path | None = None, limit: int | None = None,
                       ) -> tuple[list[Observation], list[Rejected]]:
    """Walk scenes newest-first; return up to ``limit`` whose AOI cloud fraction is acceptable."""
    found: list[Observation] = []
    rejected: list[Rejected] = []
    for scene in search_scenes(aoi.bbox, start, end, max_cloud=max_scene_cloud):
        if limit is not None and len(found) >= limit:
            break
        raster = read_aoi_cached(scene, aoi.bbox, cache_dir)
        cf = cloud_fraction(raster.bands["scl"])
        if cf > max_aoi_cloud:
            rejected.append(Rejected(scene, f"AOI cloud {cf:.0%} > {max_aoi_cloud:.0%}"))
            log.info("skip %s: AOI cloud %.0f%%", scene.id, cf * 100)
            continue
        found.append(Observation(raster, detect_vessels(raster, aoi.id, params)))
    return found, rejected


def first_clear_observation(aoi: AOI, start: date | str, end: date | str, max_scene_cloud: float = 30,
                            max_aoi_cloud: float = 0.1, params: ClassicalParams | None = None,
                            cache_dir: Path | None = None,
                            ) -> tuple[Observation | None, list[Rejected]]:
    found, rejected = clear_observations(aoi, start, end, max_scene_cloud, max_aoi_cloud, params,
                                         cache_dir, limit=1)
    return (found[0] if found else None), rejected


def write_outputs(obs: Observation, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = obs.result
    stem = f"{r.aoi_id}_{r.datetime:%Y%m%d}"
    png = save_annotated(true_color(obs.raster.bands), r, out_dir / f"{stem}_detections.png")
    js = out_dir / f"{stem}_detections.json"
    js.write_text(json.dumps(r.model_dump(mode="json"), indent=2), encoding="utf-8")
    return {"png": png, "json": js}

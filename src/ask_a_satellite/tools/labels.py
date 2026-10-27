"""Label kits for hand-labelling (Roboflow-compatible YOLO format) and reading the labels back.

Layout of an eval directory:
  images/<stem>.png   upscaled true colour (not committed; regenerable)
  labels/<stem>.txt   machine pre-annotations from the current detector (not committed)
  data.yaml           class names for Roboflow import
  meta/<stem>.json    scene + AOI needed to re-run detection on exactly the same pixels (committed)
  truth/<stem>.txt    hand-corrected labels exported from Roboflow as YOLO (committed: the test set)

YOLO coordinates are normalised to image size, so the upscale factor does not affect them.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .. import IMAGERY_CREDIT
from ..models import BBox, DetectionResult, SceneInfo
from .preprocess import true_color
from .raster import AOIRaster

CLASS_NAMES = ("vessel",)

BoxPx = tuple[float, float, float, float]  # row_min, col_min, row_max, col_max (exclusive)


def to_yolo(box: BoxPx, shape: tuple[int, int], cls: int = 0) -> str:
    h, w = shape
    r0, c0, r1, c1 = box
    return f"{cls} {(c0 + c1) / 2 / w:.6f} {(r0 + r1) / 2 / h:.6f} {(c1 - c0) / w:.6f} {(r1 - r0) / h:.6f}"


def from_yolo(line: str, shape: tuple[int, int]) -> BoxPx:
    h, w = shape
    _, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
    return (h * (cy - bh / 2), w * (cx - bw / 2), h * (cy + bh / 2), w * (cx + bw / 2))


def stem_for(result: DetectionResult) -> str:
    return f"{result.aoi_id}_{result.datetime:%Y%m%d}"


def write_label_kit(raster: AOIRaster, result: DetectionResult, bbox: BBox, eval_dir: Path,
                    scale: int = 3) -> str:
    eval_dir = Path(eval_dir)
    stem = stem_for(result)
    for sub in ("images", "labels", "meta"):
        (eval_dir / sub).mkdir(parents=True, exist_ok=True)
    rgb = true_color(raster.bands)
    h, w = raster.shape
    img = Image.fromarray(rgb).resize((w * scale, h * scale), Image.NEAREST)
    img.save(eval_dir / "images" / f"{stem}.png")
    lines = [to_yolo(d.bbox_px, (h, w)) for d in result.detections]
    (eval_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                      encoding="utf-8")
    meta = {
        "stem": stem,
        "aoi_id": result.aoi_id,
        "aoi_bbox": bbox.model_dump(),
        "scene": raster.scene.model_dump(mode="json"),
        "shape": [h, w],
        "upscale": scale,
        "preannotation_method": result.method,
        "credit": IMAGERY_CREDIT.format(year=result.datetime.year),
    }
    (eval_dir / "meta" / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (eval_dir / "data.yaml").write_text(
        f"names: {list(CLASS_NAMES)}\nnc: {len(CLASS_NAMES)}\n", encoding="utf-8")
    return stem


def load_meta(path: Path) -> tuple[dict, SceneInfo, BBox]:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    return meta, SceneInfo.model_validate(meta["scene"]), BBox.model_validate(meta["aoi_bbox"])


def read_yolo_boxes(path: Path, shape: tuple[int, int]) -> list[BoxPx]:
    text = Path(path).read_text(encoding="utf-8")
    return [from_yolo(line, shape) for line in text.splitlines() if line.strip()]

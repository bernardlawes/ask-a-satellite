"""Build the before/after image for the SWIR cloud rule (docs/assets/swir-cloud-rule.png).

PortMiami on 2026-10-11, where small cumulus that the SCL cloud mask missed were boxed as vessels.
"Before" replays detector v1: no shortwave-infrared band, and every SCL cloud pixel masked (which also
cut holes in bright hulls). "After" is the current detector. Both panels are real runs on real pixels.

    uv run python scripts/post_assets/swir_cloud_rule.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from ask_a_satellite import IMAGERY_CREDIT
from ask_a_satellite.tools.labels import load_meta
from ask_a_satellite.tools.preprocess import true_color
from ask_a_satellite.tools.raster import read_aoi_cached
from ask_a_satellite.tools.vessels import ClassicalParams, detect_vessels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week1_progression import _font  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STEM = "port-of-miami_20261011"
OUT = ROOT / "docs" / "assets" / "swir-cloud-rule.png"
BG, INK, MUTED = (247, 245, 240), (27, 31, 36), (90, 96, 104)
KEPT, REMOVED = (255, 214, 0), (255, 64, 129)

Box = tuple[int, int, int, int]


def _removed(before: list[Box], after: list[Box], tol: int = 2) -> list[Box]:
    """Boxes from v1 whose centre falls inside no v2 box."""
    def inside(b: Box, a: Box) -> bool:
        r, c = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return a[0] - tol <= r <= a[2] + tol and a[1] - tol <= c <= a[3] + tol
    return [b for b in before if not any(inside(b, a) for a in after)]


def main() -> int:
    meta, scene, bbox = load_meta(ROOT / "eval" / "meta" / f"{STEM}.json")
    aoi = read_aoi_cached(scene, bbox, ROOT / "data" / "cache")
    no_swir = replace(aoi, bands={k: v for k, v in aoi.bands.items() if k != "swir16"})
    v1 = detect_vessels(no_swir, meta["aoi_id"], ClassicalParams(min_scl_cloud_px=0))
    v2 = detect_vessels(aoi, meta["aoi_id"])
    before = [d.bbox_px for d in v1.detections]
    after = [d.bbox_px for d in v2.detections]
    gone = _removed(before, after)
    print(f"v1 {len(before)} detections, v2 {len(after)}, v1 boxes with no v2 match: {len(gone)}, "
          f"v2 rejected as cloud: {v2.params.get('rejected_cloud_puff')}")

    rgb = true_color(aoi.bands)
    h, w = rgb.shape[:2]
    margin, gap, head_h, title_h = 44, 40, 70, 250
    scale = 2.6
    pw, ph = round(w * scale), round(h * scale)
    W = pw + 2 * margin
    H = title_h + 2 * (head_h + ph) + gap + 150
    out = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(out)
    d.text((margin, 40), "Small clouds that looked like ships", font=_font(64, bold=True), fill=INK)
    sub = _font(36)
    d.text((margin, 128), f"PortMiami, {scene.datetime:%d %B %Y}. The cloud mask that comes with the imagery "
           "missed these", font=sub, fill=MUTED)
    d.text((margin, 176), "small clouds. A shortwave-infrared test catches them: clouds are bright there, "
           "hulls much less.", font=sub, fill=MUTED)

    base = Image.fromarray(rgb).resize((pw, ph), Image.NEAREST)
    panels = [
        ("Before: no shortwave-infrared test", before, [], len(before)),
        ("After: with the shortwave-infrared test", after, gone, len(after)),
    ]
    for k, (title, keep, removed, n) in enumerate(panels):
        y = title_h + k * (head_h + ph + gap)
        d.text((margin, y + 8), title, font=_font(40, bold=True), fill=INK)
        tile = base.copy()
        td = ImageDraw.Draw(tile)
        for boxes, colour in ((removed, REMOVED), (keep, KEPT)):
            for r0, c0, r1, c1 in boxes:
                td.rectangle((c0 * scale - 4, r0 * scale - 4, c1 * scale + 3, r1 * scale + 3),
                             outline=colour, width=4)
        badge = f"{n} detections"
        bf = _font(44, bold=True)
        bw = int(td.textlength(badge, font=bf)) + 36
        td.rectangle((pw - bw - 18, ph - 84, pw - 18, ph - 18), fill=(20, 20, 20))
        td.text((pw - bw, ph - 77), badge, font=bf, fill=KEPT)
        out.paste(tile, (margin, y + head_h))

    fy = H - 126
    lf = _font(32)
    d.rectangle((margin, fy + 6, margin + 34, fy + 40), outline=KEPT, width=5)
    d.text((margin + 50, fy), "detection (not yet validated)", font=lf, fill=MUTED)
    lx = margin + 60 + int(d.textlength("detection (not yet validated)", font=lf)) + 50
    d.rectangle((lx, fy + 6, lx + 34, fy + 40), outline=REMOVED, width=5)
    d.text((lx + 50, fy), "dropped by the new rules", font=lf, fill=MUTED)
    d.text((margin, fy + 56), f"Sentinel-2 L2A, {scene.datetime:%Y-%m-%d}, true colour. Nearest-neighbour "
           f"upscaling. {IMAGERY_CREDIT.format(year=scene.datetime.year)}.", font=_font(30), fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} {out.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the four-panel "what real pixels taught me" image (docs/assets/week1-progression.png).

Replays four states of the detector from week one on the same Sentinel-2 scene, so every panel is a
real run on real pixels, not a mock-up. States 1-3 were intermediate versions that never shipped; their
logic is re-implemented here as it was, on top of today's helpers. See docs/dev-log.md.

    uv run python scripts/post_assets/week1_progression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from ask_a_satellite import IMAGERY_CREDIT
from ask_a_satellite.tools.labels import load_meta
from ask_a_satellite.tools.preprocess import cloud_mask, ndwi, true_color, valid_mask
from ask_a_satellite.tools.raster import REFLECTANCE_BANDS, read_aoi_cached
from ask_a_satellite.tools.vessels import _disk, detect_vessels, sea_mask, water_background

ROOT = Path(__file__).resolve().parents[2]
STEM = "port-everglades_20261021"
OUT = ROOT / "docs" / "assets" / "week1-progression.png"

Box = tuple[int, int, int, int]


def _boxes(cand: np.ndarray, keep=lambda rr, cc: True) -> list[Box]:
    labels, _ = ndimage.label(cand, structure=np.ones((3, 3), bool))
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        rr, cc = np.nonzero(labels[sl] == i)
        if not 4 <= rr.size <= 1500:
            continue
        rr, cc = rr + sl[0].start, cc + sl[1].start
        if keep(rr, cc):
            out.append((int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1))
    return out


def _background(bands, water, cloud, valid):
    med, sigma = water_background(bands["nir"], water)
    thr = max(med + 6.0 * sigma, 0.04)
    with np.errstate(invalid="ignore"):
        return bands["nir"] > thr


def state_v0(bands: dict[str, np.ndarray]) -> list[Box]:
    """First detector: closing r=5 + unrestricted hole filling; candidates must be 'not water'."""
    cloud, valid = cloud_mask(bands["scl"]), valid_mask(bands)
    water = (ndwi(bands["green"], bands["nir"]) > 0) & valid & ~cloud
    pad = 6
    padded = np.pad(water, pad, mode="edge")
    closed = ndimage.binary_closing(padded, structure=_disk(5))[pad:-pad, pad:-pad]
    sea = ndimage.binary_fill_holes(closed | water)
    bright = _background(bands, water, cloud, valid)
    return _boxes(sea & ~water & ~cloud & valid & bright)


def state_shore_buffer(bands: dict[str, np.ndarray]) -> list[Box]:
    """Open-water sea mask, no 'not water' term, drop blobs mostly within 30 m of land."""
    cloud, valid = cloud_mask(bands["scl"]), valid_mask(bands)
    water = (ndwi(bands["green"], bands["nir"]) > 0) & valid & ~cloud
    sea = sea_mask(water, 2, 2000, 600)
    land = valid & ~sea & ~cloud
    near = ndimage.binary_dilation(land, structure=_disk(3))
    bright = _background(bands, water, cloud, valid)
    return _boxes(sea & ~cloud & valid & bright, keep=lambda rr, cc: near[rr, cc].mean() <= 0.5)


def main() -> int:
    meta, scene, bbox = load_meta(ROOT / "eval" / "meta" / f"{STEM}.json")
    aoi = read_aoi_cached(scene, bbox, ROOT / "data" / "cache")
    good = aoi.bands
    # Bug: Earth Search had already removed the BOA offset; applying -0.1 again shifted every band.
    bugged = {k: (v - 0.1 if k in REFLECTANCE_BANDS else v) for k, v in good.items()}

    current = detect_vessels(aoi, meta["aoi_id"])
    panels = [
        ("1", "Offset applied twice", "the ocean fails the water test; boxes land on roofs",
         state_v0(bugged)),
        ("2", "Offset fixed", "but the 'sea' mask leaks into neighbourhoods", state_v0(good)),
        ("3", "Open water only, drop blobs at the shore", "surf lines and the jetty remain",
         state_shore_buffer(good)),
        ("4", "Rules measured on real blobs", "land contact, surf strips, clouds, shallows",
         [d.bbox_px for d in current.detections]),
    ]
    for n, title, _, boxes in panels:
        print(f"panel {n}: {len(boxes):3d} detections  ({title})")

    rgb = true_color(good)
    img = compose(rgb, panels, scene.datetime)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} {img.size}")
    return 0


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else (
        "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(rgb: np.ndarray, panels, when, scale: int = 2) -> Image.Image:
    h, w = rgb.shape[:2]
    pw, ph = w * scale, h * scale
    head_h, gap, margin = 96, 24, 40
    title_h, foot_h = 150, 90
    W = margin * 2 + pw * 2 + gap
    H = title_h + (head_h + ph) * 2 + gap + foot_h
    bg, ink, muted, accent = (247, 245, 240), (27, 31, 36), (90, 96, 104), (255, 214, 0)
    out = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(out)
    d.text((margin, 34), "Week one on real pixels: same scene, four versions of the detector",
           font=_font(46, bold=True), fill=ink)
    d.text((margin, 96), f"Port Everglades, Sentinel-2 L2A, {when:%Y-%m-%d}. Yellow boxes are detections, "
           "not validated ships.", font=_font(28), fill=muted)

    base = Image.fromarray(rgb).resize((pw, ph), Image.NEAREST)
    for k, (n, title, sub, boxes) in enumerate(panels):
        x = margin + (k % 2) * (pw + gap)
        y = title_h + (k // 2) * (head_h + ph + gap)
        d.text((x, y + 6), f"{n}  {title}", font=_font(34, bold=True), fill=ink)
        d.text((x + 34, y + 52), sub, font=_font(26), fill=muted)
        tile = base.copy()
        td = ImageDraw.Draw(tile)
        for r0, c0, r1, c1 in boxes:
            td.rectangle((c0 * scale - 3, r0 * scale - 3, c1 * scale + 2, r1 * scale + 2),
                         outline=accent, width=3)
        badge = f"{len(boxes)} detections"
        bf = _font(40, bold=True)
        bw = int(td.textlength(badge, font=bf)) + 32
        td.rectangle((pw - bw - 16, ph - 76, pw - 16, ph - 16), fill=(20, 20, 20))
        td.text((pw - bw, ph - 70), badge, font=bf, fill=accent)
        out.paste(tile, (x, y + head_h))

    d.text((margin, H - foot_h + 26), "Accuracy is measured next, against hand labels. "
           + IMAGERY_CREDIT.format(year=when.year) + ".", font=_font(26), fill=muted)
    return out


if __name__ == "__main__":
    sys.exit(main())

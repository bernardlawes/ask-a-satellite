"""Build the "what 10 m per pixel looks like" image (docs/assets/pixel-scale.png).

A true-colour Sentinel-2 crop of Port Everglades with a zoomed inset on the largest vessel the detector
found. Upscaling is nearest-neighbour so every 10 m pixel stays a visible square. The ship's length in
the caption is the detector's own measurement on this scene, not an assumed value.

    uv run python scripts/post_assets/pixel_scale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from ask_a_satellite import IMAGERY_CREDIT
from ask_a_satellite.tools.labels import load_meta
from ask_a_satellite.tools.preprocess import true_color
from ask_a_satellite.tools.raster import read_aoi_cached
from ask_a_satellite.tools.vessels import detect_vessels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week1_progression import _font  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STEM = "port-everglades_20261021"
OUT = ROOT / "docs" / "assets" / "pixel-scale.png"
GSD_M = 10.0
SIZE = 2400
WINDOW = 36  # inset window, in source pixels
BG, INK, MUTED, ACCENT = (247, 245, 240), (27, 31, 36), (90, 96, 104), (255, 214, 0)


def _scale_bar(d: ImageDraw.ImageDraw, x: int, y: int, px: int, label: str) -> None:
    d.rectangle((x - 10, y - 46, x + px + 10, y + 22), fill=(20, 20, 20))
    d.rectangle((x, y, x + px, y + 8), fill=(255, 255, 255))
    d.text((x, y - 40), label, font=_font(28, bold=True), fill=(255, 255, 255))


def main() -> int:
    meta, scene, bbox = load_meta(ROOT / "eval" / "meta" / f"{STEM}.json")
    aoi = read_aoi_cached(scene, bbox, ROOT / "data" / "cache")
    result = detect_vessels(aoi, meta["aoi_id"])
    ship = max(result.detections, key=lambda d: d.length_m)
    length_px = ship.length_m / GSD_M
    print(f"largest detection #{ship.id}: {ship.length_m:.0f} m, {length_px:.1f} px, bbox {ship.bbox_px}")

    rgb = true_color(aoi.bands)
    h, w = rgb.shape[:2]
    margin, title_h = 40, 290
    scale = (SIZE - 2 * margin) / w
    ow, oh = SIZE - 2 * margin, round(h * scale)
    out = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(out)

    d.text((margin, 40), "What 10 m per pixel looks like", font=_font(64, bold=True), fill=INK)
    sub = _font(38)
    d.text((margin, 132), "Port Everglades from Sentinel-2. In the inset, a moving ship and its wake blur "
           "into one", font=sub, fill=MUTED)
    d.text((margin, 184), f"bright streak about {length_px:.0f} pixels long (≈{ship.length_m:.0f} m, as the "
           "detector measures it).", font=sub, fill=MUTED)

    overview = Image.fromarray(rgb).resize((ow, oh), Image.NEAREST)
    od = ImageDraw.Draw(overview)
    r0 = max(0, min(h - WINDOW, round(ship.row) - WINDOW // 2))
    c0 = max(0, min(w - WINDOW, round(ship.col) - WINDOW // 2))
    zx0, zy0 = round(c0 * scale), round(r0 * scale)
    zx1, zy1 = round((c0 + WINDOW) * scale), round((r0 + WINDOW) * scale)
    od.rectangle((zx0, zy0, zx1, zy1), outline=ACCENT, width=5)
    bar_px = round(1000 / GSD_M * scale)
    _scale_bar(od, ow - bar_px - 40, oh - 50, bar_px, "1 km")

    # Inset: nearest-neighbour zoom with a faint grid so each square reads as one 10 m pixel.
    cell = 18
    plain = Image.fromarray(rgb[r0:r0 + WINDOW, c0:c0 + WINDOW]).resize((WINDOW * cell,) * 2, Image.NEAREST)
    gridded = plain.copy()
    gd = ImageDraw.Draw(gridded)
    for k in range(1, WINDOW):
        gd.line((k * cell, 0, k * cell, WINDOW * cell), fill=(255, 255, 255), width=1)
        gd.line((0, k * cell, WINDOW * cell, k * cell), fill=(255, 255, 255), width=1)
    inset = Image.blend(plain, gridded, 0.35)
    idr = ImageDraw.Draw(inset)
    _scale_bar(idr, 24, inset.height - 34, round(100 / GSD_M * cell), "100 m")
    label, lf = "Each square = one 10 m pixel", _font(28, bold=True)
    idr.rectangle((14, 14, 34 + int(idr.textlength(label, font=lf)), 62), fill=(20, 20, 20))
    idr.text((24, 20), label, font=lf, fill=(255, 255, 255))

    oy = title_h
    out.paste(overview, (margin, oy))
    ix, iy = margin + 30, oy + 30
    d.rectangle((ix - 6, iy - 6, ix + inset.width + 5, iy + inset.height + 5), fill=ACCENT)
    out.paste(inset, (ix, iy))
    d.line((ix + inset.width + 5, iy + inset.height + 5, margin + zx0, oy + zy1), fill=ACCENT, width=4)
    d.line((ix + inset.width + 5, iy - 6, margin + zx0, oy + zy0), fill=ACCENT, width=4)

    fy = oy + oh + 40
    foot = _font(34)
    d.text((margin, fy), f"Sentinel-2 L2A, {scene.datetime:%Y-%m-%d}, true colour. Nearest-neighbour "
           "upscaling, no sharpening.", font=foot, fill=MUTED)
    d.text((margin, fy + 52), IMAGERY_CREDIT.format(year=scene.datetime.year) + ".", font=foot, fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} {out.size}, overview {ow}x{oh}, footer at y={fy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

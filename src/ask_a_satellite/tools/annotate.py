"""Render true-colour AOI crops with numbered detection boxes and the Copernicus credit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import IMAGERY_CREDIT
from ..models import DetectionResult

BOX_COLOR = (255, 214, 0)
FOOTER_BG = (20, 20, 20)
FOOTER_FG = (235, 235, 235)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_annotated(rgb: np.ndarray, result: DetectionResult, scale: int = 3,
                     title: str | None = None) -> Image.Image:
    """Upscale with nearest neighbour (keeps pixels honest), draw boxes, add a caption footer."""
    img = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    label_font = _font(max(10, 4 * scale))
    pad = 2
    for d in result.detections:
        r0, c0, r1, c1 = d.bbox_px
        box = (c0 * scale - pad, r0 * scale - pad, c1 * scale + pad - 1, r1 * scale + pad - 1)
        draw.rectangle(box, outline=BOX_COLOR, width=max(1, scale // 2))
        draw.text((box[2] + 2, box[1] - 2), str(d.id), fill=BOX_COLOR, font=label_font)

    year = result.datetime.year
    lines = [
        title or f"{result.aoi_id}  {result.datetime:%Y-%m-%d %H:%M} UTC",
        f"Large vessels detected: {result.count}  |  AOI cloud: {result.aoi_cloud_fraction:.0%}  "
        f"|  method: {result.method}",
        IMAGERY_CREDIT.format(year=year),
    ]
    font = _font(max(12, 5 * scale))
    line_h = int(font.size * 1.4) if hasattr(font, "size") else 14
    footer = Image.new("RGB", (img.width, line_h * len(lines) + 8), FOOTER_BG)
    fd = ImageDraw.Draw(footer)
    for i, text in enumerate(lines):
        fd.text((6, 4 + i * line_h), text, fill=FOOTER_FG, font=font)
    out = Image.new("RGB", (img.width, img.height + footer.height))
    out.paste(img, (0, 0))
    out.paste(footer, (0, img.height))
    return out


def save_annotated(rgb: np.ndarray, result: DetectionResult, path: Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    render_annotated(rgb, result, **kwargs).save(path)
    return path

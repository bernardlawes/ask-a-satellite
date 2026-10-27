"""Sentinel-2 scene search on the Earth Search STAC API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from ..config import get_settings
from ..models import BandAsset, BBox, SceneInfo

# Earth Search v1 asset keys for the bands the pipeline needs, and ones it uses when present.
BANDS = ("blue", "green", "red", "nir", "scl")
OPTIONAL_BANDS = ("swir16",)


def scene_from_item(item: Any) -> SceneInfo:
    """Convert a pystac Item (or STAC item dict) into a SceneInfo with per-band scale/offset."""
    d = item if isinstance(item, dict) else item.to_dict()
    props = d.get("properties", {})
    # Earth Search harmonises processing-baseline >= 04.00 DNs (removes the +1000 BOA offset) and sets
    # this flag, but still advertises offset -0.1 in raster:bands. Applying it again makes water negative.
    offset_already_applied = bool(props.get("earthsearch:boa_offset_applied", False))
    assets: dict[str, BandAsset] = {}
    for key in (*BANDS, *OPTIONAL_BANDS):
        a = d.get("assets", {}).get(key)
        if a is None:
            continue
        rb = (a.get("raster:bands") or [{}])[0]
        assets[key] = BandAsset(
            href=a["href"],
            scale=rb.get("scale", 1.0),
            offset=0.0 if offset_already_applied else rb.get("offset", 0.0),
        )
    missing = set(BANDS) - assets.keys()
    if missing:
        raise ValueError(f"item {d.get('id')} is missing assets: {sorted(missing)}")
    return SceneInfo(
        id=d["id"],
        datetime=props["datetime"],
        cloud_cover=props.get("eo:cloud_cover"),
        platform=props.get("platform"),
        mgrs_tile=props.get("grid:code") or props.get("s2:mgrs_tile"),
        assets=assets,
    )


def _interval(start: date | datetime | str, end: date | datetime | str) -> str:
    def iso(x: date | datetime | str) -> str:
        return x if isinstance(x, str) else x.isoformat()

    return f"{iso(start)}/{iso(end)}"


def search_scenes(
    bbox: BBox,
    start: date | datetime | str,
    end: date | datetime | str,
    max_cloud: float = 30.0,
    limit: int = 50,
    client: Any = None,
) -> list[SceneInfo]:
    """Find scenes intersecting ``bbox`` in [start, end] with scene cloud cover below ``max_cloud``.

    Returned newest first. Scene-level cloud cover is a coarse pre-filter; the AOI-level cloud fraction
    is measured later from the SCL band.
    """
    settings = get_settings()
    if client is None:
        from pystac_client import Client

        client = Client.open(settings.stac_api_url)
    search = client.search(
        collections=[settings.stac_collection],
        bbox=bbox.as_tuple(),
        datetime=_interval(start, end),
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    scenes = []
    for item in search.items():
        try:
            scenes.append(scene_from_item(item))
        except ValueError:
            continue
    return dedupe_by_date(sorted(scenes, key=lambda s: s.datetime, reverse=True))


def dedupe_by_date(scenes: list[SceneInfo]) -> list[SceneInfo]:
    """An AOI on a tile boundary returns one item per MGRS tile for the same overpass; keep the
    least cloudy item per acquisition date (input order is otherwise preserved)."""
    best: dict[date, SceneInfo] = {}
    for s in scenes:
        key = s.datetime.date()
        cur = best.get(key)
        if cur is None or (s.cloud_cover or 0) < (cur.cloud_cover or 0):
            best[key] = s
    kept = {id(s) for s in best.values()}
    return [s for s in scenes if id(s) in kept]

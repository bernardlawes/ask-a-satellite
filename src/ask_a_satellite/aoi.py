"""Registry of named areas of interest.

Bounding boxes are hand-drawn approximations from public maps, sized to include the berths, the
channel, and nearby anchorage water. They are intentionally small (a few km) so a crop is ~0.5 MB.
"""

from __future__ import annotations

from .models import AOI, BBox

_AOIS: dict[str, AOI] = {
    a.id: a
    for a in [
        AOI(
            id="port-of-miami",
            name="PortMiami (Dodge Island) and Government Cut",
            bbox=BBox(west=-80.200, south=25.755, east=-80.120, north=25.790),
            description="Cruise and cargo berths on Dodge Island plus the entrance channel.",
        ),
        AOI(
            id="port-everglades",
            name="Port Everglades and offshore anchorage",
            bbox=BBox(west=-80.135, south=26.065, east=-80.070, north=26.110),
            description="Port basins plus open water east of the inlet where ships anchor.",
        ),
    ]
}


def list_aois() -> list[AOI]:
    return list(_AOIS.values())


def get_aoi(aoi_id: str) -> AOI:
    key = aoi_id.strip().lower().replace(" ", "-").replace("_", "-")
    try:
        return _AOIS[key]
    except KeyError:
        known = ", ".join(sorted(_AOIS))
        raise KeyError(f"unknown AOI {aoi_id!r}; known: {known}") from None

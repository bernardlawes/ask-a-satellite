"""Typed inputs/outputs shared by the tools, the MCP server, and the report renderer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    """Geographic bounding box in WGS84 degrees."""

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if self.west >= self.east or self.south >= self.north:
            raise ValueError(f"invalid bbox ordering: {self.as_tuple()}")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


class AOI(BaseModel):
    id: str
    name: str
    bbox: BBox
    description: str = ""


class BandAsset(BaseModel):
    """A single-band COG and the scale/offset that converts its DNs to surface reflectance."""

    href: str
    scale: float = 1.0
    offset: float = 0.0


class SceneInfo(BaseModel):
    id: str
    datetime: datetime
    cloud_cover: float | None = None
    platform: str | None = None
    mgrs_tile: str | None = None
    assets: dict[str, BandAsset]


class Detection(BaseModel):
    id: int
    row: float
    col: float
    bbox_px: tuple[int, int, int, int] = Field(description="row_min, col_min, row_max, col_max (exclusive)")
    lon: float
    lat: float
    area_px: int
    area_m2: float
    length_m: float
    score: float = Field(ge=0, le=1, description="Heuristic 0-1 strength, not a probability")


class DetectionResult(BaseModel):
    aoi_id: str
    scene_id: str
    datetime: datetime
    method: str
    count: int
    detections: list[Detection]
    aoi_cloud_fraction: float
    water_fraction: float
    pixel_size_m: float
    params: dict[str, float | int | str]
    notes: list[str] = Field(default_factory=list)


class CountComparison(BaseModel):
    aoi_id: str
    before_scene_id: str
    before_datetime: datetime
    before_count: int
    after_scene_id: str
    after_datetime: datetime
    after_count: int
    difference: int
    percent_change: float | None
    caveats: list[str]

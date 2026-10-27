import copy

import pytest

from ask_a_satellite.models import BBox
from ask_a_satellite.tools.imagery import BANDS, scene_from_item, search_scenes


def stac_item(item_id="S2B_17RNJ_20261020_0_L2A", dt="2026-10-20T16:05:00Z", cloud=3.2):
    assets = {
        k: {"href": f"https://example.com/{item_id}/{k}.tif",
            "raster:bands": [{"scale": 0.0001, "offset": -0.1}] if k != "scl" else [{}]}
        for k in BANDS
    }
    assets["visual"] = {"href": "https://example.com/visual.tif"}
    return {"type": "Feature", "id": item_id, "assets": assets,
            "properties": {"datetime": dt, "eo:cloud_cover": cloud, "platform": "sentinel-2b",
                           "grid:code": "MGRS-17RNJ"}}


def test_scene_from_item_reads_scale_and_offset():
    s = scene_from_item(stac_item())
    assert s.id == "S2B_17RNJ_20261020_0_L2A"
    assert s.cloud_cover == 3.2
    assert s.mgrs_tile == "MGRS-17RNJ"
    assert s.assets["nir"].scale == 0.0001 and s.assets["nir"].offset == -0.1
    assert s.assets["scl"].scale == 1.0 and s.assets["scl"].offset == 0.0
    assert "visual" not in s.assets


def test_scene_from_item_skips_offset_already_applied_by_earth_search():
    # Regression: real items carry both the flag and offset -0.1; applying it twice made water negative.
    item = stac_item()
    item["properties"]["earthsearch:boa_offset_applied"] = True
    s = scene_from_item(item)
    assert s.assets["nir"].scale == 0.0001 and s.assets["nir"].offset == 0.0


def test_scene_from_item_requires_all_bands():
    item = stac_item()
    del item["assets"]["scl"]
    with pytest.raises(ValueError, match="scl"):
        scene_from_item(item)


class FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class FakeClient:
    def __init__(self, items):
        self.items = items
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        return FakeSearch(self.items)


def test_search_passes_filters_sorts_and_dedupes():
    broken = stac_item("BROKEN", "2026-10-05T16:00:00Z")
    del broken["assets"]["red"]
    items = [
        stac_item("A_old", "2026-10-10T16:00:00Z", 8.0),
        stac_item("B_new", "2026-10-20T16:00:00Z", 12.0),
        stac_item("B_new_othertile", "2026-10-20T16:00:03Z", 4.0),
        broken,
    ]
    client = FakeClient(copy.deepcopy(items))
    bbox = BBox(west=-80.1, south=26.0, east=-80.0, north=26.1)
    scenes = search_scenes(bbox, "2026-10-01", "2026-10-27", max_cloud=20, client=client)

    assert [s.id for s in scenes] == ["B_new_othertile", "A_old"]
    assert client.kwargs["query"] == {"eo:cloud_cover": {"lt": 20}}
    assert client.kwargs["datetime"] == "2026-10-01/2026-10-27"
    assert client.kwargs["bbox"] == bbox.as_tuple()
    assert client.kwargs["collections"] == ["sentinel-2-l2a"]


@pytest.mark.network
def test_live_search_port_everglades():
    from ask_a_satellite.aoi import get_aoi

    scenes = search_scenes(get_aoi("port-everglades").bbox, "2026-01-01", "2026-03-31", max_cloud=20)
    assert scenes, "expected at least one low-cloud scene in Q1 2026"
    assert all(s.cloud_cover < 20 for s in scenes)

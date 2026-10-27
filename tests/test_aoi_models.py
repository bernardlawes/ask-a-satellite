import pytest
from pydantic import ValidationError

from ask_a_satellite.aoi import get_aoi, list_aois
from ask_a_satellite.models import BBox


def test_registry_has_two_ports():
    ids = {a.id for a in list_aois()}
    assert {"port-of-miami", "port-everglades"} <= ids


@pytest.mark.parametrize("name", ["port-everglades", "Port Everglades", "PORT_EVERGLADES"])
def test_get_aoi_normalises_names(name):
    assert get_aoi(name).id == "port-everglades"


def test_unknown_aoi_lists_known_ids():
    with pytest.raises(KeyError, match="port-of-miami"):
        get_aoi("atlantis")


def test_bbox_rejects_inverted_coordinates():
    with pytest.raises(ValidationError):
        BBox(west=-80.0, south=26.0, east=-80.1, north=26.1)


def test_aoi_bboxes_are_small():
    for a in list_aois():
        w, s, e, n = a.bbox.as_tuple()
        assert 0 < e - w < 0.2 and 0 < n - s < 0.2

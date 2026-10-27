import numpy as np
import pytest

from ask_a_satellite.tools.raster import AOIRaster
from ask_a_satellite.tools.vessels import METHOD, ClassicalParams, detect_vessels, sea_mask

from .conftest import SHIPS, synthetic_bands


def test_detects_open_water_ships_only(synthetic_aoi):
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert res.method == METHOD
    # three ships; 1-px boat below min area; ship under cloud and the cloud itself excluded; land ignored
    assert res.count == len(SHIPS) == len(res.detections)

    found = sorted(d.bbox_px for d in res.detections)
    expected = sorted((r, c, r + h, c + w) for r, c, h, w in SHIPS)
    assert found == expected


def test_detection_measurements(synthetic_aoi):
    res = detect_vessels(synthetic_aoi, "test-aoi")
    by_len = sorted(res.detections, key=lambda d: d.length_m)
    # ships are 6, 12 and 22 px long -> ~60, 120, 220 m (moment estimate within ~15%)
    for d, true_len in zip(by_len, (60, 120, 220), strict=True):
        assert d.length_m == pytest.approx(true_len, rel=0.15)
        assert 0 < d.score <= 1
        assert d.area_m2 == d.area_px * 100
    # near Port Everglades
    for d in res.detections:
        assert -80.2 < d.lon < -80.0 and 26.0 < d.lat < 26.15
    assert [d.id for d in res.detections] == [1, 2, 3]


def test_reports_cloud_and_params(synthetic_aoi):
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert 0 < res.aoi_cloud_fraction < 0.1
    assert 0.5 < res.water_fraction < 0.8
    assert res.params["k_mad"] == ClassicalParams().k_mad
    assert "nir_threshold" in res.params


def test_heavy_cloud_adds_note(synthetic_aoi):
    b = synthetic_aoi.bands
    b["scl"][:140, :] = 9
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert any("cloud" in n for n in res.notes)
    assert res.count == 0  # all three ships are under the cloud mask


def test_no_water_returns_zero_with_note(synthetic_aoi):
    b = synthetic_aoi.bands
    b["green"][:] = 0.08
    b["nir"][:] = 0.30
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert res.count == 0
    assert res.notes


def test_noise_only_water_has_no_false_positives(synthetic_aoi):
    aoi = AOIRaster(scene=synthetic_aoi.scene, bands=synthetic_bands(seed=7, ships=[], cloud=False),
                    transform=synthetic_aoi.transform, crs=synthetic_aoi.crs)
    # remove the tiny boat too
    aoi.bands["nir"][20, 170] = 0.02
    aoi.bands["green"][20, 170] = 0.06
    assert detect_vessels(aoi, "test-aoi").count == 0


def test_bright_blob_against_shore_is_rejected(synthetic_aoi):
    b = synthetic_aoi.bands
    b["nir"][100:104, 60:66] = 0.2  # surf-like bright strip touching the land edge (col 60)
    b["green"][100:104, 60:66] = 0.15
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert res.count == len(SHIPS)
    assert res.params["rejected_touches_land"] >= 1


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        # 1-px wide, 20-px long bright line in open water (surf line / wake)
        ({"rows": slice(70, 71), "cols": slice(150, 170), "nir": 0.2, "green": 0.1, "red": 0.1},
         "thin_strip"),
        # vegetated islet: NIR >> red
        ({"rows": slice(60, 66), "cols": slice(150, 156), "nir": 0.3, "green": 0.06, "red": 0.03},
         "vegetation"),
        # bright shallow water: green >> NIR but NIR above threshold
        ({"rows": slice(60, 66), "cols": slice(150, 156), "nir": 0.06, "green": 0.2, "red": 0.12},
         "shallow_bright_water"),
        # bright blob 3 px from land (surf zone)
        ({"rows": slice(100, 104), "cols": slice(63, 67), "nir": 0.2, "green": 0.1, "red": 0.1},
         "near_shore"),
    ],
)
def test_rejection_rules(synthetic_aoi, patch, reason):
    b = synthetic_aoi.bands
    for band in ("nir", "green", "red"):
        b[band][patch["rows"], patch["cols"]] = patch[band]
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert res.count == len(SHIPS)
    assert res.params[f"rejected_{reason}"] == 1


def _paint(b, rows, cols, nir, green, red, swir_nir):
    b["nir"][rows, cols] = nir
    b["green"][rows, cols] = green
    b["red"][rows, cols] = red
    b["swir16"][rows, cols] = nir * swir_nir


def test_compact_swir_bright_puff_is_rejected_as_cloud(synthetic_aoi):
    # small cumulus that SCL left as water: round, bright, SWIR/NIR ~ 0.9
    _paint(synthetic_aoi.bands, slice(60, 68), slice(150, 158), 0.12, 0.13, 0.12, swir_nir=0.9)
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert res.count == len(SHIPS)
    assert res.params["rejected_cloud_puff"] == 1


def test_compact_hull_with_low_swir_is_kept(synthetic_aoi):
    # same shape and brightness but a hull-like SWIR/NIR of 0.35 -> counted
    _paint(synthetic_aoi.bands, slice(60, 68), slice(150, 158), 0.12, 0.13, 0.12, swir_nir=0.35)
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert res.count == len(SHIPS) + 1
    assert res.params["rejected_cloud_puff"] == 0


def test_elongated_swir_bright_ship_is_kept(synthetic_aoi):
    # container ships can be SWIR-bright (0.79 measured), but they are elongated, not puffs
    _paint(synthetic_aoi.bands, slice(60, 63), slice(140, 160), 0.15, 0.08, 0.12, swir_nir=0.8)
    assert detect_vessels(synthetic_aoi, "test-aoi").count == len(SHIPS) + 1


def test_scl_cloud_speck_on_bright_hull_does_not_split_it(synthetic_aoi):
    # SCL marks part of a bright hull as cloud (seen on real scenes); the ship must stay whole
    r, c, h, w = SHIPS[1]
    synthetic_aoi.bands["scl"][r:r + h, c + 5:c + 12] = 8
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert (r, c, r + h, c + w) in {d.bbox_px for d in res.detections}


def test_missing_swir_adds_note(synthetic_aoi):
    del synthetic_aoi.bands["swir16"]
    res = detect_vessels(synthetic_aoi, "test-aoi")
    assert res.params["swir_cloud_test"] == 0
    assert any("SWIR" in n for n in res.notes)


def test_sea_mask_drops_small_water_and_large_holes():
    water = np.zeros((100, 100), bool)
    water[:, 40:] = True  # open water
    water[60:90, 60:90] = False  # 900 px island -> stays land
    water[20:22, 70:80] = False  # 20 px ship -> filled into sea
    water[5:10, 5:10] = True  # 25 px pool on land -> dropped
    sea = sea_mask(water, radius=1, min_body_px=100, max_hole_px=600)
    assert sea[20:22, 70:80].all()
    assert not sea[62:88, 62:88].any()  # closing may round the island's corners
    assert not sea[5:10, 5:10].any()


def test_sea_mask_fills_ship_holes_but_not_land():
    water = np.ones((50, 50), bool)
    water[:, :10] = False  # land strip on the edge
    water[25:27, 30:40] = False  # ship
    sea = sea_mask(water, radius=3)
    assert sea[25:27, 30:40].all()
    assert not sea[:, :8].any()

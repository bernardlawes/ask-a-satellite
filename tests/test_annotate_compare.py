from datetime import timedelta

import pytest

from ask_a_satellite.tools.annotate import render_annotated, save_annotated
from ask_a_satellite.tools.compare import compare_counts
from ask_a_satellite.tools.preprocess import true_color
from ask_a_satellite.tools.vessels import detect_vessels


@pytest.fixture
def result(synthetic_aoi):
    return detect_vessels(synthetic_aoi, "test-aoi")


def test_render_annotated_adds_boxes_and_footer(synthetic_aoi, result):
    rgb = true_color(synthetic_aoi.bands)
    img = render_annotated(rgb, result, scale=3)
    h, w = synthetic_aoi.shape
    assert img.width == w * 3
    assert img.height > h * 3  # footer
    r0, c0, r1, c1 = result.detections[0].bbox_px
    assert img.getpixel((c0 * 3 - 2, r0 * 3 - 2)) == (255, 214, 0)


def test_save_annotated_writes_png(tmp_path, synthetic_aoi, result):
    p = save_annotated(true_color(synthetic_aoi.bands), result, tmp_path / "x" / "out.png")
    assert p.exists() and p.read_bytes()[:4] == b"\x89PNG"


def _shifted(result, days, count, cloud=None):
    return result.model_copy(update={
        "datetime": result.datetime + timedelta(days=days), "count": count,
        "scene_id": f"{result.scene_id}_{days}",
        **({"aoi_cloud_fraction": cloud} if cloud is not None else {}),
    })


def test_compare_counts_orders_by_date_and_computes_change(result):
    later = _shifted(result, 30, 6)
    cmp = compare_counts(later, result)  # passed in reverse order
    assert cmp.before_count == 3 and cmp.after_count == 6
    assert cmp.difference == 3 and cmp.percent_change == 100.0
    assert any("snapshot" in c for c in cmp.caveats)


def test_compare_counts_flags_condition_differences(result):
    cmp = compare_counts(result, _shifted(result, 7, 2, cloud=0.4))
    assert any("Cloud cover differs" in c for c in cmp.caveats)


def test_compare_counts_zero_baseline(result):
    cmp = compare_counts(_shifted(result, 0, 0), _shifted(result, 5, 2))
    assert cmp.percent_change is None


def test_compare_counts_rejects_different_aois(result):
    with pytest.raises(ValueError):
        compare_counts(result, result.model_copy(update={"aoi_id": "other"}))

import pytest
from PIL import Image

from ask_a_satellite.models import BBox
from ask_a_satellite.tools.evaluate import markdown_table, match, score, total
from ask_a_satellite.tools.labels import (
    from_yolo,
    load_meta,
    read_yolo_boxes,
    to_yolo,
    write_label_kit,
)
from ask_a_satellite.tools.vessels import detect_vessels

from .conftest import SHAPE, SHIPS


def test_yolo_round_trip():
    box = (40, 100, 43, 112)
    line = to_yolo(box, SHAPE)
    assert line.startswith("0 ")
    assert from_yolo(line, SHAPE) == pytest.approx(box, abs=1e-3)


def test_match_is_one_to_one_and_greedy_by_distance():
    truth = [(10, 10, 12, 20), (50, 50, 54, 60)]
    preds = [(11, 15), (11.2, 14), (52, 55), (90, 90)]  # two preds compete for truth 0
    pairs, fp, fn = match(preds, truth, tol_px=2)
    assert sorted(pairs) == [(0, 0), (2, 1)]
    assert sorted(fp) == [1, 3] and fn == []


def test_match_tolerance():
    truth = [(10, 10, 12, 20)]
    assert match([(13.5, 15)], truth, tol_px=2)[0] == [(0, 0)]
    assert match([(15, 15)], truth, tol_px=2)[0] == []


def test_score_perfect_on_synthetic(synthetic_aoi):
    res = detect_vessels(synthetic_aoi, "t")
    truth = [(r, c, r + h, c + w) for r, c, h, w in SHIPS]
    s = score(res, truth)
    assert (s.tp, s.fp, s.fn) == (3, 0, 0)
    assert s.precision == s.recall == s.f1 == 1.0


def test_score_counts_misses_and_false_alarms(synthetic_aoi):
    res = detect_vessels(synthetic_aoi, "t")
    truth = [(r, c, r + h, c + w) for r, c, h, w in SHIPS[:2]] + [(5, 190, 7, 195)]
    s = score(res, truth)
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    t = total([s, s])
    assert (t.tp, t.fp, t.fn) == (4, 2, 2)
    assert "| total |" in markdown_table([s, t])


def test_empty_cases_have_undefined_metrics():
    from ask_a_satellite.tools.evaluate import EvalScore

    s = EvalScore(name="x", tp=0, fp=0, fn=0)
    assert s.precision is None and s.recall is None and s.f1 is None
    assert "n/a" in markdown_table([s])


def test_label_kit_round_trip(tmp_path, synthetic_aoi):
    res = detect_vessels(synthetic_aoi, "test-aoi")
    bbox = BBox(west=-80.13, south=26.06, east=-80.07, north=26.11)
    stem = write_label_kit(synthetic_aoi, res, bbox, tmp_path, scale=3)

    img = Image.open(tmp_path / "images" / f"{stem}.png")
    assert img.size == (SHAPE[1] * 3, SHAPE[0] * 3)
    assert (tmp_path / "data.yaml").read_text().startswith("names: ['vessel']")

    meta, scene, bbox2 = load_meta(tmp_path / "meta" / f"{stem}.json")
    assert scene == synthetic_aoi.scene and bbox2 == bbox
    assert "Copernicus Sentinel-2" in meta["credit"]

    # pre-annotations read back as truth score perfectly against the same detections
    boxes = read_yolo_boxes(tmp_path / "labels" / f"{stem}.txt", tuple(meta["shape"]))
    s = score(res, boxes)
    assert (s.tp, s.fp, s.fn) == (res.count, 0, 0)

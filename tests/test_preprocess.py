import numpy as np

from ask_a_satellite.tools.preprocess import cloud_fraction, ndwi, true_color, valid_mask


def test_ndwi_signs():
    green = np.array([0.06, 0.08, 0.0], np.float32)
    nir = np.array([0.02, 0.30, 0.0], np.float32)
    out = ndwi(green, nir)
    assert out[0] > 0 > out[1]
    assert np.isnan(out[2])  # 0/0


def test_cloud_fraction_ignores_nodata():
    scl = np.array([[0, 0, 6, 6], [9, 3, 6, 10]], np.uint8)
    # 6 data pixels, 3 cloudy (9, 3, 10)
    assert cloud_fraction(scl) == 0.5
    assert cloud_fraction(np.zeros((2, 2), np.uint8)) == 1.0


def test_valid_mask_combines_nan_and_scl(synthetic_aoi):
    b = dict(synthetic_aoi.bands)
    b["nir"] = b["nir"].copy()
    b["nir"][0, 0] = np.nan
    b["scl"] = b["scl"].copy()
    b["scl"][1, 1] = 0
    v = valid_mask(b)
    assert not v[0, 0] and not v[1, 1] and v[2, 2]


def test_true_color_is_uint8_rgb(synthetic_aoi):
    rgb = true_color(synthetic_aoi.bands)
    assert rgb.shape == (*synthetic_aoi.shape, 3)
    assert rgb.dtype == np.uint8
    assert rgb.max() == 255 and rgb.min() == 0

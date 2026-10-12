"""Row by row: sample row n ÷ quartz row n, the rows binned after the division."""

import numpy as np
import pytest

from hdsfg_engine import Settings
from hdsfg_engine.data import sort_files
from hdsfg_engine.rowwise import RowInput, RowPipeline, bright_rows, retrieve_rows

from test_chosen_files import write_image

N_PIXELS, N_ROWS = 400, 8
RTOL = 1e-3        # the test files hold 4 decimals, so weak points carry rounding
ROW_SCALE = np.array([0, 0, 0.3, 1.0, 0.7, 0.2, 0, 0])      # rows 3-6 carry the signal
ROW_FACTOR = np.array([1, 1, 2.0, 1.0, 3.0, 5.0, 1, 1])      # sample row n = factor × quartz row n


@pytest.fixture
def files(tmp_path):
    """Three quartz and two sample images with any names; no noise, so the ratios are exact."""
    wavelength = np.linspace(760, 800, N_PIXELS)
    fringes = 1000 * np.exp(-((wavelength - 780) / 6) ** 2) * (1 + np.cos(wavelength * 9))
    quartz = fringes[:, None] * ROW_SCALE
    for k in range(3):
        write_image(tmp_path / f"qz {k}.asc", quartz, wavelength, f"16:1{k}:00.000")
    for k in range(2):
        write_image(tmp_path / f"water run {k}.asc", quartz * ROW_FACTOR, wavelength, f"16:2{k}:00.000")
    return (sort_files(tmp_path.glob("qz*.asc")), sort_files(tmp_path.glob("water*.asc")))


def settings():
    return Settings(remove_cosmic_rays=False, weak_reference_fraction=0.05)


def strong(result, fraction=1e-3):
    """Where quartz is not zero; "strong rows" binning is NaN where no row passes its cut."""
    strength = result.reference_strength
    keep = (strength > fraction * strength.max()) & np.isfinite(result.relative.mean)
    assert keep.sum() > 50
    return keep


def test_each_row_is_its_own_ratio(files):
    quartz, water = files
    # quartz as the sample material: no rotation, so relative is the plain ratio
    result = retrieve_rows(settings(), RowInput(quartz, water, water_material="zqz", chip_rows=[3, 4, 5, 6]))
    assert result.rows.tolist() == [3, 4, 5, 6]
    assert result.quartz.n_scans == 3 and result.water.n_scans == 2
    for k, row in enumerate(result.rows):
        q = np.abs(result.quartz_rows[:, k])
        keep = q > 1e-3 * q.max()
        np.testing.assert_allclose(result.relative_rows[keep, k], ROW_FACTOR[row - 1], rtol=RTOL)


def test_binning_after_division_differs_from_summing_first(files):
    quartz, water = files
    pipeline = RowPipeline()
    rows = np.arange(3, 7)
    scale, factor = ROW_SCALE[rows - 1], ROW_FACTOR[rows - 1]
    expected = {
        "weighted": (scale ** 2 * factor).sum() / (scale ** 2).sum(),
        "mean": factor.mean(),
    }
    for binning, value in expected.items():
        result = pipeline.run(settings(), RowInput(quartz, water, water_material="zqz", chip_rows=[3, 4, 5, 6],
                                                   binning=binning))
        keep = strong(result)
        np.testing.assert_allclose(result.relative.mean[keep], value, rtol=RTOL)
        # summing the rows first gives the signal-weighted mean instead
        np.testing.assert_allclose(result.fvb_relative.mean[keep], (scale * factor).sum() / scale.sum(),
                                   rtol=RTOL)


def test_one_row_is_the_usual_way(files):
    quartz, water = files
    result = retrieve_rows(settings(), RowInput(quartz, water, water_material="zqz", chip_rows=[4]))
    np.testing.assert_allclose(result.relative.mean, result.fvb_relative.mean)


def test_quartz_over_itself_is_one(files):
    quartz, _ = files
    for binning in ("weighted", "strong", "mean"):
        result = retrieve_rows(settings(), RowInput(quartz, quartz, water_material="zqz", chip_rows=[3, 4, 5, 6],
                                                    binning=binning))
        keep = strong(result)
        np.testing.assert_allclose(result.relative.mean[keep], 1, atol=1e-9)


def test_file_numbers_and_exposure(files):
    quartz, water = files
    result = retrieve_rows(settings(), RowInput(quartz, water, quartz_numbers=[1, 3], water_numbers=[2],
                                                quartz_exposure_sec=2, quartz_frames=5,
                                                water_exposure_sec=1, water_frames=5,
                                                water_material="zqz", chip_rows=[4]))
    assert result.quartz.numbers.tolist() == [1, 3] and result.water.numbers.tolist() == [2]
    keep = strong(result)
    # quartz exposed twice as long: the ratio is scaled by 10 s / 5 s
    np.testing.assert_allclose(result.relative.mean[keep], 2 * ROW_FACTOR[3], rtol=RTOL)


def test_bright_rows(files):
    quartz, _ = files
    from hdsfg_engine.data import load_files
    assert bright_rows(load_files(quartz, [], (325.0, 2805.0), 1.0)) == [3, 4, 5, 6]


def test_rows_need_not_be_next_to_each_other(files):
    quartz, water = files
    rows = np.array([3, 5, 6])
    scale, factor = ROW_SCALE[rows - 1], ROW_FACTOR[rows - 1]
    result = retrieve_rows(settings(), RowInput(quartz, water, water_material="zqz", chip_rows=[6, 3, 5]))
    assert result.rows.tolist() == [3, 5, 6]
    keep = strong(result)
    np.testing.assert_allclose(result.relative.mean[keep], (scale ** 2 * factor).sum() / (scale ** 2).sum(),
                               rtol=RTOL)
    np.testing.assert_allclose(result.fvb_relative.mean[keep], (scale * factor).sum() / scale.sum(), rtol=RTOL)


def test_no_rows_is_the_whole_chip_and_outside_rows_fail(files):
    quartz, water = files
    result = retrieve_rows(settings(), RowInput(quartz, water, water_material="zqz"))
    assert result.rows.tolist() == list(range(1, N_ROWS + 1))
    with pytest.raises(ValueError, match="outside"):
        retrieve_rows(settings(), RowInput(quartz, water, chip_rows=[3, N_ROWS + 1]))

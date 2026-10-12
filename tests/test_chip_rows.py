"""Full-chip image files: slices cut from the chip, and which of them is processed."""

from pathlib import Path

import numpy as np
import pytest

from hdsfg_engine import Settings, retrieve
from hdsfg_engine.data import load_channel, slice_channel

FVB_DATA = Path(__file__).parent.parent.parent / "20261005_50kHz_pinhole" / "data"
FOOTER = "\n\nDate and Time:                Wed Oct 7 16:11:25.335 2026\n"
WINDOW = (325.0, 2805.0)


def write_image(path: Path, counts: np.ndarray, wavelength: np.ndarray) -> None:
    """Andor layout: wavelength, then one column per chip row, trailing tab."""
    lines = ["\t".join([f"{w:.5f}"] + [f"{c:.4f}" for c in row]) + "\t"
             for w, row in zip(wavelength, counts)]
    path.write_text("\n".join(lines) + FOOTER)


@pytest.fixture
def image_run(tmp_path):
    rng = np.random.default_rng(0)
    wavelength = np.linspace(700, 820, 64)
    images = [rng.normal(100, 5, (64, 8)) for _ in range(2)]
    for number, image in enumerate(images, start=1):
        write_image(tmp_path / f"zqzimg_ssp_1s_10frames_s{number}.asc", image, wavelength)
    return tmp_path, images


def test_image_loads_as_fvb(image_run):
    folder, images = image_run
    channel = load_channel(folder, "zqzimg", [], WINDOW)
    assert channel.is_image and channel.images.shape == (2, 64, 8)
    np.testing.assert_allclose(channel.counts[:, 0], images[0].sum(axis=1), rtol=1e-6)


def test_one_slice_with_window(image_run):
    folder, images = image_run
    channel = load_channel(folder, "zqzimg", [], WINDOW)
    mean = slice_channel(channel, [4, 7], 1, "mean", "4")
    np.testing.assert_allclose(mean.counts[:, 1], images[1][:, 2:5].mean(axis=1), rtol=1e-5)
    total = slice_channel(channel, [4, 7], 1, "sum", "4")
    np.testing.assert_allclose(total.counts[:, 0], images[0][:, 2:5].sum(axis=1), rtol=1e-5)


def test_window_clipped_at_chip_edge(image_run):
    folder, images = image_run
    channel = load_channel(folder, "zqzimg", [], WINDOW)
    edge = slice_channel(channel, [8], 2, "sum", "8")
    np.testing.assert_allclose(edge.counts[:, 0], images[0][:, 5:8].sum(axis=1), rtol=1e-5)


def test_sum_of_slices(image_run):
    folder, images = image_run
    channel = load_channel(folder, "zqzimg", [], WINDOW)
    summed = slice_channel(channel, [2, 5], 0, "mean", "sum")
    assert summed.counts.shape == (64, 2)
    np.testing.assert_allclose(summed.counts[:, 0], images[0][:, 1] + images[0][:, 4], rtol=1e-5)


def test_each_slice_as_a_scan_averages_files(image_run):
    folder, images = image_run
    channel = load_channel(folder, "zqzimg", [], WINDOW)
    each = slice_channel(channel, [2, 7], 0, "mean", "each")
    assert each.rows_as_scans and each.prefix == "px "
    assert each.numbers.tolist() == [2, 7]
    np.testing.assert_allclose(each.counts[:, 1], (images[0][:, 6] + images[1][:, 6]) / 2, rtol=1e-5)


def test_slice_outside_chip(image_run):
    folder, _ = image_run
    channel = load_channel(folder, "zqzimg", [], WINDOW)
    with pytest.raises(ValueError, match="chip rows 1-8"):
        slice_channel(channel, [9], 0, "mean", "sum")


def test_image_data_off_ignores_slices(image_run):
    folder, images = image_run
    result = retrieve(Settings(data_dir=str(folder), reference_name="zqzimg", sample_name="zqzimg",
                               image_data=False, slices=[3], process_slice="3"))
    np.testing.assert_allclose(result.sample.raw_counts[:, 0], images[0].sum(axis=1), rtol=1e-5)


def test_each_slice_through_pipeline(image_run):
    folder, _ = image_run
    result = retrieve(Settings(data_dir=str(folder), reference_name="zqzimg", sample_name="zqzimg",
                               image_data=True, slices=[1, 2, 3], process_slice="each"))
    assert result.sample.n_scans == 3
    assert result.sample.n_spikes_replaced == 0
    assert "cosmic-ray removal skipped for each slice as a scan" in result.warnings


@pytest.mark.skipif(not FVB_DATA.is_dir(), reason="needs the lab's 2026-10-05 data")
def test_fvb_files_ignore_slices():
    result = retrieve(Settings(data_dir=str(FVB_DATA), reference_name="zqz2", sample_name="h2o2",
                               reference_file_numbers=[1, 2, 3], sample_file_numbers=[1, 2, 3],
                               image_data=True, slices=[140], process_slice="140"))
    assert not result.sample.is_image and result.sample.n_scans == 3
    assert any("slices ignored" in w for w in result.warnings)


def test_retrieve_slices_matches_single_slice(image_run):
    from hdsfg_engine.pipeline import retrieve_slices
    folder, _ = image_run
    settings = Settings(data_dir=str(folder), reference_name="zqzimg", sample_name="zqzimg",
                        image_data=True, slices=[5, 2])
    results = retrieve_slices(settings)
    assert list(results) == [2, 5]
    alone = retrieve(Settings(data_dir=str(folder), reference_name="zqzimg", sample_name="zqzimg",
                              image_data=True, slices=[5, 2], process_slice="5"))
    np.testing.assert_allclose(results[5].sample.counts, alone.sample.counts)
    np.testing.assert_allclose(results[5].absolute.mean, alone.absolute.mean, equal_nan=True)


def test_retrieve_slices_divided_by_first(image_run):
    from hdsfg_engine.pipeline import retrieve_slices
    folder, _ = image_run
    settings = Settings(data_dir=str(folder), reference_name="zqzimg", sample_name="zqzimg",
                        image_data=True, slices=[6, 3])
    results = retrieve_slices(settings, divide_by="first")
    assert all(r.settings.reference_slice == "3" for r in results.values())
    assert "slice px 3" in results[6].reference.processed
    assert "slice px 6" in results[6].sample.processed

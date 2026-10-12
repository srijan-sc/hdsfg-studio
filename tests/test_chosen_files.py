"""Chosen files with any names (File → Open files…), used as both reference and sample."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hdsfg_engine import Settings, retrieve
from hdsfg_engine.data import load_files, read_footer_exposure, sort_files

WINDOW = (325.0, 2805.0)


def write_image(path: Path, counts: np.ndarray, wavelength: np.ndarray, time: str) -> None:
    """Andor layout and footer: wavelength, one column per chip row, then the acquisition details."""
    lines = ["\t".join([f"{w:.5f}"] + [f"{c:.4f}" for c in row]) + "\t"
             for w, row in zip(wavelength, counts)]
    footer = (f"\n\nDate and Time:                Wed Oct 7 {time} 2026\n"
              "Exposure Time (secs):         1.00002\n"
              "Number of Accumulations:      10\n")
    path.write_text("\n".join(lines) + footer)


@pytest.fixture
def chosen(tmp_path):
    """Three image files with unrelated names, written out of time order."""
    rng = np.random.default_rng(1)
    wavelength = np.linspace(760, 800, 400)
    fringes = 1000 * np.exp(-((wavelength - 780) / 6) ** 2) * (1 + np.cos(wavelength * 9))
    names_times = (("quartz scan B.asc", "16:12:00.000"), ("first.asc", "16:10:00.000"),
                   ("qz_again_03.asc", "16:14:00.000"))
    images = {}
    for name, time in names_times:
        image = rng.normal(5, 1, (400, 6)) + fringes[:, None] * np.array([0, 0.2, 1, 0.8, 0, 0])
        write_image(tmp_path / name, image, wavelength, time)
        images[name] = image
    return tmp_path, images


def test_sorted_by_footer_time(chosen):
    folder, _ = chosen
    paths = sort_files(folder.glob("*.asc"))
    assert [Path(p).name for p in paths] == ["first.asc", "quartz scan B.asc", "qz_again_03.asc"]


def test_footer_exposure(chosen):
    folder, _ = chosen
    assert read_footer_exposure(folder / "first.asc") == (1.00002, 10)


def test_numbered_in_list_order_with_typed_exposure(chosen):
    folder, images = chosen
    paths = sort_files(folder.glob("*.asc"))
    channel = load_files(paths, [], WINDOW, 25.0)
    assert channel.numbers.tolist() == [1, 2, 3]
    assert channel.images.shape == (3, 400, 6)
    np.testing.assert_allclose(channel.file_exposure_sec, 25.0)
    np.testing.assert_allclose(channel.images[1], images["quartz scan B.asc"], atol=1e-3)
    some = load_files(paths, [1, 3], WINDOW, 25.0)
    assert some.numbers.tolist() == [1, 3]
    assert [p.name for p in some.files] == ["first.asc", "qz_again_03.asc"]


def test_files_are_reference_and_sample(chosen):
    folder, _ = chosen
    settings = Settings(data_dir=str(folder), files=sort_files(folder.glob("*.asc")),
                        file_exposure_sec=1.0, file_frames=10, remove_cosmic_rays=False,
                        laser_wavelength_nm=1031.5)
    result = retrieve(settings)
    assert result.reference.n_scans == result.sample.n_scans == 3
    assert result.sample.exposure_sec == 10.0
    assert result.fresnel_sample.material.name == "zqz"
    # each file ÷ the mean of the same files: the mean over the files is exactly 1
    np.testing.assert_allclose(result.relative.mean, 1, atol=1e-9)
    # a slice of the chip, and a number left out
    sliced = retrieve(replace(settings, image_data=True, slices=[3], process_slice="3",
                              sample_file_numbers=[2, 3]))
    assert sliced.sample.numbers.tolist() == [2, 3]
    np.testing.assert_allclose(sliced.relative.mean, 1, atol=1e-9)


def test_material_is_typed_in(chosen):
    folder, _ = chosen
    settings = Settings(data_dir=str(folder), files=sort_files(folder.glob("*.asc")),
                        file_material="h2o", remove_cosmic_rays=False)
    assert retrieve(settings).fresnel_sample.material.name == "h2o"

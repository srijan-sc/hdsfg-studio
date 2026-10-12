"""The homodyne tab's engine function against plain numpy, and on real data."""

from pathlib import Path

import numpy as np
import pytest

from hdsfg_engine import Settings, retrieve
from hdsfg_engine.homodyne import homodyne_per_scan

DATA = Path(__file__).parent.parent.parent / "20261005_50kHz_pinhole" / "data"


def test_divisors():
    rng = np.random.default_rng(0)
    analytic = rng.normal(size=(50, 6)) + 1j * rng.normal(size=(50, 6))
    first = homodyne_per_scan(analytic, "first")
    assert np.allclose(first[:, 0], 1)
    assert np.allclose(first, np.abs(analytic / analytic[:, [0]]) ** 2)
    assert np.allclose(homodyne_per_scan(analytic, "mean"),
                       np.abs(analytic / analytic.mean(axis=1, keepdims=True)) ** 2)
    assert np.allclose(homodyne_per_scan(analytic, "none"), np.abs(analytic) ** 2)
    with pytest.raises(ValueError):
        homodyne_per_scan(analytic, "last")


@pytest.mark.skipif(not DATA.is_dir(), reason="needs the lab's 2026-10-05 data")
def test_mean_matches_pipeline_when_quartz_is_both_channels():
    """Quartz against itself: |scan / mean|^2 is the pipeline's own |chi2|^2 per scan."""
    result = retrieve(Settings(data_dir=str(DATA), reference_name="zqz2", sample_name="zqz2"))
    ours = homodyne_per_scan(result.sample.analytic, "mean")
    pipeline = np.abs(result.relative.per_scan) ** 2
    assert np.max(np.abs(ours - pipeline)) / np.max(pipeline) < 1e-9

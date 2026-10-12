"""The Python engine against the validated MATLAB v37b, stage by stage.

The MATLAB reference files come from make_matlab_reference.m. MATLAB v37b in
turn matches the group's v37b Python script to ~1e-10 % (v37b/validation), so
passing here ties the engine to the original Python as well. Needs the lab's
2026-10-05 data and the reference files; skipped when either is missing.

Run:  .venv/bin/python -m pytest tests -q
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from hdsfg_engine import Settings, retrieve

TESTS = Path(__file__).parent
DATA = TESTS.parent.parent / "20261005_50kHz_pinhole" / "data"

pytestmark = pytest.mark.skipif(
    not DATA.is_dir() or not all((TESTS / f"matlab_reference_despike_{s}.mat").is_file() for s in ("on", "off")),
    reason="needs the lab's 2026-10-05 data and the MATLAB reference files")

# relative to the largest value of each quantity: floating-point agreement
TOLERANCE = 1e-9


def relative_error(python, matlab):
    python, matlab = np.asarray(python), np.asarray(matlab)
    return np.max(np.abs(python - matlab)) / np.max(np.abs(matlab))


@pytest.fixture(scope="module", params=["on", "off"])
def pair(request):
    reference = loadmat(TESTS / f"matlab_reference_despike_{request.param}.mat", squeeze_me=True)
    result = retrieve(Settings(data_dir=str(DATA), remove_cosmic_rays=request.param == "on"))
    return result, reference


def test_axes(pair):
    result, m = pair
    assert relative_error(result.grid.wavenumber_uniform, m["waveNumberUniform"]) < 1e-12
    assert relative_error(result.grid.time_ps, m["timePs"]) < 1e-12


def test_file_order_and_times(pair):
    result, m = pair
    for channel, prefix in ((result.reference, "reference"), (result.sample, "sample")):
        np.testing.assert_array_equal(channel.numbers, m[prefix + "Numbers"])
        np.testing.assert_allclose(channel.time_sec, m[prefix + "TimeSec"], atol=1e-6)


def test_cosmic_rays(pair):
    result, m = pair
    for channel, prefix in ((result.reference, "reference"), (result.sample, "sample")):
        np.testing.assert_array_equal(channel.spike_mask, m[prefix + "SpikeMask"].astype(bool))
        assert relative_error(channel.counts, m[prefix + "Counts"]) < TOLERANCE


def test_gates(pair):
    result, m = pair
    assert np.max(np.abs(result.gates["reference"] - m["gateReference"])) < 1e-12
    assert np.max(np.abs(result.gates["sample"] - m["gateSample"])) < 1e-12


def test_transform(pair):
    result, m = pair
    for channel, prefix in ((result.reference, "reference"), (result.sample, "sample")):
        assert relative_error(channel.resampled, m[prefix + "Resampled"]) < TOLERANCE
        assert relative_error(channel.analytic, m[prefix + "Analytic"]) < TOLERANCE


def test_fresnel(pair):
    result, m = pair
    assert relative_error(result.fresnel_reference.conversion_scale, m["conversionReference"]) < TOLERANCE
    assert relative_error(result.fresnel_sample.conversion_scale, m["conversionSample"]) < TOLERANCE
    assert result.rotation_deg == m["rotationDeg"]


@pytest.mark.parametrize("which", ["relative", "absolute"])
def test_chi2(pair, which):
    result, m = pair
    summary = getattr(result, which)
    assert relative_error(summary.per_scan, m[which + "PerScan"]) < TOLERANCE
    assert relative_error(summary.mean, m[which + "Mean"]) < TOLERANCE
    assert relative_error(summary.imag_error, m[which + "ImagError"]) < TOLERANCE
    assert relative_error(summary.real_error, m[which + "RealError"]) < TOLERANCE
    assert relative_error(summary.homodyne, m[which + "Homodyne"]) < TOLERANCE
    assert relative_error(summary.homodyne_error, m[which + "HomodyneError"]) < TOLERANCE


def test_phase(pair):
    """Mean phase, compared where it is not sitting on the 0/360 wrap."""
    result, m = pair
    for which in ("relative", "absolute"):
        python = getattr(result, which).phase_deg
        matlab = m[which + "PhaseDeg"]
        assert np.max(np.abs(python - matlab)) < 1e-6

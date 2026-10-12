"""The numerical steps of the v37b retrieval.

Each function is a port of the matching local function in ``v37bRetrieve.m``,
which itself reproduces the group's v37b Python script (plus cosmic-ray removal).
Arrays are [n_pixels x n_scans] throughout, as in the MATLAB.

Porting notes, all checked by the tests against the MATLAB output:
  * MATLAB interp1(...,'spline') and scipy's CubicSpline both default to
    not-a-knot end conditions.
  * MATLAB std defaults to N-1; every std here is ddof=0, as in numpy and the
    MATLAB port, which passes the population flag.
  * MATLAB medfilt1 and scipy medfilt both zero-pad the ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt, savgol_filter
from scipy.signal.windows import kaiser_bessel_derived

SPEED_OF_LIGHT_CM_PER_S = 2.99792458e10


# ---------------------------------------------------------------------------------
#  Cosmic rays
# ---------------------------------------------------------------------------------

def _medfilt_columns(values: np.ndarray, width: int) -> np.ndarray:
    """MATLAB medfilt1(values, width, [], 1): zero-padded running median per column."""
    if values.ndim == 1:
        return medfilt(values, width)
    return np.column_stack([medfilt(values[:, k], width) for k in range(values.shape[1])])


def remove_cosmic_rays(counts: np.ndarray, width: int, sigma: float, min_scans: int):
    """Replace single-pixel outliers with the median across scans.

    Returns (cleaned counts, spike mask). Exposures are first detrended against
    the first one with a running median, the test

        |counts - median| / median  >  sigma * medfilt(std / median)

    runs on the detrended spectra, and the drift is added back. The division is
    signed, as in the v3 Python, so the test fires on noise where the median
    is near zero; the replaced pixels are reported, not hidden.
    """
    mask = np.zeros(counts.shape, dtype=bool)
    if counts.shape[1] < min_scans:
        return counts.copy(), mask

    drift = _medfilt_columns(counts - counts[:, :1], width)
    detrended = counts - drift

    median = np.median(detrended, axis=1, keepdims=True)
    spread = np.std(detrended, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_scatter = (spread / median).ravel()
        threshold = medfilt(relative_scatter, width)[:, None]
        mask = np.abs(detrended - median) / median > sigma * threshold
    mask = np.nan_to_num(mask, nan=False).astype(bool)

    cleaned = np.where(mask, np.broadcast_to(median, counts.shape), detrended) + drift
    return cleaned, mask


# ---------------------------------------------------------------------------------
#  Axes
# ---------------------------------------------------------------------------------

@dataclass
class Grid:
    wavelength_nm: np.ndarray
    wavenumber: np.ndarray            # per pixel, descending
    wavenumber_uniform: np.ndarray    # the FFT grid, descending
    wavenumber_step: float
    time_ps: np.ndarray


def make_grid(wavelength_nm: np.ndarray, laser_wavelength_nm: float, sfg_or_dfg: str) -> Grid:
    """IR wavenumber of each pixel, the uniform grid, and the conjugate time axis.

    The time axis is built with linspace, as the Python does, so its spacing is
    1/(dNu (N-1)) rather than the FFT's exact 1/(dNu N).
    """
    n_pixels = len(wavelength_nm)
    if sfg_or_dfg == "sfg":
        wavenumber = (1 / wavelength_nm - 1 / laser_wavelength_nm) * 1e7
    elif sfg_or_dfg == "dfg":
        wavenumber = (1 / laser_wavelength_nm - 1 / wavelength_nm) * 1e7
    else:
        raise ValueError('sfg_or_dfg must be "sfg" or "dfg"')
    uniform = np.linspace(wavenumber.max(), wavenumber.min(), n_pixels)
    step = (wavenumber.max() - wavenumber.min()) / (n_pixels - 1)
    frequency_step = step * SPEED_OF_LIGHT_CM_PER_S
    time_ps = np.linspace(-0.5 / frequency_step, 0.5 / frequency_step, n_pixels) * 1e12
    return Grid(wavelength_nm, wavenumber, uniform, step, time_ps)


def resample(counts: np.ndarray, grid: Grid) -> np.ndarray:
    """Counts onto the uniform wavenumber grid, not-a-knot cubic spline."""
    order = np.argsort(grid.wavenumber)
    spline = CubicSpline(grid.wavenumber[order], counts[order], axis=0, bc_type="not-a-knot")
    return spline(grid.wavenumber_uniform)


# ---------------------------------------------------------------------------------
#  Time gate and transform
# ---------------------------------------------------------------------------------

def make_time_gate(time_ps: np.ndarray, range_ps, edges_ps, shape: int) -> np.ndarray:
    """Window keeping the local-oscillator sideband (generate_fft_filter).

    Inclusive and strict comparisons follow the Python branch by branch.
    """
    t = time_ps
    start, stop = min(range_ps), max(range_ps)
    rise, fall = edges_ps
    gate = np.zeros_like(t)

    if shape == 0:
        gate[:] = 1
    elif shape == 1:
        gate[(start < t) & (t < stop)] = 1
    elif shape == 2:
        rising = (start < t) & (t < start + rise)
        gate[rising] = 0.54 + 0.46 * np.cos(np.pi * (t[rising] - start - rise) / rise)
        gate[(start + rise < t) & (t < stop)] = 1
    elif shape in (3, 4):
        a, b = (0.54, 0.46) if shape == 3 else (0.5, 0.5)
        rising = (start <= t) & (t <= start + rise)
        falling = (stop - fall <= t) & (t <= stop)
        gate[rising] = a + b * np.cos(np.pi * (t[rising] - start - rise) / rise)
        gate[falling] = a + b * np.cos(np.pi * (-t[falling] + stop + fall) / fall)
        gate[(start + rise <= t) & (t <= stop - fall)] = 1
    elif shape == 5:
        inside = (start < t) & (t < stop)
        n_inside = int(inside.sum())
        if n_inside % 2:
            gate[inside] = kaiser_bessel_derived(n_inside + 1, 30)[:-1]
        else:
            gate[inside] = kaiser_bessel_derived(n_inside, 30)
    else:
        raise ValueError(f"gate shape must be 0 to 5, not {shape}")
    return gate


def transform(resampled: np.ndarray, gate: np.ndarray):
    """FFT, gate, inverse FFT, column by column. Returns (time domain, analytic)."""
    time_domain = np.fft.fftshift(np.fft.fft(resampled, axis=0), axes=0)
    analytic = np.fft.ifft(np.fft.ifftshift(time_domain * gate[:, None], axes=0), axis=0)
    return time_domain, analytic


def gate_spectra(spectra: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """FFT, multiply by a time gate, inverse FFT (the post-division filter)."""
    return transform(spectra, gate)[1]


def smooth_complex(spectra: np.ndarray, order: int, length: int) -> np.ndarray:
    """Savitzky-Golay on the real and imaginary parts separately."""
    return (savgol_filter(spectra.real, length, order, axis=0)
            + 1j * savgol_filter(spectra.imag, length, order, axis=0))


def reference_weights(reference_time: np.ndarray, sample_time: np.ndarray, time_weighted: bool):
    """Weights, summing to 1, for averaging the reference scans."""
    n = len(reference_time)
    if not time_weighted:
        return np.full(n, 1 / n)
    with np.errstate(divide="ignore"):
        inverse_distance = 1 / np.abs(reference_time - sample_time.mean())
    if np.any(np.isinf(inverse_distance)):
        inverse_distance = np.isinf(inverse_distance).astype(float)
    return inverse_distance / inverse_distance.sum()


# ---------------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------------

@dataclass
class Summary:
    per_scan: np.ndarray
    mean: np.ndarray
    real_error: np.ndarray
    imag_error: np.ndarray
    phase_per_scan: np.ndarray
    phase_deg: np.ndarray
    phase_error: np.ndarray
    homodyne_per_scan: np.ndarray
    homodyne: np.ndarray
    homodyne_error: np.ndarray


def summarise(per_scan: np.ndarray) -> Summary:
    """Mean and 95% confidence interval (std * 1.96 / sqrt N) across scans.

    Phase and |chi2|^2 are averaged per scan, as in the Python; a phase wrapped
    to [0, 360) and then averaged is meaningless where scans straddle the wrap.
    """
    factor = 1.96 / np.sqrt(per_scan.shape[1])
    phase = np.mod(np.degrees(np.angle(per_scan)), 360)
    homodyne = per_scan.real ** 2 + per_scan.imag ** 2
    return Summary(
        per_scan=per_scan,
        mean=per_scan.mean(axis=1),
        real_error=per_scan.real.std(axis=1) * factor,
        imag_error=per_scan.imag.std(axis=1) * factor,
        phase_per_scan=phase,
        phase_deg=phase.mean(axis=1),
        phase_error=phase.std(axis=1) * factor,
        homodyne_per_scan=homodyne,
        homodyne=homodyne.mean(axis=1),
        homodyne_error=homodyne.std(axis=1) * factor,
    )

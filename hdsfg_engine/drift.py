"""Phase change since the first scan, at every wavenumber, with no fitting.

Port of ``20261005_50kHz_pinhole/analysis/plot_phase_drift.m``:

    phase change = angle( group / first group )

on the complex spectrum after the gate and inverse FFT. The fringe phase common
to every scan cancels in the ratio. Grouping averages consecutive complex
spectra; smoothing averages conj(first) * group over a wavenumber window before
taking the angle, so strong pixels count more than weak ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Drift:
    time_min: np.ndarray        # per point, from the first point
    labels: list[str]
    phase_change_deg: np.ndarray    # [n_pixels x n_points]
    amplitude: np.ndarray       # |mean grouped spectrum|, for shading weak regions


def _movmean_shrink(values: np.ndarray, width: int) -> np.ndarray:
    """MATLAB movmean(values, width, 1) with shrinking end windows."""
    if width <= 1:
        return values
    n = values.shape[0]
    half_before = (width - 1) // 2 if width % 2 else width // 2
    half_after = width - 1 - half_before
    cumulative = np.concatenate([np.zeros((1,) + values.shape[1:], values.dtype),
                                 np.cumsum(values, axis=0)])
    index = np.arange(n)
    low = np.clip(index - half_before, 0, n)
    high = np.clip(index + half_after + 1, 0, n)
    counts = (high - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return (cumulative[high] - cumulative[low]) / counts


def phase_drift(analytic: np.ndarray, time_sec: np.ndarray, numbers: np.ndarray,
                scans_per_group: int = 1, smooth_pixels: int = 1, prefix: str = "s") -> Drift:
    n_scans = analytic.shape[1]
    group_of = np.arange(n_scans) // max(scans_per_group, 1)
    n_points = group_of[-1] + 1

    grouped = np.column_stack([analytic[:, group_of == k].mean(axis=1) for k in range(n_points)])
    times = np.array([time_sec[group_of == k].mean() for k in range(n_points)])
    labels = []
    for k in range(n_points):
        members = numbers[group_of == k]
        label = f"{prefix}{members[0]}" if len(members) == 1 else f"{prefix}{members[0]}–{members[-1]}"
        labels.append(label)

    product = np.conj(grouped[:, :1]) * grouped
    product = _movmean_shrink(product, smooth_pixels)
    return Drift(time_min=(times - times[0]) / 60, labels=labels,
                 phase_change_deg=np.degrees(np.angle(product)),
                 amplitude=np.abs(grouped.mean(axis=1)))

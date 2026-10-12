"""Calculated homodyne of every scan, |E(t) / divisor|^2.

Port of ``20261006_50kHz/analysis/plot_homodyne_all_scans.m``. E(t) is a
scan's gated complex spectrum (Re + i Im) after the inverse FFT. The divisor
is the first scan, the mean of all scans, or nothing:

    "first"   |E(t) / E(0)|^2      how each scan has changed since the start
    "mean"    |E(t) / <E>|^2       how each scan differs from the average
    "none"    |E(t)|^2             |LO|^2 x |signal|^2, so the shape also carries
                                   the IR, visible and detector spectra
"""

from __future__ import annotations

import numpy as np

DIVISORS = ("first", "mean", "none")


def homodyne_per_scan(analytic: np.ndarray, divide_by: str = "first") -> np.ndarray:
    """|analytic / divisor|^2 for every scan, [n_pixels x n_scans]."""
    if divide_by == "first":
        analytic = analytic / analytic[:, :1]
    elif divide_by == "mean":
        analytic = analytic / analytic.mean(axis=1, keepdims=True)
    elif divide_by != "none":
        raise ValueError(f'divide_by must be one of {DIVISORS}, not "{divide_by}"')
    return analytic.real ** 2 + analytic.imag ** 2

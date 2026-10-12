"""Row by row: quartz row n against water row n, the rows binned only at the end.

The usual pipeline adds every chip row first (Full Vertical Binning) and then
divides the summed water spectrum by the summed quartz spectrum. Here each chip
row goes through the whole retrieval on its own (cosmic rays, resample, gate,
transform, Fresnel, division): water row n ÷ quartz row n. The rows are
combined after the division, one of three ways (``RowInput.binning``):

  "weighted"  each row's ratio weighted by |quartz row|² at that wavenumber, so
              a row that is dark at some wavenumber hardly counts there. This
              equals  Σ conj(Q_n) W_n / Σ |Q_n|²  (times the same constants).
  "strong"    plain mean over the rows whose quartz is above the weak-reference
              fraction of the brightest quartz row, per wavenumber.
  "mean"      plain mean over every chosen row.

The quartz files chosen by number are averaged into one quartz image (as the
usual pipeline averages the reference scans); each water file is divided by
it, binned, and the files give the mean and 95 % interval. For comparison the
same files are also processed the usual way (rows summed first, then divided),
over the same rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import data, fresnel, processing
from .settings import Settings

BINNINGS = {
    "weighted": "Weighted by quartz² (bright rows count more)",
    "strong": "Mean of rows where quartz is strong",
    "mean": "Plain mean of the rows",
}


@dataclass
class RowInput:
    """What the row-by-row view adds to the sidebar's settings."""
    quartz_files: list[str] = field(default_factory=list)
    water_files: list[str] = field(default_factory=list)
    quartz_numbers: list[int] = field(default_factory=list)    # positions in quartz_files; empty = all
    water_numbers: list[int] = field(default_factory=list)
    quartz_exposure_sec: float = 1.0      # one frame
    quartz_frames: int = 1
    water_exposure_sec: float = 1.0
    water_frames: int = 1
    water_material: str = "h2o"
    chip_rows: list[int] = field(default_factory=list)   # chip rows used, 1-based; empty = whole chip
    binning: str = "weighted"


@dataclass
class RowResult:
    settings: Settings
    rows_input: RowInput
    quartz: data.Channel                  # as loaded; images [n_files x n_pixels x n_chip_rows]
    water: data.Channel
    rows: np.ndarray                      # chip rows used, 1-based
    grid: processing.Grid
    gates: dict
    fresnel_quartz: fresnel.Fresnel
    fresnel_water: fresnel.Fresnel
    rotation_deg: float
    quartz_resampled: np.ndarray          # quartz per row on the uniform grid, before the gate [n_pixels x n_rows]
    water_resampled: np.ndarray           # the same for water, mean of its files
    quartz_rows: np.ndarray               # gated quartz per row, mean of its files [n_pixels x n_rows]
    water_rows: np.ndarray                # gated water per row, mean of its files [n_pixels x n_rows]
    weights: np.ndarray                   # binning weight of each row [n_pixels x n_rows]
    relative_rows: np.ndarray             # water row ÷ quartz row, mean of the water files
    absolute_rows: np.ndarray
    relative: processing.Summary          # binned, across the water files
    absolute: processing.Summary
    fvb_relative: processing.Summary      # rows summed first, then divided
    fvb_absolute: processing.Summary
    spikes_replaced: tuple[int, int] = (0, 0)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def reference_strength(self) -> np.ndarray:
        """Quartz over the chosen rows, sqrt(Σ |Q_n|²): where small, every row is noise ÷ noise."""
        return np.sqrt((np.abs(self.quartz_rows) ** 2).sum(axis=1))

    def usable_range(self) -> tuple[float, float]:
        strength = self.reference_strength
        strong = strength >= self.settings.weak_reference_fraction * strength.max()
        wavenumber = self.grid.wavenumber_uniform[strong]
        return float(wavenumber.min()), float(wavenumber.max())


def row_profile(channel: data.Channel) -> np.ndarray:
    """Mean counts of each chip row, mean of the files: the brightness of each row."""
    return channel.image.mean(axis=0)


def bright_rows(channel: data.Channel, fraction: float = 0.1) -> list[int]:
    """Chip rows (1-based), first to last, whose mean counts above the median row exceed
    ``fraction`` of the brightest row's. Empty (the whole chip) if nothing stands out."""
    profile = row_profile(channel)
    profile = profile - np.median(profile)
    rows = np.flatnonzero(profile > fraction * profile.max()) + 1
    if not len(rows) or profile.max() <= 0:
        return []
    return list(range(int(rows.min()), int(rows.max()) + 1))


def _bin(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Σ_n w_n values_n / Σ_n w_n over the row axis (1); NaN where no row has weight."""
    shape = (slice(None), slice(None)) + (None,) * (values.ndim - 2)
    w = weights[shape]
    total = np.where(w > 0, w * values, 0).sum(axis=1)
    norm = weights.sum(axis=1)[(slice(None),) + (None,) * (values.ndim - 2)]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(norm > 0, total / norm, complex(np.nan, np.nan))     # no data: Re and Im both NaN


def _after_division(spectra: np.ndarray, s: Settings, grid: processing.Grid) -> np.ndarray:
    """The sidebar's post-division filter and smoothing; NaN points are left out and kept NaN."""
    if not s.post_division_filter and s.smoothing_order == 0:
        return spectra
    missing = ~np.isfinite(spectra)
    filled = np.where(missing, 0, spectra)
    if s.post_division_filter:
        gate = processing.make_time_gate(grid.time_ps, (-s.post_division_limit_ps, s.post_division_limit_ps),
                                         (s.post_division_width_ps, s.post_division_width_ps), 4)
        filled = processing.gate_spectra(filled, gate)
    if s.smoothing_order > 0:
        filled = processing.smooth_complex(filled, s.smoothing_order, s.smoothing_length)
    return np.where(missing, np.nan, filled)


class RowPipeline:
    """Reads each file set once and keeps the gated rows until a setting before the division changes."""

    def __init__(self):
        self._cache: dict[str, tuple[tuple, object]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def _stage(self, name, key, compute):
        cached = self._cache.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]
        value = compute()
        self._cache[name] = (key, value)
        return value

    def read(self, files, window, name: str) -> data.Channel:
        """Every file of the list, read once and kept (the exposure is applied by ``_choose``)."""
        def load():
            channel = data.load_files(files, [], window, 0.0, name=name)
            if not channel.is_image:
                raise ValueError(f"The {name} files are Full Vertical Binning spectra: "
                                 "row by row needs full-chip images")
            return channel
        return self._stage(f"read {name}", (tuple(files), tuple(window)), load)

    def quartz_bright_rows(self, s: Settings, r: RowInput) -> list[int]:
        """``bright_rows`` of the chosen quartz files (read through the same cache as ``run``)."""
        channel = self.read(r.quartz_files, tuple(s.wavelength_window_nm), "quartz")
        return bright_rows(_choose(channel, r.quartz_numbers, 0.0, "quartz"))

    def run(self, s: Settings, r: RowInput) -> RowResult:
        started = time.perf_counter()
        warnings: list[str] = []
        if not r.quartz_files or not r.water_files:
            raise ValueError("Choose the quartz files and the water files")
        quartz_total = r.quartz_exposure_sec * r.quartz_frames
        water_total = r.water_exposure_sec * r.water_frames

        # 1. read every file of each list once; choosing files by number, the exposure and the
        #    material never read them again
        window = tuple(s.wavelength_window_nm)
        quartz_all = self.read(r.quartz_files, window, "quartz")
        water_all = self.read(r.water_files, window, "water")
        if not np.array_equal(quartz_all.wavelength_nm, water_all.wavelength_nm):
            raise ValueError("Quartz and water were recorded on different wavelength axes")
        if quartz_all.images.shape[2] != water_all.images.shape[2]:
            raise ValueError(f"Quartz has {quartz_all.images.shape[2]} chip rows, "
                             f"water {water_all.images.shape[2]}")
        quartz = _choose(quartz_all, r.quartz_numbers, quartz_total, "quartz")
        water = _choose(water_all, r.water_numbers, water_total, r.water_material)
        data.sort_by_time(quartz, water)

        n_chip_rows = quartz.images.shape[2]
        rows = np.array(sorted(set(int(v) for v in r.chip_rows)) or range(1, n_chip_rows + 1))
        outside = rows[(rows < 1) | (rows > n_chip_rows)]
        if len(outside):
            raise ValueError(f"The chip has rows 1-{n_chip_rows}; row {outside[0]} is outside")

        # 2. cosmic rays, row by row across the files; then the quartz files are averaged
        #    (every later step is linear, so averaging first gives the same quartz)
        def despike():
            cut, replaced = [], []
            for channel in (quartz, water):
                counts = np.transpose(channel.images[:, :, rows - 1], (1, 2, 0)).astype(float)
                n_spikes = 0
                if s.remove_cosmic_rays:
                    for k in range(counts.shape[1]):
                        counts[:, k, :], mask = processing.remove_cosmic_rays(
                            counts[:, k, :], s.median_filter_width, s.spike_threshold_sigma,
                            s.min_exposures_for_despike)
                        n_spikes += int(mask.sum())
                cut.append(counts)
                replaced.append(n_spikes)
            return cut[0].mean(axis=2), cut[1], tuple(replaced)

        despike_key = (tuple(r.quartz_files), tuple(r.water_files), tuple(quartz.numbers),
                       tuple(water.numbers), window, tuple(rows), s.remove_cosmic_rays,
                       s.median_filter_width, s.spike_threshold_sigma, s.min_exposures_for_despike)
        quartz_counts, water_counts, spikes = self._stage("despike", despike_key, despike)
        if s.remove_cosmic_rays:
            few = [name for name, channel in (("quartz", quartz), ("water", water))
                   if channel.n_scans < s.min_exposures_for_despike]
            if few:
                warnings.append(f"cosmic-ray removal needs {s.min_exposures_for_despike} files; "
                                f"skipped for {' and '.join(few)}")

        # 3. resample, gate and transform: the averaged quartz once per row, every water file;
        #    the water in blocks of rows so the spline never holds the whole chip at once
        def transform():
            grid = processing.make_grid(quartz.wavelength_nm, s.laser_wavelength_nm, s.sfg_or_dfg)
            gates = {
                "reference": processing.make_time_gate(grid.time_ps, s.gate_range_reference_ps,
                                                       s.gate_edges_reference_ps, s.gate_shape),
                "sample": processing.make_time_gate(grid.time_ps, s.gate_range_sample_ps,
                                                    s.gate_edges_sample_ps, s.gate_shape),
            }
            quartz_resampled = processing.resample(quartz_counts, grid)
            quartz_rows = processing.transform(quartz_resampled, gates["reference"])[1]
            n_pixels, n_rows, n_files = water_counts.shape
            water_gated = np.empty((n_pixels, n_rows, n_files), dtype=complex)
            water_resampled = np.empty((n_pixels, n_rows))           # mean of the files, for the time domain
            block = max(1, 2048 // n_files)
            for low in range(0, n_rows, block):
                part = water_counts[:, low:low + block, :]
                resampled = processing.resample(part.reshape(n_pixels, -1), grid)
                water_resampled[:, low:low + block] = resampled.reshape(part.shape).mean(axis=2)
                water_gated[:, low:low + block, :] = processing.transform(
                    resampled, gates["sample"])[1].reshape(part.shape)
            return grid, gates, quartz_rows, water_gated, quartz_resampled, water_resampled

        transform_key = despike_key + (s.laser_wavelength_nm, s.sfg_or_dfg, s.gate_shape,
                                       tuple(s.gate_range_reference_ps), tuple(s.gate_range_sample_ps),
                                       tuple(s.gate_edges_reference_ps), tuple(s.gate_edges_sample_ps))
        grid, gates, quartz_rows, water_gated, quartz_resampled, water_resampled = self._stage(
            "transform", transform_key, transform)

        def compute_fresnel():
            common = dict(wavenumber=grid.wavenumber_uniform, laser_wavelength_nm=s.laser_wavelength_nm,
                          angles_rad=s.angles_rad, polarization=s.polarization,
                          sfg_or_dfg=s.sfg_or_dfg, reference_dir=s.reference_dir)
            return (fresnel.compute("zqz", s.interface_model_reference, **common),
                    fresnel.compute(r.water_material, s.interface_model_sample, **common))

        fresnel_quartz, fresnel_water = self._stage(
            "fresnel", (transform_key, r.water_material, s.interface_model_reference,
                        s.interface_model_sample, s.polarization, s.geometry, s.reference_dir),
            compute_fresnel)

        # ---- divide row by row, then bin ----------------------------------------------
        rotation_deg = (s.phase_correction_deg + fresnel_water.material.phase_deg
                        - fresnel_quartz.material.phase_deg)
        if s.sfg_or_dfg == "dfg":
            rotation_deg -= 180
        factor = quartz_total / water_total * np.exp(1j * np.deg2rad(rotation_deg))
        conversion = (fresnel_quartz.conversion_scale / fresnel_water.conversion_scale
                      * s.absolute_scale * fresnel_quartz.material.known_chi2)

        with np.errstate(divide="ignore", invalid="ignore"):
            relative_each = water_gated / quartz_rows[:, :, None] * factor     # [pixels x rows x files]

        strength = np.abs(quartz_rows)
        if r.binning == "weighted":
            weights = strength ** 2
        elif r.binning == "strong":
            weights = (strength >= s.weak_reference_fraction * strength.max()).astype(float)
        elif r.binning == "mean":
            weights = np.ones_like(strength)
        else:
            raise ValueError(f'binning must be one of {", ".join(BINNINGS)}, not "{r.binning}"')

        # the Fresnel conversion is the same for every row, so it is applied after binning
        binned = _bin(relative_each, weights)
        relative = _after_division(binned, s, grid)
        absolute = _after_division(binned * conversion[:, None], s, grid)
        relative_rows = relative_each.mean(axis=2)

        # the usual way, for comparison: rows summed first (FVB over the same rows), then divided
        with np.errstate(divide="ignore", invalid="ignore"):
            fvb = water_gated.sum(axis=1) / quartz_rows.sum(axis=1)[:, None] * factor
        fvb_relative = _after_division(fvb, s, grid)
        fvb_absolute = _after_division(fvb * conversion[:, None], s, grid)

        for material in (fresnel_quartz.material, fresnel_water.material):
            if material.placeholder:
                warnings.append(f"{material.name}: material constants are placeholders")

        return RowResult(settings=s, rows_input=r, quartz=quartz, water=water, rows=rows, grid=grid,
                         gates=gates, fresnel_quartz=fresnel_quartz, fresnel_water=fresnel_water,
                         rotation_deg=rotation_deg, quartz_resampled=quartz_resampled,
                         water_resampled=water_resampled, quartz_rows=quartz_rows,
                         water_rows=water_gated.mean(axis=2), weights=weights,
                         relative_rows=relative_rows, absolute_rows=relative_rows * conversion[:, None],
                         relative=processing.summarise(relative), absolute=processing.summarise(absolute),
                         fvb_relative=processing.summarise(fvb_relative),
                         fvb_absolute=processing.summarise(fvb_absolute),
                         spikes_replaced=spikes, warnings=warnings,
                         seconds=time.perf_counter() - started)


def _choose(channel: data.Channel, numbers, total_exposure: float, name: str) -> data.Channel:
    """The files picked by number (positions in the list, 1-based; empty = all), with the
    exposure typed in: each file's time moves to the middle of its acquisition."""
    keep = np.isin(channel.numbers, numbers) if numbers else np.ones(len(channel.numbers), dtype=bool)
    if not keep.any():
        raise FileNotFoundError(f"None of the chosen {name} file numbers are in the list")
    return replace(channel, name=name, files=[f for f, k in zip(channel.files, keep) if k],
                   numbers=channel.numbers[keep], counts=channel.counts[:, keep],
                   raw_counts=channel.raw_counts[:, keep],
                   time_sec=channel.time_sec[keep] - total_exposure / 2,
                   file_exposure_sec=np.full(int(keep.sum()), float(total_exposure)),
                   images=channel.images[keep])


def retrieve_rows(settings: Settings, rows_input: RowInput) -> RowResult:
    """One-shot run without caching."""
    return RowPipeline().run(settings, rows_input)

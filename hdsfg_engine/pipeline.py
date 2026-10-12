"""The v37b retrieval as six cached stages.

    load -> slice -> despike -> resample -> transform -> fresnel -> divide

Each stage remembers the settings it last ran with. ``Pipeline.run`` reruns
only the first stage whose settings changed and everything after it, so moving
a gate edge redoes the transform and the division but never re-reads files.
That is what lets the GUI update while a dropdown is being changed.

The numbers are those of ``v37bRetrieve.m``; ``tests/test_against_matlab.py``
checks every stage against the MATLAB output.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import data, fresnel, processing
from .settings import Settings

STAGES = ("load", "slice", "despike", "resample", "transform", "fresnel", "divide")


@dataclass
class Result:
    settings: Settings
    reference: data.Channel
    sample: data.Channel
    grid: processing.Grid
    gates: dict
    fresnel_reference: fresnel.Fresnel
    fresnel_sample: fresnel.Fresnel
    rotation_deg: float
    relative: processing.Summary
    absolute: processing.Summary
    warnings: list[str] = field(default_factory=list)
    stage_seconds: dict = field(default_factory=dict)
    rerun_from: str = "load"

    @property
    def reference_strength(self) -> np.ndarray:
        """|mean analytic reference|: where it is small the division amplifies noise."""
        return np.abs(self.reference.analytic.mean(axis=1))

    def usable_range(self) -> tuple[float, float]:
        strength = self.reference_strength
        strong = strength >= self.settings.weak_reference_fraction * strength.max()
        wavenumber = self.grid.wavenumber_uniform[strong]
        return float(wavenumber.min()), float(wavenumber.max())


def _copy_channel(channel: data.Channel) -> data.Channel:
    return replace(channel, counts=channel.counts.copy(), extra=dict(channel.extra))


class Pipeline:
    def __init__(self):
        self._cache: dict[str, tuple[tuple, object]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def _stage(self, name: str, key: tuple, compute):
        cached = self._cache.get(name)
        if cached is not None and cached[0] == key:
            return cached[1], False
        value = compute()
        self._cache[name] = (key, value)
        return value, True

    def run(self, s: Settings) -> Result:
        warnings: list[str] = []
        seconds: dict[str, float] = {}
        rerun_from = None
        upstream_changed = False

        def stage(name, own_key, compute):
            nonlocal rerun_from, upstream_changed
            started = time.perf_counter()
            if upstream_changed:          # an earlier stage reran: this one must too
                self._cache.pop(name, None)
            value, recomputed = self._stage(name, own_key, compute)
            if recomputed:
                upstream_changed = True
                rerun_from = rerun_from or name
                seconds[name] = time.perf_counter() - started
            return value

        # 1. load and sort by time
        def load():
            if s.files:          # chosen files: one set, used as both reference and sample
                sample = data.load_files(s.files, s.sample_file_numbers, s.wavelength_window_nm,
                                         s.file_exposure_sec * s.file_frames)
                reference = copy.deepcopy(sample)
                data.sort_by_time(reference, sample)
                return reference, sample
            reference = data.load_channel(s.data_dir, s.reference_name, s.reference_file_numbers,
                                          s.wavelength_window_nm)
            sample = data.load_channel(s.data_dir, s.sample_name, s.sample_file_numbers,
                                       s.wavelength_window_nm)
            if not np.array_equal(reference.wavelength_nm, sample.wavelength_nm):
                raise ValueError("Reference and sample were recorded on different wavelength axes.")
            data.sort_by_time(reference, sample)
            return reference, sample

        reference_raw, sample_raw = stage(
            "load", (s.data_dir, s.reference_name, s.sample_name,
                     tuple(s.reference_file_numbers), tuple(s.sample_file_numbers),
                     tuple(s.wavelength_window_nm), tuple(s.files), s.file_exposure_sec,
                     s.file_frames), load)

        # 1b. image runs: cut the slices (files are not read again)
        def cut_slices():
            if not s.image_data:
                return reference_raw, sample_raw
            reference_process = s.process_slice if s.reference_slice == "same" else s.reference_slice
            return (data.slice_channel(reference_raw, s.slices, s.slice_half_width, s.slice_combine,
                                       reference_process),
                    data.slice_channel(sample_raw, s.slices, s.slice_half_width, s.slice_combine,
                                       s.process_slice))

        reference_raw, sample_raw = stage("slice", (s.image_data, tuple(s.slices), s.slice_half_width,
                                                    s.slice_combine, s.process_slice, s.reference_slice),
                                          cut_slices)
        if s.image_data:
            for raw in (reference_raw, sample_raw):
                if not raw.is_image:
                    warnings.append(f"{raw.name}: Full Vertical Binning files, slices ignored")
            if not s.slices and (reference_raw.is_image or sample_raw.is_image):
                warnings.append("no slices chosen: whole chip summed")
        if s.remove_cosmic_rays and (reference_raw.rows_as_scans or sample_raw.rows_as_scans):
            warnings.append("cosmic-ray removal skipped for each slice as a scan")

        # 2. cosmic rays
        def despike():
            channels = []
            for raw in (reference_raw, sample_raw):
                channel = _copy_channel(raw)
                if s.remove_cosmic_rays and not raw.rows_as_scans:   # rows are not repeats
                    channel.counts, channel.spike_mask = processing.remove_cosmic_rays(
                        raw.counts, s.median_filter_width, s.spike_threshold_sigma,
                        s.min_exposures_for_despike)
                else:
                    channel.spike_mask = np.zeros(raw.counts.shape, dtype=bool)
                channel.n_spikes_replaced = int(channel.spike_mask.sum())
                channels.append(channel)
            return tuple(channels)

        reference, sample = stage("despike", (s.remove_cosmic_rays, s.median_filter_width,
                                              s.spike_threshold_sigma, s.min_exposures_for_despike),
                                  despike)

        # 3. axes and resampling
        def resample():
            grid = processing.make_grid(reference.wavelength_nm, s.laser_wavelength_nm, s.sfg_or_dfg)
            return grid, processing.resample(reference.counts, grid), processing.resample(sample.counts, grid)

        grid, reference_resampled, sample_resampled = stage(
            "resample", (s.laser_wavelength_nm, s.sfg_or_dfg), resample)

        # 4. gate and transform
        def transform():
            gates = {
                "reference": processing.make_time_gate(grid.time_ps, s.gate_range_reference_ps,
                                                       s.gate_edges_reference_ps, s.gate_shape),
                "sample": processing.make_time_gate(grid.time_ps, s.gate_range_sample_ps,
                                                    s.gate_edges_sample_ps, s.gate_shape),
            }
            ref_time, ref_analytic = processing.transform(reference_resampled, gates["reference"])
            sam_time, sam_analytic = processing.transform(sample_resampled, gates["sample"])
            return gates, ref_time, ref_analytic, sam_time, sam_analytic

        gates, ref_time, ref_analytic, sam_time, sam_analytic = stage(
            "transform", (s.gate_shape, tuple(s.gate_range_reference_ps), tuple(s.gate_range_sample_ps),
                          tuple(s.gate_edges_reference_ps), tuple(s.gate_edges_sample_ps)), transform)

        reference.resampled, reference.time_domain, reference.analytic = \
            reference_resampled, ref_time, ref_analytic
        sample.resampled, sample.time_domain, sample.analytic = \
            sample_resampled, sam_time, sam_analytic

        # 5. Fresnel: chosen files are the material typed in, runs go by their name
        reference_material = s.file_material if s.files else s.reference_name
        sample_material = s.file_material if s.files else s.sample_name

        def compute_fresnel():
            common = dict(wavenumber=grid.wavenumber_uniform, laser_wavelength_nm=s.laser_wavelength_nm,
                          angles_rad=s.angles_rad, polarization=s.polarization,
                          sfg_or_dfg=s.sfg_or_dfg, reference_dir=s.reference_dir)
            return (fresnel.compute(reference_material, s.interface_model_reference, **common),
                    fresnel.compute(sample_material, s.interface_model_sample, **common))

        fresnel_reference, fresnel_sample = stage(
            "fresnel", (reference_material, sample_material, s.interface_model_reference,
                        s.interface_model_sample, s.polarization, s.geometry, s.sfg_or_dfg,
                        s.laser_wavelength_nm, s.reference_dir), compute_fresnel)

        # 6. divide, filter, average
        reference.exposure_sec, warning = data.resolve_exposure(s.reference_exposure_sec, reference)
        warnings += [warning] if warning else []
        sample.exposure_sec, warning = data.resolve_exposure(s.sample_exposure_sec, sample)
        warnings += [warning] if warning else []

        rotation_deg = (s.phase_correction_deg + fresnel_sample.material.phase_deg
                        - fresnel_reference.material.phase_deg)
        if s.sfg_or_dfg == "dfg":
            rotation_deg -= 180

        def divide():
            rotation = np.exp(1j * np.deg2rad(rotation_deg))
            ratio = reference.exposure_sec / sample.exposure_sec
            weights = processing.reference_weights(reference.time_sec, sample.time_sec,
                                                   s.time_weighted_reference)
            reference_mean = ref_analytic @ weights
            relative = sam_analytic / reference_mean[:, None] * ratio * rotation

            reference_converted = (ref_analytic / fresnel_reference.conversion_scale[:, None]) @ weights
            sample_converted = sam_analytic / fresnel_sample.conversion_scale[:, None]
            absolute = (s.absolute_scale * sample_converted / reference_converted[:, None]
                        * fresnel_reference.material.known_chi2 * ratio * rotation)

            if s.post_division_filter:
                post_gate = processing.make_time_gate(
                    grid.time_ps, (-s.post_division_limit_ps, s.post_division_limit_ps),
                    (s.post_division_width_ps, s.post_division_width_ps), 4)
                relative = processing.gate_spectra(relative, post_gate)
                absolute = processing.gate_spectra(absolute, post_gate)
            if s.smoothing_order > 0:
                relative = processing.smooth_complex(relative, s.smoothing_order, s.smoothing_length)
                absolute = processing.smooth_complex(absolute, s.smoothing_order, s.smoothing_length)
            return processing.summarise(relative), processing.summarise(absolute)

        relative, absolute = stage(
            "divide", (rotation_deg, reference.exposure_sec, sample.exposure_sec,
                       s.time_weighted_reference, s.absolute_scale, s.post_division_filter,
                       s.post_division_limit_ps, s.post_division_width_ps,
                       s.smoothing_order, s.smoothing_length), divide)

        for material in (fresnel_reference.material, fresnel_sample.material):
            if material.placeholder:
                warnings.append(f"{material.name}: material constants are placeholders")

        return Result(settings=s, reference=reference, sample=sample, grid=grid, gates=gates,
                      fresnel_reference=fresnel_reference, fresnel_sample=fresnel_sample,
                      rotation_deg=rotation_deg, relative=relative, absolute=absolute,
                      warnings=warnings, stage_seconds=seconds, rerun_from=rerun_from or "none")


def retrieve(settings: Settings) -> Result:
    """One-shot run without caching."""
    return Pipeline().run(settings)


def retrieve_slices(settings: Settings, pipeline: Pipeline | None = None,
                    divide_by: str = "same") -> dict[int, Result]:
    """The full retrieval for every slice on its own.

    Slice c of the sample is divided by slice c of the reference
    (``divide_by="same"``), by the sum of the reference slices ("sum") or by
    the reference's first slice, the lowest chip row ("first"). One
    pipeline is reused, so the files are read once and each slice reruns from
    the slice stage. Keys are the slices' centre rows.
    """
    pipeline = pipeline or Pipeline()
    if divide_by == "first" and settings.slices:
        divide_by = str(min(settings.slices))
    return {centre: pipeline.run(replace(settings, image_data=True, process_slice=str(centre),
                                         reference_slice=divide_by))
            for centre in sorted(settings.slices)}

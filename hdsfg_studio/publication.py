"""Publication figures (matplotlib) and the results export.

The same ``publication_figure`` draws the in-app preview and the exported
SVG/PDF, so what the tab shows is what gets written. Text output follows
``v37bRetrieve.m``: tab-separated *_ax_and_spec.txt files that the ag_*
comparison spectra and the group's tools already read. Every export also
writes settings.json, so the parameters behind a result are never lost.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from hdsfg_engine.pipeline import Result

IMAG, REAL = "#2F6FBA", "#C8552B"
# one colour per chip-image slice, shared by the Chip image tab and its figure
SLICE_COLOURS = ("#3B82F6", "#E8622C", "#16A34A", "#D99A00", "#8B5CF6", "#0E9F9C", "#DC2626", "#DB2777")


def publication_figure(result: Result, quantity: str = "chi2", units: str = "absolute",
                       figure: Figure | None = None, x_range=None) -> Figure:
    """Im and Re chi2 (or |chi2|^2) with 95% bands, journal styling."""
    figure = figure or Figure(figsize=(5.0, 3.5), dpi=100)
    figure.clear()
    axes = figure.add_subplot(111)
    summary = getattr(result, units)
    wavenumber = result.grid.wavenumber_uniform
    x_range = x_range or result.settings.plot_wavenumber_lim
    exponent = int(round(np.log10(result.settings.absolute_scale)))
    frame = result.fresnel_sample.molecular_frame

    visible = (wavenumber >= x_range[0]) & (wavenumber <= x_range[1])
    # scale to where quartz is strong enough to divide by; elsewhere the ratio
    # is amplified noise and would squash the real spectrum
    strength = result.reference_strength
    trusted = visible & (strength >= result.settings.weak_reference_fraction * strength.max())
    if not trusted.any():
        trusted = visible
    if quantity == "chi2":
        for values, errors, colour, name in ((summary.mean.imag, summary.imag_error, IMAG, "Im"),
                                             (summary.mean.real, summary.real_error, REAL, "Re")):
            axes.fill_between(wavenumber, values - errors, values + errors, color=colour, alpha=0.22, lw=0)
            axes.plot(wavenumber, values, color=colour, lw=2.0, label=name)
        axes.axhline(0, color="0.2", lw=0.8)
        span = np.max(np.abs(np.concatenate([summary.mean.imag[trusted], summary.mean.real[trusted]])))
        axes.set_ylim(-1.15 * span, 1.15 * span)
        axes.set_ylabel(f"$\\chi^{{(2)}}_{{{frame}}}$ ($\\times10^{{-{exponent}}}$ m$^2$V$^{{-1}}$)"
                        if units == "absolute" else "$\\chi^{(2)}/\\chi^{(2)}_{quartz}$")
        axes.legend(frameon=False, loc="best")
    else:
        axes.fill_between(wavenumber, summary.homodyne - summary.homodyne_error,
                          summary.homodyne + summary.homodyne_error, color=IMAG, alpha=0.22, lw=0)
        axes.plot(wavenumber, summary.homodyne, color=IMAG, lw=2.0)
        axes.set_ylim(0, 1.1 * np.max((summary.homodyne + summary.homodyne_error)[trusted]))
        axes.set_ylabel("$|\\chi^{(2)}|^2$")

    axes.set_xlim(*x_range)
    axes.set_xlabel("Frequency (cm$^{-1}$)")
    _journal_axes(axes)
    figure.tight_layout()
    return figure


def _journal_axes(axes) -> None:
    axes.tick_params(direction="in", top=True, right=True, length=5, width=1)
    for spine in axes.spines.values():
        spine.set_linewidth(1.2)


def slices_figure(results: dict, quantity: str = "chi2", units: str = "absolute",
                  strong_only: bool = True, figure: Figure | None = None) -> Figure:
    """Every slice's retrieval in one journal figure, one colour per slice.

    ``quantity`` is "chi2" (Im above Re), "homodyne" (|chi2|^2) or "phase".
    Bands are 95 % intervals across the files of the run, drawn when a slice
    has more than one. With ``strong_only`` each slice is drawn only where its
    own quartz is above the weak-reference fraction, so no curve shows a ratio
    of noise.
    """
    figure = figure or Figure(figsize=(6.0, 4.8), dpi=100)
    figure.clear()
    if not results:
        figure.text(0.5, 0.5, "No slices yet: add them on the Detector tab.", ha="center", va="center")
        return figure
    first = next(iter(results.values()))
    s = first.settings
    wavenumber = first.grid.wavenumber_uniform
    x_range = s.plot_wavenumber_lim
    visible = (wavenumber >= x_range[0]) & (wavenumber <= x_range[1])
    exponent = int(round(np.log10(s.absolute_scale)))
    frame = first.fresnel_sample.molecular_frame
    chi_label = (f"$\\chi^{{(2)}}_{{{frame}}}$ ($\\times10^{{-{exponent}}}$ m$^2$V$^{{-1}}$)"
                 if units == "absolute" else "$\\chi^{(2)}/\\chi^{(2)}_{quartz}$")

    if quantity == "chi2":
        axes_list = figure.subplots(2, 1, sharex=True)
        parts = (("Im", lambda m: m.mean.imag, lambda m: m.imag_error),
                 ("Re", lambda m: m.mean.real, lambda m: m.real_error))
    else:
        axes_list = [figure.subplots(1, 1)]
        if quantity == "homodyne":
            parts = (("", lambda m: m.homodyne, lambda m: m.homodyne_error),)
        else:          # wrapped to (-180, 180]: a phase near 0 would otherwise jump between 0 and 360
            parts = (("", lambda m: (m.phase_deg + 180) % 360 - 180, lambda m: m.phase_error),)

    for axes, (name, value_of, error_of) in zip(axes_list, parts):
        shown = []
        for index, (centre, result) in enumerate(results.items()):
            summary = getattr(result, units)
            values, errors = value_of(summary).astype(float), error_of(summary).astype(float)
            keep = np.ones(len(wavenumber), dtype=bool)
            if strong_only:
                strengths = [result.reference_strength]
                if result.settings.reference_slice != "same":   # a shared reference is strong everywhere
                    strengths.append(np.abs(result.sample.analytic.mean(axis=1)))
                for strength in strengths:
                    keep &= strength >= s.weak_reference_fraction * strength.max()
            values, errors = np.where(keep, values, np.nan), np.where(keep, errors, np.nan)
            if quantity == "phase":          # break the line where it wraps instead of drawing a jump
                values[1:][np.abs(np.diff(values)) > 180] = np.nan
            colour = SLICE_COLOURS[index % len(SLICE_COLOURS)]
            if result.sample.n_scans > 1:
                axes.fill_between(wavenumber, values - errors, values + errors, color=colour, alpha=0.18, lw=0)
            axes.plot(wavenumber, values, color=colour, lw=1.6, label=f"px {centre}")
            shown.append(values[visible])
        data = np.concatenate(shown)
        data = data[np.isfinite(data)]
        if quantity == "chi2":
            span = 1.15 * np.max(np.abs(data)) if data.size else 1.0
            axes.set_ylim(-span, span)
            axes.axhline(0, color="0.2", lw=0.8)
            axes.set_ylabel(f"{name} {chi_label}")
        elif quantity == "homodyne":
            axes.set_ylim(0, 1.1 * np.max(data) if data.size else 1.0)
            axes.set_ylabel("$|\\chi^{(2)}|^2$")
        else:
            axes.set_ylim(-180, 180)
            axes.set_yticks(range(-180, 181, 90))
            axes.axhline(0, color="0.2", lw=0.8)
            axes.set_ylabel("phase (°)")
        axes.set_xlim(*x_range)
        _journal_axes(axes)
    axes_list[0].legend(frameon=False, loc="best", fontsize=9, ncol=2 if len(results) > 4 else 1)
    axes_list[-1].set_xlabel("Frequency (cm$^{-1}$)")
    figure.tight_layout()
    return figure


def export_slices(results: dict, folder: str | Path, units: str = "absolute",
                  strong_only: bool = True) -> list[Path]:
    """Each slice's full export in its own px<row> folder, plus the combined figures."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for centre, result in results.items():
        written += export_results(result, folder / f"px{centre}")
    for quantity, name in (("chi2", "slices"), ("homodyne", "slices_homodyne"), ("phase", "slices_phase")):
        figure = slices_figure(results, quantity, units, strong_only)
        for extension in ("svg", "pdf"):
            path = folder / f"{name}.{extension}"
            figure.savefig(path, transparent=True)
            written.append(path)
    return written


def export_results(result: Result, folder: str | Path) -> list[Path]:
    """Write text columns, settings and publication figures into ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    s = result.settings
    written = []

    def write(name, columns, fmt):
        path = folder / name
        np.savetxt(path, np.column_stack(columns), fmt=fmt, delimiter="\t")
        written.append(path)

    wavenumber = result.grid.wavenumber_uniform
    write("axis_wn.txt", [wavenumber], "%.18e")
    write("sample_time_sec.txt", [result.sample.numbers, result.sample.time_sec], ["%d", "%.3f"])
    for prefix, summary in (("div_fresnel", result.absolute), ("div_filter", result.relative)):
        stem = f"{prefix}_{s.interface_model_sample}_{s.sample_name}"
        write(f"{stem}_ax_and_spec.txt", [wavenumber, summary.mean.real, summary.mean.imag], "%.5f")
        write(f"{stem}_ax_and_spec_err.txt", [wavenumber, summary.real_error, summary.imag_error], "%.5f")
        write(f"{stem}_phase.txt", [wavenumber, summary.phase_deg, summary.phase_error],
              ["%.5f", "%.1f", "%.1f"])
        write(f"{stem}_homo.txt", [wavenumber, summary.homodyne, summary.homodyne_error],
              ["%.5f", "%.5e", "%.5e"])
        if result.sample.rows_as_scans:
            # one real and one imaginary column per slice
            path = folder / f"{stem}_per_slice.txt"
            rows = result.sample.numbers
            header = "\t".join(["wavenumber"] + [f"{part}_px{row}" for row in rows for part in ("re", "im")])
            columns = [wavenumber] + [part for k in range(len(rows))
                                      for part in (summary.per_scan[:, k].real, summary.per_scan[:, k].imag)]
            np.savetxt(path, np.column_stack(columns), fmt="%.5f", delimiter="\t", header=header, comments="")
            written.append(path)

    for quantity, name in (("chi2", "output"), ("homodyne", "output_homodyne")):
        figure = publication_figure(result, quantity)
        for extension in ("svg", "pdf"):
            path = folder / f"{name}.{extension}"
            figure.savefig(path, transparent=True)
            written.append(path)

    record = {"exported": datetime.now().isoformat(timespec="seconds"),
              "software": "HDSFG Studio (engine validated against MATLAB v37b)",
              "settings": asdict(s),
              "reference_files": [p.name for p in result.reference.files],
              "sample_files": [p.name for p in result.sample.files],
              "processed": {"reference": result.reference.processed, "sample": result.sample.processed},
              "rotation_deg": result.rotation_deg,
              "spikes_replaced": {"reference": result.reference.n_spikes_replaced,
                                  "sample": result.sample.n_spikes_replaced},
              "warnings": result.warnings}
    path = folder / "settings.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    written.append(path)
    return written


REFERENCE_GREY = "0.45"


def reference_water(settings):
    """The ag_h2o reference water spectrum (as on the χ⁽²⁾ tab) for the sample's interface model:
    (wavenumber, χ⁽²⁾, 95 % error as complex), absolute units; None if the files are missing."""
    stem = "ag_h2olorentz_ax_and_spec" if settings.interface_model_sample == "lorentz" else "ag_h2oslab_ax_and_spec"
    folder = Path(settings.reference_dir)
    try:
        values = np.loadtxt(folder / f"{stem}.txt")
        errors = np.loadtxt(folder / f"{stem}_err.txt")
    except OSError:
        return None
    return values[:, 0], values[:, 1] + 1j * values[:, 2], errors[:, 1] + 1j * errors[:, 2]


def rows_figure(result, units: str = "absolute", show_fvb: bool = True, figure: Figure | None = None,
                quantity: str = "chi2", strong_only: bool = True, show_reference: bool = True) -> Figure:
    """Row by row in journal style: the binned spectrum with its 95 % band (shaded, across the
    sample files), and the same files with the rows summed first, dashed.

    ``quantity`` is "chi2" (Im and Re), "homodyne" (|χ⁽²⁾|²) or "phase" (wrapped to ±180°).
    With ``strong_only`` nothing is drawn where quartz is below the weak-reference fraction,
    where the ratio is noise ÷ noise. With ``show_reference`` (absolute units only) the ag_h2o
    reference water spectrum is drawn in grey with its 95 % band, as on the χ⁽²⁾ tab.
    """
    figure = figure or Figure(figsize=(5.0, 3.5), dpi=100)
    figure.clear()
    axes = figure.add_subplot(111)
    s = result.settings
    summary = getattr(result, units)
    fvb = getattr(result, f"fvb_{units}")
    wavenumber = result.grid.wavenumber_uniform
    x_range = s.plot_wavenumber_lim
    strength = result.reference_strength
    strong = strength >= s.weak_reference_fraction * strength.max()
    trusted = (wavenumber >= x_range[0]) & (wavenumber <= x_range[1]) & strong

    def blank(values):
        values = np.asarray(values, dtype=float)
        return np.where(strong, values, np.nan) if strong_only else values

    def wrap(phase):
        return (np.asarray(phase) + 180) % 360 - 180

    if quantity == "chi2":
        curves = ((summary.mean.imag, summary.imag_error, fvb.mean.imag, IMAG, "Im"),
                  (summary.mean.real, summary.real_error, fvb.mean.real, REAL, "Re"))
    elif quantity == "homodyne":
        curves = ((summary.homodyne, summary.homodyne_error, fvb.homodyne, IMAG, "$|\\chi^{(2)}|^2$"),)
    else:
        curves = ((wrap(summary.phase_deg), summary.phase_error, wrap(fvb.phase_deg), IMAG, "phase"),)

    shown = []
    reference = reference_water(s) if show_reference and units == "absolute" else None
    if reference is not None:
        ref_x, ref_chi, ref_err = reference
        if quantity == "chi2":
            ref_curves = ((ref_chi.imag, ref_err.imag, "-", "Im, reference water"),
                          (ref_chi.real, ref_err.real, (0, (3, 1.5)), "Re, reference water"))
        elif quantity == "homodyne":
            ref_curves = ((np.abs(ref_chi) ** 2, None, "-", "reference water"),)
        else:
            ref_curves = ((wrap(np.degrees(np.angle(ref_chi))), None, "-", "reference water"),)
        for values, errors, style, label in ref_curves:
            values = np.asarray(values, dtype=float).copy()
            if quantity == "phase":
                values[1:][np.abs(np.diff(values)) > 180] = np.nan
            if errors is not None:
                axes.fill_between(ref_x, values - errors, values + errors, color=REFERENCE_GREY, alpha=0.18, lw=0)
            axes.plot(ref_x, values, color=REFERENCE_GREY, lw=1.3, ls=style, label=label)
            if trusted.any():      # scale the axis by the reference only where the data is usable
                inside = (ref_x >= wavenumber[trusted].min()) & (ref_x <= wavenumber[trusted].max())
                shown.append(values[inside])
    for values, errors, other, colour, name in curves:
        values, errors, other = blank(values), blank(errors), blank(other)
        if quantity == "phase":          # break the line at the wrap instead of drawing a jump
            for line in (values, other):
                line[1:][np.abs(np.diff(line)) > 180] = np.nan
        if result.water.n_scans > 1:
            axes.fill_between(wavenumber, values - errors, values + errors, color=colour, alpha=0.25, lw=0)
        axes.plot(wavenumber, values, color=colour, lw=2.0, label=f"{name}, row by row")
        if show_fvb:
            axes.plot(wavenumber, other, color=colour, lw=1.1, ls="--", label=f"{name}, rows summed first")
        shown.append((values + np.nan_to_num(errors))[trusted])
        shown.append((values - np.nan_to_num(errors))[trusted])
    shown = np.concatenate(shown)
    shown = shown[np.isfinite(shown)]

    exponent = int(round(np.log10(s.absolute_scale)))
    frame = result.fresnel_water.molecular_frame
    chi_label = (f"$\\chi^{{(2)}}_{{{frame}}}$ ($\\times10^{{-{exponent}}}$ m$^2$V$^{{-1}}$)"
                 if units == "absolute" else "$\\chi^{(2)}/\\chi^{(2)}_{quartz}$")
    if quantity == "chi2":
        if shown.size:
            span = np.max(np.abs(shown))
            axes.set_ylim(-1.1 * span, 1.1 * span)
        axes.axhline(0, color="0.2", lw=0.8)
        axes.set_ylabel(chi_label)
    elif quantity == "homodyne":
        axes.set_ylim(0, 1.1 * np.max(shown) if shown.size else 1.0)
        axes.set_ylabel("$|\\chi^{(2)}|^2$" + (f" ($\\times10^{{-{2 * exponent}}}$ m$^4$V$^{{-2}}$)"
                                               if units == "absolute" else ""))
    else:
        axes.set_ylim(-180, 180)
        axes.set_yticks(range(-180, 181, 90))
        axes.axhline(0, color="0.2", lw=0.8)
        axes.set_ylabel("phase (°)")
    axes.set_xlim(*x_range)
    axes.set_xlabel("Frequency (cm$^{-1}$)")
    axes.legend(frameon=False, loc="best", fontsize=8)
    _journal_axes(axes)
    figure.tight_layout()
    return figure


def export_rows(result, folder: str | Path, strong_only: bool = True, show_fvb: bool = True,
                show_reference: bool = True) -> list[Path]:
    """Row by row: the binned spectra (as *_ax_and_spec.txt), the usual way for comparison,
    every row's ratio and weight, the figure, and settings.json."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    s, r = result.settings, result.rows_input
    written = []

    def write(name, columns, fmt, header=""):
        path = folder / name
        np.savetxt(path, np.column_stack(columns), fmt=fmt, delimiter="\t", header=header, comments="")
        written.append(path)

    wavenumber = result.grid.wavenumber_uniform
    write("axis_wn.txt", [wavenumber], "%.18e")
    for method, summaries in (("rows", (result.absolute, result.relative)),
                              ("fvb", (result.fvb_absolute, result.fvb_relative))):
        for prefix, summary in zip(("div_fresnel", "div_filter"), summaries):
            stem = f"{prefix}_{method}_{s.interface_model_sample}_{r.water_material}"
            write(f"{stem}_ax_and_spec.txt", [wavenumber, summary.mean.real, summary.mean.imag], "%.5f")
            write(f"{stem}_ax_and_spec_err.txt", [wavenumber, summary.real_error, summary.imag_error], "%.5f")
            write(f"{stem}_phase.txt", [wavenumber, summary.phase_deg, summary.phase_error],
                  ["%.5f", "%.1f", "%.1f"])
            write(f"{stem}_homo.txt", [wavenumber, summary.homodyne, summary.homodyne_error],
                  ["%.5f", "%.5e", "%.5e"])
    for name, per_row in (("per_row_absolute.txt", result.absolute_rows),
                          ("per_row_relative.txt", result.relative_rows)):
        header = "\t".join(["wavenumber"] + [f"{part}_px{row}" for row in result.rows for part in ("re", "im")])
        columns = [wavenumber] + [part for k in range(len(result.rows))
                                  for part in (per_row[:, k].real, per_row[:, k].imag)]
        write(name, columns, "%.6g", header)
    write("per_row_weight.txt", [wavenumber] + [result.weights[:, k] for k in range(len(result.rows))],
          "%.6g", "\t".join(["wavenumber"] + [f"px{row}" for row in result.rows]))

    for units, suffix in (("absolute", ""), ("relative", "_relative")):
        for quantity, name in (("chi2", "rows"), ("homodyne", "rows_homodyne"), ("phase", "rows_phase")):
            figure = rows_figure(result, units, show_fvb, quantity=quantity, strong_only=strong_only,
                                 show_reference=show_reference)
            for extension in ("svg", "pdf"):
                path = folder / f"{name}{suffix}.{extension}"
                figure.savefig(path, transparent=True)
                written.append(path)

    record = {"exported": datetime.now().isoformat(timespec="seconds"),
              "software": "HDSFG Studio, row by row (quartz row n ÷ water row n, rows binned after)",
              "settings": asdict(s),
              "row_by_row": asdict(r),
              "rows_used": [int(row) for row in result.rows],
              "quartz_files": [p.name for p in result.quartz.files],
              "water_files": [p.name for p in result.water.files],
              "rotation_deg": result.rotation_deg,
              "spikes_replaced": {"quartz": result.spikes_replaced[0], "water": result.spikes_replaced[1]},
              "warnings": result.warnings}
    path = folder / "settings.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    written.append(path)
    return written

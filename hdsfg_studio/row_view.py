"""The "Row by row" view inside the Chip image tab.

Quartz image files and water image files are chosen here, any names, each set
with a number list saying which files to use. Chip row n of water is divided by
chip row n of quartz, every row through the whole retrieval on its own, and the
rows are binned only after the division (``hdsfg_engine.rowwise``). Every other
setting (gates, laser, geometry, Fresnel models, cosmic rays, ...) is the
sidebar's. It does not use the sidebar's runs or chosen files.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSettings, Qt, QTimer, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout,
                               QWidget)

from hdsfg_engine import Settings
from hdsfg_engine.data import read_footer_exposure, sort_files
from hdsfg_engine.rowwise import BINNINGS, RowInput, RowPipeline, RowResult, bright_rows, row_profile
from hdsfg_engine.settings import format_scan_list, parse_scan_list

from .background import Job
from .pages import BLUES, DIVERGING, TURBO, Page, _section, with_alpha
from .parameter_panel import guard_wheel
from .publication import export_rows, rows_figure

MATERIALS = (("h2o", "Water"), ("d2o", "D₂O"), ("au", "Gold"), ("diamond", "Diamond"), ("zqz", "Quartz"))
MAP_PARTS = (("imag", "Im χ⁽²⁾"), ("real", "Re χ⁽²⁾"), ("homodyne", "|χ⁽²⁾|²"), ("phase", "Phase (°)"))


class NumberList(QLineEdit):
    """File numbers as text, "1-15" or "1-5, 8"; empty means all. Red border if it cannot be read."""

    def __init__(self, on_change, placeholder: str = "all files"):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self._numbers: list[int] = []
        self._on_change = on_change
        self.editingFinished.connect(self._read)

    def _read(self) -> None:
        try:
            numbers = parse_scan_list(self.text())
        except ValueError as error:
            self.setStyleSheet("border: 1px solid #C62828;")
            self.setToolTip(str(error))
            return
        self.setStyleSheet("")
        self.setText(format_scan_list(numbers))
        if numbers != self._numbers:
            self._numbers = numbers
            self._on_change()

    def value(self) -> list[int]:
        return list(self._numbers)

    def set_value(self, numbers) -> None:
        self._numbers = sorted(set(int(n) for n in numbers or []))
        self.setStyleSheet("")
        self.setText(format_scan_list(self._numbers))


class RowByRowView(Page):
    """Runs in a background thread, so the window keeps responding: reading 171 image files
    takes ~3 s and the whole chip several more. An edit waits 250 ms for the next one before
    a run starts; edits during a run start one more run, with the latest values, when it ends.
    """
    title = "Row by row"
    gate_moved = Signal(str, tuple)      # ("reference" | "sample", (start, stop)): a gate dragged here

    def __init__(self, theme):
        super().__init__(theme)
        self._settings = Settings()
        self._input = RowInput()
        self._pipeline = RowPipeline()
        self._result: RowResult | None = None
        self._dirty = True
        self._auto_rows_pending = False      # set the rows from the quartz image on the next run
        self._job = Job("Row by row", self._finished, self)      # runs in a worker thread
        self._debounce = QTimer(self, singleShot=True, interval=250)
        self._debounce.timeout.connect(self._run)
        self._store = QSettings("HDSFG", "Studio")
        self._footers: dict[str, tuple] = {}      # path -> (exposure per frame, frames) from the footer

        body = QHBoxLayout()
        body.setSpacing(10)
        self.layout_.addLayout(body, 1)

        # ---- controls ---------------------------------------------------------------
        controls = QWidget()
        controls.setObjectName("chipControls")
        form = QVBoxLayout(controls)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(310)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body.addWidget(scroll)

        self._sets = {}
        for role, title in (("quartz", "Quartz (reference)"), ("water", "Sample")):
            form.addWidget(_section(title))
            choose = QPushButton(f"Choose {role} files…" if role == "quartz" else "Choose sample files…")
            choose.setObjectName("accent")
            choose.setToolTip("Any .asc full-chip image files, several at once. They are numbered 1, 2, ... "
                              "in acquisition order (footer time).")
            choose.clicked.connect(lambda _=False, role=role: self._choose(role))
            form.addWidget(choose, alignment=Qt.AlignLeft)
            files = QLabel("No files")
            files.setObjectName("runInfo")
            files.setWordWrap(True)
            form.addWidget(files)
            row = QHBoxLayout()
            row.addWidget(QLabel("Use files" if role == "water" else "Average files"))
            numbers = NumberList(self._changed)
            numbers.setToolTip("File numbers, e.g. 1-15 or 1-5, 8; empty uses all. "
                               + ("The quartz files are averaged into one quartz image."
                                  if role == "quartz" else
                                  "Each sample file is divided by the averaged quartz; the files give the "
                                  "mean and the 95 % band."))
            row.addWidget(numbers, 1)
            form.addLayout(row)
            row = QHBoxLayout()
            exposure = QDoubleSpinBox(minimum=0.001, maximum=10000, decimals=3, suffix=" s")
            exposure.setKeyboardTracking(False)
            exposure.setToolTip("Exposure of one frame; filled in from the first file's footer")
            frames = QSpinBox(minimum=1, maximum=100000)
            frames.setKeyboardTracking(False)
            frames.setToolTip("Frames (accumulations) per file")
            row.addWidget(QLabel("Exposure per frame"))
            row.addWidget(exposure, 1)
            form.addLayout(row)
            row = QHBoxLayout()
            row.addWidget(QLabel("Frames"))
            row.addWidget(frames, 1)
            form.addLayout(row)
            exposure.valueChanged.connect(self._changed)
            frames.valueChanged.connect(self._changed)
            self._sets[role] = dict(files=files, numbers=numbers, exposure=exposure, frames=frames)
        row = QHBoxLayout()
        row.addWidget(QLabel("Material"))
        self.material = QComboBox()
        for value, label in MATERIALS:
            self.material.addItem(label, value)
        self.material.setToolTip("What the sample files are; sets its Fresnel factors")
        self.material.currentIndexChanged.connect(self._changed)
        row.addWidget(self.material, 1)
        form.addLayout(row)

        form.addWidget(_section("Chip rows"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Rows"))
        self.chip_rows = NumberList(self._changed, "whole chip")
        self.chip_rows.setToolTip("Chip rows (pixel rows) to process, e.g. 140-144 or 139, 141-143, 146. "
                                  "Only these rows are divided and binned, and only these are summed for "
                                  "the dashed comparison. Empty uses the whole chip (about 6 s for 171 "
                                  "files; the dark rows then add a little noise). Click the plot below to add or remove a row.")
        row.addWidget(self.chip_rows, 1)
        form.addLayout(row)
        row = QHBoxLayout()
        auto = QPushButton("Auto")
        auto.setToolTip("Rows where the quartz image is brighter than 10 % of its brightest row")
        auto.clicked.connect(self._auto_rows)
        whole = QPushButton("Whole chip")
        whole.setToolTip("Every chip row (about 6 s for 171 files)")
        whole.clicked.connect(lambda: self._set_rows([]))
        row.addWidget(auto)
        row.addWidget(whole)
        row.addStretch(1)
        form.addLayout(row)
        self.profile = pg.PlotWidget()
        self.profile.setFixedHeight(150)
        self.profile.setBackground(None)
        self.profile.setMenuEnabled(False)
        self.profile.setToolTip("Mean quartz counts of each chip row (all quartz files). Shaded: the rows "
                                "used. Click a row to add or remove it; wheel or drag to zoom.")
        for axis in ("left", "bottom"):
            self.profile.getAxis(axis).setTextPen(theme.plot_foreground)
            self.profile.getAxis(axis).setPen(pg.mkPen(theme.border))
        self.profile.getAxis("left").setWidth(40)
        self.profile.setLabel("bottom", "chip row (px)", color=theme.plot_foreground, size="9pt")
        self.profile.scene().sigMouseClicked.connect(self._profile_clicked)
        form.addWidget(self.profile)
        self._profile_key = None
        self.rows_note = QLabel()
        self.rows_note.setObjectName("runInfo")
        self.rows_note.setWordWrap(True)
        form.addWidget(self.rows_note)

        form.addWidget(_section("Binning, after the division"))
        self.binning = QComboBox()
        for value, label in BINNINGS.items():
            self.binning.addItem(label, value)
        self.binning.setToolTip(
            "How the rows' χ⁽²⁾ are combined into one spectrum, per wavenumber:\n"
            "Weighted: each row counts in proportion to its quartz intensity squared, so rows that are "
            "dark at a wavenumber hardly count there.\n"
            "Strong rows: plain mean of the rows whose quartz is above Display → Weak reference of the "
            "brightest quartz row.\n"
            "Plain mean: every chosen row counts the same.")
        self.binning.currentIndexChanged.connect(self._changed)
        form.addWidget(self.binning)
        note = QLabel("Water row n ÷ quartz row n, each row through the whole retrieval on its own; "
                      "the rows are combined only at the end. Gates, laser, geometry, Fresnel models "
                      "and cosmic rays are the sidebar's.")
        note.setObjectName("runInfo")
        note.setWordWrap(True)
        form.addWidget(note)

        form.addWidget(_section("Show"))
        self.units = QComboBox()
        self.units.addItem("Absolute units (Fresnel)", "absolute")
        self.units.addItem("Relative to quartz", "relative")
        self.units.currentIndexChanged.connect(self._redraw)
        form.addWidget(self.units)
        self.show_fvb = QCheckBox("Rows summed first, for comparison", checked=True)
        self.show_fvb.setToolTip("Dashed: the same files and rows the usual way, rows added before "
                                 "dividing (what Full Vertical Binning gives)")
        self.show_fvb.toggled.connect(self._redraw)
        form.addWidget(self.show_fvb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Map"))
        self.map_part = QComboBox()
        for value, label in MAP_PARTS:
            self.map_part.addItem(label, value)
        self.map_part.currentIndexChanged.connect(self._redraw)
        row.addWidget(self.map_part, 1)
        form.addLayout(row)
        self.strong_only = QCheckBox("Map only where quartz is strong", checked=True)
        self.strong_only.setToolTip("Blank the map where a row's quartz is below Display → Weak reference "
                                    "of the brightest quartz row")
        self.strong_only.toggled.connect(self._redraw)
        form.addWidget(self.strong_only)
        export = QPushButton("Export…")
        export.setToolTip("Binned spectra (*_ax_and_spec.txt), rows summed first, every row's χ⁽²⁾ and "
                          "weight, rows.svg/.pdf and settings.json")
        export.clicked.connect(self._export)
        form.addWidget(export, alignment=Qt.AlignLeft)
        form.addStretch(1)
        for combo in controls.findChildren(QComboBox):      # long item texts must not widen the column
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        guard_wheel(scroll)                                  # scrolling the column must not change numbers

        # ---- plots ------------------------------------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(6)
        bar = QHBoxLayout()
        self.info = QLabel()
        self.info.setObjectName("runInfo")
        self.info.setWordWrap(True)
        bar.addWidget(self.info, 1)
        right.addLayout(bar)
        self.outputs = QTabWidget()
        self.outputs.currentChanged.connect(self._redraw)
        right.addWidget(self.outputs, 1)
        body.addLayout(right, 1)

        plots = QWidget()                     # the working plots (pyqtgraph, zoomable)
        plots_layout = QVBoxLayout(plots)
        plots_layout.setContentsMargins(0, 6, 0, 0)
        zoom_bar = QHBoxLayout()
        zoom_bar.addStretch(1)
        self.add_zoom_controls(zoom_bar)
        plots_layout.addLayout(zoom_bar)
        self.graphics = pg.GraphicsLayoutWidget()
        plots_layout.addWidget(self.graphics, 1)
        self.outputs.addTab(plots, "Plots")

        journal = QWidget()                   # the journal figure, as Export writes it
        journal_layout = QVBoxLayout(journal)
        journal_layout.setContentsMargins(0, 6, 0, 0)
        journal_bar = QHBoxLayout()
        self.figure_quantity = QComboBox()
        self.figure_quantity.addItem("Im and Re χ⁽²⁾", "chi2")
        self.figure_quantity.addItem("|χ⁽²⁾|²", "homodyne")
        self.figure_quantity.addItem("Phase", "phase")
        self.figure_quantity.currentIndexChanged.connect(self._redraw)
        self.figure_strong = QCheckBox("Only where quartz is strong", checked=True)
        self.figure_strong.setToolTip("Leave out wavenumbers where quartz is below Display → Weak reference "
                                      "of its peak: there the ratio is noise ÷ noise. Export uses the same.")
        self.figure_strong.toggled.connect(self._redraw)
        self.figure_reference = QCheckBox("Reference water (ag_h2o)", checked=True)
        self.figure_reference.setToolTip("The ag_h2o reference spectrum, as on the χ⁽²⁾ tab, in grey with "
                                         "its 95 % band (slab or Lorentz, as Fresnel → Sample model). "
                                         "Absolute units only. Export uses the same.")
        self.figure_reference.toggled.connect(self._redraw)
        journal_bar.addWidget(self.figure_quantity)
        journal_bar.addWidget(self.figure_strong)
        journal_bar.addWidget(self.figure_reference)
        note = QLabel("Shaded: 95 % band across the sample files. Dashed: rows summed first (left, Show). "
                      "Export writes this figure as rows.svg / .pdf.")
        note.setObjectName("runInfo")
        note.setWordWrap(True)
        journal_bar.addWidget(note, 1)
        journal_layout.addLayout(journal_bar)
        self.figure = Figure(figsize=(5.0, 3.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedSize(620, 434)
        holder = QHBoxLayout()
        holder.addStretch(1)
        holder.addWidget(self.canvas)
        holder.addStretch(1)
        journal_layout.addLayout(holder)
        journal_layout.addStretch(1)
        self.outputs.addTab(journal, "Publication")

        gates = QWidget()                     # time domain with the gates, and what is divided
        gates_layout = QVBoxLayout(gates)
        gates_layout.setContentsMargins(0, 6, 0, 0)
        gates_bar = QHBoxLayout()
        gates_bar.addWidget(QLabel("Rows"))
        self.gate_rows = QComboBox()
        self.gate_rows.setToolTip("The sum of the chosen rows, or one chip row")
        self.gate_rows.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.gate_rows.currentIndexChanged.connect(self._redraw)
        self.gate_part = QComboBox()
        for value, label in (("real", "Re (fringes)"), ("imag", "Im (fringes)"),
                             ("abs", "|amplitude| (envelope only)"), ("phase", "Phase (°)")):
            self.gate_part.addItem(label, value)
        self.gate_part.currentIndexChanged.connect(self._redraw)
        self.gate_envelope = QCheckBox("Envelope ±|amplitude|", checked=True)
        self.gate_envelope.toggled.connect(self._redraw)
        for widget in (self.gate_rows, self.gate_part, self.gate_envelope):
            gates_bar.addWidget(widget)
        hint = QLabel("Drag a green gate to move it")
        hint.setObjectName("runInfo")
        hint.setToolTip("The gates are the sidebar's (Time gate): dragging one, or its edges, changes them "
                        "there and everything reruns.")
        gates_bar.addWidget(hint)
        gates_bar.addStretch(1)
        reset = QPushButton("Reset zoom")
        reset.clicked.connect(self.reset_zoom)
        gates_bar.addWidget(reset)
        gates_layout.addLayout(gates_bar)
        self.gate_info = QLabel()
        self.gate_info.setObjectName("runInfo")
        self.gate_info.setWordWrap(True)
        gates_layout.addWidget(self.gate_info)
        self.gate_graphics = pg.GraphicsLayoutWidget()
        gates_layout.addWidget(self.gate_graphics, 1)
        self.outputs.addTab(gates, "Time domain and before division")

        self._restore()

    # ---- input ----------------------------------------------------------------------
    def _restore(self) -> None:
        """The files and choices of the last session, if the files are still there."""
        try:
            stored = json.loads(self._store.value("row_by_row", "") or "{}")
            known = set(RowInput.__dataclass_fields__)
            r = RowInput(**{k: v for k, v in stored.items() if k in known})
            if "rows" in stored and "chip_rows" not in stored:      # saved as first, last before
                r.chip_rows = list(range(min(stored["rows"]), max(stored["rows"]) + 1))
            r.quartz_files = [f for f in r.quartz_files if Path(f).is_file()]
            r.water_files = [f for f in r.water_files if Path(f).is_file()]
        except (ValueError, TypeError):
            r = RowInput()
        self._show_input(r)

    def _show_input(self, r: RowInput) -> None:
        self._input = r
        widgets = [self.material, self.chip_rows, self.binning] + \
            [w for role in self._sets.values() for w in (role["numbers"], role["exposure"], role["frames"])]
        for widget in widgets:
            widget.blockSignals(True)
        for role in ("quartz", "water"):
            controls, files = self._sets[role], getattr(r, f"{role}_files")
            controls["numbers"].set_value(getattr(r, f"{role}_numbers"))
            controls["exposure"].setValue(getattr(r, f"{role}_exposure_sec"))
            controls["frames"].setValue(getattr(r, f"{role}_frames"))
            controls["files"].setText(self._describe(files))
        self.material.setCurrentIndex(max(self.material.findData(r.water_material), 0))
        self.chip_rows.set_value(r.chip_rows)
        self.binning.setCurrentIndex(max(self.binning.findData(r.binning), 0))
        for widget in widgets:
            widget.blockSignals(False)

    @staticmethod
    def _describe(files) -> str:
        if not files:
            return "No files"
        names = [Path(f).name for f in files]
        shown = ", ".join(f"{k} {name}" for k, name in enumerate(names[:3], start=1))
        more = f", … {len(names)} {names[-1]}" if len(names) > 3 else ""
        return f"{len(files)} file{'s' if len(files) > 1 else ''} in {Path(files[0]).parent.name}: {shown}{more}"

    def _read_input(self) -> RowInput:
        q, w = self._sets["quartz"], self._sets["water"]
        return replace(self._input,
                       quartz_numbers=q["numbers"].value(), water_numbers=w["numbers"].value(),
                       quartz_exposure_sec=q["exposure"].value(), quartz_frames=q["frames"].value(),
                       water_exposure_sec=w["exposure"].value(), water_frames=w["frames"].value(),
                       water_material=self.material.currentData(),
                       chip_rows=self.chip_rows.value(),
                       binning=self.binning.currentData())

    def _choose(self, role: str) -> None:
        current = getattr(self._input, f"{role}_files")
        start = str(Path(current[0]).parent) if current else self._settings.data_dir or str(Path.home())
        caption = "Quartz image files" if role == "quartz" else "Sample image files"
        paths, _ = QFileDialog.getOpenFileNames(self, caption, start, "Andor spectra (*.asc)")
        if not paths:
            return
        paths = sort_files(paths)
        changes = {f"{role}_files": paths, f"{role}_numbers": []}
        exposure, frames = read_footer_exposure(Path(paths[0]))
        if exposure:
            changes[f"{role}_exposure_sec"] = round(exposure, 3)
        if frames:
            changes[f"{role}_frames"] = frames
        r = replace(self._read_input(), **changes)
        if role == "quartz":
            self._auto_rows_pending = True       # rows from the new quartz image, found in the background
        self._show_input(r)
        self._changed()

    def _auto_rows(self) -> None:
        if self._result is not None and not self._dirty:
            self._set_rows(bright_rows(self._result.quartz))
        elif self._input.quartz_files:
            self._auto_rows_pending = True
            self._changed()

    def _set_rows(self, rows) -> None:
        self._show_input(replace(self._read_input(), chip_rows=sorted(set(int(r) for r in rows))))
        self._changed()

    def _profile_clicked(self, event) -> None:
        """Add or remove the clicked chip row."""
        if event.button() != Qt.LeftButton or self._result is None:
            return
        view = self.profile.getPlotItem().getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        row = int(round(view.mapSceneToView(event.scenePos()).x()))
        n_rows = self._result.quartz.images.shape[2]
        if not 1 <= row <= n_rows:
            return
        current = set(self._input.chip_rows or range(1, n_rows + 1))
        current ^= {row}
        if current:
            self._set_rows(current)

    def _draw_profile(self) -> None:
        """Brightness of each quartz chip row; the rows used shaded."""
        self.profile.clear()
        result = self._result
        if result is None:
            return
        profile = row_profile(result.quartz)
        n_rows = len(profile)
        rows = np.arange(1, n_rows + 1)
        used = np.zeros(n_rows, dtype=bool)
        used[result.rows - 1] = True
        bars = pg.BarGraphItem(x=rows[used], height=profile[used], width=1.0,
                               brush=with_alpha(self.theme.accent, 110), pen=pg.mkPen(None))
        self.profile.addItem(bars)
        self.profile.plot(rows, profile, pen=pg.mkPen(self.theme.text, width=1.2))
        key = (tuple(result.quartz.files), tuple(result.quartz.numbers))
        if key != self._profile_key:             # new quartz: zoom to its bright rows, keep it after that
            self._profile_key = key
            bright = bright_rows(result.quartz) or list(rows)
            self.profile.setXRange(bright[0] - 12, bright[-1] + 12, padding=0)
            self.profile.setYRange(min(0.0, float(profile.min())), 1.1 * float(profile.max()), padding=0)

    def _changed(self, *_) -> None:
        self._input = self._read_input()
        self._store.setValue("row_by_row", json.dumps(asdict(self._input)))
        self._schedule()

    # ---- running ----------------------------------------------------------------------
    def _schedule(self) -> None:
        """Something changed: run once the edits pause (and only while the tab is shown)."""
        self._dirty = True
        if self.isVisible():
            self._debounce.start()

    def set_settings(self, settings: Settings) -> None:
        """The sidebar's settings changed; recompute when shown."""
        if settings is not self._settings:
            self._settings = settings
            self._schedule()

    def update(self, result) -> None:
        self.set_settings(result.settings)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self._debounce.start()

    def _run(self) -> None:
        if not self._dirty or self._job.busy:     # a run going on starts the next one when it ends
            return
        self._dirty = False
        r, s = self._input, self._settings
        if not r.quartz_files or not r.water_files:
            self._result = None
            self.info.setText("Choose the quartz files and the sample files on the left. Chip row n of "
                              "the sample is divided by chip row n of quartz; the rows are binned after "
                              "the division.")
            self._draw()
            return
        auto = self._auto_rows_pending
        self._auto_rows_pending = False
        self.info.setText("⏳ Processing… (reading the files takes a few seconds the first time; "
                          "the window stays usable)")
        self._job.submit(self._work, s, r, auto)

    def _work(self, s: Settings, r: RowInput, auto: bool) -> dict:
        """Worker thread: the pipeline and the footer check only, no widgets."""
        job = {"input": r}
        try:
            if auto:
                job["auto_rows"] = self._pipeline.quartz_bright_rows(s, r)
                r = job["input"] = replace(r, chip_rows=job["auto_rows"])
            result = job["result"] = self._pipeline.run(s, r)
            job["mismatch"] = {
                name: self._footer_mismatch(channel.files, typed)
                for name, channel, typed in (("quartz", result.quartz, (r.quartz_exposure_sec, r.quartz_frames)),
                                             ("sample", result.water, (r.water_exposure_sec, r.water_frames)))}
        except Exception as error:      # shown in the window; the rest of the app keeps working
            traceback.print_exc()
            job["error"] = error
        return job

    def _finished(self, ok: bool, job) -> None:
        if not ok:
            job = {"error": job}
        if "auto_rows" in job:              # show the rows that were found, keep any later edits
            self._show_input(replace(self._read_input(), chip_rows=job["auto_rows"]))
            self._input = self._read_input()
            self._store.setValue("row_by_row", json.dumps(asdict(self._input)))
        if "error" in job:
            self._result = None
            self.info.setText(f"Could not process: {job['error']}")
        else:
            self._result = job["result"]
            self._describe_result(job["input"], job["mismatch"])
        self._draw()
        self._draw_profile()
        if self._dirty:                     # edited while it ran: once more with the latest values
            self._debounce.start()

    def _describe_result(self, r: RowInput, mismatch: dict) -> None:
        result = self._result
        n_rows = result.quartz.images.shape[2]
        low, high = result.usable_range()
        self.rows_note.setText(f"{len(result.rows)} of {n_rows} rows: px {format_scan_list(result.rows)}. "
                               "Auto: rows brighter than 10 % of the brightest quartz row.")
        q, w = result.quartz, result.water
        spikes = (f"cosmic rays: {result.spikes_replaced[0]} + {result.spikes_replaced[1]} pixels replaced"
                  if result.settings.remove_cosmic_rays else "cosmic-ray removal off")
        message = (f"Quartz: {q.n_scans} file{'s' if q.n_scans > 1 else ''} averaged "
                   f"({q.file_exposure_sec[0]:g} s each). "
                   f"Sample: {w.n_scans} file{'s' if w.n_scans > 1 else ''}, each divided and binned, "
                   f"then mean ± 95 %. {BINNINGS[r.binning]}. Usable {low:.0f}–{high:.0f} cm⁻¹; "
                   f"rotation {result.rotation_deg:+.1f}°; {spikes}; {result.seconds * 1000:.0f} ms.")
        warnings = list(result.warnings)
        background = [path.name for path in list(q.files) + list(w.files) if "bg" in path.name.lower()]
        if background:      # names are not read otherwise; this only flags a likely mistake
            warnings.append(f"{', '.join(background)} looks like a background file")
        for name in ("quartz", "sample"):
            if mismatch.get(name):
                typed = ((r.quartz_exposure_sec, r.quartz_frames) if name == "quartz"
                         else (r.water_exposure_sec, r.water_frames))
                warnings.append(f"{name} exposure typed {typed[0]:g} s × {typed[1]}, {mismatch[name]}")
        if warnings:
            message += "  ⚠ " + "; ".join(warnings)
        self.info.setText(message)

    def _footer_mismatch(self, files, typed) -> str:
        """"" if every file's footer agrees with the typed exposure × frames, else what the footers say."""
        for path in files:
            if str(path) not in self._footers:
                self._footers[str(path)] = read_footer_exposure(path)
        found = {self._footers[str(path)] for path in files} - {(None, None)}
        total = typed[0] * typed[1]
        if all(e is not None and f is not None and abs(e * f - total) <= 1e-3 * total for e, f in found):
            return ""
        return "footers say " + ", ".join(f"{e:g} s × {f}" for e, f in sorted(found, key=str))

    # ---- drawing ------------------------------------------------------------------------
    def _redraw(self, *_) -> None:
        self._draw()

    def reset_zoom(self) -> None:
        self._zoom.clear()
        self._draw()

    def _draw(self) -> None:
        if self.outputs.currentIndex() == 2:
            self._draw_gates()
        elif self.outputs.currentIndex() == 1:
            self._draw_figure()
        else:
            self._draw_plots()

    def _fill_gate_rows(self, rows) -> None:
        current = self.gate_rows.currentData()
        self.gate_rows.blockSignals(True)
        self.gate_rows.clear()
        self.gate_rows.addItem(f"Sum of the chosen rows ({len(rows)})", None)
        for row in rows:
            self.gate_rows.addItem(f"px {row}", int(row))
        self.gate_rows.setCurrentIndex(max(self.gate_rows.findData(current), 0))
        self.gate_rows.blockSignals(False)

    def _draw_gates(self) -> None:
        """Left: |FFT| of the resampled spectra with the gates. Right: the gated spectra, fringes kept,
        that go into the division. Quartz above, sample below; the sum of the rows or one row."""
        self.begin_draw()
        self.gate_graphics.clear()
        result = self._result
        if result is None:
            self.end_draw()
            return
        self._fill_gate_rows(result.rows)
        chosen = self.gate_rows.currentData()
        pick = (slice(None) if chosen is None else int(np.flatnonzero(result.rows == chosen)[0]))
        where = "sum of the chosen rows" if chosen is None else f"px {chosen}"

        def one(array):
            return array[:, pick].sum(axis=1) if chosen is None else array[:, pick]

        s = result.settings
        t = result.grid.time_ps
        wavenumber = result.grid.wavenumber_uniform
        part = self.gate_part.currentData()
        names = {"real": "Re", "imag": "Im", "abs": "|amplitude|", "phase": "phase"}
        self.gate_info.setText(
            f"{where.capitalize()}; quartz mean of {result.quartz.n_scans} files, {result.water.name} mean of "
            f"{result.water.n_scans} files. Left: |FFT| of the spectrum on the uniform grid, with the gate "
            "(green). Right: the gated spectrum transformed back, fringes kept: what is divided, row by row. "
            "Grey: quartz too weak over the chosen rows.")
        top_time = top_spectrum = None
        for row, (role, label, n_files, resampled, gated, colour) in enumerate((
                ("reference", "Quartz", result.quartz.n_scans, result.quartz_resampled, result.quartz_rows,
                 self.theme.real),
                ("sample", result.water.name, result.water.n_scans, result.water_resampled, result.water_rows,
                 self.theme.imag))):
            gate = tuple(getattr(s, f"gate_range_{role}_ps"))
            # time domain
            magnitude = np.abs(np.fft.fftshift(np.fft.fft(one(resampled))))
            item = self.plot(self.gate_graphics, row, 0, f"{label}: |FFT|, gate {gate[0]:.2f}–{gate[1]:.2f} ps",
                             "time (ps)", "|FFT|")
            item.setLogMode(y=True)
            item.plot(t, np.maximum(magnitude, magnitude.max() * 1e-7), pen=pg.mkPen(colour, width=1.2))
            region = pg.LinearRegionItem(values=gate, brush=with_alpha(self.theme.ok, 45),
                                         pen=pg.mkPen(self.theme.ok, width=2))
            region.setZValue(-5)
            region.sigRegionChangeFinished.connect(
                lambda r, role=role: self.gate_moved.emit(role, tuple(round(v, 3) for v in r.getRegion())))
            item.addItem(region)
            item.setXRange(t[0], t[-1], padding=0)
            if top_time is not None:
                item.setXLink(top_time)
            top_time = top_time or item
            self.register_plot(("time", role, chosen), item)
            # before division
            spectrum = one(gated)
            amplitude = np.abs(spectrum)
            values = {"real": spectrum.real, "imag": spectrum.imag, "abs": amplitude,
                      "phase": np.degrees(np.angle(spectrum))}[part]
            item = self.plot(self.gate_graphics, row, 1, f"{label}: {names[part]}, gated", "wavenumber (cm⁻¹)",
                             "phase (°)" if part == "phase" else "counts")
            item.plot(wavenumber, values, pen=pg.mkPen(colour, width=1.1))
            if self.gate_envelope.isChecked() and part in ("real", "imag"):
                dashed = pg.mkPen(with_alpha(colour, 150), width=1, style=Qt.DashLine)
                item.plot(wavenumber, amplitude, pen=dashed)
                item.plot(wavenumber, -amplitude, pen=dashed)
            self._shade_weak(item, result)
            item.setXRange(*s.plot_wavenumber_lim, padding=0)
            if top_spectrum is not None:
                item.setXLink(top_spectrum)
            top_spectrum = top_spectrum or item
            self.register_plot(("spectrum", role, chosen, part), item)
        layout = self.gate_graphics.ci.layout
        layout.setColumnStretchFactor(0, 1)
        layout.setColumnStretchFactor(1, 1)
        self.end_draw()

    def _draw_figure(self) -> None:
        if self._result is None:
            self.figure.clear()
        else:
            rows_figure(self._result, self.units.currentData(), self.show_fvb.isChecked(), self.figure,
                        self.figure_quantity.currentData(), self.figure_strong.isChecked(),
                        self.figure_reference.isChecked())
        self.canvas.draw_idle()

    def _draw_plots(self) -> None:
        self.begin_draw()
        self.graphics.clear()
        result = self._result
        if result is None:
            self.end_draw()
            return
        s = result.settings
        units = self.units.currentData()
        summary, fvb = getattr(result, units), getattr(result, f"fvb_{units}")
        wavenumber = result.grid.wavenumber_uniform
        lim = s.plot_wavenumber_lim
        if units == "absolute":
            exponent = int(round(np.log10(s.absolute_scale)))
            y_label = f"χ⁽²⁾ {result.fresnel_water.molecular_frame} (×10⁻{exponent} m²/V)"
        else:
            y_label = "χ⁽²⁾ / χ⁽²⁾ quartz"

        main = self.plot(self.graphics, 0, 0, f"{result.water.name} ÷ quartz, row by row, binned "
                         f"(px {format_scan_list(result.rows)})", "wavenumber (cm⁻¹)", y_label)
        main.addLegend(offset=(-10, 10), labelTextColor=self.theme.text, brush=with_alpha(self.theme.base, 215))
        self._shade_weak(main, result)
        main.addLine(y=0, pen=pg.mkPen(self.theme.border))
        for values, errors, other, colour, name in (
                (summary.mean.imag, summary.imag_error, fvb.mean.imag, self.theme.imag, "Im"),
                (summary.mean.real, summary.real_error, fvb.mean.real, self.theme.real, "Re")):
            if result.water.n_scans > 1:
                upper = main.plot(wavenumber, values + errors, pen=None, connect="finite")
                lower = main.plot(wavenumber, values - errors, pen=None, connect="finite")
                main.addItem(pg.FillBetweenItem(upper, lower, brush=with_alpha(colour, self.theme.band_alpha)))
            main.plot(wavenumber, values, pen=pg.mkPen(colour, width=2), connect="finite", name=name)
            if self.show_fvb.isChecked():
                main.plot(wavenumber, other, pen=pg.mkPen(colour, width=1.2, style=Qt.DashLine),
                          connect="finite", name=f"{name}, rows summed first")
        strength = result.reference_strength
        trusted = ((wavenumber >= lim[0]) & (wavenumber <= lim[1])
                   & (strength >= s.weak_reference_fraction * strength.max()))
        shown = np.concatenate([summary.mean.imag[trusted], summary.mean.real[trusted]])
        shown = shown[np.isfinite(shown)]
        if shown.size:
            span = float(np.max(np.abs(shown))) or 1.0
            main.setYRange(-1.25 * span, 1.25 * span, padding=0)

        # every row's χ⁽²⁾ (what is binned) and each row's share of the bin
        part = self.map_part.currentData()
        per_row = getattr(result, f"{units}_rows")
        values = {"imag": per_row.imag, "real": per_row.real, "homodyne": np.abs(per_row) ** 2,
                  "phase": np.degrees(np.angle(per_row))}[part]
        quartz = np.abs(result.quartz_rows)
        if self.strong_only.isChecked():
            values = np.where(quartz >= s.weak_reference_fraction * quartz.max(), values, np.nan)
        names = dict(MAP_PARTS)
        rows_map = self.plot(self.graphics, 1, 0, f"{names[part]} of each row (mean of the sample files)",
                             "wavenumber (cm⁻¹)", "chip row (px)")
        self._map(rows_map, result, values, part, main, names[part])
        self.register_plot(("rows", units, part, self.strong_only.isChecked()), rows_map)

        with np.errstate(divide="ignore", invalid="ignore"):
            share = result.weights / result.weights.sum(axis=1, keepdims=True)
        share_map = self.plot(self.graphics, 2, 0, "Share of each row in the bin", "wavenumber (cm⁻¹)",
                              "chip row (px)")
        self._map(share_map, result, np.nan_to_num(share), "share", main, "share")
        self.register_plot(("share", result.rows_input.binning), share_map)

        layout = self.graphics.ci.layout
        layout.setRowStretchFactor(0, 5)
        layout.setRowStretchFactor(1, 2)
        layout.setRowStretchFactor(2, 2)
        main.setXRange(*lim, padding=0)              # the maps follow it
        self.register_plot(("main", units), main)
        self.end_draw()

    def _map(self, item, result, values, part, main, label) -> None:
        """Rows up, wavenumber across; x follows the main plot."""
        wavenumber = result.grid.wavenumber_uniform[::-1]          # ascending for the image
        values = values[::-1]
        finite = values[np.isfinite(values)]
        if part in ("imag", "real", "phase"):
            limit = 180.0 if part == "phase" else (float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0)
            levels, colours, blank = (-limit, limit), DIVERGING, 0.0
        else:
            high = float(np.percentile(finite, 99.5)) if finite.size else 1.0
            levels, colours, blank = (0.0, high or 1.0), (BLUES if part == "share" else TURBO), 0.0
        rows = result.rows
        full = np.full((len(wavenumber), rows[-1] - rows[0] + 1), np.nan)    # rows not used stay blank
        full[:, rows - rows[0]] = values
        image = pg.ImageItem(np.nan_to_num(full, nan=blank), levels=levels)
        image.setColorMap(colours)
        step = wavenumber[1] - wavenumber[0]
        image.setRect(QRectF(wavenumber[0] - step / 2, rows[0] - 0.5, step * len(wavenumber), full.shape[1]))
        item.addItem(image)
        item.disableAutoRange()
        item.setXLink(main)
        item.setYRange(rows[0] - 0.5, rows[-1] + 0.5, padding=0)
        bar = pg.ColorBarItem(values=levels, colorMap=colours, interactive=False, label=label)
        bar.setImageItem(image, insert_in=item)

    def _shade_weak(self, item, result) -> None:
        strength = result.reference_strength
        weak = strength < result.settings.weak_reference_fraction * strength.max()
        wavenumber = result.grid.wavenumber_uniform
        edges = np.diff(np.concatenate([[0], weak.astype(int), [0]]))
        for start, stop in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
            band = pg.LinearRegionItem(values=sorted((wavenumber[start], wavenumber[stop])), movable=False,
                                       brush=with_alpha(self.theme.weak_region, 40), pen=pg.mkPen(None))
            band.setZValue(-10)
            item.addItem(band)

    def _export(self) -> None:
        if self._result is None:
            return
        files = self._result.water.files
        default = str(Path(files[0]).parent.parent / "results" / "row_by_row")
        folder = QFileDialog.getExistingDirectory(self, "Export row by row into", default)
        if folder:
            written = export_rows(self._result, folder, self.figure_strong.isChecked(),
                                  self.show_fvb.isChecked(), self.figure_reference.isChecked())
            self.info.setText(f"Exported {len(written)} files to {folder}")

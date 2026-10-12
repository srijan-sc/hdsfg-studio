"""Chosen files: the "Files" sub-tab of the Chip image tab and the window that compares them.

File → Open files… loads any .asc files, whatever their names; each file is one
scan, numbered 1, 2, ... in acquisition order, and the set is used as both
reference and sample. The Files sub-tab lists them, chooses which are used and
which one the detector view shows. "Compare across files…" opens a window that
draws one slice of every file together: raw counts, the gated spectrum, or
χ⁽²⁾ (each file ÷ the mean of the files, as the reference is the same files).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QTableWidget, QTableWidgetItem, QWidget)

from hdsfg_engine import processing
from hdsfg_engine.data import read_end_time
from hdsfg_engine.pipeline import Pipeline, Result

from .background import Job
from .pages import DIVERGING, TURBO, Page, scan_colours, with_alpha


class FilesView(Page):
    """The chosen files as a table: which are used, and which one the detector view shows."""
    title = "Files"
    open_requested = Signal()            # Add files…
    use_changed = Signal(list)           # file numbers to use; [] = all
    remove_requested = Signal(list)      # file numbers to drop from the list
    show_requested = Signal(int)         # show this file on the Detector tab
    compare_requested = Signal()

    def __init__(self, theme):
        super().__init__(theme)
        bar = self.toolbar()
        add = QPushButton("Open files…")
        add.setObjectName("accent")
        add.setToolTip("Choose .asc files, any names (several at once). They replace the current list.")
        add.clicked.connect(self.open_requested)
        remove = QPushButton("Remove unticked")
        remove.setToolTip("Take the unticked files out of the list")
        remove.clicked.connect(lambda: self.remove_requested.emit(
            [n for n, box in self._boxes.items() if not box.isChecked()]))
        everything = QPushButton("Tick all")
        everything.clicked.connect(lambda: self.use_changed.emit([]))
        compare = QPushButton("Compare across files…")
        compare.setObjectName("accent")
        compare.setToolTip("A separate window with one slice of every file drawn together")
        compare.clicked.connect(self.compare_requested)
        for widget in (add, remove, everything, compare):
            bar.addWidget(widget)
        bar.addStretch(1)

        self.info = QLabel()
        self.info.setObjectName("runInfo")
        self.info.setWordWrap(True)
        self.layout_.addWidget(self.info)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Use", "#", "File", "Time", "Data", "Total counts"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.table.horizontalHeader()
        for column in range(6):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(self._double_clicked)
        self.layout_.addWidget(self.table, 1)
        hint = QLabel("Double-click a file to show it on Detector and slices. Unticked files are not "
                      "loaded; Data → Files to use in the sidebar is the same choice as text.")
        hint.setObjectName("runInfo")
        self.layout_.addWidget(hint)
        self._boxes: dict[int, QCheckBox] = {}
        self._times: dict[str, datetime | None] = {}

    def _ticked(self) -> list[int]:
        numbers = [n for n, box in self._boxes.items() if box.isChecked()]
        return [] if len(numbers) == len(self._boxes) else numbers

    def _double_clicked(self, row: int, _column: int) -> None:
        self.show_requested.emit(row + 1)

    def update(self, result: Result) -> None:
        self._last = result
        s = result.settings
        if not s.files:
            self.table.setRowCount(0)
            self._boxes = {}
            self.info.setText("No files chosen. Open files… (also under File, ⇧⌘O) picks any .asc files; "
                              "they are processed as repeats of one sample, each file one scan, and used "
                              "as both reference and sample.")
            return
        used = set(s.sample_file_numbers) or set(range(1, len(s.files) + 1))
        loaded = {int(n): k for k, n in enumerate(result.sample.numbers)}
        raw = result.sample.images if result.sample.images is not None else None
        self.info.setText(
            f"{len(s.files)} files from {Path(s.files[0]).parent}, numbered in acquisition order. "
            f"{len(loaded)} used, each as one scan; exposure {s.file_exposure_sec:g} s × {s.file_frames} "
            "frames (sidebar, Chosen files).")
        self.table.setRowCount(len(s.files))
        self._boxes = {}
        for row, path in enumerate(s.files):
            number = row + 1
            box = QCheckBox()
            box.setChecked(number in used)
            box.toggled.connect(lambda *_: self.use_changed.emit(self._ticked()))
            holder = QWidget()
            layout = QHBoxLayout(holder)
            layout.setContentsMargins(8, 0, 0, 0)
            layout.addWidget(box)
            self.table.setCellWidget(row, 0, holder)
            self._boxes[number] = box
            if path not in self._times:
                self._times[path] = read_end_time(Path(path))
            when = self._times[path]
            cells = [str(number), Path(path).name, when.strftime("%H:%M:%S") if when else "–"]
            if number in loaded and raw is not None:
                image = raw[loaded[number]]
                cells += [f"image {image.shape[1]} rows × {image.shape[0]} px", f"{image.sum():.4g}"]
            elif number in loaded:
                counts = result.sample.raw_counts[:, loaded[number]]
                cells += [f"spectrum {len(counts)} px", f"{counts.sum():.4g}"]
            else:
                cells += ["not used", ""]
            for column, text in enumerate(cells, start=1):
                self.table.setItem(row, column, QTableWidgetItem(text))


QUANTITIES = (
    ("counts", "Counts (slice, as read)"),
    ("real", "Gated spectrum: Re (fringes)"),
    ("imag", "Gated spectrum: Im"),
    ("abs", "Gated spectrum: |amplitude|"),
    ("gphase", "Gated spectrum: phase"),
    ("chi_imag", "χ⁽²⁾: Im"),
    ("chi_real", "χ⁽²⁾: Re"),
    ("chi_homodyne", "χ⁽²⁾: |χ⁽²⁾|²"),
    ("chi_phase", "χ⁽²⁾: phase"),
)
SIGNED = {"real", "imag", "gphase", "chi_imag", "chi_real", "chi_phase"}
PHASES = {"gphase", "chi_phase"}


class CompareWindow(Page):
    """One slice of every file in one plot, as lines or as a map (file against wavenumber)."""
    title = "Compare across files"

    def __init__(self, theme):
        super().__init__(theme)
        self.setWindowTitle("Compare slices across files")
        self.resize(1150, 780)
        bar = self.toolbar()
        self.slice = QComboBox()
        self.slice.setToolTip("Which part of the chip: one slice, the slices added, or the whole chip")
        self.quantity = QComboBox()
        for value, label in QUANTITIES:
            self.quantity.addItem(label, value)
        self.units = QComboBox()
        self.units.addItem("Absolute units (Fresnel)", "absolute")
        self.units.addItem("Relative: file ÷ mean of the files", "relative")
        self.scale = QComboBox()
        self.scale.addItem("As is", "as is")
        self.scale.addItem("Each file ÷ its peak", "peak")
        self.style = QComboBox()
        self.style.addItem("Lines", "lines")
        self.style.addItem("Map (file × wavenumber)", "map")
        for combo in (self.slice, self.quantity, self.units, self.scale, self.style):
            combo.currentIndexChanged.connect(self._redraw)
            bar.addWidget(combo)
        bar.addStretch(1)
        self.add_zoom_controls(bar)

        bar = self.toolbar()
        self.mean = QCheckBox("Show the mean of the ticked files", checked=True)
        self.strong_only = QCheckBox("Only where quartz is strong", checked=True)
        self.strong_only.setToolTip("Hide wavenumbers where the mean of the files is below Display → "
                                    "Weak reference below (gated spectrum and χ⁽²⁾)")
        for box in (self.mean, self.strong_only):
            box.toggled.connect(self._redraw)
            bar.addWidget(box)
        export = QPushButton("Export…")
        export.setToolTip("What is drawn: a text table (one column per file) and an SVG of the plot")
        export.clicked.connect(self._export)
        bar.addWidget(export)
        bar.addStretch(1)

        files_row = QHBoxLayout()
        files_row.addWidget(QLabel("Files"))
        for text, on in (("All", True), ("None", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, on=on: self._tick_all(on))
            files_row.addWidget(button)
        self.file_grid = QGridLayout()
        self.file_grid.setHorizontalSpacing(10)
        files_row.addLayout(self.file_grid)
        files_row.addStretch(1)
        self.layout_.addLayout(files_row)
        self._file_boxes: dict[int, QCheckBox] = {}

        self.info = QLabel()
        self.info.setObjectName("runInfo")
        self.info.setWordWrap(True)
        self.layout_.addWidget(self.info)
        self.graphics = pg.GraphicsLayoutWidget()
        self.layout_.addWidget(self.graphics, 1)
        self._pipeline = Pipeline()          # its own cache: the main run keeps its own
        self._job = Job("Compare across files", self._computed, self)
        self._key = None
        self._part: Result | None = None
        self._table: tuple | None = None     # x, {file: y}, labels, of what is drawn

    # ---- choices -----------------------------------------------------------------
    def _fill_slices(self, result: Result) -> None:
        s = result.settings
        current = self.slice.currentData()
        self.slice.blockSignals(True)
        self.slice.clear()
        if result.sample.is_image:
            for centre in sorted(s.slices):
                window = f" ±{s.slice_half_width}" if s.slice_half_width else ""
                self.slice.addItem(f"Slice px {centre}{window}", centre)
            if len(s.slices) > 1:
                self.slice.addItem("Sum of the slices", "sum")
        self.slice.addItem("Whole chip (all rows, as FVB)", "chip")
        index = self.slice.findData(current)
        self.slice.setCurrentIndex(index if index >= 0 else 0)
        self.slice.blockSignals(False)

    def _fill_files(self, numbers) -> None:
        numbers = [int(n) for n in numbers]
        if numbers == list(self._file_boxes):
            return
        was = {n: box.isChecked() for n, box in self._file_boxes.items()}
        while self.file_grid.count():
            box = self.file_grid.takeAt(0).widget()
            box.hide()
            box.deleteLater()
        self._file_boxes = {}
        colours = scan_colours(len(numbers))
        for index, number in enumerate(numbers):
            box = QCheckBox(f"#{number}", checked=was.get(number, True))
            box.setStyleSheet(f"color: {colours[index].name()};")
            box.toggled.connect(self._redraw)
            self.file_grid.addWidget(box, index // 12, index % 12)
            self._file_boxes[number] = box

    def _tick_all(self, on: bool) -> None:
        for box in self._file_boxes.values():
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
        self._redraw()

    # ---- computing -----------------------------------------------------------------
    def _settings_for(self, s, choice):
        if choice == "chip":
            return replace(s, image_data=False)
        process = "sum" if choice == "sum" else str(choice)
        return replace(s, image_data=True, process_slice=process, reference_slice="same")

    def _values(self, part: Result, quantity: str, units: str):
        """x, values [n_x x n_files], y label."""
        sample = part.sample
        if quantity == "counts":
            x = processing.make_grid(sample.wavelength_nm, part.settings.laser_wavelength_nm,
                                     part.settings.sfg_or_dfg).wavenumber
            label = "mean counts" if part.settings.image_data and part.settings.slice_combine == "mean" \
                else "counts"
            return x, sample.raw_counts.astype(float), label
        x = part.grid.wavenumber_uniform
        if quantity in ("real", "imag", "abs", "gphase"):
            a = sample.analytic
            values = {"real": a.real, "imag": a.imag, "abs": np.abs(a),
                      "gphase": np.degrees(np.angle(a))}[quantity]
            return x, values, "phase (°)" if quantity == "gphase" else "counts"
        summary = part.absolute if units == "absolute" else part.relative
        chi = summary.per_scan
        values = {"chi_imag": chi.imag, "chi_real": chi.real,
                  "chi_homodyne": summary.homodyne_per_scan,
                  "chi_phase": np.degrees(np.angle(chi))}[quantity]
        unit = "×10⁻²² m²/V" if units == "absolute" else "÷ mean of the files"
        label = {"chi_imag": f"Im χ⁽²⁾ ({unit})", "chi_real": f"Re χ⁽²⁾ ({unit})",
                 "chi_homodyne": "|χ⁽²⁾|²", "chi_phase": "phase (°)"}[quantity]
        return x, values, label

    def update(self, result: Result) -> None:
        self._last = result
        if not self.isVisible():
            return
        s = result.settings
        self._fill_slices(result)
        self.units.setEnabled(self.quantity.currentData().startswith("chi"))
        choice = self.slice.currentData()
        key = (repr(s), choice)
        if key != self._key:
            self._key = key                 # computed in the background, drawn when done
            self.info.setText("⏳ Processing…")
            self._job.submit(self._pipeline.run, self._settings_for(s, choice))
            return
        self._render(result)

    def _computed(self, ok, value) -> None:
        self._part = value if ok else None
        if not ok:                          # show it here; the main window keeps working
            self.info.setText(f"Could not process: {value}")
        if self._last is not None and self.isVisible():
            self._render(self._last)

    def _render(self, result: Result) -> None:
        s = result.settings
        self.begin_draw()
        self.graphics.clear()
        self._table = None
        if self._part is None:
            self.end_draw()
            return
        part = self._part
        numbers = [int(n) for n in part.sample.numbers]
        self._fill_files(numbers)
        ticked = [k for k, n in enumerate(numbers) if self._file_boxes[n].isChecked()]
        quantity, units = self.quantity.currentData(), self.units.currentData()
        x, values, y_label = self._values(part, quantity, units)
        if quantity != "counts" and self.strong_only.isChecked():
            strength = part.reference_strength
            values = np.where((strength < s.weak_reference_fraction * strength.max())[:, None],
                              np.nan, values)
        if self.scale.currentData() == "peak" and quantity not in PHASES:
            with np.errstate(invalid="ignore", divide="ignore"):
                values = values / np.nanmax(np.abs(values), axis=0)
            y_label = "÷ peak"

        what = self.slice.currentText().lower()
        files = f"{len(ticked)} of {len(numbers)} files"
        source = {"counts": "counts as read (before cosmic-ray removal)",
                  "chi": "each file's χ⁽²⁾ (file ÷ mean of the used files, the reference being the same "
                         "files)",
                  "gated": "each file's spectrum after the time gate and the back-transform"}
        kind = "counts" if quantity == "counts" else "chi" if quantity.startswith("chi") else "gated"
        self.info.setText(f"{what}: {source[kind]}; {files}. Every other setting is the sidebar's.")

        labels = [f"#{numbers[k]}" for k in ticked]
        shown = values[:, ticked]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # columns masked out entirely
            mean = np.nanmean(shown, axis=1) if ticked else None
        self._table = (x, shown, labels, mean, y_label)
        title = f"{self.quantity.currentText()}, {what}"
        if self.style.currentData() == "map":
            self._draw_map(x, shown, ticked, numbers, title, y_label, quantity)
        else:
            self._draw_lines(x, shown, ticked, numbers, mean, title, y_label)
        self.end_draw()

    def _draw_lines(self, x, shown, ticked, numbers, mean, title, y_label) -> None:
        item = self.plot(self.graphics, 0, 0, title, "IR wavenumber (cm⁻¹)", y_label)
        item.addLegend(offset=(-5, 5), labelTextColor=self.theme.text, brush=with_alpha(self.theme.base, 215))
        colours = scan_colours(len(numbers))
        for column, k in enumerate(ticked):
            item.plot(x, shown[:, column], pen=pg.mkPen(colours[k], width=1.0), connect="finite",
                      name=f"#{numbers[k]}" if len(ticked) <= 16 else None)
        if mean is not None and self.mean.isChecked() and len(ticked) > 1:
            item.plot(x, mean, pen=pg.mkPen(self.theme.text, width=1.8, style=Qt.DashLine), connect="finite",
                      name="mean")
        item.setXRange(*self._last.settings.plot_wavenumber_lim, padding=0)
        self.register_plot(("lines", self.quantity.currentData(), self.scale.currentData()), item)

    def _draw_map(self, x, shown, ticked, numbers, title, y_label, quantity) -> None:
        item = self.plot(self.graphics, 0, 0, title, "IR wavenumber (cm⁻¹)", "file")
        if not ticked:
            self.register_plot(("map",), item)
            return
        order = np.argsort(x)
        grid = np.linspace(x.min(), x.max(), len(x))
        uniform = np.column_stack([np.interp(grid, x[order], shown[order, c]) for c in range(len(ticked))])
        finite = uniform[np.isfinite(uniform)]
        if quantity in SIGNED and finite.size:
            limit = 180.0 if quantity in PHASES else float(np.percentile(np.abs(finite), 99.5)) or 1.0
            levels, colours = (-limit, limit), DIVERGING
        else:
            levels = (float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5))) \
                if finite.size else (0, 1)
            colours = TURBO
        image = pg.ImageItem(np.nan_to_num(uniform, nan=levels[0] if colours is TURBO else 0.0),
                             levels=levels)
        image.setColorMap(colours)
        step = grid[1] - grid[0]
        image.setRect(QRectF(grid[0] - step / 2, 0.5, step * len(grid), len(ticked)))
        item.addItem(image)
        item.getAxis("left").setTicks([[(row + 1, f"#{numbers[k]}") for row, k in enumerate(ticked)]])
        item.setXRange(*self._last.settings.plot_wavenumber_lim, padding=0)
        item.setYRange(0.5, len(ticked) + 0.5, padding=0)
        bar = pg.ColorBarItem(values=levels, colorMap=colours, interactive=False, label=y_label)
        bar.setImageItem(image, insert_in=item)
        self.register_plot(("map", self.quantity.currentData(), self.scale.currentData()), item)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._last is not None:
            self.update(self._last)

    def _export(self) -> None:
        if self._table is None:
            return
        x, shown, labels, mean, y_label = self._table
        default = str(Path(self._last.settings.files[0]).parent / "compare_files.txt") \
            if self._last.settings.files else "compare_files.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Export comparison", default, "Text (*.txt)")
        if not path:
            return
        columns = [x] + [shown[:, c] for c in range(shown.shape[1])] + ([mean] if mean is not None else [])
        header = (f"{self.quantity.currentText()}, {self.slice.currentText()} ({y_label})\n"
                  + "\t".join(["wavenumber_cm-1"] + labels + (["mean"] if mean is not None else [])))
        np.savetxt(path, np.column_stack(columns), delimiter="\t", header=header, fmt="%.8g")
        from pyqtgraph.exporters import SVGExporter
        SVGExporter(self.graphics.scene()).export(str(Path(path).with_suffix(".svg")))
        self.info.setText(f"Exported {path} and {Path(path).with_suffix('.svg').name}")

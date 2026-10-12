"""The plot tabs. Each page draws one stage of a pipeline Result.

Pages are told the theme once and then receive ``update(result)`` after every
run. Plots are pyqtgraph: mouse wheel zooms, drag pans, right-click resets,
and the time-domain page lets the gate edges be dragged directly.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QGraphicsOpacityEffect, QGridLayout,
                               QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox, QTabWidget,
                               QVBoxLayout, QWidget)

from hdsfg_engine import processing
from hdsfg_engine.data import slice_rows
from hdsfg_engine.drift import phase_drift
from hdsfg_engine.homodyne import homodyne_per_scan
from hdsfg_engine.pipeline import Pipeline, Result, retrieve_slices

from .background import Job
from .publication import SLICE_COLOURS
from .slice_view import SliceChi2View
from .theme import Theme

TURBO = pg.colormap.getFromMatplotlib("turbo")
DIVERGING = pg.colormap.getFromMatplotlib("RdBu_r")
# detector image: background -> blue -> near white, as the chip viewer the lab uses
CHIP_LIGHT = pg.ColorMap(np.linspace(0, 1, 4), np.array([[255, 255, 255], [120, 170, 225],
                                                          [25, 85, 180], [8, 30, 80]], dtype=np.ubyte))
CHIP_DARK = pg.ColorMap(np.linspace(0, 1, 4), np.array([[12, 22, 48], [20, 70, 160],
                                                         [90, 160, 240], [240, 248, 255]], dtype=np.ubyte))
# light blue -> dark navy, for a time axis; starts above white so early lines show
BLUES = pg.ColorMap(np.linspace(0, 1, 4), np.array([[158, 201, 230], [66, 145, 199],
                                                    [8, 82, 156], [8, 33, 89]], dtype=np.ubyte))


def scan_colours(n: int) -> list[QColor]:
    """One colour per scan, blue (first) to red (last)."""
    positions = np.linspace(0.08, 0.92, max(n, 1))
    return [TURBO.map(p, mode="qcolor") for p in positions]


def with_alpha(colour: str, alpha: int) -> QColor:
    c = QColor(colour)
    c.setAlpha(alpha)
    return c


class Page(QWidget):
    title = ""

    def __init__(self, theme: Theme):
        super().__init__()
        self.theme = theme
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(10, 8, 10, 8)
        self.layout_.setSpacing(6)
        self._last: Result | None = None
        self._plots: dict = {}          # key -> PlotItem, as registered by the page
        self._zoom: dict = {}           # key -> [x range, y range] the user zoomed to
        self._drawing = False
        self.zoom_box: QPushButton | None = None

    def toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.layout_.addLayout(row)
        return row

    # ---- zoom, shared by every plot page ----------------------------------------
    # A page calls begin_draw() before rebuilding its plots, register_plot() for
    # each one once its default ranges are set, and end_draw() at the end. A zoom
    # the user makes is stored under the plot's key and put back on every redraw;
    # keys include whatever changes the axis units, so a zoom never lands on the
    # wrong axis.
    def add_zoom_controls(self, bar: QHBoxLayout) -> None:
        self.zoom_box = QPushButton("Zoom box", checkable=True)
        self.zoom_box.setToolTip("On: drag a rectangle to zoom into it.  Off: drag pans.\n"
                                 "The mouse wheel zooms either way.")
        self.zoom_box.toggled.connect(self._apply_mouse_mode)
        reset = QPushButton("Reset zoom")
        reset.setToolTip("Back to the full view")
        reset.clicked.connect(self.reset_zoom)
        bar.addWidget(self.zoom_box)
        bar.addWidget(reset)

    def begin_draw(self) -> None:
        self._drawing = True
        self._plots = {}

    def register_plot(self, key, item: pg.PlotItem) -> None:
        self._plots[key] = item
        view = item.getViewBox()
        if key in self._zoom:
            x_range, y_range = self._zoom[key]
            view.setRange(xRange=x_range, yRange=y_range, padding=0)
        view.sigRangeChangedManually.connect(lambda *_, key=key: self._remember_zoom(key))

    def end_draw(self) -> None:
        self._apply_mouse_mode()
        self._drawing = False

    def _remember_zoom(self, key) -> None:
        if not self._drawing and key in self._plots:
            self._zoom[key] = self._plots[key].getViewBox().viewRange()

    def _apply_mouse_mode(self, *_) -> None:
        rect = self.zoom_box is not None and self.zoom_box.isChecked()
        for item in self._plots.values():
            item.getViewBox().setMouseMode(pg.ViewBox.RectMode if rect else pg.ViewBox.PanMode)

    def reset_zoom(self) -> None:
        self._zoom.clear()
        if self._last is not None:
            self.update(self._last)

    def _redraw(self, *_) -> None:
        if self._last is not None:
            self.update(self._last)

    def plot(self, graphics: pg.GraphicsLayoutWidget, row: int, col: int, title: str,
             x_label: str, y_label: str, **kwargs) -> pg.PlotItem:
        item = graphics.addPlot(row=row, col=col, **kwargs)
        item.setTitle(title, size="10pt", color=self.theme.text)
        item.showGrid(x=True, y=True, alpha=0.12)
        for axis in ("left", "bottom"):
            item.getAxis(axis).setTextPen(self.theme.plot_foreground)
            item.getAxis(axis).setPen(pg.mkPen(self.theme.border))
        item.getAxis("left").setWidth(72)       # room for the label beside wide tick numbers
        # titles last: setTextPen resets the axis title's style
        label_style = {"color": self.theme.plot_foreground, "font-size": "10pt"}
        item.setLabel("bottom", x_label, **label_style)
        item.setLabel("left", y_label, **label_style)
        return item

    def shade_weak(self, item: pg.PlotItem, result: Result) -> None:
        """Grey bands where quartz is below the weak-reference fraction."""
        strength = result.reference_strength
        weak = strength < result.settings.weak_reference_fraction * strength.max()
        wavenumber = result.grid.wavenumber_uniform
        edges = np.diff(np.concatenate([[0], weak.astype(int), [0]]))
        for start, stop in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1):
            band = pg.LinearRegionItem(values=sorted((wavenumber[start], wavenumber[stop])),
                                       movable=False,
                                       brush=with_alpha(self.theme.weak_region, 40),
                                       pen=pg.mkPen(None))
            band.setZValue(-10)
            item.addItem(band)

    def update(self, result: Result) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# ----------------------------------------------------------------------------------
MAX_SLICES = len(SLICE_COLOURS)


class Segmented(QWidget):
    """Two or more buttons of which exactly one is pressed."""
    changed = Signal(object)

    def __init__(self, options):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._values = []
        for index, (value, label) in enumerate(options):
            button = QPushButton(label, checkable=True)
            button.setObjectName("segment")
            self._group.addButton(button, index)
            self._values.append(value)
            layout.addWidget(button)
        layout.addStretch(1)
        self._group.button(0).setChecked(True)
        self._group.idClicked.connect(lambda index: self.changed.emit(self._values[index]))

    def value(self):
        return self._values[self._group.checkedId()]

    def set_value(self, value) -> None:
        if value in self._values:
            self._group.button(self._values.index(value)).setChecked(True)


def _section(title: str) -> QLabel:
    label = QLabel(title.upper())
    label.setObjectName("chipSection")
    return label


class SliceFringesView(Page):
    """Inside the Chip image tab: each slice after gate and back-transform, fringes kept.

    The Before division tab draws |gated spectrum|, the envelope. Here the real
    (or imaginary) part is drawn, so the fringes the local oscillator delay puts
    on the spectrum stay visible. Every slice is processed on its own (as on the
    χ⁽²⁾ per slice tab); each line is the mean of the run's files.

    "One slice ÷ another" divides two of these complex spectra, slice A ÷ slice
    B of the same run, as they are: no Fresnel factors, exposure ratio or phase
    rotation.
    """
    title = "Before division per slice"

    def __init__(self, theme):
        super().__init__(theme)
        bar = self.toolbar()
        self.mode = QComboBox()
        self.mode.addItem("Each slice", "each")
        self.mode.addItem("One slice ÷ another", "ratio")
        self.part = QComboBox()
        self.part.addItem("Re (fringes)", "real")
        self.part.addItem("Im (fringes)", "imag")
        self.part.addItem("|amplitude| (envelope only)", "abs")
        self.part.addItem("Phase (°)", "phase")
        self.scale = QComboBox()
        self.scale.addItem("Counts", "counts")
        self.scale.addItem("Each slice ÷ its peak amplitude", "peak")
        self.envelope = QCheckBox("Show envelope ±|amplitude|", checked=True)
        for widget in (self.mode, self.part, self.scale):
            widget.currentIndexChanged.connect(self._redraw)
            bar.addWidget(widget)
        self.envelope.toggled.connect(self._redraw)
        bar.addWidget(self.envelope)
        bar.addStretch(1)
        self.add_zoom_controls(bar)

        self.ratio_bar = QWidget()
        rows = QVBoxLayout(self.ratio_bar)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(4)
        row = QHBoxLayout()
        rows.addLayout(row)
        self.numerator, self.denominator = QComboBox(), QComboBox()
        row.addWidget(QLabel("Divide"))
        row.addWidget(self.numerator)
        row.addWidget(QLabel("÷"))
        row.addWidget(self.denominator)
        add = QPushButton("Add")
        add.setObjectName("accent")
        add.setToolTip("Keep this division on the plot and pick another")
        add.clicked.connect(lambda: self._add_pairs([(self.numerator.currentData(),
                                                      self.denominator.currentData())]))
        row.addWidget(add)
        every = QPushButton("Every slice ÷ this")
        every.setToolTip("Add every other slice divided by the slice chosen after ÷")
        every.clicked.connect(lambda: self._add_pairs([(c, self.denominator.currentData())
                                                       for c in self._results]))
        row.addWidget(every)
        clear = QPushButton("Remove all")
        clear.clicked.connect(lambda: (self._pairs.clear(), self._redraw()))
        row.addWidget(clear)
        self.strong_only = QCheckBox("Only where both slices are strong", checked=True)
        self.strong_only.setToolTip("Hide wavenumbers where either slice is below Display → Weak "
                                    "reference of its own peak: there the ratio is noise ÷ noise")
        row.addWidget(self.strong_only)
        row.addStretch(1)
        self.pair_chips = QHBoxLayout()
        self.pair_chips.setSpacing(6)
        rows.addLayout(self.pair_chips)
        for combo in (self.numerator, self.denominator):
            combo.currentIndexChanged.connect(self._redraw)
        self.strong_only.toggled.connect(self._redraw)
        self.layout_.addWidget(self.ratio_bar)
        self._pairs: list[tuple[int, int]] = []      # divisions kept on the plot, (A, B)

        self.info = QLabel()
        self.info.setObjectName("runInfo")
        self.info.setWordWrap(True)
        self.layout_.addWidget(self.info)
        self.graphics = pg.GraphicsLayoutWidget()
        self.layout_.addWidget(self.graphics, 1)
        self._pipeline = Pipeline()          # its own cache: the main run keeps its own
        self._results: dict = {}
        self._key = None
        self._job = Job("Before division per slice", self._computed, self)

    def _computed(self, ok, value) -> None:
        self._results = value if ok else {}
        if not ok:                          # show it here; the rest of the app keeps working
            self.info.setText(f"Could not process the slices: {value}")
        if self._last is not None:
            self._render(self._last)

    def _fill_pickers(self, centres) -> None:
        """Slice choices for A and B; keep the current ones if they are still slices."""
        for combo, fallback in ((self.numerator, 0), (self.denominator, 1)):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for centre in centres:
                combo.addItem(f"px {centre}", centre)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else min(fallback, len(centres) - 1))
            combo.blockSignals(False)

    def _add_pairs(self, pairs) -> None:
        for a, b in pairs:
            if a is not None and b is not None and a != b and (a, b) not in self._pairs \
                    and len(self._pairs) < MAX_SLICES:
                self._pairs.append((a, b))
        self._redraw()

    def _remove_pair(self, pair) -> None:
        self._pairs.remove(pair)
        self._redraw()

    def _rebuild_pair_chips(self) -> None:
        while self.pair_chips.count():
            item = self.pair_chips.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for index, (a, b) in enumerate(self._pairs):
            chip = QPushButton(f"●  px {a} ÷ px {b}   ✕")
            chip.setObjectName("sliceChip")
            chip.setStyleSheet(f"QPushButton#sliceChip {{ color: {SLICE_COLOURS[index % MAX_SLICES]}; }}")
            chip.setToolTip("Remove this division")
            chip.clicked.connect(lambda _=False, pair=(a, b): self._remove_pair(pair))
            self.pair_chips.addWidget(chip)
        self.pair_chips.addStretch(1)

    def update(self, result: Result) -> None:
        self._last = result
        s = result.settings
        if repr(s) != self._key:
            self._key = repr(s)
            if s.image_data:                # every slice retrieved in the background
                self.info.setText("⏳ Processing the slices…")
                self._job.submit(retrieve_slices, s, self._pipeline, "same")
                return
            self._results = {}
        self._render(result)

    def _render(self, result: Result) -> None:
        s = result.settings
        ratio = self.mode.currentData() == "ratio"
        self.ratio_bar.setVisible(ratio)
        self.scale.setEnabled(not ratio)
        self._fill_pickers(list(self._results))
        self._pairs = [(a, b) for a, b in self._pairs if a in self._results and b in self._results]
        self._rebuild_pair_chips()

        self.begin_draw()
        self.graphics.clear()
        if not self._results:
            if s.image_data:
                self.info.setText("No slices yet: add them on the Detector tab.")
            self.end_draw()
            return
        first = next(iter(self._results.values()))
        files = f"mean of {first.sample.n_scans} file{'s' if first.sample.n_scans > 1 else ''} per slice"
        if ratio:
            pairs = self._pairs or [(self.numerator.currentData(), self.denominator.currentData())]
            shown = ("the divisions added below" if self._pairs
                     else "the one chosen above; Add keeps it and lets you add more")
            self.info.setText(
                f"Showing {shown}. Each is the gated spectrum of one slice divided by another's, in the "
                f"same run ({files}). Nothing else is applied: no Fresnel factors, exposure ratio or "
                "phase rotation. Fringes common to both slices cancel; fringes left over mean the two "
                "rows do not share the same spectral phase.")
            self._draw_ratio(result, pairs)
        else:
            self.info.setText(
                f"{len(self._results)} slices, each gated and transformed back on its own "
                f"(gates as on the Time domain tab); {files}. This is what is divided, "
                "before division; Before division draws only its envelope.")
            self._draw_each(result)
        self.end_draw()

    def _values(self, spectrum: np.ndarray, part: str) -> np.ndarray:
        return {"real": spectrum.real, "imag": spectrum.imag, "abs": np.abs(spectrum),
                "phase": np.degrees(np.angle(spectrum))}[part]

    def _finish(self, item: pg.PlotItem, top, key) -> pg.PlotItem:
        item.setXRange(*self._last.settings.plot_wavenumber_lim, padding=0)
        if top is not None:
            item.setXLink(top)
        self.register_plot(key, item)
        return top or item

    def _draw_each(self, result: Result) -> None:
        part, per_peak = self.part.currentData(), self.scale.currentData() == "peak"
        names = {"real": "Re", "imag": "Im", "abs": "|amplitude|", "phase": "phase"}
        first = next(iter(self._results.values()))
        top = None
        for row, role in enumerate(("reference", "sample")):
            name = getattr(first, role).name
            y_label = "phase (°)" if part == "phase" else "÷ peak" if per_peak else "counts"
            item = self.plot(self.graphics, row, 0,
                             f"{role.capitalize()} run ({name}): {names[part]} of the gated spectrum, "
                             "per slice", "wavenumber (cm⁻¹)", y_label)
            item.addLegend(offset=(-5, 5), labelTextColor=self.theme.text,
                           brush=with_alpha(self.theme.base, 215))
            for index, (centre, slice_result) in enumerate(self._results.items()):
                colour = SLICE_COLOURS[index % MAX_SLICES]
                analytic = getattr(slice_result, role).analytic.mean(axis=1)
                amplitude = np.abs(analytic)
                factor = 1 / (amplitude.max() or 1) if per_peak and part != "phase" else 1
                wavenumber = slice_result.grid.wavenumber_uniform
                item.plot(wavenumber, self._values(analytic, part) * factor,
                          pen=pg.mkPen(colour, width=1.1), name=f"px {centre}")
                if self.envelope.isChecked() and part in ("real", "imag"):
                    dashed = pg.mkPen(with_alpha(colour, 150), width=1, style=Qt.DashLine)
                    item.plot(wavenumber, amplitude * factor, pen=dashed)
                    item.plot(wavenumber, -amplitude * factor, pen=dashed)
            top = self._finish(item, top, ("each", role, part, per_peak))

    def _draw_ratio(self, result: Result, pairs) -> None:
        part = self.part.currentData()
        names = {"real": "Re", "imag": "Im", "abs": "|ratio|", "phase": "phase of the ratio"}
        fraction = result.settings.weak_reference_fraction
        wavenumber = next(iter(self._results.values())).grid.wavenumber_uniform
        top = None
        for row, role in enumerate(("reference", "sample")):
            name = getattr(next(iter(self._results.values())), role).name
            title = (f"px {pairs[0][0]} ÷ px {pairs[0][1]}" if len(pairs) == 1
                     else f"{len(pairs)} divisions")
            item = self.plot(self.graphics, row, 0,
                             f"{role.capitalize()} run ({name}): {names[part]}, {title}",
                             "wavenumber (cm⁻¹)", "phase (°)" if part == "phase" else "ratio")
            item.addLegend(offset=(-5, 5), labelTextColor=self.theme.text,
                           brush=with_alpha(self.theme.base, 215))
            item.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(self.theme.border)))
            for index, (a, b) in enumerate(pairs):
                colour = SLICE_COLOURS[index % MAX_SLICES]
                upper = getattr(self._results[a], role).analytic.mean(axis=1)
                lower = getattr(self._results[b], role).analytic.mean(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    quotient = upper / lower
                if self.strong_only.isChecked():
                    weak = ((np.abs(upper) < fraction * np.abs(upper).max())
                            | (np.abs(lower) < fraction * np.abs(lower).max()))
                    quotient = np.where(weak, np.nan, quotient)
                item.plot(wavenumber, self._values(quotient, part), pen=pg.mkPen(colour, width=1.3),
                          connect="finite", name=f"px {a} ÷ px {b}")
                if self.envelope.isChecked() and part in ("real", "imag"):
                    dashed = pg.mkPen(with_alpha(colour, 150), width=1, style=Qt.DashLine)
                    item.plot(wavenumber, np.abs(quotient), pen=dashed, connect="finite")
                    item.plot(wavenumber, -np.abs(quotient), pen=dashed, connect="finite")
            top = self._finish(item, top, ("ratio", role, part))


class ChipImagePage(Page):
    """Image runs: the whole chip, its row profile, and the slices cut out of it.

    Slices, their averaging window and Mean/Sum are settings (they decide what
    the processing gets); the channel shown, the x axis and the sum line are
    only views. Everything is greyed out while Data → Image data is off.
    """
    title = "Chip image"
    slices_changed = Signal(dict)        # Settings fields to replace

    def __init__(self, theme):
        super().__init__(theme)
        self._result: Result | None = None
        self._heat: pg.PlotItem | None = None
        self._profile_rows: np.ndarray | None = None

        self.banner = QLabel("Image data is off. Turn on Data → Image data in the sidebar to "
                             "choose slices; until then each image counts as the sum of all its rows.")
        self.banner.setObjectName("chipBanner")
        self.banner.setWordWrap(True)
        self.layout_.addWidget(self.banner)
        self.body = QWidget()
        body = QHBoxLayout(self.body)
        body.setContentsMargins(0, 6, 0, 0)
        body.setSpacing(10)
        self.slice_view = SliceChi2View()
        self.fringe_view = SliceFringesView(theme)
        self.views = QTabWidget()
        self.views.addTab(self.body, "Detector and slices")
        self.views.addTab(self.fringe_view, "Before division per slice")
        self.views.addTab(self.slice_view, "χ⁽²⁾ per slice")
        self.views.currentChanged.connect(self._view_changed)
        self.layout_.addWidget(self.views, 1)
        self._always_on: list[QWidget] = []
        self._greyed = {}                    # one fixed effect per view, switched on while Image data is off
        for view in (self.body, self.fringe_view, self.slice_view):
            effect = QGraphicsOpacityEffect(view)
            effect.setOpacity(0.4)
            effect.setEnabled(False)
            view.setGraphicsEffect(effect)
            self._greyed[view] = effect

        # ---- controls -------------------------------------------------------------
        controls = QWidget()
        controls.setObjectName("chipControls")
        form = QVBoxLayout(controls)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(290)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body.addWidget(scroll)

        form.addWidget(_section("Data"))
        self.channel = QComboBox()
        self.channel.addItem("Sample run", "sample")
        self.channel.addItem("Reference run", "reference")
        self.channel.currentIndexChanged.connect(self._redraw)
        form.addWidget(self.channel)
        row = QHBoxLayout()
        self.file_pick = QComboBox()
        self.file_pick.setToolTip("Which file the image, profile and slice spectra show")
        self.file_pick.currentIndexChanged.connect(self._redraw)
        row.addWidget(self.file_pick, 1)
        for text, step in (("◀", -1), ("▶", 1)):
            button = QPushButton(text)
            button.setFixedWidth(30)
            button.setToolTip("Previous file" if step < 0 else "Next file")
            button.clicked.connect(lambda _=False, step=step: self._step_file(step))
            row.addWidget(button)
        form.addLayout(row)
        self.data_info = QLabel()
        self.data_info.setObjectName("runInfo")
        self.data_info.setWordWrap(True)
        form.addWidget(self.data_info)

        self.slices_title = _section("Slices")
        form.addWidget(self.slices_title)
        self.chips = QGridLayout()
        self.chips.setSpacing(6)
        form.addLayout(self.chips)
        row = QHBoxLayout()
        row.addWidget(QLabel("Pixel"))
        self.pixel = QSpinBox(minimum=1, maximum=256, value=142)
        row.addWidget(self.pixel)
        add = QPushButton("Add")
        add.setObjectName("accent")
        add.clicked.connect(lambda: self._add([self.pixel.value()]))
        row.addWidget(add)
        peak = QPushButton("Add peak")
        peak.setToolTip("Add the brightest chip row that is not a slice yet")
        peak.clicked.connect(self._add_peak)
        row.addWidget(peak)
        form.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Fan"))
        self.fan_centre = QSpinBox(minimum=1, maximum=256, value=142)
        self.fan_step = QSpinBox(minimum=1, maximum=50, value=2)
        self.fan_count = QSpinBox(minimum=1, maximum=3, value=2)
        row.addWidget(self.fan_centre)
        row.addWidget(QLabel("step"))
        row.addWidget(self.fan_step)
        form.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("±"))
        row.addWidget(self.fan_count)
        fan = QPushButton("Add fan")
        fan.setToolTip("Add slices at centre, centre ± step, centre ± 2 step, ...")
        fan.clicked.connect(self._add_fan)
        row.addWidget(fan)
        row.addStretch(1)
        form.addLayout(row)
        clear = QPushButton("Remove all")
        clear.clicked.connect(lambda: self.slices_changed.emit({"slices": []}))
        form.addWidget(clear, alignment=Qt.AlignLeft)

        form.addWidget(_section("X axis"))
        self.x_axis = Segmented((("ir", "IR cm⁻¹"), ("nm", "nm")))
        self.x_axis.changed.connect(self._redraw)
        form.addWidget(self.x_axis)
        self.x_note = QLabel()
        self.x_note.setObjectName("runInfo")
        self.x_note.setWordWrap(True)
        form.addWidget(self.x_note)

        form.addWidget(_section("Averaging window"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Half-width ±"))
        self.half_width = QSpinBox(minimum=0, maximum=20, suffix=" px")
        self.half_width.setKeyboardTracking(False)
        self.half_width.valueChanged.connect(lambda v: self.slices_changed.emit({"slice_half_width": v}))
        row.addWidget(self.half_width)
        row.addStretch(1)
        form.addLayout(row)
        self.combine = Segmented((("mean", "Mean"), ("sum", "Sum")))
        self.combine.changed.connect(lambda v: self.slices_changed.emit({"slice_combine": v}))
        form.addWidget(self.combine)
        note = QLabel("A slice is its centre row and the rows within the half-width, "
                      "averaged or summed. The processing uses what is chosen under Data → Process.")
        note.setObjectName("runInfo")
        note.setWordWrap(True)
        form.addWidget(note)

        form.addWidget(_section("Sum of slices"))
        self.show_sum = QCheckBox("Show the sum of the slices", checked=True)
        self.show_sum.toggled.connect(self._redraw)
        form.addWidget(self.show_sum)
        form.addStretch(1)

        # ---- plots ------------------------------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(6)
        bar = QHBoxLayout()
        zoom_signal = QPushButton("Zoom to signal")
        zoom_signal.clicked.connect(self._zoom_to_signal)
        full = QPushButton("Full frame")
        full.clicked.connect(self.reset_zoom)
        self.colour = QComboBox()
        self.colour.addItem("Colour: linear", "linear")
        self.colour.addItem("Colour: log", "log")
        self.colour.currentIndexChanged.connect(self._redraw)
        for widget in (zoom_signal, full, self.colour):
            bar.addWidget(widget)
        hint = QLabel("Click the image to add a slice. Drag to zoom.")
        hint.setObjectName("runInfo")
        bar.addStretch(1)
        bar.addWidget(hint)
        right.addLayout(bar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.scene().sigMouseClicked.connect(self._clicked)
        right.addWidget(self.graphics, 1)
        body.addLayout(right, 1)

    def add_view(self, widget: QWidget, title: str) -> None:
        """A further sub-tab, usable whether or not Image data is on (e.g. Files)."""
        self.views.addTab(widget, title)
        self._always_on.append(widget)

    def show_file(self, number: int) -> None:
        """Show file ``number`` on Detector and slices."""
        index = self.file_pick.findData(number)
        if index >= 0:
            self.file_pick.setCurrentIndex(index)
        self.views.setCurrentWidget(self.body)

    def _step_file(self, step: int) -> None:
        count = self.file_pick.count()
        if count > 1:
            self.file_pick.setCurrentIndex((self.file_pick.currentIndex() + step) % count)

    def _fill_file_pick(self, channel) -> None:
        """Mean of the files, then each file by number; keeps the chosen file if it is still there."""
        current = self.file_pick.currentData()
        self.file_pick.blockSignals(True)
        self.file_pick.clear()
        self.file_pick.addItem("Mean of the files", None)
        if channel.images is not None and channel.images.shape[0] > 1:
            for number, path in zip(channel.file_numbers, channel.files):
                self.file_pick.addItem(f"File #{number} · {path.name}", int(number))
        index = self.file_pick.findData(current)
        self.file_pick.setCurrentIndex(max(index, 0))
        self.file_pick.blockSignals(False)

    def _show_banner(self) -> None:
        """The Image data banner, except over the sub-tabs that do not need it (Files, Row by row)."""
        off = self._result is not None and not self._result.settings.image_data
        self.banner.setVisible(off and self.views.currentWidget() not in self._always_on)

    def _view_changed(self, *_) -> None:
        self._show_banner()
        if self._result is None:
            return
        for widget in self._always_on:
            if self.views.currentWidget() is widget:
                widget.update(self._result)
        if self.views.currentWidget() is self.slice_view:
            self.slice_view.update_from(self._result.settings)
        elif self.views.currentWidget() is self.fringe_view:
            self.fringe_view.update(self._result)

    # ---- slice editing -----------------------------------------------------------
    def _current_slices(self) -> list[int]:
        return list(self._result.settings.slices) if self._result is not None else []

    def _add(self, centres) -> None:
        if self._result is None or not self._result.settings.image_data:
            return
        slices = self._current_slices()
        for centre in centres:
            if centre not in slices and len(slices) < MAX_SLICES:
                slices.append(int(centre))
        self.slices_changed.emit({"slices": sorted(slices)})

    def _remove(self, centre: int) -> None:
        self.slices_changed.emit({"slices": [c for c in self._current_slices() if c != centre]})

    def _add_peak(self) -> None:
        if self._profile_rows is None:
            return
        taken = set(self._current_slices())
        for index in np.argsort(self._profile_rows)[::-1]:
            if index + 1 not in taken:
                self._add([int(index) + 1])
                return

    def _add_fan(self) -> None:
        centre, step, count = self.fan_centre.value(), self.fan_step.value(), self.fan_count.value()
        n_rows = len(self._profile_rows) if self._profile_rows is not None else 256
        offsets = sorted(range(-count, count + 1), key=abs)          # nearest the centre first
        self._add([centre + k * step for k in offsets if 1 <= centre + k * step <= n_rows])

    def _clicked(self, event) -> None:
        if self._heat is None or event.button() != Qt.LeftButton or not self.body.isEnabled():
            return
        position = event.scenePos()
        if not self._heat.getViewBox().sceneBoundingRect().contains(position):
            return
        row = int(round(self._heat.getViewBox().mapSceneToView(position).y()))
        if self._profile_rows is not None and 1 <= row <= len(self._profile_rows):
            QTimer.singleShot(0, lambda: self._add([row]))     # after the click is handled

    def _rebuild_chips(self, slices) -> None:
        while self.chips.count():
            chip = self.chips.takeAt(0).widget()
            chip.hide()                       # out of the layout now, deleted later
            chip.deleteLater()
        for index, centre in enumerate(slices):
            colour = SLICE_COLOURS[index % MAX_SLICES]
            chip = QPushButton(f"●  px {centre}   ✕")
            chip.setObjectName("sliceChip")
            chip.setStyleSheet(f"QPushButton#sliceChip {{ color: {colour}; }}")
            chip.setToolTip(f"Remove the slice at chip row {centre}")
            chip.clicked.connect(lambda _=False, c=centre: self._remove(c))
            self.chips.addWidget(chip, index // 2, index % 2)
        self.slices_title.setText(f"SLICES  ({len(slices)} of {MAX_SLICES})")

    # ---- zoom --------------------------------------------------------------------
    def _apply_mouse_mode(self, *_) -> None:
        for item in self._plots.values():
            item.getViewBox().setMouseMode(pg.ViewBox.RectMode)      # drag zooms, as the hint says

    def _zoom_to_signal(self) -> None:
        if self._heat is None or self._profile_rows is None:
            return
        profile = self._profile_rows - np.median(self._profile_rows)
        rows = np.flatnonzero(profile > 0.1 * profile.max()) + 1
        if not len(rows):
            return
        y_range = (rows.min() - 8, rows.max() + 8)
        columns = self._image[:, rows - 1].mean(axis=1)
        strong = self._x[columns > 0.05 * columns.max()]
        margin = 0.1 * (strong.max() - strong.min())
        x_range = (strong.min() - margin, strong.max() + margin)
        for key, item, ranges in ((self._keys["image"], self._heat, (x_range, y_range)),
                                  (self._keys["slices"], self._slice_plot, (x_range, None))):
            item.setXRange(*ranges[0], padding=0)
            if ranges[1] is not None:
                item.setYRange(*ranges[1], padding=0)
            self._zoom[key] = item.getViewBox().viewRange()

    # ---- drawing -------------------------------------------------------------------
    def update(self, result: Result) -> None:
        self._last = self._result = result
        s = result.settings
        enabled = s.image_data
        self._show_banner()
        for view in (self.body, self.fringe_view, self.slice_view):
            view.setEnabled(enabled)
            self._greyed[view].setEnabled(not enabled)
        if self.views.currentWidget() is self.slice_view:
            self.slice_view.update_from(s)     # only when shown: it runs one retrieval per slice
        elif self.views.currentWidget() is self.fringe_view:
            self.fringe_view.update(result)
        for widget in self._always_on:
            if self.views.currentWidget() is widget:
                widget.update(result)
        for widget, value in ((self.half_width, s.slice_half_width), (self.combine, s.slice_combine)):
            widget.blockSignals(True)
            widget.set_value(value) if isinstance(widget, Segmented) else widget.setValue(value)
            widget.blockSignals(False)
        self._rebuild_chips(sorted(s.slices))

        self.begin_draw()
        self.graphics.clear()
        self._heat = None
        role = self.channel.currentData()
        channel = getattr(result, role)
        if channel.images is None:
            self._profile_rows = None
            self.data_info.setText(f"{channel.name}: Full Vertical Binning spectra, no chip image.")
            self.graphics.addLabel(f"{channel.name} holds Full Vertical Binning spectra, so there is no "
                                   "chip image. Save a run in Full Resolution Image mode to use this tab.",
                                   color=self.theme.muted)
            self.end_draw()
            return

        self._fill_file_pick(channel)
        number = self.file_pick.currentData()
        n_files = channel.images.shape[0]
        if number is None:
            self._image = image = channel.image
            showing = "the mean of the files" if n_files > 1 else "it"
        else:
            k = int(np.flatnonzero(channel.file_numbers == number)[0])
            self._image = image = channel.images[k]
            showing = f"file {number} ({channel.files[k].name})"
        n_pixels, n_rows = image.shape
        self.data_info.setText(f"{channel.name}: {n_files} file{'s' if n_files > 1 else ''}, "
                               f"{n_rows} chip rows × {n_pixels} pixels; showing {showing}.")
        for spin in (self.pixel, self.fan_centre):
            spin.setMaximum(n_rows)

        wavelength = channel.wavelength_nm
        if self.x_axis.value() == "ir":
            x = processing.make_grid(wavelength, s.laser_wavelength_nm, s.sfg_or_dfg).wavenumber
            x_label = "IR wavenumber (cm⁻¹)"
            formula = (f"IR = 10⁷/λ − 10⁷/{s.laser_wavelength_nm:g} nm" if s.sfg_or_dfg == "sfg"
                       else f"IR = 10⁷/{s.laser_wavelength_nm:g} nm − 10⁷/λ")
            self.x_note.setText(f"{formula} (Laser wavelength, under Measurement). "
                                f"The detector covers {x.min():.0f} to {x.max():.0f} cm⁻¹.")
        else:
            x = wavelength
            x_label = "wavelength (nm)"
            self.x_note.setText(f"The detector covers {x.min():.1f} to {x.max():.1f} nm.")
        self._x = x

        # the image item needs evenly spaced columns; neither axis is exactly linear in pixels
        order = np.argsort(x)
        grid = np.linspace(x.min(), x.max(), n_pixels)
        uniform = np.column_stack([np.interp(grid, x[order], image[order, k]) for k in range(n_rows)])
        log = self.colour.currentData() == "log"
        shown = np.log10(np.clip(uniform, 1, None)) if log else uniform
        low = 0.0 if log else min(0.0, float(np.percentile(shown, 1)))
        high = float(np.percentile(shown, 99.95))
        view = (self.x_axis.value(), log, role)
        self._keys = {"image": ("image",) + view, "profile": ("profile",) + view, "slices": ("slices",) + view}

        heat = self._heat = self.plot(self.graphics, 0, 0, "Detector image", x_label, "chip row (px)")
        colours = CHIP_LIGHT if self.theme.name == "light" else CHIP_DARK
        image_item = pg.ImageItem(shown, levels=(low, high))
        image_item.setColorMap(colours)
        step = grid[1] - grid[0]
        image_item.setRect(QRectF(grid[0] - step / 2, 0.5, step * n_pixels, n_rows))
        heat.addItem(image_item)
        heat.setXRange(grid[0], grid[-1], padding=0)
        heat.setYRange(0.5, n_rows + 0.5, padding=0)

        self._profile_rows = image.mean(axis=0)
        profile = self.plot(self.graphics, 0, 1, "Profile", "mean counts", "")
        profile.plot(self._profile_rows, np.arange(1, n_rows + 1), pen=pg.mkPen(self.theme.text, width=1.2))
        profile.setYLink(heat)
        profile.getAxis("left").setWidth(10)
        profile.getAxis("left").setStyle(showValues=False)
        peak = float(np.abs(self._profile_rows).max()) or 1.0      # two or three ticks in a narrow plot
        major = next(m * 10 ** np.floor(np.log10(peak)) for m in (1, 2, 5, 10)
                     if peak / (m * 10 ** np.floor(np.log10(peak))) <= 2.5)
        profile.getAxis("bottom").setTickSpacing(major, major)

        centres = sorted(s.slices)
        combine = np.mean if s.slice_combine == "mean" else np.sum
        self._slice_plot = slices = self.plot(self.graphics, 1, 0, "Slices", x_label,
                                              "mean counts" if s.slice_combine == "mean" else "counts",
                                              colspan=2)
        slices.addLegend(offset=(-5, 5), labelTextColor=self.theme.text, brush=with_alpha(self.theme.base, 215))
        total = np.zeros(n_pixels)
        for index, centre in enumerate(centres):
            colour = SLICE_COLOURS[index % MAX_SLICES]
            rows = slice_rows(centre, s.slice_half_width, n_rows)
            for item in (heat, profile):
                if len(rows) > 1:
                    band = pg.LinearRegionItem(values=(rows[0] + 0.5, rows[-1] + 1.5), orientation="horizontal",
                                               movable=False, brush=with_alpha(colour, 45), pen=pg.mkPen(None))
                    item.addItem(band)
                item.addItem(pg.InfiniteLine(pos=centre, angle=0, pen=pg.mkPen(colour, width=1.5)))
            spectrum = combine(image[:, rows], axis=1)
            total += spectrum
            slices.plot(x, spectrum, pen=pg.mkPen(colour, width=1.2), name=f"px {centre}")
        if self.show_sum.isChecked() and len(centres) > 1:
            slices.plot(x, total, pen=pg.mkPen(self.theme.text, width=1.4), name="sum")
        if not centres:
            slices.setTitle("Slices: none yet. Click the image, or use Add, Add peak or Add fan.",
                            size="10pt", color=self.theme.muted)
        slices.setXRange(grid[0], grid[-1], padding=0)

        layout = self.graphics.ci.layout
        layout.setColumnStretchFactor(0, 5)
        layout.setColumnStretchFactor(1, 1)
        layout.setRowStretchFactor(0, 1)
        layout.setRowStretchFactor(1, 1)
        for name, item in (("image", heat), ("profile", profile), ("slices", slices)):
            self.register_plot(self._keys[name], item)
        self.end_draw()


# ----------------------------------------------------------------------------------
class RawPage(Page):
    title = "Raw spectra"
    SCAN_PRESETS = (("All scans", "all"), ("Mean of all scans", "mean"),
                    ("First and last", "ends"), ("Every 10th scan", "every10"))

    def __init__(self, theme):
        super().__init__(theme)
        bar = self.toolbar()
        self.channels = QComboBox()
        self.channels.addItem("Reference and sample", ("reference", "sample"))
        self.channels.addItem("Reference only", ("reference",))
        self.channels.addItem("Sample only", ("sample",))
        self.scans = QComboBox()
        self.scans.setMinimumContentsLength(18)
        self.scans.setMaxVisibleItems(25)
        for label, key in self.SCAN_PRESETS:
            self.scans.addItem(label, key)
        self.x_scale = QComboBox()
        self.x_scale.addItem("x: linear", "linear")
        self.x_scale.addItem("x: log", "log")
        self.y_scale = QComboBox()
        self.y_scale.addItem("y: linear", "linear")
        self.y_scale.addItem("y: log", "log")
        self.y_scale.addItem("y: each line ÷ its peak", "normalised")
        self.show_spikes = QCheckBox("Mark replaced pixels", checked=True)
        for widget in (self.channels, self.scans, self.x_scale, self.y_scale):
            widget.currentIndexChanged.connect(self._redraw)
            bar.addWidget(widget)
        self.show_spikes.toggled.connect(self._redraw)
        bar.addWidget(self.show_spikes)
        bar.addStretch(1)
        self.add_zoom_controls(bar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.layout_.addWidget(self.graphics)
        self._scan_numbers: tuple = ()

    def _fill_scan_list(self, result: Result) -> None:
        """Presets, then one entry per scan number; keeps the current choice if it still exists."""
        prefix = result.sample.prefix
        numbers = tuple(sorted(set(result.reference.numbers.tolist()) | set(result.sample.numbers.tolist())))
        if (prefix,) + numbers == self._scan_numbers:
            return
        self._scan_numbers = (prefix,) + numbers
        current = self.scans.currentData()
        self.scans.blockSignals(True)
        while self.scans.count() > len(self.SCAN_PRESETS):
            self.scans.removeItem(self.scans.count() - 1)
        self.scans.insertSeparator(self.scans.count())
        for number in numbers:
            self.scans.addItem(f"Slice px {number}" if prefix == "px " else f"Scan s{number}", int(number))
        index = self.scans.findData(current)
        self.scans.setCurrentIndex(index if index >= 0 else 0)
        self.scans.blockSignals(False)

    def _chosen(self, channel) -> tuple[list[int], bool]:
        """Column indices to draw, and whether to draw the mean instead."""
        choice = self.scans.currentData()
        n = channel.n_scans
        if choice == "mean":
            return [], True
        if choice == "ends":
            return sorted({0, n - 1}), False
        if choice == "every10":
            return list(range(0, n, 10)), False
        if isinstance(choice, int):
            return [int(k) for k in np.flatnonzero(channel.numbers == choice)], False
        return list(range(n)), False

    def _prepare(self, y: np.ndarray) -> np.ndarray:
        if self.y_scale.currentData() == "normalised":
            peak = np.nanmax(np.abs(y))
            y = y / peak if peak > 0 else y
        if self.y_scale.currentData() == "log":
            y = np.where(y > 0, y, np.nan)        # log of counts <= 0 is undefined; leave a gap
        return y

    def update(self, result: Result) -> None:
        self._last = result
        self._fill_scan_list(result)
        scales = (self.x_scale.currentData(), self.y_scale.currentData())
        self.begin_draw()
        self.graphics.clear()
        lim = result.settings.plot_wavenumber_lim
        log_x = self.x_scale.currentData() == "log"
        log_y = self.y_scale.currentData() == "log"
        y_label = "counts ÷ peak" if self.y_scale.currentData() == "normalised" else "counts"
        mean_pen = pg.mkPen(self.theme.text, width=1.5)
        for row, role in enumerate(self.channels.currentData()):
            channel = getattr(result, role)
            colours = scan_colours(channel.n_scans)      # a scan keeps its colour whatever is shown
            columns, show_mean = self._chosen(channel)
            raw = self.plot(self.graphics, row, 0, f"{channel.name}, counts as recorded",
                            "wavelength (nm)", y_label)
            resampled = self.plot(self.graphics, row, 1, f"{channel.name}, resampled to wavenumber",
                                  "wavenumber (cm⁻¹)", y_label)
            for item in (raw, resampled):
                item.setLogMode(x=log_x, y=log_y)
            if show_mean:
                raw.plot(channel.wavelength_nm, self._prepare(channel.counts.mean(axis=1)),
                         pen=mean_pen, connect="finite")
                resampled.plot(result.grid.wavenumber_uniform, self._prepare(channel.resampled.mean(axis=1)),
                               pen=mean_pen, connect="finite")
            for k in columns:
                # one scan alone gets the theme's blue: its time colour can be pale on white
                pen = pg.mkPen(self.theme.imag if len(columns) == 1 else colours[k], width=1)
                raw.plot(channel.wavelength_nm, self._prepare(channel.counts[:, k]), pen=pen, connect="finite")
                resampled.plot(result.grid.wavenumber_uniform, self._prepare(channel.resampled[:, k]),
                               pen=pen, connect="finite")
            if (self.show_spikes.isChecked() and not show_mean and channel.spike_mask is not None
                    and self.y_scale.currentData() != "normalised"):
                mask = np.zeros_like(channel.spike_mask)
                mask[:, columns] = channel.spike_mask[:, columns]
                if mask.any():
                    pixel, scan = np.nonzero(mask)
                    raw.plot(channel.wavelength_nm[pixel], channel.raw_counts[pixel, scan], pen=None,
                             symbol="x", symbolSize=6, symbolPen=pg.mkPen(self.theme.real, width=1.5),
                             name="replaced")
            x_range = np.log10(lim) if log_x else lim
            resampled.setXRange(*x_range, padding=0)
            self.register_plot((role, "raw") + scales, raw)
            self.register_plot((role, "resampled") + scales, resampled)
            if len(columns) == 1:
                one = f"{channel.name} {channel.prefix}{channel.numbers[columns[0]]}"
                raw.setTitle(f"{one}, counts as recorded", size="10pt", color=self.theme.text)
                resampled.setTitle(f"{one}, resampled to wavenumber",
                                   size="10pt", color=self.theme.text)
        self.end_draw()


# ----------------------------------------------------------------------------------
class TimeDomainPage(Page):
    """|FFT| on a log scale, full axis, with the two gates as draggable regions."""
    title = "Time domain"
    gate_moved = Signal(str, tuple)      # ("reference" | "sample", (start, stop))

    def __init__(self, theme):
        super().__init__(theme)
        bar = self.toolbar()
        hint = QLabel("Drag a shaded gate or its edges to move it; the spectra follow.")
        hint.setObjectName("runInfo")
        bar.addWidget(hint)
        bar.addStretch(1)
        self.add_zoom_controls(bar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.layout_.addWidget(self.graphics)
        self._regions: dict[str, pg.LinearRegionItem] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._items: dict[str, pg.PlotItem] = {}
        for row, role in enumerate(("reference", "sample")):
            item = self.plot(self.graphics, row, 0, role, "time (ps)", "|FFT|")
            item.setLogMode(y=True)
            curve = item.plot(pen=pg.mkPen(self.theme.imag if role == "sample" else self.theme.real, width=1.2))
            region = pg.LinearRegionItem(brush=with_alpha(self.theme.ok, 45),
                                         pen=pg.mkPen(self.theme.ok, width=2))
            region.setZValue(-5)
            region.sigRegionChangeFinished.connect(lambda r, role=role: self._moved(role))
            item.addItem(region)
            self._regions[role], self._curves[role], self._items[role] = region, curve, item
        self._items["sample"].setXLink(self._items["reference"])
        for role, item in self._items.items():
            self.register_plot(role, item)
        self.end_draw()
        self._initialised = False

    def reset_zoom(self) -> None:
        self._zoom.clear()
        self._initialised = False          # update() then shows the whole time axis again
        for item in self._items.values():
            item.enableAutoRange()
        super().reset_zoom()

    def _moved(self, role: str) -> None:
        start, stop = self._regions[role].getRegion()
        self.gate_moved.emit(role, (round(start, 3), round(stop, 3)))

    def update(self, result: Result) -> None:
        self._last = result
        t = result.grid.time_ps
        for role, channel in (("reference", result.reference), ("sample", result.sample)):
            magnitude = np.abs(channel.time_domain.mean(axis=1))
            self._curves[role].setData(t, np.maximum(magnitude, magnitude.max() * 1e-7))
            gate = getattr(result.settings, f"gate_range_{role}_ps")
            region = self._regions[role]
            region.blockSignals(True)
            region.setRegion(gate)
            region.blockSignals(False)
            self._items[role].setTitle(
                f"{channel.name}: |FFT| of the mean of {channel.n_scans} scans, "
                f"gate {gate[0]:.2f}–{gate[1]:.2f} ps", size="10pt", color=self.theme.text)
        if not self._initialised:
            self._items["reference"].setXRange(t[0], t[-1], padding=0)
            self._initialised = True


# ----------------------------------------------------------------------------------
class BeforeDivisionPage(Page):
    title = "Before division"

    def __init__(self, theme):
        super().__init__(theme)
        bar = self.toolbar()
        bar.addStretch(1)
        self.add_zoom_controls(bar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.layout_.addWidget(self.graphics)

    def update(self, result: Result) -> None:
        self._last = result
        self.begin_draw()
        self.graphics.clear()
        wavenumber = result.grid.wavenumber_uniform
        lim = result.settings.plot_wavenumber_lim
        first = None
        for row, channel in enumerate((result.reference, result.sample)):
            colours = scan_colours(channel.n_scans)
            item = self.plot(self.graphics, row, 0,
                             f"{channel.name}: amplitude of the gated spectrum, the curve that is divided",
                             "wavenumber (cm⁻¹)", "amplitude")
            self.shade_weak(item, result)
            amplitude = np.abs(channel.analytic)
            for k in range(channel.n_scans):
                item.plot(wavenumber, amplitude[:, k], pen=pg.mkPen(colours[k], width=1))
            item.plot(wavenumber, np.abs(channel.analytic.mean(axis=1)),
                      pen=pg.mkPen(self.theme.text, width=2))
            item.setXRange(*lim, padding=0)
            if first is None:
                first = item
            else:
                item.setXLink(first)
            self.register_plot(channel.name + str(row), item)
        self.end_draw()


# ----------------------------------------------------------------------------------
class Chi2Page(Page):
    title = "χ⁽²⁾"

    def __init__(self, theme):
        super().__init__(theme)
        bar = self.toolbar()
        self.units = QComboBox()
        self.units.addItem("Absolute units (Fresnel)", "absolute")
        self.units.addItem("Relative to quartz", "relative")
        self.view = QComboBox()
        self.view.addItem("Mean ± 95% CI", "mean")
        self.view.addItem("Every scan", "scans")
        self.compare = QCheckBox("Comparison spectrum (ag_h2o)", checked=True)
        for widget in (self.units, self.view):
            widget.currentIndexChanged.connect(lambda: self._last and self.update(self._last))
            bar.addWidget(widget)
        self.compare.toggled.connect(lambda: self._last and self.update(self._last))
        bar.addWidget(self.compare)
        bar.addStretch(1)
        self.add_zoom_controls(bar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.layout_.addWidget(self.graphics)
        self._comparison_cache = {}

    def _comparison(self, result: Result):
        model = result.settings.interface_model_sample
        stem = "ag_h2olorentz_ax_and_spec" if model == "lorentz" else "ag_h2oslab_ax_and_spec"
        if stem not in self._comparison_cache:
            from pathlib import Path
            folder = Path(result.settings.reference_dir)
            try:
                values = np.loadtxt(folder / f"{stem}.txt")
                errors = np.loadtxt(folder / f"{stem}_err.txt")
                self._comparison_cache[stem] = (values[:, 0], values[:, 1] + 1j * values[:, 2],
                                                errors[:, 1] + 1j * errors[:, 2])
            except OSError:
                self._comparison_cache[stem] = None
        return self._comparison_cache[stem]

    def _band(self, item, x, centre, half, colour):
        upper = item.plot(x, centre + half, pen=None)
        lower = item.plot(x, centre - half, pen=None)
        item.addItem(pg.FillBetweenItem(upper, lower, brush=with_alpha(colour, self.theme.band_alpha)))

    def update(self, result: Result) -> None:
        self._last = result
        self.begin_draw()
        self.graphics.clear()
        units = self.units.currentData()
        summary = getattr(result, units)
        wavenumber = result.grid.wavenumber_uniform
        lim = result.settings.plot_wavenumber_lim
        frame = result.fresnel_sample.molecular_frame
        if units == "absolute":
            exponent = int(round(np.log10(result.settings.absolute_scale)))
            y_label = f"χ⁽²⁾ {frame} (×10⁻{exponent} m²/V)"
        else:
            y_label = "χ⁽²⁾ / χ⁽²⁾ quartz"

        main = self.plot(self.graphics, 0, 0, f"{result.sample.name} against {result.reference.name}",
                         "wavenumber (cm⁻¹)", y_label, colspan=2)
        main.addLegend(offset=(-10, 10), labelTextColor=self.theme.text, brush=with_alpha(self.theme.base, 215))
        self.shade_weak(main, result)
        main.addLine(y=0, pen=pg.mkPen(self.theme.border))

        if units == "absolute" and self.compare.isChecked():
            comparison = self._comparison(result)
            if comparison is not None:
                x, values, errors = comparison
                grey = self.theme.weak_region
                self._band(main, x, values.imag, errors.imag, grey)
                main.plot(x, values.imag, pen=pg.mkPen(grey, width=1.2), name="ag (Im, Re)")
                main.plot(x, values.real, pen=pg.mkPen(grey, width=1.2, style=Qt.DashLine))

        if self.view.currentData() == "scans":
            colours = scan_colours(result.sample.n_scans)
            for k in range(result.sample.n_scans):
                main.plot(wavenumber, summary.per_scan[:, k].imag, pen=pg.mkPen(colours[k], width=1))
            main.plot(wavenumber, summary.mean.imag, pen=pg.mkPen(self.theme.text, width=2), name="Im, mean")
        else:
            self._band(main, wavenumber, summary.mean.imag, summary.imag_error, self.theme.imag)
            self._band(main, wavenumber, summary.mean.real, summary.real_error, self.theme.real)
            main.plot(wavenumber, summary.mean.imag, pen=pg.mkPen(self.theme.imag, width=2), name="Im")
            main.plot(wavenumber, summary.mean.real, pen=pg.mkPen(self.theme.real, width=2), name="Re")
        main.setXRange(*lim, padding=0)
        visible = (wavenumber >= lim[0]) & (wavenumber <= lim[1])
        strength = result.reference_strength
        trusted = visible & (strength >= result.settings.weak_reference_fraction * strength.max())
        if trusted.any():
            span = np.max(np.abs(np.concatenate([summary.mean.imag[trusted], summary.mean.real[trusted]])))
            main.setYRange(-1.25 * span, 1.25 * span, padding=0)

        phase = self.plot(self.graphics, 1, 0, "phase", "wavenumber (cm⁻¹)", "phase (°)")
        self.shade_weak(phase, result)
        self._band(phase, wavenumber, summary.phase_deg, summary.phase_error, self.theme.imag)
        phase.plot(wavenumber, summary.phase_deg, pen=pg.mkPen(self.theme.imag, width=1.5))
        phase.addLine(y=180, pen=pg.mkPen(self.theme.border))
        phase.setYRange(0, 360, padding=0)
        phase.setXRange(*lim, padding=0)

        homodyne = self.plot(self.graphics, 1, 1, "|χ⁽²⁾|²", "wavenumber (cm⁻¹)", "|χ⁽²⁾|²")
        self.shade_weak(homodyne, result)
        self._band(homodyne, wavenumber, summary.homodyne, summary.homodyne_error, self.theme.imag)
        homodyne.plot(wavenumber, summary.homodyne, pen=pg.mkPen(self.theme.imag, width=1.5))
        if trusted.any():
            homodyne.setYRange(0, 1.1 * np.max(summary.homodyne[trusted] + summary.homodyne_error[trusted]),
                               padding=0)
        homodyne.setXRange(*lim, padding=0)
        self.graphics.ci.layout.setRowStretchFactor(0, 3)
        self.graphics.ci.layout.setRowStretchFactor(1, 2)
        self.register_plot(("main", units), main)
        self.register_plot(("phase", units), phase)
        self.register_plot(("homodyne", units), homodyne)
        self.end_draw()


# ----------------------------------------------------------------------------------
class DriftPage(Page):
    """Phase change since the first scan, every pixel, no fitting."""
    title = "Phase drift"

    def __init__(self, theme):
        super().__init__(theme)
        bar = self.toolbar()
        self.channel = QComboBox()
        self.channel.addItem("Sample", "sample")
        self.channel.addItem("Reference", "reference")
        self.group = QComboBox()
        for n in (1, 2, 3, 4, 5):
            self.group.addItem(f"{n} scan{'s' if n > 1 else ''} per point", n)
        self.group.setCurrentIndex(2)
        self.smooth = QComboBox()
        for width in (0, 5, 10, 20, 40):
            self.smooth.addItem("No smoothing" if width == 0 else f"Smooth {width} cm⁻¹", width)
        self.smooth.setCurrentIndex(2)
        for widget in (self.channel, self.group, self.smooth):
            widget.currentIndexChanged.connect(lambda: self._last and self.update(self._last))
            bar.addWidget(widget)
        note = QLabel("angle(point ÷ first point); grouping and smoothing are plain averages")
        note.setObjectName("runInfo")
        bar.addWidget(note)
        bar.addStretch(1)
        self.add_zoom_controls(bar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.layout_.addWidget(self.graphics)

    def update(self, result: Result) -> None:
        self._last = result
        self.begin_draw()
        self.graphics.clear()
        channel = getattr(result, self.channel.currentData())
        smooth_px = max(1, round(self.smooth.currentData() / result.grid.wavenumber_step))
        drift = phase_drift(channel.analytic, channel.time_sec, channel.numbers,
                            self.group.currentData(), smooth_px, channel.prefix)
        if channel.rows_as_scans:        # slices of one exposure: chip row across, not time
            group_of = np.arange(channel.n_scans) // max(self.group.currentData(), 1)
            t = np.array([channel.numbers[group_of == k].mean() for k in range(group_of[-1] + 1)])
            across = "slice (chip row)"
        else:
            t = drift.time_min
            across = "time (min)"
        wavenumber = result.grid.wavenumber_uniform
        lim = result.settings.plot_wavenumber_lim
        weak = drift.amplitude < result.settings.weak_reference_fraction * drift.amplitude.max()
        strong = ~weak & (wavenumber >= lim[0]) & (wavenumber <= lim[1])
        span = max(1.0, 1.1 * np.max(np.abs(drift.phase_change_deg[strong]))) if strong.any() else 10.0

        # map: time across, wavenumber up
        heat = self.plot(self.graphics, 0, 0, f"{channel.name}: phase change since {drift.labels[0]} (°)",
                         across, "wavenumber (cm⁻¹)", rowspan=2)
        order = np.argsort(wavenumber)
        t_edges = np.concatenate([[t[0] - 0.5 * (t[1] - t[0]) if len(t) > 1 else -0.5],
                                  (t[:-1] + t[1:]) / 2,
                                  [t[-1] + 0.5 * (t[-1] - t[-2]) if len(t) > 1 else 0.5]])
        wn = wavenumber[order]
        wn_edges = np.concatenate([[wn[0]], (wn[:-1] + wn[1:]) / 2, [wn[-1]]])
        values = drift.phase_change_deg[order].T
        x_mesh, y_mesh = np.meshgrid(t_edges, wn_edges, indexing="ij")
        mesh = pg.PColorMeshItem(x_mesh, y_mesh, values, colorMap=DIVERGING, levels=(-span, span),
                                 enableAutoLevels=False)
        heat.addItem(mesh)
        heat.setYRange(*lim, padding=0)
        heat.setXRange(t_edges[0], t_edges[-1], padding=0)
        bar = pg.ColorBarItem(values=(-span, span), colorMap=DIVERGING, interactive=False, width=12)
        bar.setImageItem(mesh, insert_in=heat)

        colours = scan_colours(len(t))
        lines = self.plot(self.graphics, 0, 1, "against wavenumber, one line per point",
                          "wavenumber (cm⁻¹)", "phase change (°)")
        lines.addLegend(offset=(-5, 5), labelTextColor=self.theme.text, brush=with_alpha(self.theme.base, 215))
        for start, stop in _runs(weak):
            band = pg.LinearRegionItem(values=sorted((wavenumber[start], wavenumber[stop])), movable=False,
                                       brush=with_alpha(self.theme.weak_region, 40), pen=pg.mkPen(None))
            band.setZValue(-10)
            lines.addItem(band)
        # width 1: Qt draws antialiased lines wider than 1 px as outlines, which takes
        # seconds per redraw once 100+ noisy phase lines run off the plot
        for k in range(len(t)):
            lines.plot(wavenumber, drift.phase_change_deg[:, k], pen=pg.mkPen(colours[k], width=1),
                       name=drift.labels[k] if channel.rows_as_scans else f"{drift.labels[k]}, {t[k]:.1f} min")
        lines.setXRange(*lim, padding=0)
        lines.setYRange(-span, span, padding=0)

        traces = self.plot(self.graphics, 1, 1, f"against {across.split(' (')[0]}, at fixed wavenumbers",
                           across, "phase change (°)")
        traces.addLegend(offset=(-5, 5), labelTextColor=self.theme.text, brush=with_alpha(self.theme.base, 215))
        low, high = result.usable_range()
        targets = low + (high - low) * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        palette = [self.theme.imag, self.theme.real, self.theme.ok, self.theme.warn, self.theme.muted]
        strong_pixels = np.flatnonzero(~weak)       # where this run's own signal is strong
        for target, colour in zip(targets, palette):
            pixel = int(strong_pixels[np.argmin(np.abs(wavenumber[strong_pixels] - target))])
            traces.plot(t, drift.phase_change_deg[pixel], pen=pg.mkPen(colour, width=1.6),
                        symbol="o", symbolSize=5, symbolBrush=colour, symbolPen=None,
                        name=f"{wavenumber[pixel]:.0f} cm⁻¹")
        traces.addLine(y=0, pen=pg.mkPen(self.theme.border))
        role = self.channel.currentData()
        for name, item in (("map", heat), ("lines", lines), ("traces", traces)):
            self.register_plot((name, role, channel.rows_as_scans), item)
        self.end_draw()


# ----------------------------------------------------------------------------------
class HomodynePage(Page):
    """|E(t) / divisor|^2 of every scan in one plot, coloured by time."""
    title = "Homodyne"
    Y_LABELS = {"first": "|E(t) / E(0)|²", "mean": "|E(t) / ⟨E⟩|²", "none": "|E(t)|²  (counts²)"}

    def __init__(self, theme):
        super().__init__(theme)
        bar = self.toolbar()
        self.channel = QComboBox()
        self.channel.addItem("Sample", "sample")
        self.channel.addItem("Reference", "reference")
        self.divide_by = QComboBox()
        self.divide_by.addItem("Divided by first scan", "first")
        self.divide_by.addItem("Divided by mean of all scans", "mean")
        self.divide_by.addItem("No division", "none")
        for widget in (self.channel, self.divide_by):
            widget.currentIndexChanged.connect(lambda: self._last and self.update(self._last))
            bar.addWidget(widget)
        note = QLabel("E(t) = gated complex spectrum (Re + i Im); light blue = first scan, "
                      "dark blue = last, red = mean")
        note.setObjectName("runInfo")
        bar.addWidget(note)
        bar.addStretch(1)
        self.add_zoom_controls(bar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.layout_.addWidget(self.graphics)

    def update(self, result: Result) -> None:
        self._last = result
        self.begin_draw()
        self.graphics.clear()
        channel = getattr(result, self.channel.currentData())
        divide_by = self.divide_by.currentData()
        homodyne = homodyne_per_scan(channel.analytic, divide_by)
        wavenumber = result.grid.wavenumber_uniform
        lim = result.settings.plot_wavenumber_lim
        if channel.rows_as_scans:        # slices of one exposure: colour by chip row, not time
            position = (channel.numbers - channel.numbers[0]).astype(float)
            span, first, bar_label = max(float(position[-1]), 1e-9), float(channel.numbers[0]), "chip row"
        else:
            position = (channel.time_sec - channel.time_sec[0]) / 60
            span, first, bar_label = max(float(position[-1]), 1e-9), 0.0, "time (min)"

        noun = "slices" if channel.rows_as_scans else "scans"
        item = self.plot(self.graphics, 0, 0, f"{channel.name}, {channel.n_scans} {noun}",
                         "wavenumber (cm⁻¹)", self.Y_LABELS[divide_by])
        item.addLegend(offset=(-10, 10), labelTextColor=self.theme.text, brush=with_alpha(self.theme.base, 215))
        self.shade_weak(item, result)
        for k in range(channel.n_scans):
            item.plot(wavenumber, homodyne[:, k],
                      pen=pg.mkPen(BLUES.map(position[k] / span, mode="qcolor"), width=1))
        if divide_by != "none":
            item.addLine(y=1, pen=pg.mkPen(self.theme.muted, width=1, style=Qt.DashLine))
        mean_colour = "#B3172B" if self.theme.name == "light" else "#F0606E"
        item.plot(wavenumber, homodyne.mean(axis=1), pen=pg.mkPen(mean_colour, width=2.5), name="mean")

        item.setXRange(*lim, padding=0)
        strength = result.reference_strength
        trusted = ((wavenumber >= lim[0]) & (wavenumber <= lim[1])
                   & (strength >= result.settings.weak_reference_fraction * strength.max()))
        if trusted.any():
            if divide_by == "none":
                item.setYRange(0, 1.05 * homodyne[trusted].max(), padding=0)
            else:
                spread = np.abs(homodyne[trusted] - 1).max()
                item.setYRange(max(0.0, 1 - 1.1 * spread), 1 + 1.1 * spread, padding=0)

        colour_bar = pg.ColorBarItem(values=(first, first + span), colorMap=BLUES, interactive=False, width=14,
                                     label=bar_label)
        colour_bar.getAxis("right").setTextPen(self.theme.plot_foreground)
        self.graphics.addItem(colour_bar, row=0, col=1)
        self.register_plot((self.channel.currentData(), divide_by, channel.rows_as_scans), item)
        self.end_draw()


def _runs(mask: np.ndarray):
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1)

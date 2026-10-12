"""The "χ⁽²⁾ per slice" view inside the Chip image tab.

Every slice is retrieved on its own (sample slice ÷ reference slice at the same
chip row, all the other settings as in the sidebar) and drawn in one journal
figure by ``publication.slices_figure``, the same function Export uses.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from hdsfg_engine.pipeline import Pipeline, retrieve_slices

from .background import Job
from .publication import export_slices, slices_figure


class SliceChi2View(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        bar = QHBoxLayout()
        self.quantity = QComboBox()
        self.quantity.addItem("Im and Re χ⁽²⁾", "chi2")
        self.quantity.addItem("|χ⁽²⁾|²", "homodyne")
        self.quantity.addItem("Phase", "phase")
        self.units = QComboBox()
        self.units.addItem("Absolute units (Fresnel)", "absolute")
        self.units.addItem("Relative to quartz", "relative")
        self.divide_by = QComboBox()
        self.divide_by.addItem("÷ same slice of the reference", "same")
        self.divide_by.addItem("÷ sum of the reference slices", "sum")
        self.divide_by.addItem("÷ first slice of the reference", "first")
        self.divide_by.setToolTip("Same slice: each chip row has its own quartz, the right choice for "
                                  "sample and quartz runs.\nSum: every slice is divided by the combined "
                                  "quartz, to see how the rows differ from each other.\nFirst slice: every slice is divided by "
                                  "the lowest chip row's slice; the first slice itself then gives 1.")
        self.divide_by.currentIndexChanged.connect(lambda: self._settings and self.update_from(self._settings))
        bar.addWidget(self.divide_by)
        self.strong_only = QCheckBox("Only where quartz is strong", checked=True)
        self.strong_only.setToolTip("Draw each slice only where its own quartz reference is above "
                                    "Display → Weak reference below")
        for combo in (self.quantity, self.units):
            combo.currentIndexChanged.connect(self._draw)
            bar.addWidget(combo)
        self.strong_only.toggled.connect(self._draw)
        bar.addWidget(self.strong_only)
        export = QPushButton("Export…")
        export.setToolTip("Each slice's full export in its own px<row> folder, plus slices.svg/.pdf")
        export.clicked.connect(self._export)
        bar.addWidget(export)
        bar.addStretch(1)
        layout.addLayout(bar)
        self.info = QLabel()
        self.info.setObjectName("runInfo")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.figure = Figure(figsize=(6.4, 5.0), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedSize(760, 590)
        holder = QHBoxLayout()
        holder.addStretch(1)
        holder.addWidget(self.canvas)
        holder.addStretch(1)
        layout.addLayout(holder)
        layout.addStretch(1)
        self._pipeline = Pipeline()          # its own cache: the main run keeps its own
        self._results: dict = {}
        self._key = None
        self._settings = None
        self._job = Job("χ⁽²⁾ per slice", self._computed, self)   # one retrieval per slice, in the background

    def update_from(self, settings) -> None:
        """Retrieve every slice for these settings in the background (skipped if nothing changed)."""
        self._settings = settings
        divide_by = self.divide_by.currentData()
        key = (repr(settings), divide_by)
        if key == self._key:
            self._draw()
            return
        self._key = key
        if not settings.image_data:
            self._computed(True, {}, settings, divide_by)
            return
        self.info.setText("⏳ Processing the slices…")
        self._job.submit(self._retrieve, settings, divide_by)

    def _retrieve(self, settings, divide_by):
        """Worker thread."""
        return retrieve_slices(settings, self._pipeline, divide_by), settings, divide_by

    def _computed(self, ok, value, settings=None, divide_by=None) -> None:
        if not ok:                          # show it here; the rest of the app keeps working
            self._results = {}
            self.info.setText(f"Could not process the slices: {value}")
            self._draw()
            return
        if settings is None:
            value, settings, divide_by = value
        self._results = value
        if self._results:
            first = next(iter(self._results.values()))
            window = (f"rows within ±{settings.slice_half_width} px, {settings.slice_combine}"
                      if settings.slice_half_width else "one chip row each")
            self.info.setText(
                f"{len(self._results)} slices, each processed on its own: {first.sample.name} slice ÷ "
                + {"same": f"{first.reference.name} slice at the same chip row",
                   "sum": f"sum of the {first.reference.name} slices",
                   "first": f"{first.reference.name} slice px {min(settings.slices)}"}[divide_by]
                + f" ({window}); "
                f"{first.sample.n_scans} file{'s' if first.sample.n_scans > 1 else ''} per slice. "
                "Every other setting is the sidebar's.")
        else:
            self.info.setText("No slices yet: add them on the Detector tab.")
        self._draw()

    def _draw(self, *_) -> None:
        slices_figure(self._results, self.quantity.currentData(), self.units.currentData(),
                      self.strong_only.isChecked(), self.figure)
        self.canvas.draw_idle()

    def _export(self) -> None:
        if not self._results:
            return
        first = next(iter(self._results.values()))
        default = str(Path(first.settings.data_dir).parent / "results" / "slices")
        folder = QFileDialog.getExistingDirectory(self, "Export slices into", default)
        if folder:
            written = export_slices(self._results, folder, self.units.currentData(),
                                    self.strong_only.isChecked())
            self.info.setText(f"Exported {len(written)} files to {folder}")

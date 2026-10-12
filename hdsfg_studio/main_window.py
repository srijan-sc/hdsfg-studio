"""The main window: toolbar, settings sidebar, summary cards, plot tabs, status bar."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QProgressBar, QSplitter, QTabWidget, QToolBar,
                               QToolButton, QVBoxLayout, QWidget)

from hdsfg_engine import Pipeline, Settings
from hdsfg_engine.data import list_runs, read_footer_exposure, sort_files

from . import theme as themes
from .background import Job, activity
from .pages import (BeforeDivisionPage, Chi2Page, ChipImagePage, DriftPage, HomodynePage, Page, RawPage,
                    TimeDomainPage)
from .files_view import CompareWindow, FilesView
from .row_view import RowByRowView
from .parameter_panel import ParameterPanel
from .publication import export_results, publication_figure

APP_NAME = "HDSFG Studio"


class Card(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(0)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.value = QLabel("–")
        self.value.setObjectName("cardValue")
        layout.addWidget(self.title)
        layout.addWidget(self.value)


class PublicationPage(QWidget):
    title = "Publication"

    def __init__(self, theme):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        bar = QHBoxLayout()
        self.quantity = QComboBox()
        self.quantity.addItem("Im and Re χ⁽²⁾", "chi2")
        self.quantity.addItem("|χ⁽²⁾|²", "homodyne")
        self.units = QComboBox()
        self.units.addItem("Absolute units", "absolute")
        self.units.addItem("Relative to quartz", "relative")
        for combo in (self.quantity, self.units):
            combo.currentIndexChanged.connect(lambda: self._last and self.update(self._last))
            bar.addWidget(combo)
        note = QLabel("Exactly what Export writes as output.svg / output.pdf")
        note.setObjectName("runInfo")
        bar.addWidget(note)
        bar.addStretch(1)
        layout.addLayout(bar)
        self.figure = Figure(figsize=(5.0, 3.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        holder = QHBoxLayout()
        holder.addStretch(1)
        holder.addWidget(self.canvas)
        holder.addStretch(1)
        self.canvas.setFixedSize(620, 434)
        layout.addLayout(holder)
        layout.addStretch(1)
        self._last = None

    def update(self, result):
        self._last = result
        publication_figure(result, self.quantity.currentData(), self.units.currentData(), self.figure)
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self, data_dir: str | None = None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        self.store = QSettings("HDSFG", "Studio")
        self.theme = themes.DARK if self.store.value("theme") == "dark" else themes.LIGHT
        themes.apply(QApplication.instance(), self.theme)

        self.pipeline = Pipeline()
        self.result = None
        self._job = Job("Processing", self._run_done, self)   # the pipeline runs in a worker thread
        self._clear_cache = False          # a new folder or file list: start from the files
        self._after_run: list = []         # things to do once the next run has finished
        self._run_started = 0.0
        self.settings = Settings()
        self._debounce = QTimer(singleShot=True, interval=120)
        self._debounce.timeout.connect(self.run)

        self._build_toolbar()
        self._build_body()
        self._build_menus()
        self._build_busy_indicator()
        self.statusBar().showMessage("Open a data folder to begin")

        files = self.store.value("files") or []
        files = [files] if isinstance(files, str) else list(files)
        if not data_dir and self.store.value("mode") == "files" and files \
                and all(Path(f).is_file() for f in files):
            self.open_files(files)
            return
        folder = data_dir or self.store.value("data_dir")
        if folder and Path(folder).is_dir():
            self.open_folder(folder)

    # ------------------------------------------------------------------ layout
    def _build_toolbar(self):
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(bar)
        self.open_action = bar.addAction("Open folder…", self.choose_folder)
        bar.addAction("Open files…", self.choose_files)
        bar.addSeparator()
        self.folder_label = QLabel("No data folder")
        self.folder_label.setObjectName("runInfo")
        bar.addWidget(self.folder_label)
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,
                             spacer.sizePolicy().verticalPolicy().Preferred)
        bar.addWidget(spacer)
        self.auto_button = QToolButton(text="Live update", checkable=True, checked=True)
        self.auto_button.setToolTip("Recompute as soon as a setting changes")
        bar.addWidget(self.auto_button)
        run = QToolButton(text="Run")
        run.setObjectName("primary")
        run.clicked.connect(self.run)
        bar.addWidget(run)
        bar.addAction("Export…", self.export)

    def _build_body(self):
        self.panel = ParameterPanel()
        self.panel.changed.connect(self._settings_changed)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.panel)
        sidebar.setMinimumWidth(380)
        sidebar.setMaximumWidth(420)

        self.cards = {name: Card(name) for name in
                      ("Scans", "Rotation", "Usable range", "Pixels replaced", "Drift in sample run")}
        card_row = QHBoxLayout()
        card_row.setContentsMargins(10, 10, 10, 0)
        card_row.setSpacing(8)
        for card in self.cards.values():
            card_row.addWidget(card)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.chip_page, self.time_page = ChipImagePage(self.theme), TimeDomainPage(self.theme)
        self.chi2_page = Chi2Page(self.theme)
        self.pages: list[Page] = [self.chip_page, RawPage(self.theme), self.time_page,
                                  BeforeDivisionPage(self.theme), self.chi2_page,
                                  DriftPage(self.theme), HomodynePage(self.theme),
                                  PublicationPage(self.theme)]
        for page in self.pages:
            self.tabs.addTab(page, page.title)
        self.tabs.setCurrentWidget(self.chi2_page)
        self.tabs.currentChanged.connect(self._refresh_current_page)
        self.time_page.gate_moved.connect(self._gate_dragged)
        self.chip_page.slices_changed.connect(self._slices_changed)
        self.row_view = RowByRowView(self.theme)
        self.row_view.gate_moved.connect(self._gate_dragged)
        self.chip_page.add_view(self.row_view, "Row by row (quartz ÷ sample)")
        self.files_view = FilesView(self.theme)
        self.chip_page.add_view(self.files_view, "Files")
        self.files_view.open_requested.connect(self.choose_files)
        self.files_view.use_changed.connect(self._files_used)
        self.files_view.remove_requested.connect(self._remove_files)
        self.files_view.show_requested.connect(self.chip_page.show_file)
        self.compare = CompareWindow(self.theme)
        self.compare.setParent(self, Qt.Window)      # closes with the main window
        self.files_view.compare_requested.connect(self._show_compare)

        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)
        centre_layout.setSpacing(6)
        centre_layout.addLayout(card_row)
        centre_layout.addWidget(self.tabs, 1)

        splitter = QSplitter()
        splitter.addWidget(sidebar)
        splitter.addWidget(centre)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 1050])
        self.setCentralWidget(splitter)
        self._dirty_pages: set[int] = set()

    def _build_busy_indicator(self):
        """Right of the status bar: what is running in the background now, with a moving bar."""
        self.busy_label = QLabel()
        self.busy_label.setObjectName("runInfo")
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)               # no end known: the bar just moves
        self.busy_bar.setFixedWidth(110)
        self.busy_bar.setFixedHeight(10)
        self.busy_bar.setTextVisible(False)
        self.statusBar().addPermanentWidget(self.busy_label)
        self.statusBar().addPermanentWidget(self.busy_bar)
        activity().changed.connect(self._show_busy)
        self._show_busy([])

    def _show_busy(self, names: list):
        self.busy_label.setText(f"⏳ Running: {', '.join(names)}" if names else "")
        self.busy_label.setVisible(bool(names))
        self.busy_bar.setVisible(bool(names))

    def _build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("Open data folder…", self, shortcut=QKeySequence.Open)
        open_action.triggered.connect(self.choose_folder)
        file_menu.addAction(open_action)
        open_files = QAction("Open files…", self, shortcut="Ctrl+Shift+O")
        open_files.setToolTip("Any .asc files, used as repeats of one sample (reference and sample)")
        open_files.triggered.connect(self.choose_files)
        file_menu.addAction(open_files)
        file_menu.addSeparator()
        load = QAction("Load settings…", self)
        load.triggered.connect(self.load_settings)
        save = QAction("Save settings…", self, shortcut=QKeySequence.Save)
        save.triggered.connect(self.save_settings)
        file_menu.addActions([load, save])
        file_menu.addSeparator()
        export = QAction("Export results…", self, shortcut="Ctrl+E")
        export.triggered.connect(self.export)
        file_menu.addAction(export)

        view = self.menuBar().addMenu("&View")
        advanced = QAction("Show advanced settings", self, checkable=True)
        advanced.toggled.connect(self.panel.set_show_advanced)
        view.addAction(advanced)
        dark = QAction("Dark theme", self, checkable=True, checked=self.theme.name == "dark")
        dark.toggled.connect(self._toggle_theme)
        view.addAction(dark)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction(f"About {APP_NAME}", self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, APP_NAME,
            f"<b>{APP_NAME}</b><p>Heterodyne-detected SFG retrieval (v37b pipeline).</p>"
            "<p>The processing engine is checked against the MATLAB v37b port, which matches "
            "the group's v37b Python script to ~1e-10 %.</p>"))
        help_menu.addAction(about)

    # ------------------------------------------------------------------ actions
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open data folder",
                                                  self.settings.data_dir or str(Path.home()))
        if folder:
            self.open_folder(folder)

    def choose_files(self):
        start = str(Path(self.settings.files[0]).parent) if self.settings.files \
            else self.settings.data_dir or str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(self, "Open .asc files", start, "Andor spectra (*.asc)")
        if paths:
            self.open_files(paths)

    def open_files(self, paths, keep_settings: bool = False):
        """Chosen files, any names: each file one scan, used as both reference and sample.

        Exposure and frames are filled in from the first file's footer (they can be
        changed in the sidebar), unless ``keep_settings`` (loading a settings file).
        """
        paths = sort_files(paths)
        settings = replace(self.settings, files=paths, data_dir=str(Path(paths[0]).parent))
        if not keep_settings:
            settings.sample_file_numbers = []
            exposure, frames = read_footer_exposure(Path(paths[0]))
            if exposure:
                settings.file_exposure_sec = round(exposure, 3)
            if frames:
                settings.file_frames = frames
        self.settings = settings
        self._clear_cache = True
        self.panel.set_file_mode(True)
        self.panel.set_settings(settings)
        self.folder_label.setText(f"{Path(paths[0]).parent.name}  ·  {len(paths)} chosen files")
        self.store.setValue("files", paths)
        self.store.setValue("mode", "files")
        self._after_run.append(lambda: (self.tabs.setCurrentWidget(self.chip_page),
                                        self.chip_page.views.setCurrentWidget(self.files_view)))
        self.run()

    def _files_used(self, numbers: list):
        self.settings = replace(self.settings, sample_file_numbers=numbers)
        self.panel.set_settings(self.settings)
        self.run()

    def _remove_files(self, numbers: list):
        kept = [path for k, path in enumerate(self.settings.files, start=1) if k not in set(numbers)]
        if not kept:
            return
        self.settings = replace(self.settings, sample_file_numbers=[])
        self.open_files(kept, keep_settings=True)

    def _show_compare(self):
        self.compare.show()
        self.compare.raise_()
        self.compare.activateWindow()

    def open_folder(self, folder: str):
        runs = list_runs(folder)
        if not runs:
            QMessageBox.warning(self, APP_NAME, f"No Andor .asc runs found in\n{folder}")
            return
        names = [run.name for run in runs]
        settings = replace(self.settings, data_dir=folder, files=[])
        if folder != self.settings.data_dir or self.settings.files:   # scan numbers belong to the old data
            settings.reference_file_numbers, settings.sample_file_numbers = [], []
        if settings.reference_name not in names:
            settings.reference_name = next((n for n in names if n.startswith("zqz")), names[0])
        if settings.sample_name not in names:
            settings.sample_name = next((n for n in names if not n.startswith("zqz")), names[-1])
        self.settings = settings
        self._clear_cache = True
        self.panel.set_file_mode(False)
        self.panel.set_settings(settings)
        self.panel.set_runs(runs)
        self.folder_label.setText(f"{Path(folder).parent.name} / {Path(folder).name}  ·  {len(runs)} runs")
        self.store.setValue("data_dir", folder)
        self.store.setValue("mode", "folder")
        self._after_run.append(lambda: self.result.sample.image is not None
                               and self.tabs.setCurrentWidget(self.chip_page))   # an image run: the chip
        self.run()

    def _settings_changed(self):
        self.settings = replace(self.panel.settings(), data_dir=self.settings.data_dir)
        if self.auto_button.isChecked():
            self._debounce.start()

    def _gate_dragged(self, role: str, value: tuple):
        self.settings = replace(self.settings, **{f"gate_range_{role}_ps": value})   # a new object: views see it changed
        self.panel.set_settings(self.settings)
        self.run()

    def _slices_changed(self, values: dict):
        """Slices, window or combine changed on the Chip image tab."""
        settings = replace(self.settings, **values)
        if settings.process_slice not in ("sum", "each") and int(settings.process_slice) not in settings.slices:
            settings.process_slice = "sum"        # the slice being processed was removed
        self.settings = settings
        self.panel.set_settings(settings)
        self.run()

    def run(self):
        """Process in the background with the current settings; the window stays usable."""
        if not self.settings.data_dir:
            self.row_view.set_settings(self.settings)     # it has its own files
            return
        self._run_started = time.perf_counter()
        self._job.submit(self._compute, self.settings, self._clear_cache)
        self._clear_cache = False

    def _compute(self, settings, clear: bool):
        """Worker thread: only the pipeline, no widgets."""
        if clear:
            self.pipeline.clear()
        return self.pipeline.run(settings)

    def _run_done(self, ok: bool, value):
        after, self._after_run = self._after_run, []
        if not ok:                     # surface any failure without losing the window
            self.statusBar().showMessage(f"Could not run: {value}")
            return
        self.result = value
        elapsed = (time.perf_counter() - self._run_started) * 1000
        self._update_cards()
        self._dirty_pages = set(range(len(self.pages)))
        self._refresh_current_page()
        self.compare.update(self.result)          # draws only while its window is open
        message = (f"Done in {elapsed:.0f} ms  ·  reran from {self.result.rerun_from}  ·  "
                   f"{self.result.reference.n_scans} + {self.result.sample.n_scans} scans")
        if self.result.warnings:
            message += "  ·  ⚠ " + "; ".join(self.result.warnings)
        self.statusBar().showMessage(message)
        for action in after:
            action()

    def _refresh_current_page(self, *_):
        index = self.tabs.currentIndex()
        if self.result is not None and index in self._dirty_pages:
            self.pages[index].update(self.result)
            self._dirty_pages.discard(index)

    def _update_cards(self):
        r = self.result
        low, high = r.usable_range()
        unit = " slices" if r.sample.rows_as_scans else ""
        self.cards["Scans"].value.setText(f"{r.reference.name} {r.reference.n_scans} · "
                                          f"{r.sample.name} {r.sample.n_scans}{unit}")
        self.cards["Rotation"].value.setText(f"{r.rotation_deg:+.1f}°")
        self.cards["Usable range"].value.setText(f"{low:.0f}–{high:.0f} cm⁻¹")
        self.cards["Pixels replaced"].value.setText(
            f"{r.reference.n_spikes_replaced} · {r.sample.n_spikes_replaced}"
            if r.settings.remove_cosmic_rays and not r.sample.rows_as_scans else "off")
        from hdsfg_engine.drift import phase_drift
        import numpy as np
        drift = phase_drift(r.sample.analytic, r.sample.time_sec, r.sample.numbers,
                            max(1, r.sample.n_scans // 7),
                            max(1, round(10 / r.grid.wavenumber_step)), r.sample.prefix)
        wavenumber = r.grid.wavenumber_uniform
        band = (wavenumber >= low) & (wavenumber <= high)
        change = np.median(drift.phase_change_deg[band, -1])
        across = (f"{drift.labels[0]} to {drift.labels[-1]}" if r.sample.rows_as_scans
                  else f"in {drift.time_min[-1]:.0f} min")
        self.cards["Drift in sample run"].value.setText(f"{change:+.1f}° {across}")

    def export(self):
        if self.result is None:
            return
        default = str(Path(self.settings.data_dir).parent / "results" / "studio")
        folder = QFileDialog.getExistingDirectory(self, "Export results into", default)
        if not folder:
            return
        written = export_results(self.result, folder)
        self.statusBar().showMessage(f"Exported {len(written)} files to {folder}")

    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "settings.json", "Settings (*.json)")
        if path:
            self.settings.to_json(path)

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "Settings (*.json)")
        if not path:
            return
        loaded = Settings.from_json(path)
        if loaded.files and all(Path(f).is_file() for f in loaded.files):
            self.settings = loaded
            self.open_files(loaded.files, keep_settings=True)
        elif loaded.data_dir and Path(loaded.data_dir).is_dir():
            self.settings = loaded
            self.open_folder(loaded.data_dir)
        else:
            self.settings = replace(loaded, data_dir=self.settings.data_dir)
            self.panel.set_settings(self.settings)
            self.run()

    def _toggle_theme(self, dark: bool):
        self.store.setValue("theme", "dark" if dark else "light")
        QMessageBox.information(self, APP_NAME, "The theme changes the next time the app starts.")

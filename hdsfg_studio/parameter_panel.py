"""The settings sidebar, generated from ``hdsfg_engine.settings.PARAMETERS``.

Every ParameterSpec becomes one row: runs and choices as dropdowns, numbers as
spin boxes, switches as check boxes, ranges as two spin boxes. Rows are grouped
into collapsible sections. Changing anything emits ``changed`` with nothing
else to wire up, so a parameter added to the engine shows up here by itself.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QSizePolicy,
                               QSpinBox, QToolButton, QVBoxLayout, QWidget)

from hdsfg_engine.data import RunInfo
from hdsfg_engine.settings import (PARAMETERS, ParameterSpec, Settings, format_scan_list,
                                   parse_scan_list, spec_for)


class WheelGuard(QObject):
    """Spin boxes and dropdowns in a scrolling column take the mouse wheel only once clicked.

    Without this, scrolling the column with the wheel changes whatever number the pointer
    passes over (an exposure of 3 s became 21 s that way). An unclicked field passes the
    wheel to the column, which scrolls.
    """

    def __init__(self, scroll: QScrollArea):
        super().__init__(scroll)
        self._scroll = scroll

    def eventFilter(self, widget, event) -> bool:
        if event.type() == QEvent.Wheel and not widget.hasFocus():
            QApplication.sendEvent(self._scroll.verticalScrollBar(), event)
            return True
        return False


def guard_wheel(scroll: QScrollArea) -> None:
    """Install WheelGuard on every spin box and dropdown inside ``scroll``."""
    guard = WheelGuard(scroll)
    for widget in scroll.widget().findChildren(QAbstractSpinBox) + scroll.widget().findChildren(QComboBox):
        widget.setFocusPolicy(Qt.StrongFocus)
        widget.installEventFilter(guard)


def spec_label(name: str) -> str:
    return spec_for(name).label


class Section(QWidget):
    """A titled block whose content folds away under its header."""

    def __init__(self, title: str):
        super().__init__()
        self.header = QToolButton(text=title, checkable=True, checked=True)
        self.header.setObjectName("sectionHeader")
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.toggled.connect(self._toggle)

        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(4, 0, 0, 6)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(6)
        self.form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)

    def _toggle(self, open_: bool) -> None:
        self.header.setArrowType(Qt.DownArrow if open_ else Qt.RightArrow)
        self.body.setVisible(open_)


class RangeEdit(QWidget):
    changed = Signal()

    def __init__(self, spec: ParameterSpec):
        super().__init__()
        self.low, self.high = (self._spin(spec) for _ in range(2))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.low)
        dash = QLabel("–")
        layout.addWidget(dash)
        layout.addWidget(self.high)

    def _spin(self, spec: ParameterSpec) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(spec.minimum, spec.maximum)
        spin.setDecimals(spec.decimals)
        spin.setSingleStep(spec.step)
        spin.setKeyboardTracking(False)
        spin.setMinimumWidth(78)          # overrides the wide hint from the +/-20 range
        if spec.unit:
            spin.setSuffix(f" {spec.unit}")
        spin.valueChanged.connect(self.changed)
        return spin

    def value(self):
        return (self.low.value(), self.high.value())

    def set_value(self, value) -> None:
        for spin, number in zip((self.low, self.high), value):
            spin.blockSignals(True)
            spin.setValue(number)
            spin.blockSignals(False)


class ScanListEdit(QLineEdit):
    """Scan numbers as text, "1-130" or "1-10, 20, 40-50"; empty means all.

    Read when editing finishes (Return or leaving the field). Text that cannot
    be read turns the border red and keeps the previous selection.
    """
    changed = Signal()

    def __init__(self, spec: ParameterSpec):
        super().__init__()
        self.setPlaceholderText(spec.placeholder)
        self.setClearButtonEnabled(True)
        self._help = spec.help
        self._numbers: list[int] = []
        self.editingFinished.connect(self._read_text)

    def _read_text(self):
        try:
            numbers = parse_scan_list(self.text())
        except ValueError as error:
            self.setStyleSheet("border: 1px solid #C62828;")
            self.setToolTip(str(error))
            return
        self.setStyleSheet("")
        self.setToolTip(self._help)
        self.setText(format_scan_list(numbers))
        if numbers != self._numbers:
            self._numbers = numbers
            self.changed.emit()

    def value(self) -> list[int]:
        return list(self._numbers)

    def set_value(self, numbers) -> None:
        self._numbers = sorted(set(int(n) for n in numbers or []))
        self.setStyleSheet("")
        self.setText(format_scan_list(self._numbers))


class ParameterPanel(QScrollArea):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("parameters")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("parameterContent")
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(12, 6, 12, 12)
        self._layout.setSpacing(2)
        self.setWidget(content)

        self._settings = Settings()
        self._editors: dict[str, QWidget] = {}
        self._rows: dict[str, tuple[QLabel, QWidget]] = {}
        self._runs: list[RunInfo] = []
        self._show_advanced = False
        self._file_mode = False

        sections: dict[str, Section] = {}
        for spec in PARAMETERS:
            if spec.group not in sections:
                sections[spec.group] = Section(spec.group)
                self._layout.addWidget(sections[spec.group])
            editor = self._make_editor(spec)
            label = QLabel(spec.label)
            label.setProperty("class", "fieldLabel")
            label.setToolTip(spec.help)
            editor.setToolTip(spec.help)
            sections[spec.group].form.addRow(label, editor)
            self._editors[spec.name] = editor
            self._rows[spec.name] = (label, editor)

        self.run_info = QLabel()
        self.run_info.setObjectName("runInfo")
        self.run_info.setWordWrap(True)
        sections["Data"].form.addRow(self.run_info)
        guard_wheel(self)
        self._layout.addStretch(1)
        self._apply_advanced_visibility()
        self.set_file_mode(False)

    # ------------------------------------------------------------------ editors
    def _make_editor(self, spec: ParameterSpec) -> QWidget:
        if spec.kind in ("choice", "run", "slice"):
            editor = QComboBox()
            editor.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            editor.setMinimumContentsLength(10)
            for value, label in spec.choices:
                editor.addItem(label, value)
            editor.currentIndexChanged.connect(self._emit)
        elif spec.kind == "float":
            editor = QDoubleSpinBox()
            editor.setRange(spec.minimum, spec.maximum)
            editor.setDecimals(spec.decimals)
            editor.setSingleStep(spec.step)
            editor.setKeyboardTracking(False)
            if spec.unit:
                editor.setSuffix(f" {spec.unit}")
            editor.valueChanged.connect(self._emit)
        elif spec.kind == "int":
            editor = QSpinBox()
            editor.setRange(int(spec.minimum), int(spec.maximum))
            editor.setSingleStep(int(spec.step))
            editor.setKeyboardTracking(False)
            if spec.unit:
                editor.setSuffix(f" {spec.unit}")
            editor.valueChanged.connect(self._emit)
        elif spec.kind == "bool":
            editor = QCheckBox()
            editor.toggled.connect(self._emit)
        elif spec.kind == "range":
            editor = RangeEdit(spec)
            editor.changed.connect(self._emit)
        elif spec.kind == "scans":
            editor = ScanListEdit(spec)
            editor.changed.connect(self._emit)
        else:
            raise ValueError(f"unknown parameter kind {spec.kind}")
        return editor

    def _emit(self, *_):
        self._settings = self._read()
        for widget in self._rows["process_slice"]:
            widget.setEnabled(self._settings.image_data)
        self._update_run_info()
        self.changed.emit()

    # ------------------------------------------------------------------ public
    def set_runs(self, runs: list[RunInfo]) -> None:
        self._runs = runs
        for name in ("reference_name", "sample_name"):
            combo: QComboBox = self._editors[name]
            combo.blockSignals(True)
            combo.clear()
            for run in runs:
                combo.addItem(run.label, run.name)
            combo.blockSignals(False)
        self.set_settings(self._settings)

    def set_file_mode(self, on: bool) -> None:
        """Chosen files are both reference and sample: the run rows go grey, the file rows come on."""
        self._file_mode = on
        if on:
            for name in ("reference_name", "sample_name"):
                combo: QComboBox = self._editors[name]
                combo.blockSignals(True)
                combo.clear()
                combo.addItem("Chosen files", None)
                combo.blockSignals(False)
        for name in ("reference_name", "reference_file_numbers", "sample_name"):
            for widget in self._rows[name]:
                widget.setEnabled(not on)
        for spec in PARAMETERS:
            if spec.group == "Chosen files":
                for widget in self._rows[spec.name]:
                    widget.setEnabled(on)
        self._rows["sample_file_numbers"][0].setText("Files to use" if on else spec_label("sample_file_numbers"))
        self._editors["sample_file_numbers"].setPlaceholderText("all files" if on else "all scans")
        self._update_run_info()

    def _fill_slice_choices(self, settings: Settings) -> None:
        """The Process dropdown lists the current slices; greyed out without image data."""
        for spec in PARAMETERS:
            if spec.kind != "slice":
                continue
            combo: QComboBox = self._editors[spec.name]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Sum of slices", "sum")
            combo.addItem("Each slice as a scan", "each")
            for centre in sorted(settings.slices):
                combo.addItem(f"Slice px {centre}", str(centre))
            combo.blockSignals(False)
            for widget in self._rows[spec.name]:
                widget.setEnabled(settings.image_data)

    def set_settings(self, settings: Settings) -> None:
        self._settings = copy.deepcopy(settings)
        self._fill_slice_choices(settings)
        for spec in PARAMETERS:
            editor = self._editors[spec.name]
            value = getattr(settings, spec.name)
            editor.blockSignals(True)
            if isinstance(editor, QComboBox):
                index = editor.findData(value)
                if index >= 0:
                    editor.setCurrentIndex(index)
            elif isinstance(editor, (QDoubleSpinBox, QSpinBox)):
                editor.setValue(value)
            elif isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, (RangeEdit, ScanListEdit)):
                editor.set_value(value)
            editor.blockSignals(False)
        self._update_run_info()

    def settings(self) -> Settings:
        return copy.deepcopy(self._settings)

    def set_show_advanced(self, show: bool) -> None:
        self._show_advanced = show
        self._apply_advanced_visibility()

    def _apply_advanced_visibility(self) -> None:
        for spec in PARAMETERS:
            if spec.advanced:
                for widget in self._rows[spec.name]:
                    widget.setVisible(self._show_advanced)

    def _read(self) -> Settings:
        values = {}
        for spec in PARAMETERS:
            editor = self._editors[spec.name]
            if isinstance(editor, QComboBox):
                values[spec.name] = editor.currentData()
            elif isinstance(editor, (QDoubleSpinBox, QSpinBox)):
                values[spec.name] = editor.value()
            elif isinstance(editor, QCheckBox):
                values[spec.name] = editor.isChecked()
            elif isinstance(editor, (RangeEdit, ScanListEdit)):
                values[spec.name] = editor.value()
        # one edge-width control drives both channels
        values["gate_edges_reference_ps"] = values["gate_edges_sample_ps"]
        values = {k: v for k, v in values.items() if v is not None}
        return replace(self._settings, **values)

    def _update_run_info(self) -> None:
        if self._file_mode:
            s = self._settings
            n = len(s.files)
            used = len(set(s.sample_file_numbers) & set(range(1, n + 1))) if s.sample_file_numbers else n
            self.run_info.setText(f"{used} of {n} chosen files × {s.file_exposure_sec * s.file_frames:g} s, "
                                  "used as both reference and sample")
            return
        by_name = {run.name: run for run in self._runs}
        lines = []
        for role, name, chosen in (("Reference", self._settings.reference_name,
                                    self._settings.reference_file_numbers),
                                   ("Sample", self._settings.sample_name,
                                    self._settings.sample_file_numbers)):
            run = by_name.get(name)
            if run is None or run.exposure_sec is None:
                continue
            total = len(run.files)
            if chosen:
                loaded = len(set(chosen) & set(run.numbers))
                count = f"{loaded} of {total} scans"
            else:
                count = f"{total} scans"
            lines.append(f"{role}: {count} × {run.exposure_sec:g} s")
        self.run_info.setText("   ".join(lines))

"""Light and dark themes: one palette for Qt widgets, one for pyqtgraph."""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Theme:
    name: str
    window: str
    panel: str
    base: str
    text: str
    muted: str
    border: str
    accent: str
    plot_background: str
    plot_foreground: str
    imag: str            # Im chi2 and "sample"
    real: str            # Re chi2 and "reference"
    band_alpha: int
    weak_region: str
    ok: str
    warn: str


LIGHT = Theme("light", window="#F4F4F2", panel="#FAFAF8", base="#FFFFFF", text="#1F1F1D",
              muted="#6B6A65", border="#DAD8D0", accent="#2F6FBA",
              plot_background="#FFFFFF", plot_foreground="#3D3D3A",
              imag="#2F6FBA", real="#C8552B", band_alpha=55, weak_region="#9C9A92",
              ok="#2E7D32", warn="#B26A00")

DARK = Theme("dark", window="#1E1E1D", panel="#252524", base="#2B2B2A", text="#ECEBE6",
             muted="#A3A29C", border="#3B3B39", accent="#6CA6E8",
             plot_background="#1B1B1A", plot_foreground="#CFCDC6",
             imag="#6CA6E8", real="#EE8A5E", band_alpha=70, weak_region="#77756F",
             ok="#7CC47F", warn="#E7A54A")


def apply(app: QApplication, theme: Theme) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(theme.window))
    palette.setColor(QPalette.WindowText, QColor(theme.text))
    palette.setColor(QPalette.Base, QColor(theme.base))
    palette.setColor(QPalette.AlternateBase, QColor(theme.panel))
    palette.setColor(QPalette.Text, QColor(theme.text))
    palette.setColor(QPalette.Button, QColor(theme.panel))
    palette.setColor(QPalette.ButtonText, QColor(theme.text))
    palette.setColor(QPalette.Highlight, QColor(theme.accent))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ToolTipBase, QColor(theme.base))
    palette.setColor(QPalette.ToolTipText, QColor(theme.text))
    palette.setColor(QPalette.PlaceholderText, QColor(theme.muted))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(theme.muted))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(theme.muted))
    app.setPalette(palette)

    app.setStyleSheet(f"""
        QMainWindow, QDialog {{ background: {theme.window}; }}
        QToolBar {{ background: {theme.panel}; border: none; border-bottom: 1px solid {theme.border};
                    spacing: 6px; padding: 4px 8px; }}
        QToolBar QToolButton {{ padding: 4px 10px; border-radius: 6px; }}
        QToolBar QToolButton:hover {{ background: {theme.border}; }}
        QToolButton#primary {{ background: {theme.accent}; color: white; font-weight: 600; }}
        QToolButton#primary:hover {{ background: {theme.accent}; }}
        QScrollArea#parameters, QWidget#parameterContent {{ background: {theme.panel}; border: none; }}
        QWidget#sidebar {{ background: {theme.panel}; border-right: 1px solid {theme.border}; }}
        QToolButton#sectionHeader {{ border: none; font-weight: 600; color: {theme.text};
                                     padding: 8px 2px 4px 2px; text-align: left; }}
        QLabel.fieldLabel {{ color: {theme.muted}; }}
        QLabel#runInfo {{ color: {theme.muted}; font-size: 11px; }}
        QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
            background: {theme.base}; border: 1px solid {theme.border}; border-radius: 6px;
            padding: 3px 6px; min-height: 20px; }}
        QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: {theme.muted}; }}
        QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border-color: {theme.accent}; }}
        QTabWidget::pane {{ border: none; background: {theme.base}; }}
        QTabBar::tab {{ background: transparent; padding: 7px 14px; color: {theme.muted};
                        border-bottom: 2px solid transparent; }}
        QTabBar::tab:selected {{ color: {theme.text}; border-bottom: 2px solid {theme.accent}; }}
        QTabBar::tab:hover {{ color: {theme.text}; }}
        QFrame#card {{ background: {theme.panel}; border: 1px solid {theme.border}; border-radius: 8px; }}
        QLabel#cardTitle {{ color: {theme.muted}; font-size: 11px; }}
        QLabel#cardValue {{ font-size: 15px; font-weight: 600; }}
        QStatusBar {{ background: {theme.panel}; border-top: 1px solid {theme.border}; color: {theme.muted}; }}
        QCheckBox {{ spacing: 6px; }}
        QWidget#chipControls {{ background: {theme.panel}; }}
        QLabel#chipSection {{ color: {theme.muted}; font-size: 11px; font-weight: 600;
                             letter-spacing: 1px; padding-top: 10px; }}
        QLabel#chipBanner {{ background: {theme.panel}; border: 1px solid {theme.warn}; border-radius: 6px;
                            color: {theme.text}; padding: 6px 10px; }}
        QPushButton#segment {{ border: 1px solid {theme.border}; padding: 4px 12px; background: {theme.base}; }}
        QPushButton#segment:checked {{ background: {theme.accent}; color: white; border-color: {theme.accent}; }}
        QPushButton#accent {{ background: {theme.accent}; color: white; border-radius: 5px; padding: 4px 12px; }}
        QPushButton#sliceChip {{ background: {theme.base}; border: 1px solid {theme.border};
                                border-radius: 12px; padding: 3px 10px; font-weight: 600; }}
    """)

    pg.setConfigOptions(antialias=True, background=theme.plot_background,
                        foreground=theme.plot_foreground)

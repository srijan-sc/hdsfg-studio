"""Render every tab of the app to PNG without showing a window (for checking layout).

    .venv/bin/python tools/screenshot.py <data folder> <output folder>
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from hdsfg_studio.main_window import MainWindow  # noqa: E402

app = QApplication(sys.argv[:1])
window = MainWindow(sys.argv[1])
window.resize(1440, 900)
window.show()
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
for index in range(window.tabs.count()):
    window.tabs.setCurrentIndex(index)
    app.processEvents()
    window.grab().save(os.path.join(out, f"tab{index}_{window.pages[index].title.replace(' ', '_')}.png"))
print("saved", window.tabs.count(), "screenshots;", window.statusBar().currentMessage())

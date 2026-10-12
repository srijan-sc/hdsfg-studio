"""Start HDSFG Studio:  python -m hdsfg_studio [data folder]"""

import sys

from PySide6.QtWidgets import QApplication

from .main_window import APP_NAME, MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("HDSFG")
    window = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

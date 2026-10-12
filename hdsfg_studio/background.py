"""Work that runs in a worker thread, so the window never freezes, and what is running now.

A ``Job`` runs one function at a time in a thread. Asking again while it runs keeps only the
newest request, which starts when the current one ends; the older result is then dropped, so
a plot never flashes stale numbers. The result is handed back on the window's thread.

``activity()`` lists the jobs running now; the main window shows them in the status bar.
"""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal


class Activity(QObject):
    changed = Signal(list)           # names of the jobs running now

    def __init__(self):
        super().__init__()
        self._running: dict[str, int] = {}

    def begin(self, name: str) -> None:
        self._running[name] = self._running.get(name, 0) + 1
        self.changed.emit(self.running())

    def end(self, name: str) -> None:
        self._running[name] -= 1
        if not self._running[name]:
            del self._running[name]
        self.changed.emit(self.running())

    def running(self) -> list[str]:
        return list(self._running)


_ACTIVITY: Activity | None = None


def activity() -> Activity:
    global _ACTIVITY
    if _ACTIVITY is None:
        _ACTIVITY = Activity()
    return _ACTIVITY


class Job(QObject):
    """``submit(function, *args)`` runs it in a thread; ``on_done(ok, value)`` gets the result
    (``ok`` False: ``value`` is the exception) on the window's thread."""
    _finished = Signal(bool, object)

    def __init__(self, name: str, on_done, parent: QObject | None = None):
        super().__init__(parent)
        self.name = name
        self._on_done = on_done
        self._pending = None
        self.busy = False
        self._finished.connect(self._deliver)

    def submit(self, function, *args) -> None:
        self._pending = (function, args)
        if not self.busy:
            self._start()

    def _start(self) -> None:
        function, args = self._pending
        self._pending = None
        self.busy = True
        activity().begin(self.name)
        threading.Thread(target=self._work, args=(function, args), daemon=True).start()

    def _work(self, function, args) -> None:
        try:
            ok, value = True, function(*args)
        except Exception as error:          # shown by on_done; the window keeps working
            traceback.print_exc()
            ok, value = False, error
        self._finished.emit(ok, value)

    def _deliver(self, ok: bool, value) -> None:
        self.busy = False
        activity().end(self.name)
        if self._pending is not None:       # asked again meanwhile: run the newest, drop this one
            self._start()
            return
        self._on_done(ok, value)

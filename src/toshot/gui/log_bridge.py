from __future__ import annotations

import logging
import queue
import sys
from typing import TextIO


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATE = "%H:%M:%S"


class QueueLogHandler(logging.Handler):
    """Push every formatted log record into a queue for the GUI."""

    def __init__(self, q: queue.Queue[str], *, level: int = logging.DEBUG) -> None:
        super().__init__(level=level)
        self.q = q
        self.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Keep exact multiline messages (tracebacks)
            for line in msg.splitlines() or [msg]:
                self.q.put(line)
        except Exception:
            self.handleError(record)


class QueueWriter:
    """File-like object that mirrors writes into the log queue (stdout/stderr)."""

    def __init__(self, q: queue.Queue[str], prefix: str = "") -> None:
        self.q = q
        self.prefix = prefix
        self._buf = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip() or line == "":
                self.q.put(f"{self.prefix}{line}" if self.prefix else line)
        return len(data)

    def flush(self) -> None:
        if self._buf:
            self.q.put(f"{self.prefix}{self._buf}" if self.prefix else self._buf)
            self._buf = ""

    def isatty(self) -> bool:
        return False


class StdRedirect:
    """Context manager: redirect stdout/stderr into the log queue."""

    def __init__(self, q: queue.Queue[str]) -> None:
        self.q = q
        self._out: TextIO | None = None
        self._err: TextIO | None = None

    def __enter__(self) -> StdRedirect:
        self._out, self._err = sys.stdout, sys.stderr
        sys.stdout = QueueWriter(self.q, prefix="")  # type: ignore[assignment]
        sys.stderr = QueueWriter(self.q, prefix="")  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            sys.stdout.flush()  # type: ignore[union-attr]
            sys.stderr.flush()  # type: ignore[union-attr]
        except Exception:
            pass
        if self._out is not None:
            sys.stdout = self._out
        if self._err is not None:
            sys.stderr = self._err


def install_gui_logging(q: queue.Queue[str]) -> QueueLogHandler:
    """Attach a DEBUG handler to the root logger for the lifetime of the GUI."""
    handler = QueueLogHandler(q, level=logging.DEBUG)
    root = logging.getLogger()
    # Avoid duplicate handlers if GUI reopened
    for h in list(root.handlers):
        if isinstance(h, QueueLogHandler):
            root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    # Keep third-party noise manageable but visible
    logging.getLogger("toshot").setLevel(logging.DEBUG)
    logging.getLogger("ultralytics").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return handler

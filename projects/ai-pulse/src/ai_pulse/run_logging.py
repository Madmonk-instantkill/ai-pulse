"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


class _Tee:
    """Stands in for sys.stdout / sys.stderr: everything written goes to
    the real console AND to the run's log file, so the log holds the whole
    run, including the CrewAI agent output."""

    def __init__(self, console, log_file):
        self._console = console
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._console.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._console.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        """Always False so colour codes and live-redraw tricks stay out of
        the log file."""
        return False

    def __getattr__(self, name):
        """Anything else a library asks of a stream (encoding, fileno...)
        is answered by the real console."""
        return getattr(self._console, name)


def _safe_stream(stream):
    """Return a stream that will not crash on characters the console
    cannot show (for example Greek letters under a Windows code page).
    Scheduled runs have no console at all, so a missing stream becomes a
    null one."""
    if stream is None:
        return open(os.devnull, "w", encoding="utf-8")
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")
    return stream


@contextmanager
def capture_run_log(started_at: datetime):
    """Copy everything the run prints (and everything sent to the logging
    module) into logs/ai_pulse_<date>_<time>.log while still showing it on
    the console. Yields the log file's path.

    The file is closed when the block ends, so the caller can delete it
    afterwards (Windows will not delete an open file). Whether to delete
    it is the caller's decision: this only collects."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ai_pulse_{started_at:%Y-%m-%d_%H%M%S}.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(_safe_stream(original_stdout), log_file)
        sys.stderr = _Tee(_safe_stream(original_stderr), log_file)
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
        )
        try:
            yield log_path
        finally:
            logging.getLogger().handlers.clear()
            sys.stdout, sys.stderr = original_stdout, original_stderr

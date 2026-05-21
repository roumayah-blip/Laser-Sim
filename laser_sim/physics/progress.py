"""Optional progress reporting for long CPA solver passes."""

from __future__ import annotations

import json
import time
from typing import Callable, Protocol

ProgressCallback = Callable[[float, str], None]


class ProgressReporter(Protocol):
    def __call__(self, fraction: float, message: str) -> None: ...


def noop_progress(_fraction: float, _message: str) -> None:
    pass


def emit_progress(
    callback: ProgressCallback | None,
    fraction: float,
    message: str,
) -> None:
    if callback is None:
        return
    callback(float(min(max(fraction, 0.0), 1.0)), message)


def file_progress_writer(path: str) -> ProgressCallback:
    """Write progress to a JSON file for a parent process (isolated Taichi worker)."""

    def _write(fraction: float, message: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"frac": float(min(max(fraction, 0.0), 1.0)), "msg": message, "t": time.time()},
                    f,
                )
        except OSError:
            pass

    return _write


def read_progress_file(path: str) -> tuple[float, str] | None:
    """Read last progress snapshot from worker; None if missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data["frac"]), str(data.get("msg", ""))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

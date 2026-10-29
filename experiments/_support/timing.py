"""Prediction timing primitive."""
from __future__ import annotations
import gc
import time
from collections.abc import Callable
from typing import Any


def timed_call(callable_: Callable[[], Any]) -> tuple[int, Any]:
    """Time only ``callable_`` while restoring the caller's GC state."""
    was_enabled = gc.isenabled()
    gc.disable()
    start = time.perf_counter_ns()
    try:
        value = callable_()
    finally:
        elapsed = time.perf_counter_ns() - start
        if was_enabled:
            gc.enable()
    return elapsed, value

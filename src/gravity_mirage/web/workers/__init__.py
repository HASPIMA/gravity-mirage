"""
Simple in-memory job queue for GIF exports.

This is intentionally lightweight and suitable for development.
Jobs are stored in `JOBS` and processed by a background worker
thread that writes the generated GIF to `exports/`.
"""

import queue as _queue
from typing import Any

JOB_QUEUE: _queue.Queue = _queue.Queue()
JOBS: dict[str, dict[str, Any]] = {}

__all__ = ["JOBS", "JOB_QUEUE"]

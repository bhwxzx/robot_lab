#!/usr/bin/env python3
"""Shared, host-local safety primitives for training and evaluation executors."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def gpu_lock_path(gpu_index: int) -> Path:
    """Return one user-scoped lock path shared by every skill executor."""
    return Path(tempfile.gettempdir()) / (
        f"monitor-tune-isaaclab-{os.getuid()}-gpu-{gpu_index}.lock"
    )

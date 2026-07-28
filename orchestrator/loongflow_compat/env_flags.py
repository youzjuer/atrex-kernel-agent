"""Shared environment parsing for LoongFlow compatibility features."""

from __future__ import annotations

import os


_TRUTHY = frozenset({"1", "true", "yes", "on"})


def enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY

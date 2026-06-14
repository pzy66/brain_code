"""Runtime bootstrap helpers shared by scripts and tests."""

from __future__ import annotations

import os

from .paths import ensure_runtime_import_paths


def configure_qt_offscreen() -> None:
    """Default Qt to offscreen rendering for headless test runs."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def bootstrap_runtime(*, qt_offscreen: bool = False) -> None:
    """Apply the minimum compatibility bootstrap for local entrypoints."""

    if qt_offscreen:
        configure_qt_offscreen()
    ensure_runtime_import_paths()

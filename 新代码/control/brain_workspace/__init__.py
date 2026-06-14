"""Shared helpers for the local brain-code workspace."""

from __future__ import annotations

from .paths import (
    BRAIN_CODE_ROOT,
    HYBRID_CONTROLLER_DIR,
    MI_COLLECTION_DIR,
    MI_PROJECT_DIR,
    MI_SHARED_DIR,
    SSVEP_PROJECT_DIR,
    WORKSPACE_ROOT,
    ensure_runtime_import_paths,
)

__all__ = [
    "BRAIN_CODE_ROOT",
    "WORKSPACE_ROOT",
    "MI_PROJECT_DIR",
    "MI_COLLECTION_DIR",
    "MI_SHARED_DIR",
    "SSVEP_PROJECT_DIR",
    "HYBRID_CONTROLLER_DIR",
    "ensure_runtime_import_paths",
]

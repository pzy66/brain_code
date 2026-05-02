"""Path constants used by the unified collection workflow."""

from __future__ import annotations

from pathlib import Path

from brain_workspace.paths import (
    BRAIN_CODE_ROOT,
    HYBRID_CONTROLLER_DIR,
    MI_COLLECTION_DIR,
    MI_DATASET_DIR,
    MI_PROJECT_DIR,
    MI_SHARED_DIR,
    SSVEP_DATASET_DIR,
    SSVEP_PROJECT_DIR,
    UNIFIED_COLLECTION_INDEX_PATH,
    WORKSPACE_ROOT,
    resolve_brain_code_path,
)

DEFAULT_MI_OUTPUT_ROOT = MI_DATASET_DIR
DEFAULT_SSVEP_DATASET_DIR = SSVEP_DATASET_DIR


def resolve_ssvep_dataset_dir(value: str | Path | None) -> Path:
    return resolve_brain_code_path(
        value,
        base=SSVEP_PROJECT_DIR,
        default=DEFAULT_SSVEP_DATASET_DIR,
        purpose="SSVEP dataset dir",
    )

__all__ = [
    "BRAIN_CODE_ROOT",
    "WORKSPACE_ROOT",
    "MI_PROJECT_DIR",
    "MI_COLLECTION_DIR",
    "MI_SHARED_DIR",
    "MI_DATASET_DIR",
    "DEFAULT_MI_OUTPUT_ROOT",
    "SSVEP_PROJECT_DIR",
    "SSVEP_DATASET_DIR",
    "HYBRID_CONTROLLER_DIR",
    "UNIFIED_COLLECTION_INDEX_PATH",
    "DEFAULT_SSVEP_DATASET_DIR",
    "resolve_ssvep_dataset_dir",
]

"""Canonical filesystem paths for the brain-code repository."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Iterable

BRAIN_CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BRAIN_CODE_ROOT.parent


def _resolve_data_root() -> Path:
    raw = os.environ.get("BRAIN_DATA_ROOT", "").strip()
    if not raw:
        return BRAIN_CODE_ROOT / "datasets"
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = BRAIN_CODE_ROOT / candidate
    return candidate.resolve()


DATASETS_ROOT = _resolve_data_root()

MI_PROJECT_DIR = BRAIN_CODE_ROOT / "01_MI" / "mi_classifier_latest"
MI_COLLECTION_DIR = MI_PROJECT_DIR / "code" / "collection"
MI_SHARED_DIR = MI_PROJECT_DIR / "code" / "shared"
MI_DATASET_DIR = DATASETS_ROOT / "MI"
SSVEP_PROJECT_DIR = BRAIN_CODE_ROOT / "02_SSVEP"
SSVEP_DATASET_DIR = DATASETS_ROOT / "SSVEP"
VISION_DATASET_DIR = DATASETS_ROOT / "vision"
PROFILE_DATASET_DIR = DATASETS_ROOT / "profiles"
SSVEP_PROFILE_DIR = PROFILE_DATASET_DIR / "SSVEP"
HYBRID_PROFILE_DIR = PROFILE_DATASET_DIR / "hybrid_controller"
HYBRID_SSVEP_PROFILE_DIR = HYBRID_PROFILE_DIR / "ssvep_profiles"
HYBRID_CONTROLLER_DIR = BRAIN_CODE_ROOT / "hybrid_controller"
UNIFIED_COLLECTION_INDEX_PATH = BRAIN_CODE_ROOT / "artifacts" / "unified_collection_index.csv"

RUNTIME_IMPORT_PATHS = (
    SSVEP_PROJECT_DIR,
    MI_COLLECTION_DIR,
    MI_SHARED_DIR,
)


def ensure_sys_path(paths: Iterable[Path], *, prepend: bool = True) -> list[Path]:
    """Add runtime-only source roots to sys.path and return the paths added."""

    added: list[Path] = []
    for path in paths:
        resolved = Path(path).resolve()
        value = str(resolved)
        if value in sys.path:
            continue
        if prepend:
            sys.path.insert(0, value)
        else:
            sys.path.append(value)
        added.append(resolved)
    return added


def ensure_runtime_import_paths() -> list[Path]:
    """Make legacy MI and SSVEP source roots importable for compatibility."""

    return ensure_sys_path(RUNTIME_IMPORT_PATHS)


def path_is_relative_to(path: Path, root: Path) -> bool:
    """Return whether path resolves inside root without requiring it to exist."""

    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def resolve_brain_code_path(value: str | Path | None, *, base: Path, default: Path, purpose: str) -> Path:
    """Resolve a user path and require it to remain inside brain_code."""

    raw = "" if value is None else str(value).strip()
    candidate = Path(raw).expanduser() if raw else Path(default)
    if not candidate.is_absolute():
        candidate = Path(base) / candidate
    resolved = candidate.resolve()
    if not path_is_relative_to(resolved, BRAIN_CODE_ROOT):
        raise ValueError(f"{purpose} must be inside brain_code: {resolved}")
    return resolved


def resolve_data_path(value: str | Path | None, *, base: Path, default: Path, purpose: str) -> Path:
    """Resolve a dataset/profile path.

    Relative paths are scoped under DATASETS_ROOT. Absolute paths are accepted
    for explicit local copies on external disks or pytest temporary roots.
    """

    raw = "" if value is None else str(value).strip()
    candidate = Path(raw).expanduser() if raw else Path(default)
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (Path(base) / candidate).resolve()
    if not path_is_relative_to(resolved, DATASETS_ROOT):
        raise ValueError(f"{purpose} must be inside BRAIN_DATA_ROOT/datasets root: {resolved}")
    return resolved


def required_workspace_paths() -> tuple[Path, ...]:
    """Paths expected by the current runtime entrypoints."""

    return (
        BRAIN_CODE_ROOT,
        MI_PROJECT_DIR,
        MI_COLLECTION_DIR,
        MI_SHARED_DIR,
        SSVEP_PROJECT_DIR,
        HYBRID_CONTROLLER_DIR,
        DATASETS_ROOT,
    )


def missing_required_paths() -> list[Path]:
    """Return required paths that are missing from this checkout."""

    return [path for path in required_workspace_paths() if not path.exists()]

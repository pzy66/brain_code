from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from brain_workspace.paths import SSVEP_DATASET_DIR, SSVEP_PROFILE_DIR

from .dataset import load_collection_dataset
from .fbcca_local_opt import (
    DEFAULT_FBCCA_LOCAL_SEARCH_PRESET,
    FBCCALocalOptConfig,
    run_fbcca_local_opt,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATASET_ROOT = SSVEP_DATASET_DIR
DEFAULT_FBCCA_BASE_PROFILE_PATH = SSVEP_PROFILE_DIR / "fbcca_base_profile.json"
DEFAULT_FBCCA_BASE_REPORT_PATH = PROJECT_DIR / "artifacts" / "runs" / "local" / "fbcca_base_profile_report.json"
EXPECTED_FBCCA_BASE_FREQS = (8.0, 10.0, 12.0, 15.0)
DEFAULT_FBCCA_BASE_PROFILE_TASK = "fbcca-base-profile-opt"


@dataclass(frozen=True)
class FBCCABaseProfileOptConfig:
    dataset_manifests: tuple[Path, ...] = ()
    dataset_root: Path = DEFAULT_BASE_DATASET_ROOT
    output_profile_path: Path = DEFAULT_FBCCA_BASE_PROFILE_PATH
    report_path: Path = DEFAULT_FBCCA_BASE_REPORT_PATH
    search_preset: str = DEFAULT_FBCCA_LOCAL_SEARCH_PRESET
    compute_backend: str = "auto"
    organize_report_dir: bool = False


def discover_fbcca_base_dataset_manifests(dataset_root: Path = DEFAULT_BASE_DATASET_ROOT) -> tuple[Path, ...]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        return ()
    return tuple(sorted(path.resolve() for path in root.rglob("session_manifest.json")))


def _normalize_manifest_paths(paths: Sequence[Path | str]) -> tuple[Path, ...]:
    dedup: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        path = Path(item).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return tuple(dedup)


def validate_fbcca_base_dataset_manifests(
    manifests: Sequence[Path | str],
    *,
    expected_freqs: Sequence[float] = EXPECTED_FBCCA_BASE_FREQS,
) -> tuple[Path, ...]:
    paths = _normalize_manifest_paths(manifests)
    if not paths:
        raise ValueError("fbcca base profile optimization found no dataset manifests")
    expected = tuple(float(item) for item in expected_freqs)
    valid: list[Path] = []
    errors: list[str] = []
    for path in paths:
        try:
            dataset = load_collection_dataset(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        freqs = tuple(float(item) for item in dataset.freqs)
        if freqs != expected:
            errors.append(f"{path}: freqs={freqs}, expected={expected}")
            continue
        valid.append(path)
    if errors:
        raise ValueError("fbcca base profile datasets rejected: " + "; ".join(errors))
    if not valid:
        raise ValueError(f"fbcca base profile optimization found no {expected} dataset manifests")
    return tuple(valid)


def run_fbcca_base_profile_opt(
    config: FBCCABaseProfileOptConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    requested = tuple(config.dataset_manifests or discover_fbcca_base_dataset_manifests(config.dataset_root))
    manifests = validate_fbcca_base_dataset_manifests(requested)
    local_config = FBCCALocalOptConfig(
        dataset_manifest_session1=manifests[0],
        dataset_manifests=manifests,
        output_profile_path=Path(config.output_profile_path),
        report_path=Path(config.report_path),
        search_preset=str(config.search_preset or DEFAULT_FBCCA_LOCAL_SEARCH_PRESET),
        compute_backend=str(config.compute_backend or "auto"),
        organize_report_dir=bool(config.organize_report_dir),
    )
    payload = run_fbcca_local_opt(local_config, log_fn=log_fn, progress_fn=progress_fn)
    return {
        **dict(payload),
        "task": DEFAULT_FBCCA_BASE_PROFILE_TASK,
        "base_profile_task": DEFAULT_FBCCA_BASE_PROFILE_TASK,
        "dataset_manifests": [str(path) for path in manifests],
        "config": asdict(config),
    }

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.run_artifacts import deployed_profile_paths


def _copy_tree(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    except Exception:
        if sys.platform.startswith("win"):
            destination.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                ["robocopy", str(source), str(destination), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode > 7:
                raise RuntimeError(
                    f"robocopy failed for {source} -> {destination}: {completed.stdout}\n{completed.stderr}"
                )
        else:
            raise
    try:
        return sum(1 for _ in destination.rglob("*"))
    except Exception:
        return 0


def _copy_children(source_dir: Path, destination_dir: Path) -> list[str]:
    copied: list[str] = []
    if not source_dir.exists():
        return copied
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(source_dir.iterdir()):
        target = destination_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        copied.append(item.name)
    return copied


def _write_profile_index(destination_root: Path, payload: dict[str, Any]) -> None:
    paths = deployed_profile_paths(destination_root)
    paths["root_dir"].mkdir(parents=True, exist_ok=True)
    paths["profile_index"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_legacy_profile_root(explicit_source: Optional[Path] = None) -> Path:
    candidates = []
    if explicit_source is not None:
        candidates.append(Path(explicit_source).expanduser().resolve())
    candidates.extend(
        [
            PROJECT_DIR / "2026-04_async_fbcca_idle_decoder" / "profiles",
            PROJECT_DIR / "_archive" / "legacy_2026-04_async_fbcca_idle_decoder" / "profiles",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("legacy profile root not found under current 02_SSVEP layout")


def migrate_legacy_artifacts(*, source_root: Optional[Path] = None) -> dict[str, Any]:
    legacy_root = resolve_legacy_profile_root(source_root)
    artifact_root = PROJECT_DIR / "artifacts"
    datasets_root = artifact_root / "datasets"
    deployed_root = artifact_root / "deployed_profiles"
    legacy_runs_root = artifact_root / "runs" / "_legacy_imported" / legacy_root.parent.name
    dev_smoke_root = artifact_root / "dev_smoke"

    copied_datasets = _copy_children(legacy_root / "datasets", datasets_root)
    copied_smoke_dirs: list[str] = []
    for item in sorted(legacy_root.iterdir()):
        if not item.is_dir():
            continue
        if item.name == "datasets" or not item.name.startswith("datasets_"):
            continue
        target = dev_smoke_root / item.name
        shutil.copytree(item, target, dirs_exist_ok=True)
        copied_smoke_dirs.append(item.name)

    imported_reports = _copy_tree(legacy_root / "reports", legacy_runs_root / "reports")
    imported_server_runs = _copy_tree(legacy_root / "server_runs", legacy_runs_root / "server_runs")
    imported_server_profiles = _copy_tree(legacy_root / "server_profiles", legacy_runs_root / "server_profiles")

    deployed_paths = deployed_profile_paths(deployed_root)
    copied_profiles: dict[str, str] = {}
    profile_json = legacy_root / "default_profile.json"
    profile_v2_json = legacy_root / "default_profile_v2.json"
    if profile_json.exists():
        deployed_paths["root_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile_json, deployed_paths["profile_json"])
        copied_profiles["profile_json"] = str(deployed_paths["profile_json"])
    if profile_v2_json.exists():
        deployed_paths["root_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile_v2_json, deployed_paths["profile_v2_json"])
        copied_profiles["profile_v2_json"] = str(deployed_paths["profile_v2_json"])
    if copied_profiles:
        _write_profile_index(
            deployed_root,
            {
                "source": "legacy_migration",
                "legacy_root": str(legacy_root),
                "published_profile_json": copied_profiles.get("profile_json", ""),
                "published_profile_v2_json": copied_profiles.get("profile_v2_json", ""),
            },
        )

    summary = {
        "legacy_root": str(legacy_root),
        "artifact_root": str(artifact_root),
        "copied_datasets": copied_datasets,
        "copied_smoke_dirs": copied_smoke_dirs,
        "imported_reports_count": int(imported_reports),
        "imported_server_runs_count": int(imported_server_runs),
        "imported_server_profiles_count": int(imported_server_profiles),
        "copied_profiles": copied_profiles,
    }
    summary_path = artifact_root / "migration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate legacy 2026-04 SSVEP artifacts into the new artifact layout.")
    parser.add_argument("--source-root", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = migrate_legacy_artifacts(source_root=args.source_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

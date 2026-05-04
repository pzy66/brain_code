from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence

from brain_workspace.bootstrap import ensure_runtime_import_paths
from brain_workspace.environment import build_environment_report, format_environment_report
from brain_workspace.paths import (
    BRAIN_CODE_ROOT,
    DATASETS_ROOT,
    HYBRID_CONTROLLER_DIR,
    MI_DATASET_DIR,
    PROFILE_DATASET_DIR,
    SSVEP_DATASET_DIR,
    VISION_DATASET_DIR,
)


def _print_asset_summary() -> None:
    print(f"brain_code_root={BRAIN_CODE_ROOT}")
    print(f"datasets_root={DATASETS_ROOT}")
    print(f"mi_dataset_dir={MI_DATASET_DIR}")
    print(f"ssvep_dataset_dir={SSVEP_DATASET_DIR}")
    print(f"vision_dataset_dir={VISION_DATASET_DIR}")
    print(f"profile_dataset_dir={PROFILE_DATASET_DIR}")
    vision_model = VISION_DATASET_DIR / "models" / "best.pt"
    if not vision_model.exists():
        print(f"missing_optional_asset[vision_model]={vision_model}")


def diagnose(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the brain-code checkout, Python modules, and local asset roots.")
    parser.add_argument("--strict", action="store_true", help="return non-zero if required paths or optional modules are missing")
    args = parser.parse_args(list(argv or ()))
    report = build_environment_report()
    print(format_environment_report(report))
    _print_asset_summary()
    if args.strict and (report.missing_paths or report.missing_modules):
        return 1
    return 0


def _run_unified_collection(argv: Sequence[str]) -> int:
    from unified_collection.app import main as unified_main

    return int(unified_main(list(argv)))


def _run_ssvep_launcher(argv: Sequence[str]) -> int:
    ensure_runtime_import_paths()
    from apps.launcher_ui import main as ssvep_main

    return int(ssvep_main(list(argv)))


def _run_hybrid(argv: Sequence[str], *, simulate: bool) -> int:
    from hybrid_controller.app import main as hybrid_main

    base_args: list[str] = []
    if simulate:
        base_args.extend(
            [
                "--robot-mode",
                "fake",
                "--vision-mode",
                "sim",
                "--move-source",
                "sim",
                "--decision-source",
                "sim",
                "--timing-profile",
                "fast",
            ]
        )
    return int(hybrid_main([*base_args, *argv]))


def _run_mi_collection(argv: Sequence[str]) -> int:
    ensure_runtime_import_paths()
    target = BRAIN_CODE_ROOT / "01_MI" / "mi_classifier_latest" / "run_01_collection_only.py"
    if argv:
        sys.argv = [str(target), *argv]
    runpy.run_path(str(target), run_name="__main__")
    return 0


def launch(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch a brain-code application entrypoint.")
    parser.add_argument(
        "--target",
        choices=("unified", "mi", "ssvep", "hybrid"),
        default="unified",
        help="application to launch",
    )
    parser.add_argument("--simulate", action="store_true", help="validate launch paths without requiring hardware")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments forwarded to the selected application")
    args = parser.parse_args(list(argv or ()))
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.simulate:
        _print_asset_summary()
        print(f"launch_target={args.target}")
        print("simulate=true")
        return 0
    if args.target == "unified":
        return _run_unified_collection(forwarded)
    if args.target == "mi":
        return _run_mi_collection(forwarded)
    if args.target == "ssvep":
        return _run_ssvep_launcher(forwarded)
    if args.target == "hybrid":
        return _run_hybrid(forwarded, simulate=False)
    raise AssertionError(args.target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified brain-code command line.")
    subparsers = parser.add_subparsers(dest="command")
    diagnose_parser = subparsers.add_parser("diagnose", help="check environment and local asset roots")
    diagnose_parser.add_argument("--strict", action="store_true")
    launch_parser = subparsers.add_parser("launch", help="launch the unified GUI or a subsystem")
    launch_parser.add_argument("--target", choices=("unified", "mi", "ssvep", "hybrid"), default="unified")
    launch_parser.add_argument("--simulate", action="store_true")
    launch_parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        return launch(())
    command = raw_args[0]
    if command == "diagnose":
        return diagnose(raw_args[1:])
    if command == "launch":
        return launch(raw_args[1:])
    parser = build_parser()
    parser.parse_args(raw_args)
    return 0

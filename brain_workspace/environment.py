"""Read-only environment diagnostics for the brain-code workspace."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import missing_required_paths, required_workspace_paths

CHECK_MODULES = {
    "core": ("numpy", "scipy"),
    "gui": ("PyQt5",),
    "ssvep": ("brainflow", "serial", "paramiko"),
    "mi": ("mne", "sklearn", "pyriemann", "joblib"),
    "hybrid": ("cv2", "ultralytics", "roslibpy", "yaml"),
    "gpu": ("cupy",),
}


@dataclass(frozen=True)
class EnvironmentReport:
    python_executable: str
    python_version: str
    required_paths: tuple[Path, ...]
    missing_paths: tuple[Path, ...]
    missing_modules: dict[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return not self.missing_paths


def build_environment_report() -> EnvironmentReport:
    missing_modules: dict[str, tuple[str, ...]] = {}
    for group, modules in CHECK_MODULES.items():
        missing = tuple(name for name in modules if importlib.util.find_spec(name) is None)
        if missing:
            missing_modules[group] = missing
    return EnvironmentReport(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        required_paths=required_workspace_paths(),
        missing_paths=tuple(missing_required_paths()),
        missing_modules=missing_modules,
    )


def format_environment_report(report: EnvironmentReport | None = None) -> str:
    report = report or build_environment_report()
    lines = [
        f"python_executable={report.python_executable}",
        f"python_version={report.python_version}",
        f"required_paths={len(report.required_paths)}",
        f"missing_paths={len(report.missing_paths)}",
    ]
    lines.extend(f"missing={path}" for path in report.missing_paths)
    if report.missing_modules:
        for group, modules in sorted(report.missing_modules.items()):
            lines.append(f"missing_modules[{group}]={','.join(modules)}")
    else:
        lines.append("missing_modules=0")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the local brain-code Python environment.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code when required paths or optional modules are missing",
    )
    args = parser.parse_args(argv)
    report = build_environment_report()
    print(format_environment_report(report))
    if args.strict and (report.missing_paths or report.missing_modules):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

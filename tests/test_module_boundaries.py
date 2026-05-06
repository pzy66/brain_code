from __future__ import annotations

import ast
from pathlib import Path

from brain_workspace.paths import BRAIN_CODE_ROOT


FORBIDDEN_HYBRID_IMPORT_ROOTS = {
    "apps",
    "ssvep_core",
    "mi_data_collector",
    "01_MI",
    "02_SSVEP",
    "04_Communication_And_Integration",
    "05_Vision_Block_Recognition",
}


def _python_files(root: Path) -> list[Path]:
    ignored_parts = {"__pycache__", ".pytest_cache", ".pytest_tmp", ".tmp_pytest", "tmp_test_dir"}
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in ignored_parts or part.startswith("pytest-cache-files-") for part in path.parts):
            continue
        files.append(path)
    return files


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_hybrid_controller_runtime_does_not_import_numbered_debug_modules() -> None:
    allowed_relative = {
        Path("hybrid_controller/tools/debug_vision_grasp_flow.py"),
    }
    offenders: list[str] = []
    for path in _python_files(BRAIN_CODE_ROOT / "hybrid_controller"):
        relative = path.relative_to(BRAIN_CODE_ROOT)
        if relative in allowed_relative:
            continue
        forbidden = sorted(_import_roots(path) & FORBIDDEN_HYBRID_IMPORT_ROOTS)
        if forbidden:
            offenders.append(f"{relative}: {', '.join(forbidden)}")

    assert offenders == []


def test_default_pytest_pythonpath_does_not_expose_submodule_private_roots() -> None:
    pyproject = (BRAIN_CODE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"02_SSVEP"' not in pyproject
    assert '"01_MI/mi_classifier_latest/code/collection"' not in pyproject
    assert '"01_MI/mi_classifier_latest/code/shared"' not in pyproject

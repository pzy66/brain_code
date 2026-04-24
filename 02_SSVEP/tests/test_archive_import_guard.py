from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _python_files() -> list[Path]:
    roots = [
        PROJECT_DIR / "apps",
        PROJECT_DIR / "entrypoints",
        PROJECT_DIR / "ssvep_core",
        PROJECT_DIR / "tools",
        PROJECT_DIR / "tests",
        PROJECT_DIR / "START_SSVEP.py",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files.extend(sorted(root.rglob("*.py")))
    return files


def test_active_mainline_does_not_import_archive() -> None:
    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if str(alias.name).startswith("_archive"):
                        offenders.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if str(node.module or "").startswith("_archive"):
                    offenders.append(f"{path}: from {node.module} import ...")
    assert offenders == []

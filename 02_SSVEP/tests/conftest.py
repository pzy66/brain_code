from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_DIR / ".tmp_test_artifacts" / "pytest_tmp"


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    case_dir = TEST_TMP_ROOT / f"case_{uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

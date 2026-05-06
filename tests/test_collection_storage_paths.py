from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from brain_workspace.bootstrap import configure_qt_offscreen
from brain_workspace.paths import (
    BRAIN_CODE_ROOT,
    DATASETS_ROOT,
    MI_DATASET_DIR,
    PROFILE_DATASET_DIR,
    SSVEP_DATASET_DIR,
    VISION_DATASET_DIR,
    path_is_relative_to,
)

configure_qt_offscreen()
from brain_workspace.paths import ensure_runtime_import_paths

ensure_runtime_import_paths()

from apps.data_collection_ui import DEFAULT_DATASET_DIR, resolve_dataset_dir  # noqa: E402
from mi_data_collector import DEFAULT_OUTPUT_ROOT, build_initial_config_from_args, resolve_output_root  # noqa: E402
from unified_collection.paths import DEFAULT_MI_OUTPUT_ROOT, DEFAULT_SSVEP_DATASET_DIR, resolve_ssvep_dataset_dir  # noqa: E402


def test_collection_default_storage_roots_are_inside_brain_code() -> None:
    assert DATASETS_ROOT == BRAIN_CODE_ROOT / "datasets"
    assert DEFAULT_OUTPUT_ROOT == MI_DATASET_DIR
    assert DEFAULT_MI_OUTPUT_ROOT == MI_DATASET_DIR
    assert DEFAULT_DATASET_DIR == SSVEP_DATASET_DIR
    assert DEFAULT_SSVEP_DATASET_DIR == SSVEP_DATASET_DIR
    assert VISION_DATASET_DIR == DATASETS_ROOT / "vision"
    assert PROFILE_DATASET_DIR == DATASETS_ROOT / "profiles"

    for path in (DATASETS_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_MI_OUTPUT_ROOT, DEFAULT_DATASET_DIR, DEFAULT_SSVEP_DATASET_DIR):
        assert path_is_relative_to(Path(path), BRAIN_CODE_ROOT)


def test_mi_relative_output_root_is_dataset_root_relative() -> None:
    resolved = resolve_output_root("MI")

    assert resolved == MI_DATASET_DIR.resolve()
    assert path_is_relative_to(resolved, DATASETS_ROOT)


def test_mi_cli_output_root_is_normalized_before_window_config() -> None:
    class Args:
        synthetic = False
        serial_port = ""
        output_root = "MI"
        subject_id = ""
        session_id = ""

    config = build_initial_config_from_args(Args())

    assert Path(config["output_root"]) == MI_DATASET_DIR.resolve()


def test_ssvep_relative_dataset_dir_is_dataset_root_relative() -> None:
    resolved = resolve_dataset_dir("SSVEP")

    assert resolved == SSVEP_DATASET_DIR.resolve()
    assert path_is_relative_to(resolved, DATASETS_ROOT)


def test_unified_ssvep_relative_dataset_dir_is_dataset_root_relative() -> None:
    resolved = resolve_ssvep_dataset_dir("SSVEP")

    assert resolved == SSVEP_DATASET_DIR.resolve()
    assert path_is_relative_to(resolved, DATASETS_ROOT)


def test_brain_data_root_override_rebases_dataset_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "brain-data"
    env = dict(os.environ)
    env["BRAIN_DATA_ROOT"] = str(data_root)
    code = (
        "import json; "
        "from brain_workspace.paths import DATASETS_ROOT, MI_DATASET_DIR, SSVEP_DATASET_DIR, "
        "VISION_DATASET_DIR, PROFILE_DATASET_DIR; "
        "print(json.dumps({"
        "'root': str(DATASETS_ROOT), "
        "'mi': str(MI_DATASET_DIR), "
        "'ssvep': str(SSVEP_DATASET_DIR), "
        "'vision': str(VISION_DATASET_DIR), "
        "'profiles': str(PROFILE_DATASET_DIR)"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BRAIN_CODE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert Path(payload["root"]) == data_root.resolve()
    assert Path(payload["mi"]) == (data_root / "MI").resolve()
    assert Path(payload["ssvep"]) == (data_root / "SSVEP").resolve()
    assert Path(payload["vision"]) == (data_root / "vision").resolve()
    assert Path(payload["profiles"]) == (data_root / "profiles").resolve()


def test_collection_ui_rejects_external_storage_roots() -> None:
    external = BRAIN_CODE_ROOT.parent / "_outside_brain_code_collection_path"
    assert resolve_output_root(external) == external.resolve()
    assert resolve_dataset_dir(external) == external.resolve()
    assert resolve_ssvep_dataset_dir(external) == external.resolve()

    with pytest.raises(ValueError, match="BRAIN_DATA_ROOT"):
        resolve_output_root("..")

    with pytest.raises(ValueError, match="BRAIN_DATA_ROOT"):
        resolve_dataset_dir("..")

    with pytest.raises(ValueError, match="BRAIN_DATA_ROOT"):
        resolve_ssvep_dataset_dir("..")

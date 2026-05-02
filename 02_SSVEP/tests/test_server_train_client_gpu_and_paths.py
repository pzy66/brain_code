from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.server_train_client import (
    REMOTE_DATA_DIR,
    REMOTE_ROOT,
    assert_remote_ssvep_path,
    build_parser,
    build_train_command,
    discover_local_datasets,
)


def test_assert_remote_path_restricts_to_data1_zkx_and_ssvep_root() -> None:
    safe_path = f"{REMOTE_ROOT}/reports/model-compare/20260413/run_x/report.json"
    assert assert_remote_ssvep_path(safe_path) == safe_path
    assert assert_remote_ssvep_path(f"{REMOTE_ROOT}/data/session1") == f"{REMOTE_ROOT}/data/session1"

    with pytest.raises(ValueError):
        assert_remote_ssvep_path("/tmp/anything.json")
    with pytest.raises(ValueError):
        assert_remote_ssvep_path("/tmp/ssvep")
    with pytest.raises(ValueError):
        assert_remote_ssvep_path("/home/zhangkexin/brain/ssvep/data")
    with pytest.raises(ValueError):
        assert_remote_ssvep_path("/data1/zkx/other_project/output.json")
    with pytest.raises(ValueError):
        assert_remote_ssvep_path("/data1/zkx/brain/mi")


def test_discover_local_datasets_requires_npz(tmp_path: Path) -> None:
    session_dir = tmp_path / "subject001_collection_r01"
    session_dir.mkdir()
    (session_dir / "session_manifest.json").write_text(
        json.dumps(
            {
                "session_id": "subject001_collection_r01",
                "subject_id": "subject001",
                "trials": [{"label": "8Hz"}, {"label": "idle"}],
            }
        ),
        encoding="utf-8",
    )

    assert discover_local_datasets(tmp_path) == []

    (session_dir / "raw_trials.npz").write_bytes(b"npz-placeholder")
    found = discover_local_datasets(tmp_path)
    assert len(found) == 1
    assert found[0].session_id == "subject001_collection_r01"
    assert found[0].trial_count == 2


def test_build_train_command_includes_session2_and_gpu_flags() -> None:
    payload = build_train_command(
        task="classifier-compare",
        dataset_manifest_remote=f"{REMOTE_DATA_DIR}/session1/session_manifest.json",
        dataset_manifest_session2_remote=f"{REMOTE_DATA_DIR}/session2/session_manifest.json",
        run_id="run_classifier_compare",
        compute_backend="cuda",
        gpu_device=0,
        gpu_precision="float32",
        gpu_warmup=True,
        gpu_cache_policy="windows",
        win_candidates="2.5,3.0,3.5,4.0",
        multi_seed_count=6,
    )
    command = str(payload["command"])
    assert "--dataset-manifest-session2" in command
    assert "--compute-backend cuda" in command
    assert "--gpu-device 0" in command
    assert "--gpu-precision float32" in command
    assert "--gpu-warmup 1" in command
    assert "--gpu-cache-policy windows" in command
    assert "--win-candidates" in command
    assert "2.5,3.0,3.5,4.0" in command
    assert "--multi-seed-count 6" in command


def test_profile_eval_remote_command_is_path_safe() -> None:
    payload = build_train_command(
        task="profile-eval",
        dataset_manifest_remote=f"{REMOTE_ROOT}/data/session1/session_manifest.json",
        run_id="profile_eval_test",
        pretrained_profile_remote=f"{REMOTE_ROOT}/profiles/run1/profile_best_fbcca_weighted.json",
        profile_eval_mode="fbcca-vs-all",
    )
    command = str(payload["command"])
    assert "--task profile-eval" in command
    assert "--pretrained-profile" in command
    assert "--freeze-profile-weights 1" in command
    for key in ("log_path", "report_dir", "report_json", "output_profile"):
        assert str(payload[key]).startswith(REMOTE_ROOT + "/")


def test_new_analysis_remote_commands_are_supported() -> None:
    for task in ("focused-compare", "classifier-compare"):
        payload = build_train_command(
            task=task,
            dataset_manifest_remote=f"{REMOTE_ROOT}/data/session1/session_manifest.json",
            run_id=f"{task}_test",
        )
        assert f"--task {task}" in str(payload["command"])
        assert str(payload["report_dir"]).startswith(REMOTE_ROOT + "/reports/")


def test_server_parser_supports_remote_gpu_and_session2_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--action",
            "model-compare",
            "--dataset-manifest",
            str(PROJECT_DIR / "artifacts" / "datasets" / "s1" / "session_manifest.json"),
            "--dataset-manifest-session2",
            str(PROJECT_DIR / "artifacts" / "datasets" / "s2" / "session_manifest.json"),
            "--compute-backend",
            "cuda",
            "--gpu-device",
            "0",
            "--gpu-precision",
            "float32",
            "--gpu-warmup",
            "1",
            "--gpu-cache-policy",
            "windows",
            "--win-candidates",
            "2.5,3.0,3.5,4.0",
            "--multi-seed-count",
            "5",
        ]
    )
    assert str(args.action) == "model-compare"
    assert str(args.compute_backend) == "cuda"
    assert int(args.gpu_device) == 0
    assert str(args.gpu_precision) == "float32"
    assert int(args.gpu_warmup) == 1
    assert str(args.gpu_cache_policy) == "windows"
    assert str(args.win_candidates) == "2.5,3.0,3.5,4.0"
    assert int(args.multi_seed_count) == 5

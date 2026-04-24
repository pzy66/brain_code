from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_server_train_client import (
    REMOTE_DATA_DIR,
    REMOTE_ROOT,
    assert_remote_ssvep_path,
    build_parser,
    build_train_command,
)


def test_assert_remote_path_restricts_to_data1_zkx_and_ssvep_root() -> None:
    safe_path = f"{REMOTE_ROOT}/reports/20260413/run_x/offline_train_eval.json"
    assert assert_remote_ssvep_path(safe_path) == safe_path

    with pytest.raises(ValueError):
        assert_remote_ssvep_path("/tmp/anything.json")
    with pytest.raises(ValueError):
        assert_remote_ssvep_path("/data1/zkx/other_project/output.json")


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


def test_server_parser_supports_remote_gpu_and_session2_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--action",
            "model-compare",
            "--dataset-manifest",
            str(PROJECT_DIR / "profiles" / "datasets" / "s1" / "session_manifest.json"),
            "--dataset-manifest-session2",
            str(PROJECT_DIR / "profiles" / "datasets" / "s2" / "session_manifest.json"),
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

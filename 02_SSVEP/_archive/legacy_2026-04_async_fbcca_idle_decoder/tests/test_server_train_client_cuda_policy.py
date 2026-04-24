from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_server_train_client import (  # noqa: E402
    build_train_command,
    preflight_cuda_or_fail,
    start_remote_task,
)


class _FakeSSH:
    def __init__(self, code: int = 0, out: str = "", err: str = "") -> None:
        self._code = int(code)
        self._out = str(out)
        self._err = str(err)
        self.commands: list[str] = []

    def exec(self, command: str, *, check: bool = True):  # type: ignore[override]
        self.commands.append(str(command))
        return self._code, self._out, self._err


def test_cuda_preflight_strict_failure_when_nvidia_smi_missing() -> None:
    ssh = _FakeSSH(code=71, out="CUDA_PREFLIGHT:NO_NVIDIA_SMI")
    with pytest.raises(RuntimeError):
        preflight_cuda_or_fail(ssh, compute_backend="cuda", gpu_device=0)


def test_cuda_preflight_skips_for_non_cuda_backend() -> None:
    ssh = _FakeSSH(code=0, out="")
    payload = preflight_cuda_or_fail(ssh, compute_backend="cpu", gpu_device=0)
    assert payload["checked"] is False
    assert payload["reason"] == "compute_backend=cpu"
    assert ssh.commands == []


def test_start_remote_task_records_reproducible_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[dict] = []

    def _capture(record: dict) -> None:
        saved.append(dict(record))

    monkeypatch.setattr("ssvep_server_train_client.save_task_record", _capture)
    ssh = _FakeSSH(code=0, out="98765\n")
    command_payload = {
        "run_id": "run_001",
        "task": "model-compare",
        "command": "echo run",
        "dataset_manifest_session1": "/data1/zkx/brain/ssvep/data/s1/session_manifest.json",
        "dataset_manifest_session2": "/data1/zkx/brain/ssvep/data/s2/session_manifest.json",
        "compute_backend": "cuda",
        "gpu_device": "0",
        "gpu_precision": "float32",
        "gpu_warmup": "1",
        "gpu_cache_policy": "windows",
        "win_candidates": "2.5,3.0,3.5,4.0",
        "multi_seed_count": "5",
    }
    metadata = {
        "session1": "C:/tmp/s1/session_manifest.json",
        "session2": "C:/tmp/s2/session_manifest.json",
        "remote_manifest_paths": {
            "session1": "/data1/zkx/brain/ssvep/data/s1/session_manifest.json",
            "session2": "/data1/zkx/brain/ssvep/data/s2/session_manifest.json",
        },
        "gpu_params": {
            "compute_backend": "cuda",
            "gpu_device": 0,
            "gpu_precision": "float32",
            "gpu_warmup": True,
            "gpu_cache_policy": "windows",
            "win_candidates": "2.5,3.0,3.5,4.0",
            "multi_seed_count": 5,
        },
    }
    record = start_remote_task(ssh, command_payload, metadata=metadata)
    assert record["pid"] == "98765"
    assert record["run_id"] == "run_001"
    assert record["task"] == "model-compare"
    assert record["session1"].endswith("session_manifest.json")
    assert record["remote_manifest_paths"]["session2"].startswith("/data1/zkx/brain/ssvep/")
    assert record["gpu_params"]["compute_backend"] == "cuda"
    assert "started_at" in record
    assert saved and saved[0]["run_id"] == "run_001"


def test_build_train_command_accepts_fbcca_weighted_compare() -> None:
    payload = build_train_command(
        task="fbcca-weighted-compare",
        dataset_manifest_remote="/data1/zkx/brain/ssvep/data/s1/session_manifest.json",
        run_id="weighted_001",
    )
    command = str(payload["command"])
    assert "--task fbcca-weighted-compare" in command
    assert payload["task"] == "fbcca-weighted-compare"

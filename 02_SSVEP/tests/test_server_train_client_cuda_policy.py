from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.server_train_client import (  # noqa: E402
    build_train_command,
    preflight_cuda_or_fail,
    read_remote_status,
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


class _FakeSFTPFile:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSFTP:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = dict(mapping)

    def open(self, path: str, mode: str = "r"):  # type: ignore[override]
        return _FakeSFTPFile(self._mapping[str(path)])


class _FakeRemoteSSH(_FakeSSH):
    def __init__(self, *, existing: set[str], files: dict[str, str], tail: str = "") -> None:
        super().__init__(code=0, out="", err="")
        self._existing = set(existing)
        self._files = dict(files)
        self._tail = str(tail)
        self._sftp = _FakeSFTP(self._files)

    @property
    def sftp(self):
        return self._sftp

    def exists(self, path: str) -> bool:  # type: ignore[override]
        return str(path) in self._existing

    def tail_file(self, path: str, *, lines: int = 50) -> str:  # type: ignore[override]
        return self._tail

    def exec(self, command: str, *, check: bool = True):  # type: ignore[override]
        self.commands.append(str(command))
        text = str(command)
        if "ps -p" in text:
            return 0, "12345 Rl 00:10 python benchmark.py\n", ""
        if "fuser /dev/nvidia" in text:
            return 0, "12345\n", ""
        return 0, "", ""


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

    monkeypatch.setattr("tools.server_train_client.save_task_record", _capture)
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


def test_build_train_command_uses_server_safe_cli_entrypoint() -> None:
    payload = build_train_command(
        task="model-compare",
        dataset_manifest_remote="/data1/zkx/brain/ssvep/data/s1/session_manifest.json",
        run_id="cli_001",
    )
    command = str(payload["command"])
    assert "/tools/training_evaluation_cli.py" in command
    assert "/entrypoints/start_training_eval.py" not in command
    assert "--headless" not in command


def test_read_remote_status_finds_nested_reports_progress_snapshot() -> None:
    report_dir = "/data1/zkx/brain/ssvep/reports/external_short_pretrain/run_x"
    nested_dir = report_dir + "/reports"
    progress_path = nested_dir + "/progress_snapshot.json"
    summary_path = nested_dir + "/summary.json"
    partial_path = nested_dir + "/partial_summary.json"
    ssh = _FakeRemoteSSH(
        existing={progress_path, summary_path, partial_path},
        files={progress_path: '{"stage":"evaluate_recipe","progress_percent":42.0}'},
        tail="tail text",
    )
    record = {
        "run_id": "run_x",
        "pid": "12345",
        "log_path": "/data1/zkx/brain/ssvep/logs/run_x.log",
        "report_dir": report_dir,
        "expected_summary": summary_path,
        "gpu_device": 0,
    }

    status = read_remote_status(ssh, record)

    assert status["active_report_dir"] == nested_dir
    assert status["progress_path"] == progress_path
    assert status["progress"]["stage"] == "evaluate_recipe"
    assert status["artifacts"]["progress_snapshot"] is True
    assert status["artifacts"]["partial_summary"] is True
    assert status["artifacts"]["summary_json"] is True

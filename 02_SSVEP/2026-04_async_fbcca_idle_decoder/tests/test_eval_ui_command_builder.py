from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_model_evaluation_ui import (
    EvalConfig,
    build_benchmark_command,
    resolve_non_conflicting_report_path,
)
from async_fbcca_idle_standalone import (
    DEFAULT_BENCHMARK_CAL_IDLE_REPEATS,
    DEFAULT_BENCHMARK_CAL_TARGET_REPEATS,
    DEFAULT_BENCHMARK_EVAL_IDLE_REPEATS,
    DEFAULT_BENCHMARK_EVAL_TARGET_REPEATS,
    DEFAULT_BENCHMARK_SWITCH_TRIALS,
    DEFAULT_CHANNEL_WEIGHT_MODE,
    DEFAULT_GATE_POLICY,
)


def test_build_benchmark_command_includes_dataset_dir_and_core_args() -> None:
    config = EvalConfig(
        serial_port="auto",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        model_names=("fbcca", "trca"),
        channel_modes=("auto", "all8"),
        multi_seed_count=5,
        output_profile_path=Path(r"C:\tmp\out_profile.json"),
        report_path=Path(r"C:\tmp\report.json"),
        dataset_dir=Path(r"C:\tmp\datasets"),
    )

    cmd = build_benchmark_command(config, python_executable=r"C:\python.exe")

    assert cmd[0] == r"C:\python.exe"
    assert "benchmark" in cmd
    assert "--serial-port" in cmd
    assert cmd[cmd.index("--serial-port") + 1] == "auto"
    assert "--dataset-dir" in cmd
    assert cmd[cmd.index("--dataset-dir") + 1] == r"C:\tmp\datasets"
    assert "--models" in cmd
    assert cmd[cmd.index("--models") + 1] == "fbcca,trca"
    assert "--channel-modes" in cmd
    assert cmd[cmd.index("--channel-modes") + 1] == "auto,all8"
    assert "--multi-seed-count" in cmd
    assert cmd[cmd.index("--multi-seed-count") + 1] == "5"
    assert "--gate-policy" in cmd
    assert cmd[cmd.index("--gate-policy") + 1] == str(DEFAULT_GATE_POLICY)
    assert "--channel-weight-mode" in cmd
    assert cmd[cmd.index("--channel-weight-mode") + 1] == str(DEFAULT_CHANNEL_WEIGHT_MODE)
    assert "--calibration-target-repeats" in cmd
    assert cmd[cmd.index("--calibration-target-repeats") + 1] == str(int(DEFAULT_BENCHMARK_CAL_TARGET_REPEATS))
    assert "--calibration-idle-repeats" in cmd
    assert cmd[cmd.index("--calibration-idle-repeats") + 1] == str(int(DEFAULT_BENCHMARK_CAL_IDLE_REPEATS))
    assert "--eval-target-repeats" in cmd
    assert cmd[cmd.index("--eval-target-repeats") + 1] == str(int(DEFAULT_BENCHMARK_EVAL_TARGET_REPEATS))
    assert "--eval-idle-repeats" in cmd
    assert cmd[cmd.index("--eval-idle-repeats") + 1] == str(int(DEFAULT_BENCHMARK_EVAL_IDLE_REPEATS))
    assert "--eval-switch-trials" in cmd
    assert cmd[cmd.index("--eval-switch-trials") + 1] == str(int(DEFAULT_BENCHMARK_SWITCH_TRIALS))


def test_resolve_non_conflicting_report_path_appends_timestamp_if_exists() -> None:
    artifacts = PROJECT_DIR / "profiles" / "datasets_test_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    case_dir = artifacts / "report_path_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "benchmark_report.json"
    report_path.write_text("{}", encoding="utf-8")
    try:
        resolved = resolve_non_conflicting_report_path(report_path, now_stamp="20260409_210000")
        assert resolved != report_path.resolve()
        assert resolved.name == "benchmark_report_20260409_210000.json"
        fresh = case_dir / "benchmark_report_fresh.json"
        assert resolve_non_conflicting_report_path(fresh) == fresh.resolve()
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

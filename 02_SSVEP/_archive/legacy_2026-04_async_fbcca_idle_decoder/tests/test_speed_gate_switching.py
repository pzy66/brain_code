from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import AsyncDecisionGate, ThresholdProfile, benchmark_rank_key


def _feature(pred_freq: float, *, top1: float = 1.0, ratio: float = 1.8, margin: float = 0.5) -> dict[str, float]:
    return {
        "pred_freq": float(pred_freq),
        "top1_score": float(top1),
        "ratio": float(ratio),
        "margin": float(margin),
        "top2_score": 0.1,
        "normalized_top1": 0.5,
        "score_entropy": 0.5,
    }


def test_speed_gate_supports_direct_selected_to_selected_switch() -> None:
    gate = AsyncDecisionGate(
        enter_score_th=0.5,
        enter_ratio_th=1.2,
        enter_margin_th=0.2,
        exit_score_th=0.2,
        exit_ratio_th=1.05,
        min_enter_windows=1,
        min_exit_windows=2,
        gate_policy="speed",
        min_switch_windows=1,
    )
    first = gate.update(_feature(8.0))
    assert first["state"] == "selected"
    assert abs(float(first["selected_freq"]) - 8.0) < 1e-8

    switched = gate.update(_feature(10.0))
    assert switched["state"] == "selected"
    assert abs(float(switched["selected_freq"]) - 10.0) < 1e-8


def test_speed_gate_switch_hysteresis_prevents_single_window_flip() -> None:
    gate = AsyncDecisionGate(
        enter_score_th=0.5,
        enter_ratio_th=1.2,
        enter_margin_th=0.2,
        exit_score_th=0.2,
        exit_ratio_th=1.05,
        min_enter_windows=1,
        min_exit_windows=2,
        gate_policy="speed",
        min_switch_windows=2,
    )
    gate.update(_feature(8.0))
    tentative = gate.update(_feature(10.0))
    assert tentative["state"] == "selected"
    assert abs(float(tentative["selected_freq"]) - 8.0) < 1e-8

    stabilized = gate.update(_feature(8.0))
    assert stabilized["state"] == "selected"
    assert abs(float(stabilized["selected_freq"]) - 8.0) < 1e-8


def test_threshold_profile_from_legacy_payload_remains_compatible() -> None:
    legacy = {
        "freqs": [8, 10, 12, 15],
        "win_sec": 2.0,
        "step_sec": 0.25,
        "enter_score_th": 0.1,
        "enter_ratio_th": 1.1,
        "enter_margin_th": 0.01,
        "exit_score_th": 0.08,
        "exit_ratio_th": 1.05,
        "min_enter_windows": 2,
        "min_exit_windows": 2,
        "model_name": "fbcca",
        "gate_policy": "balanced",
    }
    profile = ThresholdProfile.from_dict(legacy)
    assert int(profile.min_switch_windows) == 1
    assert profile.switch_enter_score_th is None
    gate = AsyncDecisionGate.from_profile(profile)
    assert gate.gate_policy == "balanced"


def test_async_speed_ranking_uses_deadline_metrics_order() -> None:
    better_recall_3s = {
        "idle_fp_per_min": 0.2,
        "control_recall_at_3s": 0.80,
        "switch_detect_rate_at_2.8s": 0.10,
        "switch_latency_s": 3.0,
        "release_latency_s": 1.2,
        "acc_4class": 0.70,
        "macro_f1_4class": 0.68,
        "itr_bpm_4class": 15.0,
        "inference_ms": 2.0,
    }
    better_switch_deadline = {
        "idle_fp_per_min": 0.2,
        "control_recall_at_3s": 0.70,
        "switch_detect_rate_at_2.8s": 0.95,
        "switch_latency_s": 2.0,
        "release_latency_s": 1.0,
        "acc_4class": 0.90,
        "macro_f1_4class": 0.90,
        "itr_bpm_4class": 22.0,
        "inference_ms": 1.0,
    }
    assert benchmark_rank_key(better_recall_3s, ranking_policy="async-speed") < benchmark_rank_key(
        better_switch_deadline, ranking_policy="async-speed"
    )

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import _calibration_objective, benchmark_rank_key


def test_benchmark_rank_key_penalizes_near_zero_recall() -> None:
    zero_recall = {
        "idle_fp_per_min": 0.0,
        "control_recall": 0.0,
        "switch_detect_rate": 0.0,
        "switch_latency_s": 6.0,
        "itr_bpm": 12.0,
        "inference_ms": 1.0,
    }
    usable_recall = {
        "idle_fp_per_min": 5.0,
        "control_recall": 0.40,
        "switch_detect_rate": 0.5,
        "switch_latency_s": 5.0,
        "itr_bpm": 10.0,
        "inference_ms": 2.0,
    }
    assert benchmark_rank_key(usable_recall) < benchmark_rank_key(zero_recall)


def test_benchmark_rank_key_keeps_idle_priority_within_valid_recall() -> None:
    better_idle = {
        "idle_fp_per_min": 0.3,
        "control_recall": 0.70,
        "switch_detect_rate": 0.2,
        "switch_latency_s": 4.0,
        "itr_bpm": 8.0,
        "inference_ms": 1.0,
    }
    worse_idle_higher_recall = {
        "idle_fp_per_min": 0.8,
        "control_recall": 0.95,
        "switch_detect_rate": 0.9,
        "switch_latency_s": 2.0,
        "itr_bpm": 20.0,
        "inference_ms": 1.0,
    }
    assert benchmark_rank_key(better_idle) < benchmark_rank_key(worse_idle_higher_recall)


def test_calibration_objective_penalizes_zero_detection_collapse() -> None:
    collapsed = {
        "idle_trial_false_positive": 0.0,
        "idle_false_positive": 0.0,
        "trial_success_rate": 0.0,
        "window_end_to_end_recall": 0.0,
        "mean_detection_latency_sec": 1.0,
    }
    detected = {
        "idle_trial_false_positive": 0.1,
        "idle_false_positive": 0.1,
        "trial_success_rate": 0.1,
        "window_end_to_end_recall": 0.1,
        "mean_detection_latency_sec": 1.0,
    }
    assert _calibration_objective(detected) < _calibration_objective(collapsed)

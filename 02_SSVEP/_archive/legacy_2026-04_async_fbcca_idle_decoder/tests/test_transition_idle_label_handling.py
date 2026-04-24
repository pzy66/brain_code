from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import ThresholdProfile, fit_threshold_profile, summarize_profile_quality


def test_summarize_profile_quality_treats_transition_idle_as_idle() -> None:
    profile = ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=2.0,
        step_sec=0.25,
        enter_score_th=0.1,
        enter_ratio_th=1.01,
        enter_margin_th=0.01,
        exit_score_th=0.05,
        exit_ratio_th=1.0,
        min_enter_windows=1,
        min_exit_windows=1,
        model_name="fbcca",
    )
    rows = [
        {
            "label": "transition_idle",
            "expected_freq": None,
            "top1_score": 0.2,
            "ratio": 1.5,
            "margin": 0.1,
            "correct": False,
            "pred_freq": 8.0,
            "window_index": 0,
            "trial_id": 1,
            "block_index": 0,
        },
        {
            "label": "transition_idle",
            "expected_freq": None,
            "top1_score": 0.2,
            "ratio": 1.5,
            "margin": 0.1,
            "correct": False,
            "pred_freq": 8.0,
            "window_index": 1,
            "trial_id": 1,
            "block_index": 0,
        },
        {
            "label": "8Hz",
            "expected_freq": 8.0,
            "top1_score": 0.3,
            "ratio": 2.0,
            "margin": 0.2,
            "correct": True,
            "pred_freq": 8.0,
            "window_index": 0,
            "trial_id": 2,
            "block_index": 0,
        },
    ]
    summary = summarize_profile_quality(rows, profile)
    assert int(summary["idle_windows"]) == 2
    assert int(summary["non_idle_windows"]) == 1
    assert int(summary["idle_trials"]) == 1
    assert int(summary["non_idle_trials"]) == 1


def test_fit_threshold_profile_accepts_transition_idle_rows_as_idle() -> None:
    rows = [
        {
            "label": "8Hz",
            "expected_freq": 8.0,
            "top1_score": 0.40,
            "ratio": 1.80,
            "margin": 0.20,
            "normalized_top1": 0.70,
            "score_entropy": 0.40,
            "correct": True,
            "window_index": 0,
            "trial_id": 1,
            "block_index": 0,
        },
        {
            "label": "10Hz",
            "expected_freq": 10.0,
            "top1_score": 0.38,
            "ratio": 1.70,
            "margin": 0.18,
            "normalized_top1": 0.68,
            "score_entropy": 0.42,
            "correct": True,
            "window_index": 0,
            "trial_id": 2,
            "block_index": 0,
        },
        {
            "label": "transition_idle",
            "expected_freq": None,
            "top1_score": 0.10,
            "ratio": 1.05,
            "margin": 0.01,
            "normalized_top1": 0.20,
            "score_entropy": 1.20,
            "correct": False,
            "window_index": 0,
            "trial_id": 3,
            "block_index": 0,
        },
        {
            "label": "transition_idle_tail",
            "expected_freq": None,
            "top1_score": 0.09,
            "ratio": 1.02,
            "margin": 0.005,
            "normalized_top1": 0.18,
            "score_entropy": 1.25,
            "correct": False,
            "window_index": 0,
            "trial_id": 4,
            "block_index": 0,
        },
    ]
    profile = fit_threshold_profile(
        rows,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=2.0,
        step_sec=0.25,
        min_enter_windows=1,
        min_exit_windows=1,
    )
    assert profile.enter_score_th >= 0.0

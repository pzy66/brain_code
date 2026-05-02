from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core._train_eval_staged import (  # noqa: E402
    _build_aggregated_fbcca_profile,
    _evaluate_pretrained_profile_result,
)
from ssvep_core.async_fbcca_idle_standalone import AsyncDecisionGate, ThresholdProfile, save_profile  # noqa: E402


def test_frequency_specific_threshold_gate_uses_per_freq_values() -> None:
    profile = ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=1.5,
        step_sec=0.25,
        enter_score_th=0.50,
        enter_ratio_th=1.30,
        enter_margin_th=0.20,
        exit_score_th=0.40,
        exit_ratio_th=1.20,
        min_enter_windows=1,
        min_exit_windows=1,
        control_state_mode="frequency-specific-threshold",
        frequency_specific_thresholds={
            "8": {
                "enter_score_th": 0.20,
                "enter_ratio_th": 1.05,
                "enter_margin_th": 0.05,
                "exit_score_th": 0.18,
                "exit_ratio_th": 1.01,
            }
        },
    )
    gate = AsyncDecisionGate.from_profile(profile)

    freq8 = gate.update(
        {
            "pred_freq": 8.0,
            "top1_score": 0.25,
            "ratio": 1.10,
            "margin": 0.08,
        }
    )
    assert freq8["selected_freq"] == 8.0

    gate.reset()
    freq12 = gate.update(
        {
            "pred_freq": 12.0,
            "top1_score": 0.25,
            "ratio": 1.10,
            "margin": 0.08,
        }
    )
    assert freq12["selected_freq"] is None
    assert freq12["state"] == "idle"


def test_profile_eval_rejects_unweighted_fbcca_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "plain_fbcca_profile.json"
    profile = ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=1.5,
        step_sec=0.25,
        enter_score_th=0.02,
        enter_ratio_th=1.1,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.05,
        min_enter_windows=2,
        min_exit_windows=2,
        model_name="fbcca_fixed_all8",
        eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    save_profile(profile, profile_path)

    with pytest.raises(ValueError, match="weighted FBCCA profile"):
        _evaluate_pretrained_profile_result(
            profile_path=profile_path,
            eval_segments=[],
            dataset_channels=(1, 2, 3, 4, 5, 6, 7, 8),
            sampling_rate=250,
            metric_scope="dual",
            decision_time_mode="fixed-window",
            async_decision_time_mode="first-correct",
            compute_backend="cpu",
            gpu_device=0,
            gpu_precision="float32",
            gpu_warmup=False,
            gpu_cache_policy="full",
            prefer_cross_session=False,
        )


def test_weighted_fbcca_profile_aggregation_uses_seed_weights() -> None:
    base_profile = ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=1.5,
        step_sec=0.25,
        enter_score_th=0.02,
        enter_ratio_th=1.1,
        enter_margin_th=0.003,
        exit_score_th=0.017,
        exit_ratio_th=1.05,
        min_enter_windows=2,
        min_exit_windows=2,
        model_name="fbcca_cw_sw_all8",
        eeg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
        channel_weight_mode="fbcca_diag",
        subband_weight_mode="chen_ab_subject",
        subband_weights=(0.30, 0.25, 0.20, 0.15, 0.10),
    )
    run_a = {
        "model_name": "fbcca_cw_sw_all8",
        "channel_mode": "all8",
        "eval_seed": 20260410,
        "metrics": {
            "idle_fp_per_min": 1.0,
            "control_recall": 0.82,
            "switch_latency_s": 2.5,
            "release_latency_s": 1.1,
            "acc_4class": 0.90,
            "macro_f1_4class": 0.89,
        },
    }
    run_b = {
        "model_name": "fbcca_cw_sw_all8",
        "channel_mode": "all8",
        "eval_seed": 20260411,
        "metrics": {
            "idle_fp_per_min": 1.2,
            "control_recall": 0.80,
            "switch_latency_s": 2.7,
            "release_latency_s": 1.2,
            "acc_4class": 0.88,
            "macro_f1_4class": 0.87,
        },
    }
    profile_by_run = {
        ("fbcca_cw_sw_all8", "all8", 20260410): ThresholdProfile.from_dict(
            {
                **base_profile.__dict__,
                "channel_weights": [0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 0.95, 1.05],
                "subband_weights": [0.32, 0.24, 0.18, 0.16, 0.10],
            }
        ),
        ("fbcca_cw_sw_all8", "all8", 20260411): ThresholdProfile.from_dict(
            {
                **base_profile.__dict__,
                "channel_weights": [0.9, 1.0, 1.1, 1.0, 1.1, 0.9, 1.0, 1.0],
                "subband_weights": [0.28, 0.26, 0.22, 0.14, 0.10],
            }
        ),
    }

    aggregated_profile, summary = _build_aggregated_fbcca_profile(
        weighted_runs=[run_a, run_b],
        profile_by_run=profile_by_run,
        prefer_cross_session=False,
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
        aggregation_mode="median",
    )

    assert aggregated_profile is not None
    assert summary is not None
    assert aggregated_profile.channel_weights is not None
    assert aggregated_profile.subband_weights is not None
    assert len(aggregated_profile.channel_weights) == 8
    assert len(aggregated_profile.subband_weights) == 5
    assert pytest.approx(sum(aggregated_profile.subband_weights), rel=1e-6) == 1.0
    assert pytest.approx(sum(aggregated_profile.channel_weights) / 8.0, rel=1e-6) == 1.0

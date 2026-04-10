from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import parse_channel_mode_list, summarize_benchmark_robustness


def test_parse_channel_mode_list_normalizes_and_deduplicates() -> None:
    modes = parse_channel_mode_list("AUTO,all8,auto")
    assert modes == ("auto", "all8")


def test_parse_channel_mode_list_rejects_invalid_mode() -> None:
    try:
        parse_channel_mode_list("auto,foo")
    except ValueError as exc:
        assert "unsupported channel mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid channel mode")


def test_summarize_benchmark_robustness_builds_mode_rankings() -> None:
    runs = [
        {
            "model_name": "fbcca",
            "channel_mode": "auto",
            "eval_seed": 1,
            "meets_acceptance": False,
            "metrics": {
                "idle_fp_per_min": 0.2,
                "control_recall": 0.6,
                "control_miss_rate": 0.4,
                "switch_detect_rate": 0.5,
                "switch_latency_s": 3.2,
                "detection_latency_s": 2.8,
                "itr_bpm": 14.0,
                "inference_ms": 2.0,
            },
            "selected_eeg_channels": [1, 2, 3, 4],
        },
        {
            "model_name": "cca",
            "channel_mode": "auto",
            "eval_seed": 1,
            "meets_acceptance": False,
            "metrics": {
                "idle_fp_per_min": 0.5,
                "control_recall": 0.5,
                "control_miss_rate": 0.5,
                "switch_detect_rate": 0.4,
                "switch_latency_s": 3.8,
                "detection_latency_s": 3.1,
                "itr_bpm": 10.0,
                "inference_ms": 1.0,
            },
            "selected_eeg_channels": [1, 2, 3, 4],
        },
        {
            "model_name": "fbcca",
            "channel_mode": "all8",
            "eval_seed": 1,
            "meets_acceptance": False,
            "metrics": {
                "idle_fp_per_min": 0.1,
                "control_recall": 0.65,
                "control_miss_rate": 0.35,
                "switch_detect_rate": 0.55,
                "switch_latency_s": 3.0,
                "detection_latency_s": 2.7,
                "itr_bpm": 13.0,
                "inference_ms": 3.0,
            },
            "selected_eeg_channels": [1, 2, 3, 4, 5, 6, 7, 8],
        },
    ]

    summary = summarize_benchmark_robustness(runs)
    assert "auto" in summary["by_mode"]
    assert "all8" in summary["by_mode"]
    auto_ranked = summary["by_mode"]["auto"]["ranked_models"]
    assert auto_ranked[0]["model_name"] == "fbcca"
    assert auto_ranked[0]["rank"] == 1
    assert auto_ranked[0]["runs_success"] == 1
    assert auto_ranked[0]["selected_eeg_channels_mode"] == [1, 2, 3, 4]
    all8_ranked = summary["by_mode"]["all8"]["ranked_models"]
    assert all8_ranked[0]["selected_eeg_channels_mode"] == [1, 2, 3, 4, 5, 6, 7, 8]

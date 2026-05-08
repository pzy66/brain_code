from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core import wang2016_frequency_selection as selector


def test_rows_for_combination_uses_command_freqs_and_marks_other_targets_idle() -> None:
    all_freqs = (8.0, 9.0, 10.0, 11.0, 12.0)
    metadata = [
        {"expected_freq": 8.0, "label": "target01_8Hz", "trial_id": 1, "block_index": 0, "window_index": 0},
        {"expected_freq": 9.0, "label": "target02_9Hz", "trial_id": 2, "block_index": 0, "window_index": 0},
        {"expected_freq": 12.0, "label": "target05_12Hz", "trial_id": 5, "block_index": 0, "window_index": 0},
    ]
    score_matrix = np.asarray(
        [
            [0.9, 0.2, 0.1, 0.1, 0.1],
            [0.1, 0.8, 0.2, 0.1, 0.1],
            [0.1, 0.2, 0.2, 0.1, 0.9],
        ],
        dtype=np.float64,
    )

    rows = selector._rows_for_combination(
        score_matrix,
        metadata,
        all_freqs=all_freqs,
        command_freqs=(8.0, 10.0, 11.0, 12.0),
    )

    assert rows[0]["expected_freq"] == 8.0
    assert rows[0]["trial_role"] == "control"
    assert rows[1]["expected_freq"] is None
    assert rows[1]["trial_role"] == "hard_idle"
    assert rows[1]["label"] == "hard_idle_wang2016_9Hz"
    assert rows[2]["expected_freq"] == 12.0
    assert rows[2]["correct"] is True


def test_combination_spacing_prefers_non_low_frequencies() -> None:
    assert selector._combination_spacing_ok((10.0, 11.0, 12.0, 15.0), min_spacing_hz=1.0, min_freq_hz=9.5)
    assert not selector._combination_spacing_ok((8.0, 10.0, 12.0, 15.0), min_spacing_hz=1.0, min_freq_hz=9.5)
    assert not selector._combination_spacing_ok((10.0, 10.2, 12.0, 15.0), min_spacing_hz=1.0, min_freq_hz=9.5)


def test_proxy_sort_key_prefers_lower_hard_idle_when_accuracy_ties() -> None:
    better = {
        "freqs": [10.0, 11.0, 12.0, 15.0],
        "min_per_freq_window_acc": 1.0,
        "raw_control_window_acc": 1.0,
        "hard_idle_top1_score_p95": 0.20,
        "hard_idle_top1_score_p99": 0.30,
        "min_per_freq_margin_p20": 0.05,
        "control_margin_p20": 0.07,
        "min_single_headroom_p50": 0.01,
        "mean_single_headroom_p50": 0.02,
    }
    worse = {**better, "hard_idle_top1_score_p95": 0.40}

    assert selector._proxy_sort_key(better) < selector._proxy_sort_key(worse)


def test_proxy_summary_includes_240hz_frame_lock_report() -> None:
    all_freqs = (8.0, 9.6, 10.0, 12.0, 15.0)
    metadata = [
        {"expected_freq": 8.0},
        {"expected_freq": 9.6},
        {"expected_freq": 10.0},
        {"expected_freq": 12.0},
        {"expected_freq": 15.0},
    ]
    score_matrix = np.eye(len(all_freqs), dtype=np.float64)
    single_stats = {
        f"{freq:g}": {"headroom_p50": 0.1}
        for freq in all_freqs
    }

    summary = selector._combination_proxy_summary(
        score_matrix,
        metadata,
        all_freqs=all_freqs,
        command_freqs=(8.0, 9.6, 10.0, 12.0),
        single_stats=single_stats,
    )

    frame_report = summary["frame_lock_240hz"]
    assert frame_report["all_integer_frames_per_cycle"] is True
    assert int(frame_report["max_frame_sequence_repeat_frames"]) == 30


def test_combination_helpers_tolerate_float_frequency_labels() -> None:
    all_freqs = (9.8, 12.600000381, 14.8, 15.8, 13.0)
    metadata = [
        {"expected_freq": 9.8, "label": "target_9p8", "trial_id": 1, "block_index": 0, "window_index": 0},
        {"expected_freq": 12.600000381, "label": "target_12p6", "trial_id": 2, "block_index": 0, "window_index": 0},
        {"expected_freq": 13.0, "label": "target_13", "trial_id": 3, "block_index": 0, "window_index": 0},
    ]
    score_matrix = np.asarray(
        [
            [0.9, 0.1, 0.2, 0.1, 0.1],
            [0.1, 0.9, 0.2, 0.1, 0.1],
            [0.1, 0.2, 0.2, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    command_freqs = (9.8, 12.6, 14.8, 15.8)

    rows = selector._rows_for_combination(
        score_matrix,
        metadata,
        all_freqs=all_freqs,
        command_freqs=command_freqs,
    )
    summary = selector._combination_proxy_summary(
        score_matrix,
        metadata,
        all_freqs=all_freqs,
        command_freqs=command_freqs,
        single_stats={f"{freq:g}": {"headroom_p50": 0.1} for freq in command_freqs},
    )

    assert rows[1]["expected_freq"] == 12.6
    assert rows[1]["correct"] is True
    assert rows[2]["trial_role"] == "hard_idle"
    assert summary["per_frequency_window_acc"]["12.6"] == 1.0

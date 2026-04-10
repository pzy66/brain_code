from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import (
    BaseSSVEPDecoder,
    TrialSpec,
    ThresholdProfile,
    _decision_latency_with_mode,
    compute_classification_metrics,
    compute_confusion_matrix_counts,
    evaluate_decoder_on_trials_v2,
    pack_evaluation_metrics_for_ranking,
)
from ssvep_core.reporting import export_evaluation_figures


class _Constant8Decoder(BaseSSVEPDecoder):
    def __init__(self) -> None:
        super().__init__(
            model_name="const8",
            sampling_rate=20,
            freqs=(8.0, 10.0, 12.0, 15.0),
            win_sec=1.0,
            step_sec=0.5,
            model_params={},
        )

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        _ = X_window
        return np.asarray([3.0, 1.0, 0.3, 0.1], dtype=float)


class _SequenceDecoder(BaseSSVEPDecoder):
    def __init__(self, sequence: list[np.ndarray]) -> None:
        super().__init__(
            model_name="sequence",
            sampling_rate=20,
            freqs=(8.0, 10.0, 12.0, 15.0),
            win_sec=1.0,
            step_sec=0.5,
            model_params={},
        )
        self._sequence = [np.asarray(item, dtype=float) for item in sequence]
        self._index = 0

    def score_window(self, X_window: np.ndarray) -> np.ndarray:
        _ = X_window
        idx = min(self._index, len(self._sequence) - 1)
        self._index += 1
        return np.asarray(self._sequence[idx], dtype=float)


def _test_profile(*, min_enter_windows: int = 1, min_exit_windows: int = 1) -> ThresholdProfile:
    return ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=1.0,
        step_sec=0.5,
        enter_score_th=2.0,
        enter_ratio_th=1.4,
        enter_margin_th=1.0,
        exit_score_th=0.5,
        exit_ratio_th=1.1,
        min_enter_windows=min_enter_windows,
        min_exit_windows=min_exit_windows,
        model_name="fbcca",
    )


def test_confusion_and_macro_f1_formula() -> None:
    labels = ["8", "10", "12", "15"]
    y_true = ["8", "10", "12", "15"]
    y_pred = ["8", "10", "10", "15"]
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        decision_time_samples_s=[1.0, 1.0, 1.0, 1.0],
        itr_class_count=4,
        decision_time_fallback_s=1.0,
    )
    assert metrics["n_total"] == 4
    assert metrics["n_correct"] == 3
    assert abs(float(metrics["acc"]) - 0.75) < 1e-9
    assert abs(float(metrics["macro_f1"]) - (2.0 / 3.0)) < 1e-6
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    assert abs(float(metrics["mean_decision_time_s"]) - 1.0) < 1e-9


def test_decision_latency_modes() -> None:
    assert (
        _decision_latency_with_mode(
            mode="first-correct",
            first_correct_latency=None,
            first_any_latency=2.0,
            trial_duration_sec=4.0,
            win_sec=3.0,
        )
        == 7.0
    )
    assert (
        _decision_latency_with_mode(
            mode="first-any",
            first_correct_latency=None,
            first_any_latency=2.0,
            trial_duration_sec=4.0,
            win_sec=3.0,
        )
        == 2.0
    )
    assert (
        _decision_latency_with_mode(
            mode="fixed-window",
            first_correct_latency=1.2,
            first_any_latency=0.8,
            trial_duration_sec=4.0,
            win_sec=3.0,
        )
        == 3.0
    )


def test_pack_ranking_metrics_uses_4class_fields() -> None:
    bundle = {
        "async_metrics": {
            "idle_fp_per_min": 0.2,
            "control_recall": 0.8,
            "switch_latency_s": 2.0,
            "release_latency_s": 1.0,
            "inference_ms": 3.0,
        },
        "metrics_4class": {"acc": 0.9, "macro_f1": 0.88, "itr_bpm": 25.0, "mean_decision_time_s": 1.2},
        "metrics_5class": {"acc": 0.7, "macro_f1": 0.68, "itr_bpm": 18.0},
    }
    merged = pack_evaluation_metrics_for_ranking(bundle, metric_scope="dual")
    assert abs(float(merged["acc_4class"]) - 0.9) < 1e-9
    assert abs(float(merged["macro_f1_4class"]) - 0.88) < 1e-9
    assert abs(float(merged["itr_bpm_4class"]) - 25.0) < 1e-9


def test_export_evaluation_figures_writes_pngs() -> None:
    tmp_root = PROJECT_DIR / ".tmp_test_artifacts" / f"fig_export_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        payload = {
            "chosen_metrics_4class": {
                "labels": ["8", "10", "12", "15"],
                "confusion_matrix": [[4, 1, 0, 0], [0, 5, 0, 0], [0, 1, 3, 0], [0, 0, 0, 4]],
                "decision_time_samples_s": [1.1, 1.3, 1.5, 1.7],
            },
            "chosen_metrics_2class": {
                "labels": ["idle", "control"],
                "confusion_matrix": [[12, 1], [2, 15]],
            },
            "model_results": [
                {
                    "rank": 1,
                    "model_name": "fbcca",
                    "metrics": {
                        "idle_fp_per_min": 0.3,
                        "control_recall": 0.85,
                        "acc_4class": 0.88,
                        "macro_f1_4class": 0.86,
                        "switch_latency_s": 2.3,
                    },
                }
            ],
        }
        figures = export_evaluation_figures(payload, output_dir=tmp_root)
        assert Path(figures["confusion_4class"]).exists()
        assert Path(figures["confusion_2class"]).exists()
        assert Path(figures["decision_time_hist"]).exists()
        assert Path(figures["model_radar_async_vs_cls"]).exists()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_confusion_matrix_counts_skips_unknown_labels() -> None:
    mat = compute_confusion_matrix_counts(
        y_true=["a", "b", "x"],
        y_pred=["a", "a", "b"],
        labels=["a", "b"],
    )
    assert mat == [[1, 0], [1, 0]]


def test_idle_false_positive_rate_is_event_based() -> None:
    decoder = _Constant8Decoder()
    profile = _test_profile(min_enter_windows=1, min_exit_windows=1)
    segment = np.zeros((80, 1), dtype=np.float64)
    trials = [(TrialSpec(label="idle", expected_freq=None, trial_id=1, block_index=0), segment)]
    bundle = evaluate_decoder_on_trials_v2(decoder, profile, trials, decision_time_mode="first-correct")
    async_metrics = dict(bundle["async_metrics"])
    idle_windows = float(async_metrics["idle_windows"])
    idle_minutes = idle_windows * profile.step_sec / 60.0
    expected_fp_per_min = 1.0 / idle_minutes
    assert int(async_metrics["idle_selected_events"]) == 1
    assert int(async_metrics["idle_selected_windows"]) > 1
    assert abs(float(async_metrics["idle_fp_per_min"]) - expected_fp_per_min) < 1e-9
    assert float(async_metrics["idle_selected_windows_per_min"]) > float(async_metrics["idle_fp_per_min"])


def test_binary_metrics_follow_first_correct_definition() -> None:
    decoder = _Constant8Decoder()
    profile = _test_profile(min_enter_windows=1, min_exit_windows=1)
    segment = np.zeros((80, 1), dtype=np.float64)
    trials = [(TrialSpec(label="10Hz", expected_freq=10.0, trial_id=1, block_index=0), segment)]
    bundle = evaluate_decoder_on_trials_v2(decoder, profile, trials, decision_time_mode="first-correct")
    metrics_2 = dict(bundle["metrics_2class"])
    assert abs(float(metrics_2["acc"]) - 0.0) < 1e-9
    assert metrics_2["confusion_matrix"] == [[0, 0], [1, 0]]


def test_four_class_prediction_uses_first_correct_when_available() -> None:
    decoder = _SequenceDecoder(
        [
            np.asarray([3.0, 1.0, 0.3, 0.1]),  # candidate 8Hz
            np.asarray([3.0, 1.0, 0.3, 0.1]),  # select 8Hz (first-any)
            np.asarray([0.2, 0.19, 0.18, 0.17]),  # force exit
            np.asarray([1.0, 3.0, 0.3, 0.1]),  # candidate 10Hz
            np.asarray([1.0, 3.0, 0.3, 0.1]),  # select 10Hz (first-correct)
        ]
    )
    profile = _test_profile(min_enter_windows=1, min_exit_windows=1)
    segment = np.zeros((60, 1), dtype=np.float64)
    trials = [(TrialSpec(label="10Hz", expected_freq=10.0, trial_id=1, block_index=0), segment)]
    bundle = evaluate_decoder_on_trials_v2(decoder, profile, trials, decision_time_mode="first-correct")
    metrics_4 = dict(bundle["metrics_4class"])
    assert abs(float(metrics_4["acc"]) - 1.0) < 1e-9
    assert metrics_4["confusion_matrix"] == [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

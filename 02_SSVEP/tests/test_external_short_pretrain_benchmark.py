from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import TrialSpec
from tools import run_external_short_pretrain_benchmark as bench


def _segments_for_blocks(blocks: int = 4) -> list[tuple[TrialSpec, np.ndarray]]:
    freqs = (9.8, 12.0, 14.8, 15.8)
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    for block in range(blocks):
        for freq in freqs:
            segments.append(
                (
                    TrialSpec(label=f"{float(freq):g}Hz", expected_freq=float(freq), trial_id=trial_id, block_index=block),
                    np.zeros((500, 8), dtype=np.float64),
                )
            )
            trial_id += 1
        for idle_index in range(6):
            segments.append(
                (
                    TrialSpec(label=f"hard_idle_{block}_{idle_index}", expected_freq=None, trial_id=trial_id, block_index=block),
                    np.zeros((500, 8), dtype=np.float64),
                )
            )
            trial_id += 1
    return segments


def _aggregate_test_row(
    *,
    subject: str,
    recipe_id: str,
    async_macro_f1_5class: float,
    async_acc_5class: float,
    idle_fp_per_min: float,
    control_recall: float,
    detection_latency_s: float,
) -> dict[str, object]:
    return {
        "method": "fbcca_ridge5",
        "recipe_id": recipe_id,
        "dataset": "beta",
        "subject": subject,
        "calibration_blocks": [0, 1],
        "split_summary": {"idle_multiplier": 2.0},
        "summary_metrics": {
            "fixed_acc_4class": async_acc_5class,
            "fixed_macro_f1_4class": async_macro_f1_5class,
            "fixed_acc_5class": async_acc_5class,
            "fixed_macro_f1_5class": async_macro_f1_5class,
            "async_acc_4class": async_acc_5class,
            "async_macro_f1_4class": async_macro_f1_5class,
            "async_acc_5class": async_acc_5class,
            "async_macro_f1_5class": async_macro_f1_5class,
            "idle_fp_per_min": idle_fp_per_min,
            "idle_selected_windows_per_min": idle_fp_per_min,
            "control_recall": control_recall,
            "control_recall_at_2s": control_recall,
            "control_recall_at_2.5s": control_recall,
            "control_recall_at_3s": control_recall,
            "detection_latency_s": detection_latency_s,
            "switch_latency_s": detection_latency_s,
            "release_latency_s": detection_latency_s,
        },
    }


def test_enumerate_external_subjects_orders_and_limits(tmp_path: Path) -> None:
    wang_raw = tmp_path / "wang"
    beta_raw = tmp_path / "beta"
    wang_raw.mkdir()
    beta_raw.mkdir()
    (wang_raw / "S2.mat").write_text("", encoding="utf-8")
    (wang_raw / "S1.mat").write_text("", encoding="utf-8")
    (beta_raw / "S16.mat").write_text("", encoding="utf-8")
    (tmp_path / "64-channels.loc").write_text("", encoding="utf-8")

    rows = bench.enumerate_external_subjects(
        datasets=("wang2016", "beta"),
        freqs=(9.8, 12.0, 14.8, 15.8),
        wang_raw_dir=wang_raw,
        wang_channels_loc=tmp_path / "64-channels.loc",
        beta_raw_dir=beta_raw,
        subject_limit_per_dataset=1,
    )

    assert [(row.dataset, row.subject) for row in rows] == [("wang2016", "S1"), ("beta", "S16")]


def test_parse_subject_whitelist_supports_global_and_dataset_scoped_tokens() -> None:
    parsed = bench._parse_subject_whitelist("S1,beta:S16,wang2016:S2")
    assert parsed == (("*", "S1"), ("beta", "S16"), ("wang2016", "S2"))


def test_method_parser_supports_short_pretrain_candidates() -> None:
    parsed = bench._csv_method_tuple("fbcca_ridge5,fbcca_lda5,itcca5,ecca5,trca5,trca_r5,tdca5,fbcca_ridge5")
    assert parsed == ("fbcca_ridge5", "fbcca_lda5", "itcca5", "ecca5", "trca5", "trca_r5", "tdca5")


def test_classifier_recipe_id_preserves_strict_names_and_marks_gap() -> None:
    assert bench._classifier_recipe_id(win_sec=2.0, min_enter_windows=1, max_gap_windows=0) == "win2_me1"
    assert bench._classifier_recipe_id(win_sec=1.75, min_enter_windows=2, max_gap_windows=1) == "win1p75_me2_gap1"


def test_classifier_threshold_policy_parser_supports_recall_guard() -> None:
    assert bench._parse_classifier_threshold_policy(" balanced_recall_guard ") == "balanced_recall_guard"


def test_tdca_candidate_windows_require_effective_raw_latency_buffer() -> None:
    pairs = bench._score_method_candidate_pairs(
        method_name="tdca5",
        win_sec_candidates=(1.5, 2.0),
        min_enter_candidates=(1, 2),
        max_supported_win_sec=1.63,
        sampling_rate=250,
    )

    assert pairs == []

    pairs = bench._score_method_candidate_pairs(
        method_name="tdca5",
        win_sec_candidates=(1.5, 2.0),
        min_enter_candidates=(1, 2),
        max_supported_win_sec=1.64,
        sampling_rate=250,
    )

    assert pairs == [(1.5, 1), (1.5, 2)]


def test_enumerate_external_subjects_respects_subject_whitelist(tmp_path: Path) -> None:
    wang_raw = tmp_path / "wang"
    beta_raw = tmp_path / "beta"
    wang_raw.mkdir()
    beta_raw.mkdir()
    (wang_raw / "S1.mat").write_text("", encoding="utf-8")
    (wang_raw / "S2.mat").write_text("", encoding="utf-8")
    (beta_raw / "S1.mat").write_text("", encoding="utf-8")
    (beta_raw / "S16.mat").write_text("", encoding="utf-8")
    (tmp_path / "64-channels.loc").write_text("", encoding="utf-8")

    rows = bench.enumerate_external_subjects(
        datasets=("wang2016", "beta"),
        freqs=(9.8, 12.0, 14.8, 15.8),
        wang_raw_dir=wang_raw,
        wang_channels_loc=tmp_path / "64-channels.loc",
        beta_raw_dir=beta_raw,
        subject_limit_per_dataset=0,
        subject_whitelist=(("wang2016", "S2"), ("beta", "S16")),
    )

    assert [(row.dataset, row.subject) for row in rows] == [("wang2016", "S2"), ("beta", "S16")]


def test_build_block_split_plans_respects_budget() -> None:
    plans = bench.build_block_split_plans(
        dataset="wang2016",
        subject="S1",
        block_indices=(0, 1, 2, 3),
        calibration_block_count=2,
        max_splits=3,
        seed=7,
    )

    assert len(plans) == 3
    assert all(len(plan.calibration_blocks) == 2 for plan in plans)
    assert all(len(plan.holdout_blocks) == 2 for plan in plans)


def test_select_split_segments_samples_idle_to_budget() -> None:
    segments = _segments_for_blocks(blocks=3)
    calibration, holdout, summary = bench.select_split_segments(
        segments,
        freqs=(9.8, 12.0, 14.8, 15.8),
        calibration_blocks=(0, 1),
        holdout_blocks=(2,),
        idle_multiplier=1.5,
        seed=42,
    )

    counts = dict(summary["calibration_counts"])
    assert counts["control"] == 8
    assert counts["idle"] == 12
    holdout_counts = dict(summary["holdout_counts"])
    assert holdout_counts["control"] == 4
    assert holdout_counts["idle"] == 6
    assert len(calibration) == 20
    assert len(holdout) == 10


def test_extract_row_metrics_reads_fixed_and_async_five_class_fields() -> None:
    payload = {
        "fixed_window_metrics_4class": {"acc": 0.95, "macro_f1": 0.94},
        "fixed_window_metrics_5class": {"acc": 0.80, "macro_f1": 0.78, "itr_bpm": 15.0},
        "async_lens_metrics_4class": {"acc": 0.90, "macro_f1": 0.88},
        "async_lens_metrics_5class": {"acc": 0.70, "macro_f1": 0.68, "itr_bpm": 18.0},
        "async_metrics": {
            "idle_fp_per_min": 0.0,
            "idle_selected_windows_per_min": 0.4,
            "control_recall": 0.9,
            "control_recall_at_2s": 0.55,
            "control_recall_at_2.5s": 0.75,
            "control_recall_at_3s": 0.85,
            "detection_latency_s": 1.75,
            "switch_latency_supported": False,
            "release_latency_supported": False,
            "switch_latency_s": 2.1,
            "release_latency_s": 1.4,
        },
    }
    metrics = bench._extract_row_metrics(payload)
    assert abs(float(metrics["fixed_acc_4class"]) - 0.95) < 1e-9
    assert abs(float(metrics["fixed_acc_5class"]) - 0.80) < 1e-9
    assert abs(float(metrics["fixed_macro_f1_5class"]) - 0.78) < 1e-9
    assert abs(float(metrics["async_acc_4class"]) - 0.90) < 1e-9
    assert abs(float(metrics["async_acc_5class"]) - 0.70) < 1e-9
    assert abs(float(metrics["idle_fp_per_min"]) - 0.0) < 1e-9
    assert abs(float(metrics["idle_selected_windows_per_min"]) - 0.4) < 1e-9
    assert abs(float(metrics["control_recall_at_2s"]) - 0.55) < 1e-9
    assert abs(float(metrics["control_recall_at_2.5s"]) - 0.75) < 1e-9
    assert abs(float(metrics["detection_latency_s"]) - 1.75) < 1e-9
    assert float(metrics["switch_latency_supported"]) == 0.0
    assert float(metrics["release_latency_supported"]) == 0.0


def test_fbcca_lda5_model_learns_idle_plus_four_commands() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    centers = {
        label: np.tile(np.eye(5, dtype=np.float64)[index], 2)
        for index, label in enumerate(labels)
    }

    def make_trial(label: str, trial_id: int) -> bench.ScoredTrial:
        expected = None if label == "idle" else float(label)
        features = np.vstack(
            [
                centers[label] + np.full(10, 0.01 * offset, dtype=np.float64)
                for offset in range(3)
            ]
        )
        return bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=expected, trial_id=trial_id, block_index=0),
            score_matrix=np.zeros((3, 4), dtype=np.float64),
            feature_matrix=features,
            duration_sec=3.0,
        )

    calibration = [make_trial(label, index) for index, label in enumerate(labels)]
    holdout = [make_trial(label, index + 10) for index, label in enumerate(labels)]

    model = bench._fit_fbcca_lda5_model(calibration, freqs=freqs)
    bundle = bench._evaluate_fbcca_lda5_model(
        model,
        holdout,
        win_sec=2.0,
        step_sec=0.25,
        min_enter_windows=1,
    )

    metrics = dict(bundle["classifier_metrics_5class"])
    assert float(metrics["acc"]) == 1.0
    assert float(bundle["async_metrics"]["idle_fp_per_min"]) == 0.0
    assert float(bundle["async_metrics"]["control_recall"]) == 1.0
    threshold_selection = dict(model.fit_summary["threshold_selection"])
    assert float(threshold_selection["selected_metrics"]["idle_fp_per_min"]) == 0.0


def test_fbcca_ridge5_model_learns_idle_plus_four_commands() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    centers = {
        label: np.tile(np.eye(5, dtype=np.float64)[index], 2)
        for index, label in enumerate(labels)
    }

    def make_trial(label: str, trial_id: int) -> bench.ScoredTrial:
        expected = None if label == "idle" else float(label)
        features = np.vstack(
            [
                centers[label] + np.full(10, 0.01 * offset, dtype=np.float64)
                for offset in range(4)
            ]
        )
        return bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=expected, trial_id=trial_id, block_index=0),
            score_matrix=np.zeros((4, 4), dtype=np.float64),
            feature_matrix=features,
            duration_sec=3.0,
        )

    calibration = [make_trial(label, index) for index, label in enumerate(labels)]
    holdout = [make_trial(label, index + 10) for index, label in enumerate(labels)]

    model = bench._fit_fbcca_ridge5_model(calibration, freqs=freqs)
    bundle = bench._evaluate_fbcca_lda5_model(
        model,
        holdout,
        win_sec=2.0,
        step_sec=0.25,
        min_enter_windows=1,
    )

    metrics = dict(bundle["classifier_metrics_5class"])
    assert float(metrics["acc"]) == 1.0
    assert float(bundle["async_metrics"]["idle_fp_per_min"]) == 0.0
    assert float(bundle["async_metrics"]["control_recall"]) == 1.0
    assert float(model.l2) in {0.03, 0.1, 0.3, 1.0, 3.0}
    assert dict(model.fit_summary)["classifier"] == "fbcca_score_ridge_5class"
    assert dict(model.fit_summary)["class_weighting"] == "balanced_window_inverse_frequency"


def test_fbcca_lda5_fixed_and_async_metrics_are_separate_with_enter_streak() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    dims = 10
    feature_mean = np.zeros(dims, dtype=np.float64)
    feature_std = np.ones(dims, dtype=np.float64)
    class_means = np.vstack([np.full(dims, float(index), dtype=np.float64) for index in range(len(labels))])
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=feature_mean,
        feature_std=feature_std,
        class_means=class_means,
        pooled_var=np.ones(dims, dtype=np.float64),
        command_confidence_th=0.50,
        fit_summary={},
    )
    trial = bench.ScoredTrial(
        trial=TrialSpec(label="9.8", expected_freq=9.8, trial_id=1, block_index=0),
        score_matrix=np.zeros((2, 4), dtype=np.float64),
        feature_matrix=np.vstack(
            [
                np.full(dims, 0.0, dtype=np.float64),
                np.full(dims, 1.0, dtype=np.float64),
            ]
        ),
        duration_sec=3.0,
    )

    bundle = bench._evaluate_fbcca_lda5_model(
        model,
        [trial],
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=2,
    )

    assert float(bundle["fixed_window_metrics_5class"]["acc"]) == 1.0
    assert float(bundle["async_lens_metrics_5class"]["acc"]) == 0.0
    assert float(bundle["async_metrics"]["control_recall"]) == 0.0


def test_fbcca_lda5_max_gap_recovers_single_window_dropout() -> None:
    labels = np.asarray(("idle", "9.8", "12", "14.8", "15.8"), dtype=object)
    model = bench.FBCCALDA5Model(
        freqs=(9.8, 12.0, 14.8, 15.8),
        labels=tuple(labels.tolist()),
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.zeros((5, 1), dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=0.60,
        fit_summary={},
    )
    probs = np.asarray(
        [
            [0.10, 0.86, 0.02, 0.01, 0.01],
            [0.70, 0.18, 0.05, 0.04, 0.03],
            [0.10, 0.84, 0.03, 0.02, 0.01],
        ],
        dtype=np.float64,
    )

    strict_label, _strict_confidence, _strict_index = bench._predict_fbcca_lda5_trial_from_probs(
        model,
        probs,
        labels,
        min_enter_windows=2,
        max_gap_windows=0,
    )
    gap_label, gap_confidence, gap_index = bench._predict_fbcca_lda5_trial_from_probs(
        model,
        probs,
        labels,
        min_enter_windows=2,
        max_gap_windows=1,
    )

    assert strict_label == "idle"
    assert gap_label == "9.8"
    assert gap_confidence >= 0.60
    assert gap_index == 2.0


def test_fbcca_lda5_fixed_window_uses_last_window() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    dims = 10
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(dims, dtype=np.float64),
        feature_std=np.ones(dims, dtype=np.float64),
        class_means=np.vstack([np.full(dims, float(index), dtype=np.float64) for index in range(len(labels))]),
        pooled_var=np.ones(dims, dtype=np.float64),
        command_confidence_th=0.50,
        fit_summary={},
    )
    trial = bench.ScoredTrial(
        trial=TrialSpec(label="9.8", expected_freq=9.8, trial_id=1, block_index=0),
        score_matrix=np.zeros((2, 4), dtype=np.float64),
        feature_matrix=np.vstack(
            [
                np.full(dims, 0.0, dtype=np.float64),
                np.full(dims, 1.0, dtype=np.float64),
            ]
        ),
        duration_sec=3.0,
    )

    pred_5, pred_4, _confidence = bench._predict_fbcca_lda5_fixed_trial(model, trial)

    assert pred_5 == "9.8"
    assert pred_4 == "9.8"


def test_fbcca_lda5_missed_control_latency_uses_penalty_not_window_length() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    dims = 10
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(dims, dtype=np.float64),
        feature_std=np.ones(dims, dtype=np.float64),
        class_means=np.vstack([np.full(dims, float(index), dtype=np.float64) for index in range(len(labels))]),
        pooled_var=np.ones(dims, dtype=np.float64),
        command_confidence_th=0.50,
        fit_summary={},
    )
    trial = bench.ScoredTrial(
        trial=TrialSpec(label="12", expected_freq=12.0, trial_id=1, block_index=0),
        score_matrix=np.zeros((2, 4), dtype=np.float64),
        feature_matrix=np.zeros((2, dims), dtype=np.float64),
        duration_sec=3.0,
    )

    bundle = bench._evaluate_fbcca_lda5_model(
        model,
        [trial],
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=1,
    )

    event = dict(bundle["classifier_trial_events"][0])
    assert event["pred"] == "idle"
    assert float(event["decision_time_s"]) == 4.5


def test_fbcca_lda5_missed_first_frequency_is_wrong_in_async_4class() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    dims = 10
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(dims, dtype=np.float64),
        feature_std=np.ones(dims, dtype=np.float64),
        class_means=np.vstack([np.full(dims, float(index), dtype=np.float64) for index in range(len(labels))]),
        pooled_var=np.ones(dims, dtype=np.float64),
        command_confidence_th=0.50,
        fit_summary={},
    )
    trial = bench.ScoredTrial(
        trial=TrialSpec(label="9.8", expected_freq=9.8, trial_id=1, block_index=0),
        score_matrix=np.zeros((2, 4), dtype=np.float64),
        feature_matrix=np.zeros((2, dims), dtype=np.float64),
        duration_sec=3.0,
    )

    bundle = bench._evaluate_fbcca_lda5_model(
        model,
        [trial],
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=1,
    )

    metrics4 = dict(bundle["async_lens_metrics_4class"])
    assert float(metrics4["acc"]) == 0.0
    assert metrics4["y_true"] == ["9.8"]
    assert metrics4["y_pred"] == ["12"]


def test_fbcca_lda5_threshold_selection_uses_idle_penalty() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=0.0,
        fit_summary={},
    )

    def make_trial(label: str, values: list[float], trial_id: int) -> bench.ScoredTrial:
        expected = None if label == "idle" else float(label)
        return bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=expected, trial_id=trial_id, block_index=0),
            score_matrix=np.zeros((len(values), 4), dtype=np.float64),
            feature_matrix=np.asarray(values, dtype=np.float64).reshape(-1, 1),
            duration_sec=3.0,
        )

    scored = [
        make_trial("9.8", [1.0, 1.0], 1),
        make_trial("12", [2.0, 2.0], 2),
        make_trial("14.8", [3.0, 3.0], 3),
        make_trial("15.8", [4.0, 4.0], 4),
        make_trial("idle", [0.6, 0.0], 5),
    ]

    zero_bundle = bench._evaluate_fbcca_lda5_model(
        model,
        scored,
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=1,
    )
    selection = bench._select_fbcca_lda5_confidence_threshold(
        model,
        scored,
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=1,
    )

    assert float(zero_bundle["async_metrics"]["idle_fp_per_min"]) > 0.0
    assert float(selection["selected_metrics"]["idle_fp_per_min"]) == 0.0
    assert float(selection["command_confidence_th"]) > 0.0


def test_balanced_recall_guard_prefers_recall_inside_idle_budget() -> None:
    valid_higher_recall = {
        "idle_fp_per_min": 1.0,
        "idle_selected_windows_per_min": 5.0,
        "control_recall": 0.82,
        "control_recall_at_2.5s": 0.75,
        "control_recall_at_3s": 0.82,
        "async_macro_f1_5class": 0.70,
        "detection_latency_s": 2.0,
    }
    zero_fp_all_idle = {
        "idle_fp_per_min": 0.0,
        "idle_selected_windows_per_min": 0.0,
        "control_recall": 0.0,
        "control_recall_at_2.5s": 0.0,
        "control_recall_at_3s": 0.0,
        "async_macro_f1_5class": 0.20,
        "detection_latency_s": float("inf"),
    }

    assert (
        bench._classifier_threshold_rank_key(
            valid_higher_recall,
            policy="balanced_recall_guard",
            tie_breaker=0.3,
        )
        < bench._classifier_threshold_rank_key(
            zero_fp_all_idle,
            policy="balanced_recall_guard",
            tie_breaker=0.9,
        )
    )


def test_balanced_recall_guard_rejects_thresholds_over_idle_budget_first() -> None:
    over_budget_high_recall = {
        "idle_fp_per_min": 1.01,
        "idle_selected_windows_per_min": 2.0,
        "control_recall": 1.0,
        "control_recall_at_2.5s": 1.0,
        "control_recall_at_3s": 1.0,
        "async_macro_f1_5class": 0.90,
        "detection_latency_s": 1.5,
    }
    in_budget_lower_recall = {
        "idle_fp_per_min": 1.0,
        "idle_selected_windows_per_min": 5.0,
        "control_recall": 0.55,
        "control_recall_at_2.5s": 0.50,
        "control_recall_at_3s": 0.55,
        "async_macro_f1_5class": 0.50,
        "detection_latency_s": 2.0,
    }

    assert (
        bench._classifier_threshold_rank_key(
            in_budget_lower_recall,
            policy="balanced_recall_guard",
            tie_breaker=0.9,
        )
        < bench._classifier_threshold_rank_key(
            over_budget_high_recall,
            policy="balanced_recall_guard",
            tie_breaker=0.1,
        )
    )


def test_fbcca_lda5_threshold_selection_reuses_probability_cache(monkeypatch) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.zeros((5, 1), dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=0.0,
        fit_summary={},
    )
    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=None if label == "idle" else float(label), trial_id=index, block_index=0),
            score_matrix=np.zeros((2, 4), dtype=np.float64),
            feature_matrix=np.full((2, 1), float(index), dtype=np.float64),
            duration_sec=3.0,
        )
        for index, label in enumerate(labels)
    ]
    call_count = 0

    def fake_predict(_model, feature_matrix):
        nonlocal call_count
        call_count += 1
        class_index = int(np.asarray(feature_matrix)[0, 0])
        probs = np.full((2, 5), 0.025, dtype=np.float64)
        probs[:, class_index] = 0.9
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs, np.asarray(labels, dtype=object)

    monkeypatch.setattr(bench, "_predict_classifier_windows", fake_predict)

    bench._select_fbcca_lda5_confidence_threshold(
        model,
        scored,
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=1,
    )

    assert call_count == len(scored)


def test_fbcca_lda5_threshold_selection_rejects_all_idle_solution() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=1.0,
        fit_summary={},
    )

    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=float(label), trial_id=index, block_index=0),
            score_matrix=np.zeros((2, 4), dtype=np.float64),
            feature_matrix=np.asarray([float(index), float(index)], dtype=np.float64).reshape(-1, 1),
            duration_sec=3.0,
        )
        for index, label in enumerate(labels[1:], start=1)
    ]
    scored.append(
        bench.ScoredTrial(
            trial=TrialSpec(label="idle", expected_freq=None, trial_id=99, block_index=0),
            score_matrix=np.zeros((2, 4), dtype=np.float64),
            feature_matrix=np.asarray([0.0, 0.0], dtype=np.float64).reshape(-1, 1),
            duration_sec=3.0,
        )
    )

    all_idle_bundle = bench._evaluate_fbcca_lda5_model(
        model,
        scored,
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=1,
    )
    selection = bench._select_fbcca_lda5_confidence_threshold(
        model,
        scored,
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=1,
    )

    assert float(all_idle_bundle["async_metrics"]["control_recall"]) == 0.0
    assert float(selection["selected_metrics"]["control_recall"]) >= 0.80
    assert float(selection["command_confidence_th"]) < 1.0


def test_evaluate_fbcca_lda5_model_reports_control_recall_deadlines() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    dims = 10
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(dims, dtype=np.float64),
        feature_std=np.ones(dims, dtype=np.float64),
        class_means=np.vstack([np.full(dims, float(index), dtype=np.float64) for index in range(len(labels))]),
        pooled_var=np.ones(dims, dtype=np.float64),
        command_confidence_th=0.50,
        fit_summary={},
    )
    fast_trial = bench.ScoredTrial(
        trial=TrialSpec(label="9.8", expected_freq=9.8, trial_id=1, block_index=0),
        score_matrix=np.zeros((4, 4), dtype=np.float64),
        feature_matrix=np.vstack([np.full(dims, 1.0, dtype=np.float64) for _ in range(4)]),
        duration_sec=4.0,
    )
    slow_trial = bench.ScoredTrial(
        trial=TrialSpec(label="12", expected_freq=12.0, trial_id=2, block_index=0),
        score_matrix=np.zeros((5, 4), dtype=np.float64),
        feature_matrix=np.vstack(
            [
                np.full(dims, 0.0, dtype=np.float64),
                np.full(dims, 0.0, dtype=np.float64),
                np.full(dims, 0.0, dtype=np.float64),
                np.full(dims, 2.0, dtype=np.float64),
                np.full(dims, 2.0, dtype=np.float64),
                np.full(dims, 2.0, dtype=np.float64),
            ]
        ),
        duration_sec=4.0,
    )

    bundle = bench._evaluate_fbcca_lda5_model(
        model,
        [fast_trial, slow_trial],
        win_sec=1.5,
        step_sec=0.5,
        min_enter_windows=1,
    )

    metrics = dict(bundle["async_metrics"])
    assert float(metrics["control_recall"]) == 1.0
    assert float(metrics["control_recall_at_2s"]) == 0.5
    assert float(metrics["control_recall_at_2.5s"]) == 0.5
    assert float(metrics["control_recall_at_3s"]) == 1.0


def test_fbcca_lda5_idle_window_selection_counts_windows_not_trials() -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    dims = 10
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(dims, dtype=np.float64),
        feature_std=np.ones(dims, dtype=np.float64),
        class_means=np.vstack([np.full(dims, float(index), dtype=np.float64) for index in range(len(labels))]),
        pooled_var=np.ones(dims, dtype=np.float64),
        command_confidence_th=0.50,
        fit_summary={},
    )
    trial = bench.ScoredTrial(
        trial=TrialSpec(label="hard_idle", expected_freq=None, trial_id=1, block_index=0),
        score_matrix=np.zeros((3, 4), dtype=np.float64),
        feature_matrix=np.vstack(
            [
                np.full(dims, 4.0, dtype=np.float64),
                np.full(dims, 3.0, dtype=np.float64),
                np.full(dims, 0.0, dtype=np.float64),
            ]
        ),
        duration_sec=6.0,
    )

    bundle = bench._evaluate_fbcca_lda5_model(
        model,
        [trial],
        win_sec=1.5,
        step_sec=0.25,
        min_enter_windows=2,
    )

    metrics = dict(bundle["async_metrics"])
    assert float(metrics["idle_fp_trials"]) == 0.0
    assert float(metrics["idle_selected_windows"]) == 2.0
    assert abs(float(metrics["idle_selected_windows_per_min"]) - 20.0) < 1e-9


def test_score_trials_for_classifier_falls_back_to_single_window_api() -> None:
    class SingleWindowDecoder:
        win_samples = 2
        step_samples = 1
        fs = 2

        def score_window(self, window: np.ndarray) -> np.ndarray:
            return np.asarray([float(np.sum(window)), 1.0, 2.0, 3.0], dtype=np.float64)

    segment = np.arange(8, dtype=np.float64).reshape(4, 2)
    trial = TrialSpec(label="9.8", expected_freq=9.8, trial_id=1, block_index=0)

    scored = bench._score_trials_for_classifier(
        trial_segments=[(trial, segment)],
        decoder=SingleWindowDecoder(),
    )

    assert len(scored) == 1
    assert scored[0].score_matrix.shape == (3, 4)
    assert scored[0].feature_matrix.shape[0] == 3
    assert float(scored[0].score_matrix[0, 0]) == 6.0


def test_score_trials_for_classifier_can_use_analyze_window_scores() -> None:
    class AnalyzeOnlyDecoder:
        win_samples = 2
        step_samples = 1
        fs = 2

        def analyze_window(self, window: np.ndarray) -> dict[str, np.ndarray]:
            return {"scores": np.asarray([float(np.mean(window)), 1.0, 2.0, 3.0], dtype=np.float64)}

    segment = np.arange(8, dtype=np.float64).reshape(4, 2)
    trial = TrialSpec(label="12", expected_freq=12.0, trial_id=1, block_index=0)

    scored = bench._score_trials_for_classifier(
        trial_segments=[(trial, segment)],
        decoder=AnalyzeOnlyDecoder(),
    )

    assert len(scored) == 1
    assert scored[0].score_matrix.shape == (3, 4)
    assert float(scored[0].score_matrix[0, 0]) == 1.5


def test_fbcca_lda5_method_uses_precomputed_scored_trials(monkeypatch, tmp_path: Path) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    feature_count = len(bench._classifier_feature_names(freqs))
    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=None if label == "idle" else float(label), trial_id=index, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.zeros((1, feature_count), dtype=np.float64),
            duration_sec=3.0,
        )
        for index, label in enumerate(labels)
    ]
    model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        class_means=np.zeros((5, feature_count), dtype=np.float64),
        pooled_var=np.ones(feature_count, dtype=np.float64),
        command_confidence_th=0.0,
        fit_summary={"source": "precomputed"},
    )

    def fail_score_once(**_kwargs):
        raise AssertionError("scoring should be skipped when precomputed scored trials are supplied")

    monkeypatch.setattr(bench, "_score_split_once", fail_score_once)
    monkeypatch.setattr(bench, "_fit_fbcca_lda5_model", lambda _scored, **_kwargs: model)
    monkeypatch.setattr(
        bench,
        "_evaluate_fbcca_lda5_model",
        lambda *_args, **_kwargs: {
            "fixed_window_metrics_4class": {"acc": 1.0, "macro_f1": 1.0},
            "fixed_window_metrics_5class": {"acc": 1.0, "macro_f1": 1.0},
            "async_lens_metrics_4class": {"acc": 1.0, "macro_f1": 1.0},
            "async_lens_metrics_5class": {"acc": 1.0, "macro_f1": 1.0},
            "async_metrics": {
                "control_recall": 1.0,
                "control_recall_at_2s": 1.0,
                "control_recall_at_2.5s": 1.0,
                "control_recall_at_3s": 1.0,
            },
        },
    )

    row = bench.run_fbcca_lda5_method(
        artifact_dir=tmp_path,
        spec=bench.ExternalSubjectSpec("wang2016", "S1", Path("S1.mat"), freqs),
        split_plan=bench.SplitPlan("S1", "wang2016", 0, 1, (0,), (1,)),
        split_summary={"idle_multiplier": 1.0},
        sampling_rate=250,
        freqs=freqs,
        calibration_segments=[],
        holdout_segments=[],
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        step_sec=0.25,
        win_sec=1.5,
        min_enter_windows=1,
        calibration_scored=scored,
        holdout_scored=scored,
    )

    assert row["calibration_profile"]["fit_summary"] == {"source": "precomputed"}
    assert float(row["summary_metrics"]["control_recall_at_2.5s"]) == 1.0
    artifact_path = Path(str(row["calibration_profile"]["candidate_artifact_path"]))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["artifact_schema_version"] == "external_fbcca_classifier_candidate_v1"
    assert artifact["model_name"] == "fbcca_score_lda_5class"
    assert artifact["runtime_loadable"] is False
    assert artifact["training_provenance"]["required_channel_names"] == list(bench.WANG2016_REQUIRED_CHANNELS)
    assert artifact["training_provenance"]["only_required_channels_used"] is True
    assert artifact["feature_contract"]["feature_names"] == bench._classifier_feature_names(freqs)
    assert len(artifact["state"]["feature_mean"]) == feature_count


def test_fbcca_ridge5_method_uses_precomputed_scored_trials(monkeypatch, tmp_path: Path) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    feature_count = len(bench._classifier_feature_names(freqs))
    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=None if label == "idle" else float(label), trial_id=index, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.zeros((1, feature_count), dtype=np.float64),
            duration_sec=3.0,
        )
        for index, label in enumerate(labels)
    ]
    model = bench.FBCCARidge5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        weights=np.zeros((feature_count + 1, 5), dtype=np.float64),
        l2=0.3,
        command_confidence_th=0.0,
        fit_summary={"source": "precomputed"},
    )

    def fail_score_once(**_kwargs):
        raise AssertionError("scoring should be skipped when precomputed scored trials are supplied")

    monkeypatch.setattr(bench, "_score_split_once", fail_score_once)
    monkeypatch.setattr(bench, "_fit_fbcca_ridge5_model", lambda _scored, **_kwargs: model)
    monkeypatch.setattr(
        bench,
        "_evaluate_fbcca_lda5_model",
        lambda *_args, **_kwargs: {
            "fixed_window_metrics_4class": {"acc": 1.0, "macro_f1": 1.0},
            "fixed_window_metrics_5class": {"acc": 1.0, "macro_f1": 1.0},
            "async_lens_metrics_4class": {"acc": 1.0, "macro_f1": 1.0},
            "async_lens_metrics_5class": {"acc": 1.0, "macro_f1": 1.0},
            "async_metrics": {
                "control_recall": 1.0,
                "control_recall_at_2s": 1.0,
                "control_recall_at_2.5s": 1.0,
                "control_recall_at_3s": 1.0,
            },
        },
    )

    row = bench.run_fbcca_ridge5_method(
        artifact_dir=tmp_path,
        spec=bench.ExternalSubjectSpec("wang2016", "S1", Path("S1.mat"), freqs),
        split_plan=bench.SplitPlan("S1", "wang2016", 0, 1, (0,), (1,)),
        split_summary={"idle_multiplier": 1.0},
        sampling_rate=250,
        freqs=freqs,
        calibration_segments=[],
        holdout_segments=[],
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        step_sec=0.25,
        win_sec=1.5,
        min_enter_windows=1,
        calibration_scored=scored,
        holdout_scored=scored,
    )

    assert row["calibration_profile"]["fit_summary"] == {"source": "precomputed"}
    assert float(row["calibration_profile"]["l2"]) == 0.3
    artifact_path = Path(str(row["calibration_profile"]["candidate_artifact_path"]))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["model_name"] == "fbcca_score_ridge_5class"
    assert artifact["state"]["l2"] == 0.3
    assert len(artifact["state"]["weights"]) == feature_count + 1


def test_score_based_method_artifact_propagates_decoder_metadata(monkeypatch, tmp_path: Path) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    score_source_name = "itcca"
    feature_count = len(bench._classifier_feature_names(freqs, score_source_name=score_source_name))
    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=None if label == "idle" else float(label), trial_id=index, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.zeros((1, feature_count), dtype=np.float64),
            duration_sec=3.0,
        )
        for index, label in enumerate(labels)
    ]
    model = bench.FBCCARidge5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        weights=np.zeros((feature_count + 1, len(labels)), dtype=np.float64),
        l2=0.3,
        command_confidence_th=0.0,
        fit_summary={"source": "precomputed"},
    )

    monkeypatch.setattr(bench, "_fit_fbcca_ridge5_model", lambda _scored, **_kwargs: model)
    monkeypatch.setattr(
        bench,
        "_evaluate_fbcca_lda5_model",
        lambda *_args, **_kwargs: {
            "fixed_window_metrics_4class": {"acc": 1.0, "macro_f1": 1.0},
            "fixed_window_metrics_5class": {"acc": 1.0, "macro_f1": 1.0},
            "async_lens_metrics_4class": {"acc": 1.0, "macro_f1": 1.0},
            "async_lens_metrics_5class": {"acc": 1.0, "macro_f1": 1.0},
            "async_metrics": {
                "control_recall": 1.0,
                "control_recall_at_2s": 1.0,
                "control_recall_at_2.5s": 1.0,
                "control_recall_at_3s": 1.0,
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.0,
                "detection_latency_s": 1.0,
            },
        },
    )

    row = bench.run_fbcca_ridge5_method(
        artifact_dir=tmp_path,
        spec=bench.ExternalSubjectSpec("wang2016", "S1", Path("S1.mat"), freqs),
        split_plan=bench.SplitPlan("S1", "wang2016", 0, 1, (0,), (1,)),
        split_summary={"idle_multiplier": 1.0},
        sampling_rate=250,
        freqs=freqs,
        calibration_segments=[],
        holdout_segments=[],
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        step_sec=0.25,
        win_sec=1.5,
        min_enter_windows=1,
        calibration_scored=scored,
        holdout_scored=scored,
        method_name="itcca5",
        score_source_name=score_source_name,
        decoder_name="itcca",
        decoder_model_params={},
    )

    artifact_path = Path(str(row["calibration_profile"]["candidate_artifact_path"]))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert row["calibration_profile"]["classifier"] == "itcca_score_ridge_5class"
    assert artifact["model_name"] == "itcca_score_ridge_5class"
    assert artifact["model_family"] == "itcca_score_classifier_5class"
    assert artifact["feature_contract"]["feature_source"] == "itcca_score_matrix"
    assert artifact["training_provenance"]["score_source_name"] == "itcca"
    assert artifact["training_provenance"]["decoder_name"] == "itcca"


def test_score_segment_subset_cached_reuses_overlap_and_preserves_order(monkeypatch) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    feature_count = len(bench._classifier_feature_names(freqs))
    idle1 = (
        TrialSpec(label="idle_a", expected_freq=None, trial_id=1, block_index=0),
        np.zeros((500, 8), dtype=np.float64),
    )
    idle2 = (
        TrialSpec(label="idle_b", expected_freq=None, trial_id=2, block_index=0),
        np.zeros((500, 8), dtype=np.float64),
    )
    commands = [
        (
            TrialSpec(label=f"{float(freq):g}Hz", expected_freq=float(freq), trial_id=10 + index, block_index=0),
            np.zeros((500, 8), dtype=np.float64),
        )
        for index, freq in enumerate(freqs)
    ]
    score_call_sizes: list[int] = []

    class FakeDecoder:
        win_samples = 1
        step_samples = 1
        fs = 250

    monkeypatch.setattr(bench, "_build_fbcca_decoder_for_scoring", lambda **_kwargs: FakeDecoder())

    def fake_score_trials(*, trial_segments, decoder):
        assert isinstance(decoder, FakeDecoder)
        score_call_sizes.append(len(trial_segments))
        scored = []
        for trial, _segment in trial_segments:
            scored.append(
                bench.ScoredTrial(
                    trial=trial,
                    score_matrix=np.zeros((1, len(freqs)), dtype=np.float64),
                    feature_matrix=np.zeros((1, feature_count), dtype=np.float64),
                    duration_sec=2.0,
                )
            )
        return scored

    monkeypatch.setattr(bench, "_score_trials_for_classifier", fake_score_trials)
    decoder_cache = {}
    scored_cache = {}

    first = bench._score_segment_subset_cached(
        freqs=freqs,
        sampling_rate=250,
        step_sec=0.25,
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        win_sec=1.5,
        segments=[idle1, *commands],
        context="first calibration",
        decoder_cache=decoder_cache,
        scored_cache=scored_cache,
    )
    second_segments = [commands[2], idle2, commands[0], commands[3], commands[1]]
    second = bench._score_segment_subset_cached(
        freqs=freqs,
        sampling_rate=250,
        step_sec=0.25,
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        win_sec=1.5,
        segments=second_segments,
        context="second calibration",
        decoder_cache=decoder_cache,
        scored_cache=scored_cache,
    )

    assert score_call_sizes == [5, 1]
    assert [item.trial.trial_id for item in first] == [1, 10, 11, 12, 13]
    assert [item.trial.trial_id for item in second] == [12, 2, 10, 13, 11]
    assert len(decoder_cache) == 1


def test_aggregate_recipe_rows_uses_subject_level_means() -> None:
    rows = [
        {
            "method": "fast_fbcca",
            "recipe_id": "win2_tw0p25",
            "dataset": "wang2016",
            "subject": "S1",
            "calibration_blocks": [0, 1],
            "split_summary": {"idle_multiplier": 1.0},
            "summary_metrics": {
                "fixed_acc_4class": 0.9,
                "fixed_macro_f1_4class": 0.89,
                "fixed_acc_5class": 0.8,
                "fixed_macro_f1_5class": 0.7,
                "async_acc_4class": 0.88,
                "async_macro_f1_4class": 0.87,
                "async_acc_5class": 0.75,
                "async_macro_f1_5class": 0.65,
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.2,
                "control_recall": 0.9,
                "control_recall_at_2s": 0.5,
                "control_recall_at_2.5s": 0.7,
                "control_recall_at_3s": 0.85,
                "detection_latency_s": 2.0,
                "switch_latency_s": 2.1,
                "release_latency_s": 1.5,
            },
        },
        {
            "method": "fast_fbcca",
            "recipe_id": "win2_tw0p25",
            "dataset": "wang2016",
            "subject": "S1",
            "calibration_blocks": [0, 1],
            "split_summary": {"idle_multiplier": 1.0},
            "summary_metrics": {
                "fixed_acc_4class": 0.7,
                "fixed_macro_f1_4class": 0.69,
                "fixed_acc_5class": 0.6,
                "fixed_macro_f1_5class": 0.5,
                "async_acc_4class": 0.68,
                "async_macro_f1_4class": 0.67,
                "async_acc_5class": 0.55,
                "async_macro_f1_5class": 0.45,
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.4,
                "control_recall": 0.7,
                "control_recall_at_2s": 0.3,
                "control_recall_at_2.5s": 0.5,
                "control_recall_at_3s": 0.65,
                "detection_latency_s": 2.4,
                "switch_latency_s": 2.4,
                "release_latency_s": 1.7,
            },
        },
        {
            "method": "fast_fbcca",
            "recipe_id": "win2_tw0p25",
            "dataset": "beta",
            "subject": "S16",
            "calibration_blocks": [0, 1],
            "split_summary": {"idle_multiplier": 1.0},
            "summary_metrics": {
                "fixed_acc_4class": 0.96,
                "fixed_macro_f1_4class": 0.95,
                "fixed_acc_5class": 0.9,
                "fixed_macro_f1_5class": 0.88,
                "async_acc_4class": 0.93,
                "async_macro_f1_4class": 0.92,
                "async_acc_5class": 0.86,
                "async_macro_f1_5class": 0.84,
                "idle_fp_per_min": 0.2,
                "idle_selected_windows_per_min": 0.3,
                "control_recall": 0.95,
                "control_recall_at_2s": 0.75,
                "control_recall_at_2.5s": 0.9,
                "control_recall_at_3s": 0.92,
                "detection_latency_s": 1.6,
                "switch_latency_s": 1.9,
                "release_latency_s": 1.3,
            },
        },
    ]

    summaries = bench.aggregate_recipe_rows(rows)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["subject_count"] == 2
    assert summary["split_count"] == 3
    assert abs(float(summary["mean_fixed_acc_4class"]) - 0.88) < 1e-9
    assert abs(float(summary["mean_fixed_acc_5class"]) - 0.8) < 1e-9
    assert abs(float(summary["mean_fixed_macro_f1_5class"]) - 0.74) < 1e-9
    assert abs(float(summary["mean_control_recall_at_2s"]) - 0.575) < 1e-9
    assert abs(float(summary["mean_control_recall_at_2.5s"]) - 0.75) < 1e-9
    assert abs(float(summary["mean_detection_latency_s"]) - 1.9) < 1e-9


def test_aggregate_recipe_rows_tracks_shared_coverage() -> None:
    rows = [
        _aggregate_test_row(
            subject="S1",
            recipe_id="win2_me1",
            async_macro_f1_5class=0.86,
            async_acc_5class=0.92,
            idle_fp_per_min=0.5,
            control_recall=0.9,
            detection_latency_s=2.0,
        ),
        _aggregate_test_row(
            subject="S16",
            recipe_id="win2_me1",
            async_macro_f1_5class=0.88,
            async_acc_5class=0.94,
            idle_fp_per_min=0.5,
            control_recall=0.9,
            detection_latency_s=2.0,
        ),
        _aggregate_test_row(
            subject="S16",
            recipe_id="win3_me1",
            async_macro_f1_5class=1.0,
            async_acc_5class=1.0,
            idle_fp_per_min=0.0,
            control_recall=1.0,
            detection_latency_s=1.5,
        ),
    ]

    summaries = bench.aggregate_recipe_rows(rows, expected_subject_count=2)
    summaries_by_recipe = {str(summary["recipe_id"]): summary for summary in summaries}

    assert summaries[0]["recipe_id"] == "win3_me1"
    assert summaries_by_recipe["win2_me1"]["coverage_subject_count"] == 2
    assert summaries_by_recipe["win2_me1"]["expected_subject_count"] == 2
    assert summaries_by_recipe["win2_me1"]["shared_eligible"] is True
    assert summaries_by_recipe["win3_me1"]["coverage_subject_count"] == 1
    assert summaries_by_recipe["win3_me1"]["expected_subject_count"] == 2
    assert summaries_by_recipe["win3_me1"]["shared_eligible"] is False

    shared_summaries = [summary for summary in summaries if summary["shared_eligible"]]
    assert [summary["recipe_id"] for summary in shared_summaries] == ["win2_me1"]


def test_render_markdown_summary_lists_shared_recipes_with_coverage() -> None:
    rows = [
        _aggregate_test_row(
            subject="S1",
            recipe_id="win2_me1",
            async_macro_f1_5class=0.86,
            async_acc_5class=0.92,
            idle_fp_per_min=0.5,
            control_recall=0.9,
            detection_latency_s=2.0,
        ),
        _aggregate_test_row(
            subject="S16",
            recipe_id="win2_me1",
            async_macro_f1_5class=0.88,
            async_acc_5class=0.94,
            idle_fp_per_min=0.5,
            control_recall=0.9,
            detection_latency_s=2.0,
        ),
        _aggregate_test_row(
            subject="S16",
            recipe_id="win3_me1",
            async_macro_f1_5class=1.0,
            async_acc_5class=1.0,
            idle_fp_per_min=0.0,
            control_recall=1.0,
            detection_latency_s=1.5,
        ),
    ]
    summaries = bench.aggregate_recipe_rows(rows, expected_subject_count=2)
    subjects = [
        bench.ExternalSubjectSpec(
            dataset="beta",
            subject="S1",
            mat_path=Path("S1.mat"),
            freqs=(9.8, 12.0, 14.8, 15.8),
        ),
        bench.ExternalSubjectSpec(
            dataset="beta",
            subject="S16",
            mat_path=Path("S16.mat"),
            freqs=(9.8, 12.0, 14.8, 15.8),
        ),
    ]

    markdown = bench.render_markdown_summary(
        run_id="coverage_test",
        freqs=(9.8, 12.0, 14.8, 15.8),
        subjects=subjects,
        rows=rows,
        summaries=summaries,
    )

    assert markdown.index("## Top Shared Recipes") < markdown.index("## Top Recipes")
    assert "| Rank | Method | Recipe | Coverage |" in markdown
    assert "| 1 | fbcca_ridge5 | `win2_me1` | 2/2 |" in markdown


def test_aggregate_recipe_rows_uses_aggregate_recipe_id_when_present() -> None:
    rows = [
        {
            "method": "threshold_pretrain",
            "recipe_id": "search_w1p5_gpbalanced_me1_mx1_csunified",
            "aggregate_recipe_id": "selected_policy_grid_search",
            "selected_recipe_id": "search_w1p5_gpbalanced_me1_mx1_csunified",
            "dataset": "beta",
            "subject": "S1",
            "calibration_blocks": [0],
            "split_summary": {"idle_multiplier": 1.0},
            "summary_metrics": {
                "fixed_acc_4class": 0.9,
                "fixed_macro_f1_4class": 0.9,
                "fixed_acc_5class": 0.7,
                "fixed_macro_f1_5class": 0.68,
                "async_acc_4class": 0.82,
                "async_macro_f1_4class": 0.81,
                "async_acc_5class": 0.62,
                "async_macro_f1_5class": 0.60,
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.0,
                "control_recall": 0.88,
                "control_recall_at_2s": 0.40,
                "control_recall_at_2.5s": 0.70,
                "control_recall_at_3s": 0.84,
                "detection_latency_s": 1.8,
                "switch_latency_s": 2.0,
                "release_latency_s": 1.5,
            },
        },
        {
            "method": "threshold_pretrain",
            "recipe_id": "search_w2_gpbalanced_me2_mx1_csunified",
            "aggregate_recipe_id": "selected_policy_grid_search",
            "selected_recipe_id": "search_w2_gpbalanced_me2_mx1_csunified",
            "dataset": "beta",
            "subject": "S1",
            "calibration_blocks": [1],
            "split_summary": {"idle_multiplier": 1.0},
            "summary_metrics": {
                "fixed_acc_4class": 0.85,
                "fixed_macro_f1_4class": 0.84,
                "fixed_acc_5class": 0.66,
                "fixed_macro_f1_5class": 0.64,
                "async_acc_4class": 0.8,
                "async_macro_f1_4class": 0.79,
                "async_acc_5class": 0.58,
                "async_macro_f1_5class": 0.56,
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.1,
                "control_recall": 0.86,
                "control_recall_at_2s": 0.30,
                "control_recall_at_2.5s": 0.60,
                "control_recall_at_3s": 0.8,
                "detection_latency_s": 2.0,
                "switch_latency_s": 2.2,
                "release_latency_s": 1.6,
            },
        },
    ]

    summaries = bench.aggregate_recipe_rows(rows)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["recipe_id"] == "selected_policy_grid_search"
    assert summary["split_count"] == 2
    assert summary["selected_recipe_counts"] == {
        "search_w1p5_gpbalanced_me1_mx1_csunified": 1,
        "search_w2_gpbalanced_me2_mx1_csunified": 1,
    }
    assert abs(float(summary["mean_control_recall_at_2s"]) - 0.35) < 1e-9
    assert abs(float(summary["mean_control_recall_at_2.5s"]) - 0.65) < 1e-9


def test_aggregate_recipe_rows_groups_same_recipe_by_calibration_block_count() -> None:
    rows = [
        {
            "method": "threshold_pretrain",
            "recipe_id": "search_w2_gpbalanced_me1_mx1_csunified",
            "dataset": "beta",
            "subject": "S1",
            "calibration_blocks": [0, 1],
            "split_summary": {"idle_multiplier": 1.0},
            "summary_metrics": {
                "fixed_acc_4class": 0.9,
                "fixed_macro_f1_4class": 0.9,
                "fixed_acc_5class": 0.5,
                "fixed_macro_f1_5class": 0.48,
                "async_acc_4class": 0.8,
                "async_macro_f1_4class": 0.79,
                "async_acc_5class": 0.6,
                "async_macro_f1_5class": 0.58,
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.0,
                "control_recall": 0.85,
                "control_recall_at_2s": 0.45,
                "control_recall_at_2.5s": 0.65,
                "control_recall_at_3s": 0.8,
                "switch_latency_s": 2.0,
                "release_latency_s": 1.5,
            },
        },
        {
            "method": "threshold_pretrain",
            "recipe_id": "search_w2_gpbalanced_me1_mx1_csunified",
            "dataset": "beta",
            "subject": "S1",
            "calibration_blocks": [1, 2],
            "split_summary": {"idle_multiplier": 1.0},
            "summary_metrics": {
                "fixed_acc_4class": 0.8,
                "fixed_macro_f1_4class": 0.8,
                "fixed_acc_5class": 0.4,
                "fixed_macro_f1_5class": 0.38,
                "async_acc_4class": 0.75,
                "async_macro_f1_4class": 0.74,
                "async_acc_5class": 0.55,
                "async_macro_f1_5class": 0.53,
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.1,
                "control_recall": 0.8,
                "control_recall_at_2s": 0.35,
                "control_recall_at_2.5s": 0.55,
                "control_recall_at_3s": 0.75,
                "switch_latency_s": 2.2,
                "release_latency_s": 1.6,
            },
        },
    ]

    summaries = bench.aggregate_recipe_rows(rows)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["calibration_block_count"] == 2
    assert summary["split_count"] == 2
    assert summary["calibration_blocks"] == []
    assert summary["calibration_block_patterns"] == [[0, 1], [1, 2]]


def test_summary_rank_rejects_all_idle_recipe_before_idle_tiebreakers() -> None:
    all_idle = {
        "method": "fbcca_ridge5",
        "recipe_id": "win3_me2",
        "mean_idle_fp_per_min": 0.0,
        "mean_idle_selected_windows_per_min": 0.0,
        "mean_control_recall": 0.0,
        "mean_control_recall_at_2s": 0.0,
        "mean_control_recall_at_2.5s": 0.0,
        "mean_control_recall_at_3s": 0.0,
        "mean_async_macro_f1_5class": 0.18,
        "mean_async_acc_5class": 0.9,
        "mean_fixed_macro_f1_5class": 0.9,
        "mean_fixed_acc_5class": 0.98,
        "mean_detection_latency_s": float("inf"),
    }
    useful_control = {
        "method": "fbcca_lda5",
        "recipe_id": "win1p5_me1",
        "mean_idle_fp_per_min": 0.0,
        "mean_idle_selected_windows_per_min": 0.5,
        "mean_control_recall": 0.85,
        "mean_control_recall_at_2s": 0.60,
        "mean_control_recall_at_2.5s": 0.75,
        "mean_control_recall_at_3s": 0.85,
        "mean_async_macro_f1_5class": 0.79,
        "mean_async_acc_5class": 0.96,
        "mean_fixed_macro_f1_5class": 0.86,
        "mean_fixed_acc_5class": 0.94,
        "mean_detection_latency_s": 1.9,
    }

    assert bench._summary_rank_key(useful_control) < bench._summary_rank_key(all_idle)


def test_summary_rank_allows_small_idle_budget_for_better_classifier_quality() -> None:
    zero_fp_lower_recall = {
        "method": "fbcca_lda5",
        "recipe_id": "win1p5_me1",
        "mean_idle_fp_per_min": 0.0,
        "mean_idle_selected_windows_per_min": 0.0,
        "mean_control_recall": 0.67,
        "mean_control_recall_at_2s": 0.67,
        "mean_control_recall_at_2.5s": 0.67,
        "mean_control_recall_at_3s": 0.67,
        "mean_async_macro_f1_5class": 0.80,
        "mean_async_acc_5class": 0.97,
        "mean_fixed_macro_f1_5class": 0.86,
        "mean_fixed_acc_5class": 0.93,
        "mean_detection_latency_s": 1.9,
    }
    low_fp_high_recall = {
        "method": "fbcca_lda5",
        "recipe_id": "win2p5_me2",
        "mean_idle_fp_per_min": 0.37,
        "mean_idle_selected_windows_per_min": 1.85,
        "mean_control_recall": 1.0,
        "mean_control_recall_at_2s": 0.80,
        "mean_control_recall_at_2.5s": 0.95,
        "mean_control_recall_at_3s": 1.0,
        "mean_async_macro_f1_5class": 0.94,
        "mean_async_acc_5class": 0.98,
        "mean_fixed_macro_f1_5class": 0.89,
        "mean_fixed_acc_5class": 0.97,
        "mean_detection_latency_s": 2.75,
    }

    assert bench._summary_rank_key(low_fp_high_recall) < bench._summary_rank_key(zero_fp_lower_recall)


def test_ridge5_l2_selection_rejects_all_idle_candidate(monkeypatch) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")

    def fake_base(_scored_trials, *, freqs, l2):
        return bench.FBCCARidge5Model(
            freqs=tuple(float(freq) for freq in freqs),
            labels=labels,
            feature_mean=np.zeros(1, dtype=np.float64),
            feature_std=np.ones(1, dtype=np.float64),
            weights=np.zeros((2, len(labels)), dtype=np.float64),
            command_confidence_th=0.0,
            l2=float(l2),
            fit_summary={"l2": float(l2)},
        )

    def fake_threshold(base_model, _scored_trials, **_kwargs):
        if abs(float(base_model.l2) - 0.03) < 1e-12:
            selected = {
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.0,
                "control_recall": 0.0,
                "control_recall_at_2s": 0.0,
                "control_recall_at_2.5s": 0.0,
                "control_recall_at_3s": 0.0,
                "async_macro_f1_5class": 0.18,
                "async_acc_5class": 0.9,
                "fixed_macro_f1_5class": 0.9,
                "fixed_acc_5class": 0.98,
                "detection_latency_s": float("inf"),
            }
        else:
            selected = {
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.4,
                "control_recall": 0.85,
                "control_recall_at_2s": 0.65,
                "control_recall_at_2.5s": 0.75,
                "control_recall_at_3s": 0.85,
                "async_macro_f1_5class": 0.78,
                "async_acc_5class": 0.95,
                "fixed_macro_f1_5class": 0.86,
                "fixed_acc_5class": 0.94,
                "detection_latency_s": 1.8,
            }
        return {"command_confidence_th": 0.5, "selected_metrics": selected}

    monkeypatch.setattr(bench, "_fit_fbcca_ridge5_base_model", fake_base)
    monkeypatch.setattr(bench, "_select_fbcca_lda5_confidence_threshold", fake_threshold)

    model = bench._fit_fbcca_ridge5_model(
        [],
        freqs=freqs,
        l2_candidates=(0.03, 0.1),
    )

    assert float(model.l2) == 0.1


def test_lda5_fit_uses_supplied_base_model(monkeypatch) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    base_model = bench.FBCCALDA5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.zeros((5, 1), dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=0.0,
        fit_summary={"source": "cached"},
    )

    def fail_base_fit(*_args, **_kwargs):
        raise AssertionError("cached base model should skip base fitting")

    monkeypatch.setattr(bench, "_fit_fbcca_lda5_base_model", fail_base_fit)
    monkeypatch.setattr(
        bench,
        "_select_fbcca_lda5_confidence_threshold",
        lambda *_args, **_kwargs: {
            "command_confidence_th": 0.42,
            "selected_metrics": {"control_recall": 1.0, "idle_fp_per_min": 0.0},
        },
    )

    model = bench._fit_fbcca_lda5_model([], freqs=freqs, base_model=base_model)

    assert float(model.command_confidence_th) == 0.42
    assert model.fit_summary["source"] == "cached"


def test_ridge5_fit_uses_supplied_base_models(monkeypatch) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    labels = ("idle", "9.8", "12", "14.8", "15.8")
    base_models = [
        bench.FBCCARidge5Model(
            freqs=freqs,
            labels=labels,
            feature_mean=np.zeros(1, dtype=np.float64),
            feature_std=np.ones(1, dtype=np.float64),
            weights=np.zeros((2, 5), dtype=np.float64),
            l2=l2,
            command_confidence_th=0.0,
            fit_summary={"source": "cached", "l2": l2},
        )
        for l2 in (0.03, 0.1)
    ]

    def fail_base_fit(*_args, **_kwargs):
        raise AssertionError("cached ridge base models should skip base fitting")

    def fake_threshold(base_model, _scored_trials, **_kwargs):
        control_recall = 0.2 if abs(float(base_model.l2) - 0.03) < 1e-12 else 1.0
        return {
            "command_confidence_th": 0.5,
            "selected_metrics": {
                "idle_fp_per_min": 0.0,
                "idle_selected_windows_per_min": 0.0,
                "control_recall": control_recall,
                "control_recall_at_2s": control_recall,
                "control_recall_at_2.5s": control_recall,
                "control_recall_at_3s": control_recall,
                "async_macro_f1_5class": control_recall,
                "async_acc_5class": control_recall,
                "fixed_macro_f1_5class": control_recall,
                "fixed_acc_5class": control_recall,
                "detection_latency_s": 1.0,
            },
        }

    monkeypatch.setattr(bench, "_fit_fbcca_ridge5_base_model", fail_base_fit)
    monkeypatch.setattr(bench, "_select_fbcca_lda5_confidence_threshold", fake_threshold)

    model = bench._fit_fbcca_ridge5_model([], freqs=freqs, base_models=base_models)

    assert float(model.l2) == 0.1
    assert model.fit_summary["source"] == "cached"

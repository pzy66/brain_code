from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

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
    ns2_fp_per_min: float = 0.0,
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
            "mixed_idle_fp_per_min": idle_fp_per_min,
            "idle_selected_windows_per_min": idle_fp_per_min,
            "control_recall": control_recall,
            "control_recall_at_2s": control_recall,
            "control_recall_at_2.5s": control_recall,
            "control_recall_at_3s": control_recall,
            "detection_latency_s": detection_latency_s,
            "switch_latency_s": detection_latency_s,
            "release_latency_s": detection_latency_s,
            "ns1_fp_per_min": 0.0,
            "ns2_fp_per_min": ns2_fp_per_min,
            "ns3_fp_per_min": 0.0,
            "ns_all_fp_per_min": ns2_fp_per_min,
        },
    }


def test_enumerate_external_subjects_orders_and_limits(tmp_path: Path) -> None:
    wang_raw = tmp_path / "wang"
    beta_raw = tmp_path / "beta"
    ysu_raw = tmp_path / "ysu"
    wang_raw.mkdir()
    beta_raw.mkdir()
    ysu_raw.mkdir()
    (wang_raw / "S2.mat").write_text("", encoding="utf-8")
    (wang_raw / "S1.mat").write_text("", encoding="utf-8")
    (beta_raw / "S16.mat").write_text("", encoding="utf-8")
    (ysu_raw / "S01").mkdir()
    (tmp_path / "64-channels.loc").write_text("", encoding="utf-8")

    rows = bench.enumerate_external_subjects(
        datasets=("wang2016", "beta", "ysu_an"),
        freqs=(9.8, 12.0, 14.8, 15.8),
        wang_raw_dir=wang_raw,
        wang_channels_loc=tmp_path / "64-channels.loc",
        beta_raw_dir=beta_raw,
        ysu_an_raw_dir=ysu_raw,
        ysu_an_channel_loc=tmp_path / "Channel Loc.xlsx",
        subject_limit_per_dataset=1,
    )

    assert [(row.dataset, row.subject) for row in rows] == [
        ("wang2016", "S1"),
        ("beta", "S16"),
        ("ysu_an", "S01"),
    ]


def test_parse_subject_whitelist_supports_global_and_dataset_scoped_tokens() -> None:
    parsed = bench._parse_subject_whitelist("S1,beta:S16,wang2016:S2,ysu_an:S01")
    assert parsed == (("*", "S1"), ("beta", "S16"), ("wang2016", "S2"), ("ysu_an", "S01"))


def test_classifier_gate_variant_parser_and_tokens_keep_baseline_recipe_id() -> None:
    parsed = bench._csv_gate_variant_tuple("baseline_lrtmw,lrtmw_margin_gate,baseline")

    baseline_recipe = bench._classifier_recipe_id_with_smoothing(
        win_sec=2.0,
        min_enter_windows=2,
        smoothing_windows=3,
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        gate_variant=bench.CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
    )
    margin_params = {
        "gate_variant": bench.CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN,
        "margin_control_quantile": 0.05,
        "margin_idle_quantile": 0.95,
        "ratio_idle_quantile": 0.975,
    }
    margin_recipe = bench._classifier_recipe_id_with_smoothing(
        win_sec=2.0,
        min_enter_windows=2,
        smoothing_windows=3,
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        gate_variant=bench.CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN,
        variant_token=bench._classifier_gate_variant_token(bench.CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN, margin_params),
    )

    assert parsed == (bench.CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW, bench.CLASSIFIER_GATE_VARIANT_LRTMW_MARGIN)
    assert baseline_recipe == "win2_me2_sm3_lrtmw"
    assert margin_recipe.startswith("win2_me2_sm3_lrtmw_mg0p05_0p95_0p975")


def test_frequency_specific_gate_variants_parser_and_grid() -> None:
    parsed = bench._csv_gate_variant_tuple("freqspec_threshold,frequency-specific-logistic,baseline")

    assert parsed == (
        bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        bench.CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
    )
    assert len(bench._gate_variant_param_grid(bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD)) == 108
    assert len(bench._gate_variant_param_grid(bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC)) == 16
    token = bench._classifier_gate_variant_token(
        bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        {
            "margin_idle_quantile": 0.95,
            "ratio_idle_quantile": 0.975,
            "entropy_control_quantile": 0.85,
            "ns2_safety_factor": 1.2,
        },
    )
    assert token == "fsth0p95_0p975_0p85_1p2"


def test_frequency_specific_gate_grid_can_be_limited_for_smoke() -> None:
    threshold_grid = bench._gate_variant_param_grid(
        bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        freqspec_margin_idle_quantiles=(0.90,),
        freqspec_ratio_idle_quantiles=(0.90, 0.95),
        freqspec_entropy_control_quantiles=(0.80,),
        freqspec_ns2_safety_factors=(1.0, 1.2),
    )
    logistic_grid = bench._gate_variant_param_grid(
        bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        freqspec_logistic_prob_thresholds=(0.6, 0.8),
        freqspec_logistic_ns2_weights=(1.0,),
    )

    assert len(threshold_grid) == 4
    assert {item["ratio_idle_quantile"] for item in threshold_grid} == {0.90, 0.95}
    assert {item["ns2_safety_factor"] for item in threshold_grid} == {1.0, 1.2}
    assert len(logistic_grid) == 2
    assert {item["prob_threshold"] for item in logistic_grid} == {0.6, 0.8}


def test_frequency_specific_threshold_priority6_combo_set_is_exact() -> None:
    grid = bench._gate_variant_param_grid(
        bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        freqspec_threshold_combo_set=bench.FREQSPEC_THRESHOLD_COMBO_SET_PRIORITY6,
        freqspec_margin_idle_quantiles=(0.975,),
        freqspec_ratio_idle_quantiles=(0.975,),
        freqspec_entropy_control_quantiles=(0.80,),
        freqspec_ns2_safety_factors=(1.3,),
    )

    assert len(grid) == 6
    assert [item["combo_name"] for item in grid] == [
        "mild",
        "balanced",
        "ns2_strict",
        "recall_safe",
        "margin_only-ish",
        "ratio_ns2",
    ]
    assert grid[0] == {
        "gate_variant": bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        "combo_name": "mild",
        "margin_idle_quantile": 0.90,
        "ratio_idle_quantile": 0.90,
        "entropy_control_quantile": 0.90,
        "ns2_safety_factor": 1.0,
    }
    assert grid[2]["ns2_safety_factor"] == 1.2


def test_ns2_and_subject_floor_gate_grids_can_be_limited_for_round2() -> None:
    ns2_grid = bench._gate_variant_param_grid(
        bench.CLASSIFIER_GATE_VARIANT_NS2_AWARE,
        ns2_safety_factors=(1.0, 1.2),
    )
    floor_grid = bench._gate_variant_param_grid(
        bench.CLASSIFIER_GATE_VARIANT_SUBJECT_THRESHOLD_FLOOR,
        subject_floor_global_quantiles=(0.90, 0.95),
        subject_floor_idle_quantiles=(0.95,),
    )
    combo_grid = bench._gate_variant_param_grid(
        bench.CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
        subject_floor_global_quantiles=(0.90,),
        subject_floor_idle_quantiles=(0.95, 0.975),
        ns2_safety_factors=(1.0, 1.2),
    )
    combo_token = bench._classifier_gate_variant_token(
        bench.CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
        combo_grid[-1],
    )

    assert ns2_grid == [
        {"gate_variant": bench.CLASSIFIER_GATE_VARIANT_NS2_AWARE, "ns2_safety_factor": 1.0},
        {"gate_variant": bench.CLASSIFIER_GATE_VARIANT_NS2_AWARE, "ns2_safety_factor": 1.2},
    ]
    assert len(floor_grid) == 2
    assert {item["subject_idle_quantile"] for item in floor_grid} == {0.95}
    assert len(combo_grid) == 4
    assert combo_token == "floorns20p9_0p975_1p2"


def test_parser_exposes_ns2_and_subject_floor_round2_options() -> None:
    parser = bench.build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "round2",
            "--output-root",
            "out",
            "--dataset-root",
            "data",
            "--wang-raw-dir",
            "wang",
            "--wang-channels-loc",
            "loc",
            "--beta-raw-dir",
            "beta",
            "--classifier-gate-variants",
            "ns2_aware_gate,subject_floor_ns2_aware_gate",
            "--ns2-safety-factors",
            "1.0,1.2",
            "--subject-floor-global-quantiles",
            "0.90,0.95",
            "--subject-floor-idle-quantiles",
            "0.95,0.975",
        ]
    )

    assert bench._csv_gate_variant_tuple(args.classifier_gate_variants) == (
        bench.CLASSIFIER_GATE_VARIANT_NS2_AWARE,
        bench.CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
    )
    assert bench._csv_float_tuple(args.ns2_safety_factors, default=()) == (1.0, 1.2)
    assert bench._csv_float_tuple(args.subject_floor_global_quantiles, default=()) == (0.90, 0.95)
    assert bench._csv_float_tuple(args.subject_floor_idle_quantiles, default=()) == (0.95, 0.975)


def test_lrt_shape_gate_rejects_low_margin_windows() -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    labels = ("idle", "8", "10.5", "12", "15")
    feature_count = len(bench._classifier_feature_names(freqs))
    model = bench.FBCCARidge5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        weights=np.zeros((feature_count + 1, len(labels)), dtype=np.float64),
        l2=0.1,
        command_confidence_th=0.1,
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        lrt_window_th=0.2,
        lrt_enter_th=0.0,
        score_shape_margin_index=bench._feature_index(bench._classifier_feature_names(freqs), "margin"),
        score_shape_margin_th=1.0,
        fit_summary={},
    )
    probs = np.asarray([[0.1, 0.8, 0.05, 0.03, 0.02]], dtype=np.float64)
    evidence = np.asarray([2.0], dtype=np.float64)
    features = np.zeros((1, feature_count), dtype=np.float64)
    features[0, int(model.score_shape_margin_index)] = 0.2

    pred_label, _confidence, _first_index = bench._predict_lrt_multiwindow_reject_trial_from_probs(
        model,
        probs,
        np.asarray(labels, dtype=object),
        evidence,
        min_enter_windows=1,
        feature_matrix=features,
    )

    assert pred_label == "idle"


def test_frequency_specific_gate_mask_rejects_selected_freq_when_threshold_not_met() -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    labels = ("idle", "8", "10.5", "12", "15")
    feature_names = bench._classifier_feature_names(freqs)
    feature_count = len(feature_names)
    margin_index = bench._feature_index(feature_names, "margin")
    ratio_index = bench._feature_index(feature_names, "ratio")
    entropy_index = bench._feature_index(feature_names, "score_entropy")
    top1_index = bench._feature_index(feature_names, "top1_score")
    top2_index = bench._feature_index(feature_names, "top2_score")
    normalized_top1_index = bench._feature_index(feature_names, "normalized_top1")
    model = bench.FBCCARidge5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        weights=np.zeros((feature_count + 1, len(labels)), dtype=np.float64),
        l2=0.1,
        command_confidence_th=0.1,
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        lrt_window_th=0.1,
        lrt_enter_th=0.0,
        smoothing_windows=1,
        gate_variant=bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        frequency_specific_control_state_gates={
            "8": {
                "type": "threshold",
                "theta_lrt_f": 0.2,
                "theta_score_f": 0.5,
                "theta_margin_f": 1.0,
                "theta_ratio_f": 1.1,
                "theta_entropy_f": 0.9,
                "theta_multiwindow_same_freq_count": 1.0,
            }
        },
        fit_summary={"score_bank_mode": "command_only"},
    )
    probs = np.asarray([[0.05, 0.90, 0.03, 0.01, 0.01]], dtype=np.float64)
    features = np.zeros((1, feature_count), dtype=np.float64)
    features[0, 0] = 0.9
    features[0, top1_index] = 0.9
    features[0, top2_index] = 0.1
    features[0, margin_index] = 0.2
    features[0, ratio_index] = 9.0
    features[0, normalized_top1_index] = 0.75
    features[0, entropy_index] = 0.2

    pred_label, _confidence, _first_index = bench._predict_lrt_multiwindow_reject_trial_from_probs(
        model,
        probs,
        np.asarray(labels, dtype=object),
        np.asarray([2.0], dtype=np.float64),
        min_enter_windows=1,
        feature_matrix=features,
    )

    assert pred_label == "idle"


def test_frequency_specific_logistic_trace_exports_score_space_fields() -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    labels = ("idle", "8", "10.5", "12", "15")
    feature_names = bench._classifier_feature_names(freqs)
    feature_count = len(feature_names)
    margin_index = bench._feature_index(feature_names, "margin")
    top1_index = bench._feature_index(feature_names, "top1_score")
    top2_index = bench._feature_index(feature_names, "top2_score")
    ratio_index = bench._feature_index(feature_names, "ratio")
    normalized_top1_index = bench._feature_index(feature_names, "normalized_top1")
    entropy_index = bench._feature_index(feature_names, "score_entropy")
    weights = np.zeros((feature_count + 1, len(labels)), dtype=np.float64)
    weights[0, 1] = 5.0
    base_model = bench.FBCCARidge5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        weights=weights,
        l2=0.1,
        command_confidence_th=0.0,
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        lrt_feature_indices=(margin_index,),
        lrt_feature_mean_control=np.asarray([3.0], dtype=np.float64),
        lrt_feature_std_control=np.ones(1, dtype=np.float64),
        lrt_feature_mean_idle=np.asarray([0.0], dtype=np.float64),
        lrt_feature_std_idle=np.ones(1, dtype=np.float64),
        lrt_window_th=0.1,
        lrt_enter_th=0.0,
        smoothing_windows=1,
        fit_summary={"score_bank_mode": "command_only"},
    )
    candidate_model = replace(
        base_model,
        gate_variant=bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_LOGISTIC,
        frequency_specific_control_state_gates={
            "8": {
                "type": "logistic",
                "weights": [3.0] + [0.0] * len(bench.FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
                "feature_mean": [0.0] * len(bench.FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
                "feature_std": [1.0] * len(bench.FREQUENCY_SPECIFIC_GATE_FEATURE_NAMES),
                "prob_threshold": 0.5,
            }
        },
    )
    features = np.zeros((2, feature_count), dtype=np.float64)
    features[:, 0] = 0.9
    features[:, top1_index] = 0.9
    features[:, top2_index] = 0.1
    features[:, margin_index] = 3.0
    features[:, ratio_index] = 9.0
    features[:, normalized_top1_index] = 0.75
    features[:, entropy_index] = 0.2
    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label="8Hz", expected_freq=8.0, trial_id=7, block_index=2),
            score_matrix=np.asarray([[0.9, 0.1, 0.05, 0.02], [0.92, 0.11, 0.05, 0.02]], dtype=np.float64),
            feature_matrix=features,
            duration_sec=2.5,
        )
    ]

    trace = bench._trace_rows_for_frequency_specific_logistic_case(
        baseline_model=base_model,
        candidate_model=candidate_model,
        scored_trials=scored,
        dataset="ysu_an",
        subject="S11",
        split_index=0,
        recipe_id="trace_recipe",
        frequency_profile="8_10p5_12_15",
        frequency_set_id="none_8_10p5_12_15",
        win_sec=2.0,
        step_sec=0.25,
        min_enter_windows=2,
    )

    assert trace["logistic_trace_windows"]
    first = trace["logistic_trace_windows"][0]
    for key in (
        "selected_freq_score",
        "top1_score",
        "top2_score",
        "top3_score",
        "margin",
        "ratio",
        "normalized_top1",
        "score_entropy",
        "lrt_evidence",
        "multiwindow_same_freq_count",
        "multiwindow_margin_mean",
        "multiwindow_entropy_mean",
        "cs_probability",
        "gate_pass",
        "transition_type",
    ):
        assert key in first
    assert first["subject"] == "S11"
    assert first["selected_freq"] == "8"
    assert first["cs_probability"] > 0.5
    assert trace["logistic_trace_trial_summary"][0]["transition_type"] == "baseline_TP_candidate_TP"


def test_subject_floor_ns2_aware_gate_uses_calibration_only_thresholds() -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    labels = ("idle", "8", "10.5", "12", "15")
    feature_names = bench._classifier_feature_names(freqs)
    feature_count = len(feature_names)
    margin_index = bench._feature_index(feature_names, "margin")
    model = bench.FBCCARidge5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        weights=np.zeros((feature_count + 1, len(labels)), dtype=np.float64),
        l2=0.1,
        command_confidence_th=0.0,
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        lrt_feature_indices=(margin_index,),
        lrt_feature_mean_control=np.asarray([3.0], dtype=np.float64),
        lrt_feature_std_control=np.ones(1, dtype=np.float64),
        lrt_feature_mean_idle=np.asarray([0.0], dtype=np.float64),
        lrt_feature_std_idle=np.asarray([3.0], dtype=np.float64),
        lrt_window_th=0.2,
        lrt_enter_th=0.0,
        fit_summary={},
    )

    def row(margin: float) -> np.ndarray:
        values = np.zeros(feature_count, dtype=np.float64)
        values[margin_index] = float(margin)
        return values

    updated = bench._apply_gate_variant_to_model(
        model,
        feature_names=feature_names,
        grouped_features={
            "control": np.vstack([row(3.0), row(3.2)]),
            "idle": np.vstack([row(0.0), row(0.2), row(0.4)]),
            "ns2": np.vstack([row(2.5), row(2.8), row(3.0)]),
        },
        params={
            "gate_variant": bench.CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE,
            "global_floor_quantile": 0.90,
            "subject_idle_quantile": 0.95,
            "ns2_safety_factor": 1.2,
        },
    )
    fit = dict(updated.fit_summary)

    assert updated.gate_variant == bench.CLASSIFIER_GATE_VARIANT_SUBJECT_FLOOR_NS2_AWARE
    assert updated.lrt_window_floor_th is not None
    assert fit["threshold_fit_split"] == "calibration_blocks"
    assert fit["test_split"] == "holdout_blocks"
    assert fit["ns2_threshold_source"] == "calibration_ns2"
    assert fit["lrt_window_floor_th"] == pytest.approx(
        max(
            model.lrt_window_th,
            fit["subject_floor_global_lrt_th"],
            fit["subject_floor_idle_lrt_th"],
            fit["ns2_lrt_window_floor_th"],
        )
    )
    assert fit["lrt_window_floor_th"] >= fit["ns2_lrt_window_floor_th"]
    assert fit["gate_variant_params"]["ns2_safety_factor"] == 1.2


def test_frequency_specific_threshold_fit_uses_selected_freq_rows_and_ns2_hard_negative() -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    labels = ("idle", "8", "10.5", "12", "15")
    feature_names = bench._classifier_feature_names(freqs)
    feature_count = len(feature_names)
    score8 = bench._feature_index(feature_names, "fbcca_score_8")
    score10 = bench._feature_index(feature_names, "fbcca_score_10.5")
    top1 = bench._feature_index(feature_names, "top1_score")
    top2 = bench._feature_index(feature_names, "top2_score")
    margin = bench._feature_index(feature_names, "margin")
    ratio = bench._feature_index(feature_names, "ratio")
    normalized = bench._feature_index(feature_names, "normalized_top1")
    entropy = bench._feature_index(feature_names, "score_entropy")

    def feature_row(selected: str, *, margin_value: float, ratio_value: float = 4.0, entropy_value: float = 0.2) -> np.ndarray:
        row = np.zeros(feature_count, dtype=np.float64)
        if selected == "8":
            row[score8] = 10.0
            row[score10] = 1.0
        else:
            row[score8] = 1.0
            row[score10] = 10.0
        row[top1] = 10.0
        row[top2] = 1.0
        row[margin] = margin_value
        row[ratio] = ratio_value
        row[normalized] = 0.8
        row[entropy] = entropy_value
        return row

    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label="8Hz", expected_freq=8.0, trial_id=1, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.vstack([feature_row("8", margin_value=4.0)]),
            duration_sec=2.0,
        ),
        bench.ScoredTrial(
            trial=TrialSpec(label="ns2_idle", expected_freq=None, trial_id=2, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.vstack([feature_row("10.5", margin_value=99.0)]),
            duration_sec=2.0,
        ),
        bench.ScoredTrial(
            trial=TrialSpec(label="ns2_idle", expected_freq=None, trial_id=3, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.vstack([feature_row("8", margin_value=4.0)]),
            duration_sec=2.0,
        ),
    ]
    model = bench.FBCCARidge5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        weights=np.zeros((feature_count + 1, len(labels)), dtype=np.float64),
        l2=0.1,
        command_confidence_th=0.0,
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        lrt_feature_indices=(margin,),
        lrt_feature_mean_control=np.asarray([4.0], dtype=np.float64),
        lrt_feature_std_control=np.ones(1, dtype=np.float64),
        lrt_feature_mean_idle=np.zeros(1, dtype=np.float64),
        lrt_feature_std_idle=np.ones(1, dtype=np.float64),
        lrt_window_th=0.0,
        lrt_enter_th=0.0,
        smoothing_windows=1,
        fit_summary={"min_enter_windows": 1, "score_bank_mode": "command_only"},
    )
    model.weights[1 + score8, 1] = 1.0
    model.weights[1 + score10, 2] = 1.0

    payload = bench._fit_frequency_specific_threshold_gate_payload(
        model,
        feature_names=feature_names,
        scored_trials=scored,
        params={
            "margin_idle_quantile": 0.90,
            "ratio_idle_quantile": 0.90,
            "entropy_control_quantile": 0.85,
            "ns2_safety_factor": 1.2,
        },
        smoothing_windows=1,
    )

    assert sorted(payload) == ["10.5", "12", "15", "8"]
    assert payload["8"]["negative_windows"] == 1
    assert payload["8"]["hard_negative_windows"] == 1
    assert payload["8"]["theta_lrt_f"] >= payload["8"]["theta_ns2_f"] * 1.2 - 1e-9
    assert payload["8"]["theta_margin_f"] < 20.0
    assert payload["8"]["gate_fit_validation"]["policy"] == "calibration_trial_holdout_alternating"
    assert payload["8"]["gate_fit_validation"]["validation_trial_ids"] == [2]


def test_frequency_specific_gate_model_records_calibration_validation_metrics() -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    labels = ("idle", "8", "10.5", "12", "15")
    feature_names = bench._classifier_feature_names(freqs)
    feature_count = len(feature_names)
    score8 = bench._feature_index(feature_names, "fbcca_score_8")
    top1 = bench._feature_index(feature_names, "top1_score")
    top2 = bench._feature_index(feature_names, "top2_score")
    margin = bench._feature_index(feature_names, "margin")
    ratio = bench._feature_index(feature_names, "ratio")
    normalized = bench._feature_index(feature_names, "normalized_top1")
    entropy = bench._feature_index(feature_names, "score_entropy")

    def row(score: float, margin_value: float) -> np.ndarray:
        values = np.zeros(feature_count, dtype=np.float64)
        values[score8] = score
        values[top1] = score
        values[top2] = 0.1
        values[margin] = margin_value
        values[ratio] = score / 0.1
        values[normalized] = 0.9
        values[entropy] = 0.1
        return values

    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label="8Hz", expected_freq=8.0, trial_id=1, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.vstack([row(5.0, 4.0)]),
            duration_sec=2.0,
        ),
        bench.ScoredTrial(
            trial=TrialSpec(label="ns2_idle", expected_freq=None, trial_id=2, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.vstack([row(5.0, 4.0)]),
            duration_sec=2.0,
        ),
        bench.ScoredTrial(
            trial=TrialSpec(label="8Hz", expected_freq=8.0, trial_id=3, block_index=1),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.vstack([row(5.0, 4.0)]),
            duration_sec=2.0,
        ),
    ]
    model = bench.FBCCARidge5Model(
        freqs=freqs,
        labels=labels,
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_std=np.ones(feature_count, dtype=np.float64),
        weights=np.zeros((feature_count + 1, len(labels)), dtype=np.float64),
        l2=0.1,
        command_confidence_th=0.0,
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        lrt_feature_indices=(margin,),
        lrt_feature_mean_control=np.asarray([4.0], dtype=np.float64),
        lrt_feature_std_control=np.ones(1, dtype=np.float64),
        lrt_feature_mean_idle=np.zeros(1, dtype=np.float64),
        lrt_feature_std_idle=np.ones(1, dtype=np.float64),
        lrt_window_th=0.0,
        lrt_enter_th=0.0,
        smoothing_windows=1,
        fit_summary={"min_enter_windows": 1, "score_bank_mode": "command_only"},
    )
    model.weights[1 + score8, 1] = 1.0

    updated = bench._apply_gate_variant_to_model(
        model,
        feature_names=feature_names,
        grouped_features={"control": np.vstack([row(5.0, 4.0)]), "idle": np.vstack([row(5.0, 4.0)])},
        scored_trials=scored,
        params={
            "gate_variant": bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
            "margin_idle_quantile": 0.90,
            "ratio_idle_quantile": 0.90,
            "entropy_control_quantile": 0.85,
            "ns2_safety_factor": 1.0,
        },
        win_sec=2.0,
        step_sec=0.25,
        min_enter_windows=1,
    )

    validation = updated.fit_summary["gate_validation_metrics"]
    assert updated.fit_summary["frequency_specific_grid_selection_policy"] == "calibration_internal_validation_first"
    assert validation["supported"] is True
    assert validation["split"] == "calibration_gate_validation_trials"
    assert validation["trial_count"] == 1
    assert "control_recall" in validation["metrics"]


def test_frequency_specific_summary_rank_uses_gate_validation_before_holdout() -> None:
    baseline = {
        "gate_variant": bench.CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        "mean_mixed_idle_fp_per_min": 0.8,
        "mean_control_recall": 0.88,
        "mean_control_recall_at_2.5s": 0.78,
        "mean_async_macro_f1_5class": 0.8,
        "mean_detection_latency_s": 2.3,
    }
    validation_good = {
        **baseline,
        "gate_variant": bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        "mean_mixed_idle_fp_per_min": 0.95,
        "gate_validation_summary": {
            "supported": True,
            "mean_idle_fp_per_min": 0.2,
            "mean_control_recall": 0.9,
            "mean_control_recall_at_2.5s": 0.8,
            "mean_async_macro_f1_5class": 0.82,
            "mean_detection_latency_s": 2.2,
        },
    }
    validation_bad = {
        **baseline,
        "gate_variant": bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        "mean_mixed_idle_fp_per_min": 0.1,
        "gate_validation_summary": {
            "supported": True,
            "mean_idle_fp_per_min": 2.0,
            "mean_control_recall": 0.5,
            "mean_control_recall_at_2.5s": 0.4,
            "mean_async_macro_f1_5class": 0.4,
            "mean_detection_latency_s": 3.0,
        },
    }

    assert bench._summary_rank_key(validation_good) < bench._summary_rank_key(validation_bad)


def test_decision_table_marks_ns2_reduced_tradeoff_and_5state_alias() -> None:
    baseline = {
        "method": "fbcca_ridge5",
        "recipe_id": "win2_me2_sm3_lrtmw",
        "gate_variant": bench.CLASSIFIER_GATE_VARIANT_BASELINE_LRTMW,
        "expected_subject_count": 24,
        "coverage_subject_count": 24,
        "split_count": 48,
        "mean_mixed_idle_fp_per_min": 0.84,
        "mean_ns2_fp_per_min": 1.98,
        "mean_control_recall": 0.885,
        "mean_control_recall_at_2.5s": 0.778,
        "mean_detection_latency_s": 2.29,
        "mean_async_macro_f1_5class": 0.8,
        "deployable_budget_pass": True,
        "subjects": [],
    }
    candidate = {
        **baseline,
        "recipe_id": "win2_me2_sm3_lrtmw_ns2sf1p2",
        "gate_variant": bench.CLASSIFIER_GATE_VARIANT_NS2_AWARE,
        "mean_ns2_fp_per_min": 1.2,
        "mean_control_recall_at_2.5s": 0.752,
    }

    metrics = bench._summary_metric_payload(candidate)
    rows = bench._decision_table_rows([baseline, candidate])

    assert metrics["async_macro_f1_5state"] == metrics["async_macro_f1_5class"]
    assert rows[1]["ns2_status"] == "ns2_reduced_tradeoff"
    assert rows[1]["deployable"] is True
    assert rows[1]["delta_ns2_fp_per_min"] < 0.0


def test_subgroup_comparison_marks_ns2_improved_and_candidate_eligible() -> None:
    baseline_rows = [
        _aggregate_test_row(
            subject=subject,
            recipe_id="win2_me2_sm3_lrtmw",
            async_macro_f1_5class=0.8,
            async_acc_5class=0.8,
            idle_fp_per_min=1.2,
            ns2_fp_per_min=2.0 if subject in bench.HIGH_FP_SUBGROUP_SUBJECTS else 0.2,
            control_recall=0.78 if subject in bench.LOW_RECALL_SUBGROUP_SUBJECTS else 0.88,
            detection_latency_s=2.3,
        )
        for subject in bench.HIGH_RISK_VALIDATION_SUBJECTS
    ]
    candidate_rows = [
        _aggregate_test_row(
            subject=subject,
            recipe_id="win2_me2_sm3_lrtmw_fsth0p95_0p90_0p90_1p1",
            async_macro_f1_5class=0.82,
            async_acc_5class=0.82,
            idle_fp_per_min=0.9,
            ns2_fp_per_min=1.1 if subject in bench.HIGH_FP_SUBGROUP_SUBJECTS else 0.2,
            control_recall=0.76 if subject in bench.LOW_RECALL_SUBGROUP_SUBJECTS else 0.86,
            detection_latency_s=2.35,
        )
        for subject in bench.HIGH_RISK_VALIDATION_SUBJECTS
    ]
    for row in candidate_rows:
        row["gate_variant"] = bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD
        row["gate_variant_params"] = {
            "gate_variant": bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
            "combo_name": "balanced",
            "margin_idle_quantile": 0.95,
            "ratio_idle_quantile": 0.90,
            "entropy_control_quantile": 0.90,
            "ns2_safety_factor": 1.1,
        }
        row["calibration_profile"] = {
            "fit_summary": {
                "gate_variant_params": dict(row["gate_variant_params"]),
                "frequency_specific_grid_selection_policy": bench.FREQSPEC_GRID_SELECTION_POLICY,
                "gate_validation_metrics": {
                    "supported": True,
                    "metrics": {
                        "mixed_idle_fp_per_min": 0.8,
                        "ns2_fp_per_min": 1.0,
                        "control_recall": 0.86,
                        "control_recall_at_2.5s": 0.76,
                        "async_macro_f1_5class": 0.8,
                        "detection_latency_s": 2.3,
                    },
                },
            }
        }

    summaries = bench.aggregate_recipe_rows(baseline_rows + candidate_rows, expected_subject_count=9)
    rows = bench._decision_table_rows(summaries)
    candidate = next(row for row in rows if row["gate_variant"] == bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD)

    assert candidate["combo_name"] == "balanced"
    assert candidate["high_risk_ns2_fp_per_min"] == pytest.approx(1.1)
    assert candidate["high_risk_delta_ns2_fp_per_min"] == pytest.approx(-0.9)
    assert candidate["high_risk_ns2_reduction_ratio"] == pytest.approx(0.45)
    assert candidate["low_recall_recall_at_2.5s"] == pytest.approx(0.76)
    assert candidate["low_recall_delta_recall_at_2.5s"] == pytest.approx(-0.02)
    assert candidate["ns2_improved"] is True
    assert candidate["recall_degraded"] is False
    assert candidate["full24_candidate_eligible"] is True
    assert candidate["recommended_profile_export"] is True
    assert candidate["frequency_specific_grid_selection_policy"] == bench.FREQSPEC_GRID_SELECTION_POLICY
    assert candidate["gate_validation_ns2_fp_per_min"] == pytest.approx(1.0)
    assert "freq_10p5_command_recall" in candidate
    assert "freq_8_recall_at_2.5s" in candidate


def test_subgroup_comparison_marks_recall_degraded_with_relaxed_large_ns2_gain() -> None:
    baseline = {
        "mean_control_recall": 0.88,
        "subjects": [
            {"subject": subject, "mean_ns2_fp_per_min": 2.0, "mean_control_recall_at_2.5s": 0.78}
            for subject in bench.HIGH_FP_SUBGROUP_SUBJECTS
        ]
        + [
            {"subject": subject, "mean_ns2_fp_per_min": 0.2, "mean_control_recall_at_2.5s": 0.78}
            for subject in bench.LOW_RECALL_SUBGROUP_SUBJECTS
        ],
    }
    candidate = {
        "mean_control_recall": 0.86,
        "failed_case_count": 0,
        "hard_failed_case_count": 0,
        "subjects": [
            {"subject": subject, "mean_ns2_fp_per_min": 1.0, "mean_control_recall_at_2.5s": 0.78}
            for subject in bench.HIGH_FP_SUBGROUP_SUBJECTS
        ]
        + [
            {"subject": subject, "mean_ns2_fp_per_min": 0.2, "mean_control_recall_at_2.5s": 0.725}
            for subject in bench.LOW_RECALL_SUBGROUP_SUBJECTS
        ],
    }

    payload = bench._candidate_subgroup_comparison_payload(candidate, baseline)

    assert payload["high_risk_ns2_reduction_ratio"] == pytest.approx(0.5)
    assert payload["allowed_low_recall_drop"] == pytest.approx(0.05)
    assert payload["recall_degraded"] is True
    assert payload["full24_candidate_eligible"] is False


def test_candidate_artifact_paths_for_recipe_reads_matching_rows() -> None:
    rows = [
        _aggregate_test_row(
            subject="S01",
            recipe_id="win2_me2_sm3_lrtmw",
            async_macro_f1_5class=0.9,
            async_acc_5class=0.9,
            idle_fp_per_min=0.5,
            control_recall=0.9,
            detection_latency_s=2.2,
        ),
        _aggregate_test_row(
            subject="S02",
            recipe_id="win2_me2_sm3_lrtmw",
            async_macro_f1_5class=0.9,
            async_acc_5class=0.9,
            idle_fp_per_min=0.5,
            control_recall=0.9,
            detection_latency_s=2.2,
        ),
        _aggregate_test_row(
            subject="S03",
            recipe_id="win2p5_me2_sm3_lrtmw",
            async_macro_f1_5class=0.9,
            async_acc_5class=0.9,
            idle_fp_per_min=0.5,
            control_recall=0.9,
            detection_latency_s=2.2,
        ),
    ]
    rows[0]["calibration_profile"] = {"candidate_artifact_path": "/remote/S01.json"}
    rows[1]["calibration_profile"] = {"candidate_artifact_path": "/remote/S02.json"}
    rows[2]["calibration_profile"] = {"candidate_artifact_path": "/remote/S03.json"}
    recipe = {
        "method": "fbcca_ridge5",
        "recipe_id": "win2_me2_sm3_lrtmw",
        "frequency_set_id": "",
        "idle_multiplier": 2.0,
        "calibration_block_count": 2,
    }

    paths = bench._candidate_artifact_paths_for_recipe(rows, recipe)

    assert paths == ["/remote/S01.json", "/remote/S02.json"]


def test_channel_compatibility_payload_marks_ysuan_strict8_match() -> None:
    payload = bench._channel_compatibility_payload(
        "ysu_an",
        {
            "all_channel_count": 63,
            "selected_channel_names": ["Oz", "O1", "O2", "PO3", "Poz", "PO7", "PO8", "PO4"],
            "selected_channel_indices_zero_based": [15, 14, 16, 45, 46, 44, 48, 47],
        },
    )

    assert payload["channel_contract"] == "strict_required_8_posterior"
    assert payload["matches_project_channel_contract"] is True
    assert payload["dataset_matches_required_order"] is True
    assert payload["only_required_channels_used"] is True
    assert payload["source_channel_count"] == 63
    assert payload["selected_channel_indices_one_based"] == [16, 15, 17, 46, 47, 45, 49, 48]


def test_channel_compatibility_summary_reports_mismatch() -> None:
    summary = bench._channel_compatibility_summary(
        [
            {
                "dataset": "ysu_an",
                "subject": "S01",
                "channel_compatibility": {
                    "matches_project_channel_contract": False,
                    "dataset_selected_channel_names": ["O1", "Oz"],
                    "dataset_required_channel_names": list(bench.YSUAN_REQUIRED_CHANNELS),
                },
            }
        ]
    )

    assert summary["all_loaded_subjects_match_project_channel_contract"] is False
    assert summary["mismatched_subjects"][0]["subject"] == "S01"


def test_evaluation_payload_preserves_ysuan_no_control_subtype_metrics() -> None:
    payload = bench._evaluation_payload(
        {
            "fixed_window_metrics_5class": {"acc": 1.0},
            "async_lens_metrics_5class": {"macro_f1": 0.8},
            "no_control_subtype_metrics": {
                "ns1": {"idle_fp_per_min": 0.1},
                "ns2": {"idle_fp_per_min": 0.2},
                "ns3": {"idle_fp_per_min": 0.3},
                "ns_all_fp_per_min": 0.25,
            },
        }
    )

    metrics = bench._extract_row_metrics(payload)

    assert abs(float(metrics["ns1_fp_per_min"]) - 0.1) < 1e-9
    assert abs(float(metrics["ns2_fp_per_min"]) - 0.2) < 1e-9
    assert abs(float(metrics["ns3_fp_per_min"]) - 0.3) < 1e-9
    assert abs(float(metrics["ns_all_fp_per_min"]) - 0.25) < 1e-9
    assert abs(float(metrics["real_idle_fp_per_min"]) - 0.25) < 1e-9
    assert abs(float(metrics["mixed_idle_fp_per_min"]) - 0.25) < 1e-9


def test_extract_row_metrics_distinguishes_real_approx_and_mixed_idle() -> None:
    payload = {
        "async_metrics": {"idle_fp_per_min": 1.25},
        "clean_idle_proxy_metrics": {"supported": True, "idle_fp_per_min": 0.4},
    }

    metrics = bench._extract_row_metrics(payload)

    assert abs(float(metrics["approx_idle_fp_per_min"]) - 1.25) < 1e-9
    assert abs(float(metrics["real_idle_fp_per_min"]) - 0.4) < 1e-9
    assert abs(float(metrics["mixed_idle_fp_per_min"]) - 0.4) < 1e-9


def test_build_ysuan_all_target_segments_keeps_explicit_ns_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    available_freqs = bench.YSUAN_TARGET_FREQUENCIES
    cs_segments = [
        (
            TrialSpec(label=f"{float(freq):g}Hz", expected_freq=float(freq), trial_id=index, block_index=0),
            np.zeros((1000, 8), dtype=np.float64),
        )
        for index, freq in enumerate(available_freqs)
    ]
    ns_segments = [
        (
            TrialSpec(label="ysu_an_ns1_trial01", expected_freq=None, trial_id=100, block_index=0),
            np.zeros((1000, 8), dtype=np.float64),
        )
    ]

    monkeypatch.setattr(bench, "load_ysuan_subject", lambda *args, **kwargs: object())
    monkeypatch.setattr(bench, "build_ysuan_cs_segments", lambda *args, **kwargs: cs_segments)
    monkeypatch.setattr(bench, "build_ysuan_ns_segments", lambda *args, **kwargs: ns_segments)
    spec = bench.ExternalSubjectSpec(dataset="ysu_an", subject="S01", mat_path=Path("S01"), freqs=freqs)

    segments = bench._build_all_target_segments_for_spec(spec, available_freqs=available_freqs)

    assert bench._count_segments(segments, freqs)["idle"] == 1


def test_ysuan_holdout_no_control_scored_uses_only_explicit_ns_trials() -> None:
    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label="ysu_an_ns1_trial05", expected_freq=None, trial_id=1, block_index=4),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.zeros((1, 4), dtype=np.float64),
            duration_sec=4.0,
        ),
        bench.ScoredTrial(
            trial=TrialSpec(label="hard_idle_9Hz", expected_freq=None, trial_id=2, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.zeros((1, 4), dtype=np.float64),
            duration_sec=4.0,
        ),
        bench.ScoredTrial(
            trial=TrialSpec(label="8Hz", expected_freq=8.0, trial_id=3, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.zeros((1, 4), dtype=np.float64),
            duration_sec=4.0,
        ),
    ]

    ns_scored, support = bench._ysuan_holdout_no_control_scored(scored, win_sec=2.0)

    assert [item.trial.label for item in ns_scored] == ["ysu_an_ns1_trial05"]
    assert support["supported"] is True
    assert support["segment_count"] == 1
    assert "holdout NS1/NS2/NS3" in str(support["note"])


def test_ysuan_no_control_payload_is_not_gated_by_idle_eval_mode() -> None:
    scored = [
        bench.ScoredTrial(
            trial=TrialSpec(label="ysu_an_ns2_trial01", expected_freq=None, trial_id=1, block_index=0),
            score_matrix=np.zeros((1, 4), dtype=np.float64),
            feature_matrix=np.zeros((1, 4), dtype=np.float64),
            duration_sec=4.0,
        )
    ]

    ns_scored, support = bench._ysuan_holdout_no_control_scored(scored, win_sec=1.5)

    assert len(ns_scored) == 1
    assert support["supported"] is True
    assert support["note"] == "YSU-an no-control metrics use holdout NS1/NS2/NS3 trials only."


def test_method_parser_supports_short_pretrain_candidates() -> None:
    parsed = bench._csv_method_tuple("fbcca_ridge5,fbcca_lda5,itcca5,ecca5,trca5,trca_r5,tdca5,fbcca_ridge5")
    assert parsed == ("fbcca_ridge5", "fbcca_lda5", "itcca5", "ecca5", "trca5", "trca_r5", "tdca5")


def test_classifier_recipe_id_preserves_strict_names_and_marks_gap() -> None:
    assert bench._classifier_recipe_id(win_sec=2.0, min_enter_windows=1, max_gap_windows=0) == "win2_me1"
    assert bench._classifier_recipe_id(win_sec=1.75, min_enter_windows=2, max_gap_windows=1) == "win1p75_me2_gap1"


def test_classifier_threshold_policy_parser_supports_recall_guard() -> None:
    assert bench._parse_classifier_threshold_policy(" balanced_recall_guard ") == "balanced_recall_guard"
    assert (
        bench._parse_classifier_threshold_policy(" lrt_multiwindow_reject_gate ")
        == bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY
    )


def test_full_reference_bank_features_add_command_vs_noncommand_evidence() -> None:
    command_scores = np.asarray(
        [
            [0.8, 0.3, 0.2, 0.1],
            [0.4, 0.9, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    all_scores = np.asarray(
        [
            [0.1, 0.8, 0.3, 0.2, 0.7, 0.1],
            [0.95, 0.4, 0.9, 0.2, 0.1, 0.05],
        ],
        dtype=np.float64,
    )
    command_freqs = (8.0, 10.0, 12.0, 15.0)
    all_freqs = (7.8, 8.0, 10.0, 12.0, 14.0, 15.0)

    features = bench._score_matrices_to_features(
        command_score_matrix=command_scores,
        command_freqs=command_freqs,
        score_bank_mode="full_reference_bank",
        all_score_matrix=all_scores,
        all_freqs=all_freqs,
    )

    assert features.shape == (2, len(command_freqs) + len(bench.CLASSIFIER_DERIVED_FEATURE_NAMES) + len(bench.FULL_REFERENCE_BANK_FEATURE_NAMES))
    full = features[:, -len(bench.FULL_REFERENCE_BANK_FEATURE_NAMES) :]
    assert abs(float(full[0, 0]) - 0.8) < 1e-9
    assert abs(float(full[0, 1]) - 0.8) < 1e-9
    assert int(full[0, 2]) == 1
    assert abs(float(full[0, 4]) - 0.1) < 1e-9
    assert abs(float(full[1, 0]) - 0.9) < 1e-9
    assert abs(float(full[1, 1]) - 0.95) < 1e-9
    assert int(full[1, 2]) == 2
    assert float(full[1, 4]) < 0.0


def test_full_reference_bank_feature_names_extend_contract() -> None:
    names = bench._classifier_feature_names(
        (8.0, 10.0, 12.0, 15.0),
        score_bank_mode="full_reference_bank",
    )

    assert names[-6:] == list(bench.FULL_REFERENCE_BANK_FEATURE_NAMES)


def test_pretrain_budget_estimate_flags_over_budget_personalized_search() -> None:
    shared = bench._budget_payload(
        freq_selection_mode="shared_fixed4",
        pretrain_budget_sec=120.0,
        personalized_candidate_count=0,
    )
    personalized = bench._budget_payload(
        freq_selection_mode="personalized_upper_bound",
        pretrain_budget_sec=120.0,
        personalized_candidate_count=12,
    )

    assert shared["pretrain_budget_pass"] is True
    assert abs(float(shared["estimated_pretrain_duration_sec"]) - 102.0) < 1e-9
    assert personalized["pretrain_budget_pass"] is False
    assert abs(float(personalized["estimated_pretrain_duration_sec"]) - 150.0) < 1e-9


def test_frequency_search_plan_lists_frame_locked_shared_sets() -> None:
    plan = bench._frequency_search_plan(
        mode="shared_fixed4",
        candidate_source="frame_locked_240",
        datasets=("beta",),
    )

    assert plan["frequency_selection_mode"] == "shared_fixed4"
    assert plan["shared_candidate_set_count"] == 5
    assert plan["shared_candidate_sets_preview"][0] == [8.0, 9.6, 10.0, 12.0]


def test_shared_frequency_sets_expand_all_frame_locked_combinations() -> None:
    plan = bench._frequency_search_plan(
        mode="shared_fixed4",
        candidate_source="frame_locked_240",
        datasets=("beta",),
    )

    sets = bench._shared_frequency_sets_for_plan(plan, fallback_freqs=(9.8, 12.0, 14.8, 15.8))

    assert len(sets) == 5
    assert sets[0] == (8.0, 9.6, 10.0, 12.0)
    assert sets[-1] == (9.6, 10.0, 12.0, 15.0)


def test_relabel_segments_for_command_freqs_turns_noncommands_into_hard_idle() -> None:
    segments = [
        (
            TrialSpec(label="8Hz", expected_freq=8.0, trial_id=1, block_index=0),
            np.zeros((10, 2), dtype=np.float64),
        ),
        (
            TrialSpec(label="hard_idle_beta_target11_10Hz", expected_freq=None, trial_id=2, block_index=0),
            np.zeros((10, 2), dtype=np.float64),
        ),
        (
            TrialSpec(label="12Hz", expected_freq=12.0, trial_id=3, block_index=0),
            np.zeros((10, 2), dtype=np.float64),
        ),
    ]

    relabeled = bench._relabel_segments_for_command_freqs(
        segments,
        command_freqs=(8.0, 9.6, 10.0, 12.0),
    )

    assert [item[0].expected_freq for item in relabeled] == [8.0, 10.0, 12.0]
    relabeled = bench._relabel_segments_for_command_freqs(
        segments,
        command_freqs=(8.0, 9.6, 11.0, 12.0),
    )
    assert [item[0].expected_freq for item in relabeled] == [8.0, None, 12.0]
    assert relabeled[1][0].label == "hard_idle_10Hz"


def test_personalized_frequency_selection_uses_calibration_blocks_only(monkeypatch) -> None:
    candidate_freqs = (8.0, 10.0, 12.0, 15.0, 20.0)
    segments = []
    trial_id = 0
    for freq in (8.0, 10.0, 12.0, 15.0):
        segments.append(
            (
                TrialSpec(label=f"{freq:g}Hz", expected_freq=freq, trial_id=trial_id, block_index=0),
                np.full((2, 1), freq, dtype=np.float64),
            )
        )
        trial_id += 1
    segments.append(
        (
            TrialSpec(label="20Hz", expected_freq=20.0, trial_id=trial_id, block_index=1),
            np.full((2, 1), 20.0, dtype=np.float64),
        )
    )

    class FakeDecoder:
        win_samples = 1
        step_samples = 1
        fs = 250

        def score_windows_batch(self, windows):
            values = np.asarray(windows, dtype=np.float64)[:, 0, 0]
            scores = np.zeros((len(values), len(candidate_freqs)), dtype=np.float64)
            for row_index, value in enumerate(values):
                for col_index, freq in enumerate(candidate_freqs):
                    if abs(float(value) - float(freq)) < 1e-9:
                        scores[row_index, col_index] = 10.0
            return scores

    monkeypatch.setattr(bench, "_build_fbcca_decoder_for_scoring", lambda **_kwargs: FakeDecoder())

    selected, summary = bench._score_personalized_frequency_candidates(
        all_target_segments=segments,
        candidate_freqs=candidate_freqs,
        calibration_blocks=(0,),
        sampling_rate=250,
        win_sec=1.0,
        max_supported_win_sec=1.0,
        step_sec=0.25,
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
    )

    assert 20.0 not in selected
    assert set(selected) == {8.0, 10.0, 12.0, 15.0}
    ranked = {float(row["freq"]): row for row in summary["ranked_candidates"]}
    assert ranked[20.0]["trial_count"] == 0


def test_clean_idle_proxy_reports_unsupported_when_prestim_is_shorter_than_window() -> None:
    segments = [
        (
            TrialSpec(label="pre_stim_idle_8Hz", expected_freq=None, trial_id=1, block_index=0),
            np.zeros((125, 8), dtype=np.float64),
        )
    ]

    payload = bench._clean_idle_proxy_support_payload(
        clean_idle_segments=segments,
        sampling_rate=250,
        win_sec=1.25,
    )

    assert payload["available"] is True
    assert payload["supported"] is False
    assert payload["max_segment_duration_sec"] == 0.5


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


def test_select_ysuan_split_uses_two_cs_repeats_and_four_ns_calibration_trials() -> None:
    freqs = (8.0, 10.5, 12.0, 15.0)
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    for block in range(12):
        for freq in freqs:
            segments.append(
                (
                    TrialSpec(label=f"{freq:g}Hz", expected_freq=freq, trial_id=trial_id, block_index=block),
                    np.zeros((1000, 8), dtype=np.float64),
                )
            )
            trial_id += 1
    for subtype, count, samples in (("ns1", 24, 1000), ("ns2", 24, 1000), ("ns3", 48, 500)):
        for index in range(count):
            segments.append(
                (
                    TrialSpec(
                        label=f"ysu_an_{subtype}_trial{index + 1:02d}",
                        expected_freq=None,
                        trial_id=trial_id,
                        block_index=index,
                    ),
                    np.zeros((samples, 8), dtype=np.float64),
                )
            )
            trial_id += 1

    calibration, holdout, summary = bench.select_ysuan_split_segments(
        segments,
        freqs=freqs,
        calibration_blocks=(0, 1),
        holdout_blocks=tuple(range(2, 12)),
        idle_multiplier=10.0,
        seed=1,
    )

    cal_control = [trial for trial, _segment in calibration if trial.expected_freq is not None]
    cal_idle = [trial for trial, _segment in calibration if trial.expected_freq is None]
    holdout_control = [trial for trial, _segment in holdout if trial.expected_freq is not None]
    holdout_idle = [trial for trial, _segment in holdout if trial.expected_freq is None]
    assert len(cal_control) == 8
    assert len(cal_idle) == 12
    assert len(holdout_control) == 40
    assert len(holdout_idle) == (20 + 20 + 44)
    assert summary["ysu_an_ns_calibration_counts"] == {"ns1": 4, "ns2": 4, "ns3": 4}
    assert summary["ysu_an_ns_holdout_counts"] == {"ns1": 20, "ns2": 20, "ns3": 44}


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
        "no_control_subtype_metrics": {
            "ns1": {"idle_fp_per_min": 0.1},
            "ns2": {"idle_fp_per_min": 0.2},
            "ns3": {"idle_fp_per_min": 0.3},
            "ns_all_fp_per_min": 0.25,
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
    assert abs(float(metrics["ns1_fp_per_min"]) - 0.1) < 1e-9
    assert abs(float(metrics["ns2_fp_per_min"]) - 0.2) < 1e-9
    assert abs(float(metrics["ns3_fp_per_min"]) - 0.3) < 1e-9
    assert abs(float(metrics["ns_all_fp_per_min"]) - 0.25) < 1e-9
    assert abs(float(metrics["cs_control_recall"]) - 0.9) < 1e-9


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


def test_classifier_probability_smoothing_recovers_weak_consecutive_command() -> None:
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
            [0.35, 0.65, 0.00, 0.00, 0.00],
            [0.45, 0.55, 0.00, 0.00, 0.00],
            [0.35, 0.65, 0.00, 0.00, 0.00],
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
    smoothed = bench._smooth_classifier_probabilities(probs, smoothing_windows=2)
    smooth_label, smooth_confidence, smooth_index = bench._predict_fbcca_lda5_trial_from_probs(
        model,
        smoothed,
        labels,
        min_enter_windows=2,
        max_gap_windows=0,
    )

    assert strict_label == "idle"
    assert smooth_label == "9.8"
    assert smooth_confidence >= 0.60
    assert smooth_index == 1.0


def test_adaptive_gate_feature_matrix_shape_and_rank_features() -> None:
    labels = np.asarray(("idle", "9.8", "12", "14.8", "15.8"), dtype=object)
    probs = np.asarray(
        [
            [0.20, 0.60, 0.10, 0.05, 0.05],
            [0.55, 0.15, 0.20, 0.05, 0.05],
        ],
        dtype=np.float64,
    )
    command_scores = np.asarray(
        [
            [0.60, 0.30, 0.20, 0.10],
            [0.20, 0.50, 0.30, 0.10],
        ],
        dtype=np.float64,
    )
    command_features = bench._score_matrix_to_features(command_scores)
    full_features = bench._full_reference_bank_features(
        command_score_matrix=command_scores,
        all_score_matrix=np.asarray(
            [
                [0.60, 0.30, 0.20, 0.10, 0.55],
                [0.20, 0.50, 0.30, 0.10, 0.70],
            ],
            dtype=np.float64,
        ),
        command_freqs=(9.8, 12.0, 14.8, 15.8),
        all_freqs=(9.8, 12.0, 14.8, 15.8, 10.0),
    )
    trial = bench.ScoredTrial(
        trial=TrialSpec(label="9.8", expected_freq=9.8, trial_id=1, block_index=0),
        score_matrix=command_scores,
        feature_matrix=np.column_stack(
            [
                command_features,
                full_features,
            ]
        ),
        duration_sec=2.0,
    )

    features = bench._adaptive_gate_feature_matrix_for_trial(
        bench.FBCCALDA5Model(
            freqs=(9.8, 12.0, 14.8, 15.8),
            labels=tuple(labels.tolist()),
            feature_mean=np.zeros(1, dtype=np.float64),
            feature_std=np.ones(1, dtype=np.float64),
            class_means=np.zeros((5, 1), dtype=np.float64),
            pooled_var=np.ones(1, dtype=np.float64),
            command_confidence_th=0.0,
            fit_summary={},
        ),
        trial,
        probs,
        labels,
    )

    assert features.shape == (2, len(bench.ADAPTIVE_EVIDENCE_FEATURE_NAMES))
    rank_column = bench.ADAPTIVE_EVIDENCE_FEATURE_NAMES.index("full_bank_inverse_command_rank")
    margin_column = bench.ADAPTIVE_EVIDENCE_FEATURE_NAMES.index("full_bank_nearest_noncommand_margin")
    assert features[0, rank_column] == 1.0
    assert features[1, rank_column] == 0.5
    assert features[0, margin_column] > 0.0
    assert features[1, margin_column] < 0.0


def test_adaptive_evidence_gate_recovers_continuous_weak_command() -> None:
    labels = np.asarray(("idle", "9.8", "12", "14.8", "15.8"), dtype=object)
    model = bench.FBCCALDA5Model(
        freqs=(9.8, 12.0, 14.8, 15.8),
        labels=tuple(labels.tolist()),
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.zeros((5, 1), dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=0.0,
        fit_summary={},
        gate_policy=bench.CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
        evidence_decision_th=0.60,
        evidence_enter_th=0.40,
        evidence_decay=0.50,
    )
    probs = np.asarray(
        [
            [0.48, 0.52, 0.00, 0.00, 0.00],
            [0.47, 0.53, 0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    gate_probs = np.asarray([0.85, 0.86], dtype=np.float64)

    label, confidence, index = bench._predict_adaptive_evidence_trial_from_probs(
        model,
        probs,
        labels,
        gate_probs,
        min_enter_windows=2,
        max_gap_windows=0,
    )

    assert label == "9.8"
    assert confidence >= 0.85
    assert index == 1.0


def test_adaptive_evidence_gate_rejects_unstable_idle_like_windows() -> None:
    labels = np.asarray(("idle", "9.8", "12", "14.8", "15.8"), dtype=object)
    model = bench.FBCCALDA5Model(
        freqs=(9.8, 12.0, 14.8, 15.8),
        labels=tuple(labels.tolist()),
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.zeros((5, 1), dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=0.0,
        fit_summary={},
        gate_policy=bench.CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
        evidence_decision_th=0.70,
        evidence_enter_th=0.60,
        evidence_decay=0.50,
    )
    probs = np.asarray(
        [
            [0.20, 0.50, 0.30, 0.00, 0.00],
            [0.20, 0.30, 0.50, 0.00, 0.00],
            [0.20, 0.50, 0.30, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    gate_probs = np.asarray([0.55, 0.56, 0.55], dtype=np.float64)

    label, confidence, index = bench._predict_adaptive_evidence_trial_from_probs(
        model,
        probs,
        labels,
        gate_probs,
        min_enter_windows=2,
        max_gap_windows=0,
    )

    assert label == "idle"
    assert confidence == 0.0
    assert index == 0.0


def test_adaptive_evidence_gate_enter_threshold_still_honors_min_enter() -> None:
    labels = np.asarray(("idle", "9.8", "12", "14.8", "15.8"), dtype=object)
    model = bench.FBCCALDA5Model(
        freqs=(9.8, 12.0, 14.8, 15.8),
        labels=tuple(labels.tolist()),
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.zeros((5, 1), dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=0.0,
        fit_summary={},
        gate_policy=bench.CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
        evidence_decision_th=0.70,
        evidence_enter_th=0.25,
        evidence_decay=0.50,
    )
    probs = np.asarray([[0.10, 0.90, 0.00, 0.00, 0.00]], dtype=np.float64)
    gate_probs = np.asarray([0.95], dtype=np.float64)

    label, confidence, index = bench._predict_adaptive_evidence_trial_from_probs(
        model,
        probs,
        labels,
        gate_probs,
        min_enter_windows=2,
        max_gap_windows=0,
    )

    assert label == "idle"
    assert confidence == 0.0
    assert index == 0.0


def _make_lrt_test_model() -> bench.FBCCALDA5Model:
    return bench.FBCCALDA5Model(
        freqs=(9.8, 12.0, 14.8, 15.8),
        labels=("idle", "9.8", "12", "14.8", "15.8"),
        feature_mean=np.zeros(1, dtype=np.float64),
        feature_std=np.ones(1, dtype=np.float64),
        class_means=np.zeros((5, 1), dtype=np.float64),
        pooled_var=np.ones(1, dtype=np.float64),
        command_confidence_th=0.0,
        fit_summary={},
        gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        lrt_feature_indices=(0, 1),
        lrt_feature_mean_control=np.asarray([2.0, 2.0], dtype=np.float64),
        lrt_feature_std_control=np.ones(2, dtype=np.float64),
        lrt_feature_mean_idle=np.zeros(2, dtype=np.float64),
        lrt_feature_std_idle=np.ones(2, dtype=np.float64),
        lrt_window_th=0.5,
        lrt_enter_th=1.0,
        lrt_decay=0.5,
    )


def test_lrt_multiwindow_gate_recovers_continuous_weak_command() -> None:
    labels = np.asarray(("idle", "9.8", "12", "14.8", "15.8"), dtype=object)
    probs = np.asarray(
        [
            [0.48, 0.52, 0.00, 0.00, 0.00],
            [0.47, 0.53, 0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    evidence = np.asarray([1.1, 1.2], dtype=np.float64)

    label, confidence, index = bench._predict_lrt_multiwindow_reject_trial_from_probs(
        _make_lrt_test_model(),
        probs,
        labels,
        evidence,
        min_enter_windows=2,
        max_gap_windows=0,
    )

    assert label == "9.8"
    assert confidence == 1.2
    assert index == 1.0


def test_lrt_multiwindow_gate_rejects_noncommand_like_evidence() -> None:
    labels = np.asarray(("idle", "9.8", "12", "14.8", "15.8"), dtype=object)
    probs = np.asarray(
        [
            [0.10, 0.90, 0.00, 0.00, 0.00],
            [0.10, 0.90, 0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    evidence = np.asarray([-0.2, -0.1], dtype=np.float64)

    label, confidence, index = bench._predict_lrt_multiwindow_reject_trial_from_probs(
        _make_lrt_test_model(),
        probs,
        labels,
        evidence,
        min_enter_windows=2,
        max_gap_windows=0,
    )

    assert label == "idle"
    assert confidence == 0.0
    assert index == 0.0


def test_lrt_window_evidence_prefers_control_distribution() -> None:
    model = _make_lrt_test_model()
    evidence = bench._lrt_window_evidence_from_features(
        model,
        np.asarray(
            [
                [2.0, 2.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    assert evidence[0] > 0.0
    assert evidence[1] < 0.0


def test_fit_lrt_multiwindow_model_carries_parameters_into_evaluation() -> None:
    freqs = (8.0, 9.6, 10.0, 12.0)
    all_freqs = (8.0, 9.6, 10.0, 12.0, 15.0)

    def scored_trial(label: str, command_scores: np.ndarray, trial_id: int) -> bench.ScoredTrial:
        expected_freq = None if label == "idle" else float(label)
        full_scores = np.column_stack(
            [
                command_scores,
                np.full(int(command_scores.shape[0]), 0.25, dtype=np.float64),
            ]
        )
        if label == "idle":
            full_scores[:, -1] = 6.0
        features = bench._score_matrices_to_features(
            command_score_matrix=command_scores,
            command_freqs=freqs,
            score_bank_mode="full_reference_bank",
            all_score_matrix=full_scores,
            all_freqs=all_freqs,
        )
        return bench.ScoredTrial(
            trial=TrialSpec(label=label, expected_freq=expected_freq, trial_id=trial_id, block_index=0),
            score_matrix=command_scores,
            feature_matrix=features,
            duration_sec=3.0,
        )

    scored: list[bench.ScoredTrial] = []
    trial_id = 1
    for command_index, freq in enumerate(freqs):
        for _repeat in range(2):
            scores = np.full((3, 4), 0.35, dtype=np.float64)
            scores[:, command_index] = 6.0
            scored.append(scored_trial(f"{freq:g}", scores, trial_id))
            trial_id += 1
    for _repeat in range(2):
        idle_scores = np.full((3, 4), 0.25, dtype=np.float64)
        scored.append(scored_trial("idle", idle_scores, trial_id))
        trial_id += 1

    model = bench._fit_fbcca_lda5_model(
        scored,
        freqs=freqs,
        win_sec=1.75,
        step_sec=0.25,
        min_enter_windows=2,
        max_gap_windows=1,
        smoothing_windows=2,
        threshold_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
    )
    bundle = bench._evaluate_fbcca_lda5_model(
        model,
        scored,
        win_sec=1.75,
        step_sec=0.25,
        min_enter_windows=2,
        max_gap_windows=1,
    )

    assert model.gate_policy == bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY
    assert model.lrt_feature_indices
    assert model.lrt_feature_mean_control is not None
    assert model.lrt_feature_std_control is not None
    assert model.lrt_feature_mean_idle is not None
    assert model.lrt_feature_std_idle is not None
    assert float(bundle["async_metrics"]["control_trials"]) == 8.0
    assert float(bundle["async_metrics"]["control_recall"]) == 1.0
    assert len(bundle["classifier_trial_events"]) == len(scored)


def test_classifier_recipe_id_records_smoothing_window() -> None:
    assert (
        bench._classifier_recipe_id_with_smoothing(
            win_sec=1.75,
            min_enter_windows=2,
            max_gap_windows=1,
            smoothing_windows=3,
        )
        == "win1p75_me2_gap1_sm3"
    )
    assert (
        bench._classifier_recipe_id_with_smoothing(
            win_sec=1.75,
            min_enter_windows=2,
            max_gap_windows=1,
            smoothing_windows=1,
        )
        == "win1p75_me2_gap1"
    )
    assert (
        bench._classifier_recipe_id_with_smoothing(
            win_sec=1.75,
            min_enter_windows=2,
            max_gap_windows=1,
            smoothing_windows=2,
            gate_policy=bench.CLASSIFIER_ADAPTIVE_EVIDENCE_GATE_POLICY,
        )
        == "win1p75_me2_gap1_sm2_aeg"
    )
    assert (
        bench._classifier_recipe_id_with_smoothing(
            win_sec=1.75,
            min_enter_windows=2,
            max_gap_windows=1,
            smoothing_windows=2,
            gate_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        )
        == "win1p75_me2_gap1_sm2_lrtmw"
    )


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


def test_safe_float_treats_inf_latency_as_missing_for_ranking() -> None:
    good = {
        "idle_fp_per_min": 0.5,
        "idle_selected_windows_per_min": 1.0,
        "control_recall": 0.8,
        "control_recall_at_2.5s": 0.8,
        "control_recall_at_3s": 0.8,
        "async_macro_f1_5class": 0.7,
        "detection_latency_s": 2.0,
    }
    inf_latency = {**good, "detection_latency_s": float("inf")}

    assert bench._classifier_threshold_rank_key(
        good,
        policy="balanced_recall_guard",
    ) < bench._classifier_threshold_rank_key(
        inf_latency,
        policy="balanced_recall_guard",
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
    assert artifact["training_provenance"]["source_subjects"] == ["S1"]
    assert artifact["training_provenance"]["target_subject"] == "S1"
    assert artifact["training_provenance"]["calibration_blocks"] == [0]
    assert artifact["training_provenance"]["excluded_test_blocks"] == [1]
    assert "test blocks are excluded" in artifact["training_provenance"]["data_leakage_guard"]
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
    assert artifact["runtime_loadable"] is False
    assert artifact["state"]["l2"] == 0.3
    assert len(artifact["state"]["weights"]) == feature_count + 1
    assert artifact["runtime_profile_model_params"]["state"]["l2"] == 0.3


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


def test_fit_decoder_methods_reject_full_reference_bank_mode() -> None:
    with pytest.raises(ValueError, match="does not support score_bank_mode=full_reference_bank"):
        bench._score_split_once_for_method(
            method_name="ecca5",
            freqs=(9.8, 12.0, 14.8, 15.8),
            sampling_rate=250,
            step_sec=0.25,
            compute_backend="cpu",
            gpu_device=0,
            gpu_precision="float32",
            calibration_segments=[],
            holdout_segments=[],
            win_sec=1.5,
            context="ecca full-bank guard",
            score_bank_mode="full_reference_bank",
            full_bank_freqs=(8.0, 8.2, 8.4, 8.6),
        )


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

    def fake_score_trials(*, trial_segments, decoder, **_kwargs):
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


def test_score_split_once_for_method_reuses_shared_cache(monkeypatch) -> None:
    freqs = (9.8, 12.0, 14.8, 15.8)
    feature_count = len(bench._classifier_feature_names(freqs))
    calibration_segments = [
        (
            TrialSpec(label="idle", expected_freq=None, trial_id=100, block_index=0),
            np.zeros((500, 8), dtype=np.float64),
        ),
        *[
            (
                TrialSpec(label=f"{float(freq):g}Hz", expected_freq=float(freq), trial_id=index, block_index=0),
                np.zeros((500, 8), dtype=np.float64),
            )
            for index, freq in enumerate(freqs)
        ],
    ]
    holdout_segments = [
        (
            TrialSpec(label="idle", expected_freq=None, trial_id=200, block_index=1),
            np.zeros((500, 8), dtype=np.float64),
        ),
        *[
            (
                TrialSpec(label=f"{float(freq):g}Hz", expected_freq=float(freq), trial_id=10 + index, block_index=1),
                np.zeros((500, 8), dtype=np.float64),
            )
            for index, freq in enumerate(freqs)
        ],
    ]
    score_call_sizes: list[int] = []

    class FakeDecoder:
        win_samples = 1
        step_samples = 1
        fs = 250

    monkeypatch.setattr(bench, "_build_fbcca_decoder_for_scoring", lambda **_kwargs: FakeDecoder())

    def fake_score_trials(*, trial_segments, decoder, **_kwargs):
        assert isinstance(decoder, FakeDecoder)
        score_call_sizes.append(len(trial_segments))
        return [
            bench.ScoredTrial(
                trial=trial,
                score_matrix=np.zeros((1, len(freqs)), dtype=np.float64),
                feature_matrix=np.zeros((1, feature_count), dtype=np.float64),
                duration_sec=2.0,
            )
            for trial, _segment in trial_segments
        ]

    monkeypatch.setattr(bench, "_score_trials_for_classifier", fake_score_trials)
    decoder_cache: dict[tuple[object, ...], object] = {}
    scored_cache: dict[tuple[object, ...], dict[tuple[object, ...], bench.ScoredTrial]] = {}

    first = bench._score_split_once_for_method(
        method_name="fbcca_ridge5",
        freqs=freqs,
        sampling_rate=250,
        step_sec=0.25,
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        calibration_segments=calibration_segments,
        holdout_segments=holdout_segments,
        win_sec=1.5,
        context="first split",
        decoder_cache=decoder_cache,
        scored_cache=scored_cache,
    )
    second = bench._score_split_once_for_method(
        method_name="fbcca_ridge5",
        freqs=freqs,
        sampling_rate=250,
        step_sec=0.25,
        compute_backend="cpu",
        gpu_device=0,
        gpu_precision="float32",
        calibration_segments=list(reversed(calibration_segments)),
        holdout_segments=list(reversed(holdout_segments)),
        win_sec=1.5,
        context="second split",
        decoder_cache=decoder_cache,
        scored_cache=scored_cache,
    )

    assert score_call_sizes == [5, 5]
    assert [item.trial.trial_id for item in first[0]] == [100, 0, 1, 2, 3]
    assert [item.trial.trial_id for item in second[0]] == [3, 2, 1, 0, 100]
    assert [item.trial.trial_id for item in second[1]] == [13, 12, 11, 10, 200]
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


def test_aggregate_recipe_rows_exposes_ysuan_ns_metrics() -> None:
    row = _aggregate_test_row(
        subject="S01",
        recipe_id="win1p5_me2",
        async_acc_5class=0.7,
        async_macro_f1_5class=0.7,
        idle_fp_per_min=0.4,
        control_recall=0.8,
        detection_latency_s=1.5,
    )
    row["dataset"] = "ysu_an"
    row["summary_metrics"].update(
        {
            "ns1_fp_per_min": 0.1,
            "ns2_fp_per_min": 0.2,
            "ns3_fp_per_min": 0.3,
            "ns_all_fp_per_min": 0.25,
            "cs_control_recall": 0.8,
        }
    )

    summary = bench.aggregate_recipe_rows([row], expected_subject_count=1)[0]

    assert abs(float(summary["mean_ns1_fp_per_min"]) - 0.1) < 1e-9
    assert abs(float(summary["mean_ns2_fp_per_min"]) - 0.2) < 1e-9
    assert abs(float(summary["mean_ns3_fp_per_min"]) - 0.3) < 1e-9
    assert abs(float(summary["mean_ns_all_fp_per_min"]) - 0.25) < 1e-9
    assert abs(float(summary["mean_cs_control_recall"]) - 0.8) < 1e-9
    assert abs(float(summary["subjects"][0]["mean_ns_all_fp_per_min"]) - 0.25) < 1e-9


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


def test_aggregate_recipe_rows_marks_deployable_shared_recipe() -> None:
    rows = [
        _aggregate_test_row(
            subject="S01",
            recipe_id="win2_me2_lrtmw",
            async_macro_f1_5class=0.88,
            async_acc_5class=0.94,
            idle_fp_per_min=0.7,
            control_recall=0.85,
            detection_latency_s=2.2,
        ),
        _aggregate_test_row(
            subject="S02",
            recipe_id="win2_me2_lrtmw",
            async_macro_f1_5class=0.90,
            async_acc_5class=0.95,
            idle_fp_per_min=0.9,
            control_recall=0.82,
            detection_latency_s=2.4,
        ),
        _aggregate_test_row(
            subject="S01",
            recipe_id="win1p5_me1_fast",
            async_macro_f1_5class=0.92,
            async_acc_5class=0.97,
            idle_fp_per_min=0.2,
            control_recall=0.90,
            detection_latency_s=1.6,
        ),
    ]

    summaries = bench.aggregate_recipe_rows(rows, expected_subject_count=2)
    shared = bench._shared_recipe_summaries(summaries)
    deployable = bench._deployable_recipe_summaries(shared)

    assert [summary["recipe_id"] for summary in shared] == ["win2_me2_lrtmw"]
    assert [summary["recipe_id"] for summary in deployable] == ["win2_me2_lrtmw"]
    assert deployable[0]["deployable_budget_pass"] is True
    assert deployable[0]["deployable_budget_failed_reasons"] == []


def test_deployable_budget_payload_reports_failed_reasons() -> None:
    payload = bench._deployable_budget_payload(
        {
            "expected_subject_count": 24,
            "coverage_subject_count": 4,
            "mean_idle_fp_per_min": 1.5,
            "mean_control_recall": 0.7,
            "mean_control_recall_at_2.5s": 0.6,
            "mean_detection_latency_s": 3.0,
        }
    )

    assert payload["deployable_budget_pass"] is False
    assert payload["deployable_budget_failed_reasons"] == [
        "full_subject_coverage",
        "idle_fp_budget",
        "control_recall_budget",
        "control_recall_at_2.5s_budget",
        "detection_latency_budget",
    ]


def test_deployable_candidate_profile_payload_records_lrt_recipe_without_runtime_claim() -> None:
    recipe = {
        "method": "fbcca_ridge5",
        "recipe_id": "win2_me2_sm3_lrtmw",
        "frequency_set_id": "none_8_10p5_12_15",
        "idle_multiplier": 3.0,
        "calibration_block_count": 2,
        "selected_freqs": [8.0, 10.5, 12.0, 15.0],
        "expected_subject_count": 24,
        "coverage_subject_count": 24,
        "split_count": 48,
        "mean_idle_fp_per_min": 0.8367,
        "mean_control_recall": 0.8854,
        "mean_control_recall_at_2.5s": 0.7781,
        "mean_detection_latency_s": 2.2969,
        "mean_ns1_fp_per_min": 0.6094,
        "mean_ns2_fp_per_min": 1.9844,
        "mean_ns3_fp_per_min": 0.0,
        "deployable_budget_pass": True,
        "deployable_budget_checks": {"idle_fp_budget": True},
        "deployable_budget": {"max_idle_fp_per_min": 1.0},
    }
    rows = [
        {
            "method": "fbcca_ridge5",
            "recipe_id": "win2_me2_sm3_lrtmw",
            "frequency_set_id": "none_8_10p5_12_15",
            "calibration_blocks": [0, 2],
            "split_summary": {"idle_multiplier": 3.0},
            "calibration_profile": {"candidate_artifact_path": "/remote/S01/win2_me2_sm3_lrtmw_candidate.json"},
        },
        {
            "method": "fbcca_ridge5",
            "recipe_id": "win2_me2_sm3_lrtmw",
            "frequency_set_id": "none_8_10p5_12_15",
            "calibration_blocks": [6, 8],
            "split_summary": {"idle_multiplier": 3.0},
            "calibration_profile": {"candidate_artifact_path": "/remote/S02/win2_me2_sm3_lrtmw_candidate.json"},
        },
        {
            "method": "fbcca_ridge5",
            "recipe_id": "win2_me3_sm3_lrtmw",
            "frequency_set_id": "none_8_10p5_12_15",
            "calibration_blocks": [0, 2],
            "split_summary": {"idle_multiplier": 3.0},
            "calibration_profile": {"candidate_artifact_path": "/remote/other.json"},
        },
    ]

    payload = bench._deployable_candidate_profile_payload(
        run_id="ysuan_full24",
        best_deployable_shared_recipe=recipe,
        rows=rows,
        channel_compatibility={"all_loaded_subjects_match_project_channel_contract": True},
        artifact_paths={"summary_json": "/remote/summary.json"},
    )

    recommended = payload["recommended_short_pretrain_recipe"]
    assert payload["schema_version"] == bench.DEPLOYABLE_CANDIDATE_PROFILE_SCHEMA_VERSION
    assert payload["runtime_loadable"] is False
    assert "do not copy this JSON" in payload["runtime_load_note"]
    assert recommended["threshold_policy"] == "lrt_multiwindow_reject_gate"
    assert recommended["win_sec"] == 2.0
    assert recommended["min_enter_windows"] == 2
    assert recommended["smoothing_windows"] == 3
    assert recommended["idle_multiplier"] == 3.0
    assert recommended["score_bank_mode"] == "full_reference_bank"
    assert recommended["channel_weight_mode"] is None
    assert payload["validation_metrics"]["coverage_subject_count"] == 24
    assert payload["validation_metrics"]["mean_ns2_fp_per_min"] == 1.9844
    assert payload["channel_contract"]["all_loaded_subjects_match_project_channel_contract"] is True
    assert payload["candidate_artifacts"]["candidate_artifact_count"] == 2


def test_artifact_manifest_includes_frequency_specific_reports(tmp_path: Path) -> None:
    paths = bench._artifact_manifest_paths(
        report_root=tmp_path / "reports",
        run_id="freqspec_smoke",
        log_path=tmp_path / "reports" / "benchmark.log",
        failed_cases_path=tmp_path / "reports" / "failed_cases.json",
        coverage_report_path=tmp_path / "reports" / "coverage_report.json",
    )

    assert paths["ns2_by_selected_freq_csv"].endswith("ns2_by_selected_freq.csv")
    assert paths["ns2_by_subject_freq_csv"].endswith("ns2_by_subject_freq.csv")
    assert paths["selected_freq_confusion_csv"].endswith("selected_freq_confusion.csv")
    assert paths["per_frequency_metrics_csv"].endswith("per_frequency_metrics.csv")
    assert paths["gate_params_by_frequency_json"].endswith("gate_params_by_frequency.json")


def test_gate_params_by_frequency_payload_extracts_four_frequency_gates() -> None:
    summary = {
        "method": "fbcca_ridge5",
        "recipe_id": "win2_me2_sm3_lrtmw_fsth0p95_0p95_0p85_1p2",
        "gate_variant": bench.CLASSIFIER_GATE_VARIANT_FREQUENCY_SPECIFIC_THRESHOLD,
        "gate_params": [
            {
                "subject": "ysu_an:S01",
                "split_index": 0,
                "frequency_specific_control_state_gates": {
                    "8": {"theta_lrt_f": 1.0},
                    "10.5": {"theta_lrt_f": 1.1},
                    "12": {"theta_lrt_f": 1.2},
                    "15": {"theta_lrt_f": 1.3},
                },
            }
        ],
    }

    payload = bench._gate_params_by_frequency_payload([summary])
    gates = payload["recipes"][0]["frequency_specific_control_state_gates"]

    assert sorted(gates) == ["10.5", "12", "15", "8"]
    assert gates["8"][0]["theta_lrt_f"] == 1.0
    assert gates["15"][0]["subject"] == "ysu_an:S01"


def test_tracked_external_fbcca_candidate_manifest_matches_ysuan_lrt_contract() -> None:
    manifest_path = PROJECT_DIR / "config" / "external_fbcca_classifier_candidate_v1.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = dict(payload["recommended_short_pretrain_recipe"])

    assert payload["schema_version"] == "external_fbcca_classifier_candidate_profile_v1"
    assert payload["runtime_loadable"] is False
    assert payload["budget_pass"] is True
    assert payload["source_run"]["expected_subject_count"] == 24
    assert payload["source_run"]["coverage_subject_count"] == 24
    assert recipe["method"] == "fbcca_ridge5"
    assert recipe["recipe_id"] == "win2_me2_sm3_lrtmw"
    assert recipe["score_bank_mode"] == "full_reference_bank"
    assert recipe["threshold_policy"] == "lrt_multiwindow_reject_gate"
    assert recipe["win_sec"] == 2.0
    assert recipe["min_enter_windows"] == 2
    assert recipe["smoothing_windows"] == 3
    assert recipe["channel_weight_mode"] is None
    assert recipe["channel_weights"] is None
    assert payload["channel_contract"]["project_channel_names"] == list(bench.PROJECT_POSTERIOR_8_CHANNELS)
    assert payload["channel_contract"]["only_required_channels_used"] is True


def test_case_tracker_records_failure_safe_coverage_and_failures() -> None:
    ctx = bench.CaseContext(
        dataset="beta",
        subject="S11",
        frequency_profile="deploy_current_profile",
        frequency_set_id="freqs_8_10_12_15",
        selected_freqs=(8.0, 10.0, 12.0, 15.0),
        method="tdca5",
        calibration_blocks=(0,),
        holdout_blocks=(1, 2),
        split_index=0,
        window_length_s=1.0,
        min_enter_windows=2,
        reject_gate="lrt_multiwindow_reject_gate:full_reference_bank",
        implementation_level="engineering-approx",
    )
    tracker = bench.CaseTracker(expected_subject_count=2)

    tracker.planned(ctx)
    tracker.skipped(ctx, reason="insufficient_training_trials", detail="one calibration block")
    tracker.planned(ctx)
    tracker.completed(ctx, row={"recipe_id": "win1_me2", "calibration_profile": {"candidate_artifact_path": "candidate.json"}})
    report = tracker.report()

    failed = tracker.failed_cases[0]
    assert failed["skip_or_fail"] == "skip"
    assert failed["skip_reason"] == "insufficient_training_trials"
    assert failed["excluded_test_blocks"] == [1, 2]
    leaf = report["by_dataset_frequency_profile_method_subject"]["beta"]["deploy_current_profile"]["tdca5"]["S11"]["0"]["split00"]
    assert leaf["planned"] == 2
    assert leaf["skipped"] == 1
    assert leaf["completed"] == 1
    flat = report["by_dataset_frequency_profile_method"][0]
    assert flat["shared_eligible"] is False
    assert flat["subjects_completed"] == 1
    assert report["event_count"] == 4
    assert report["case_count"] == 2
    assert report["planned_case_count"] == 2
    assert report["completed_case_count"] == 1
    assert report["skipped_case_count"] == 1
    assert report["hard_failed_case_count"] == 0


def test_score_bank_compatibility_skips_template_spatial_full_bank() -> None:
    assert bench._method_score_bank_skip_reason("trca5", "full_reference_bank") == "unsupported_score_bank_mode"
    assert bench._method_score_bank_skip_reason("tdca5", "full_reference_bank") == "unsupported_score_bank_mode"
    assert bench._method_score_bank_skip_reason("fbcca_ridge5", "full_reference_bank") == ""
    assert bench._method_score_bank_skip_reason("trca5", "command_only") == ""


def test_enrich_result_row_writes_contract_fields_and_levels() -> None:
    row = _aggregate_test_row(
        subject="S11",
        recipe_id="win1_me2",
        async_macro_f1_5class=0.5,
        async_acc_5class=0.6,
        idle_fp_per_min=1.2,
        control_recall=0.7,
        detection_latency_s=2.1,
    )
    row["method"] = "trca5"
    row["calibration_blocks"] = [0]
    frequency_case = bench.FrequencyEvalCase(
        mode="none",
        frequency_set_id="freqs_8_10_12_15",
        freqs=(8.0, 10.0, 12.0, 15.0),
    )

    enriched = bench._enrich_result_row(
        row,
        frequency_profile="deploy_current_profile",
        frequency_case=frequency_case,
        step_sec=0.25,
        decision_start_sec=0.5,
        decision_deadline_sec=2.5,
        min_release_windows=3,
        threshold_policy=bench.CLASSIFIER_LRT_MULTIWINDOW_REJECT_GATE_POLICY,
        score_bank_mode="full_reference_bank",
    )

    assert enriched["frequency_profile"] == "deploy_current_profile"
    assert enriched["step_size_s"] == 0.25
    assert enriched["decision_start_s"] == 0.5
    assert enriched["decision_deadline_s"] == 2.5
    assert enriched["min_release_windows"] == 3
    assert enriched["implementation_level"] == "engineering-approx"
    assert enriched["engineering_approx"] is True
    assert enriched["paper_faithful"] is False
    assert enriched["reject_gate"] == "lrt_multiwindow_reject_gate:full_reference_bank"
    assert "mixed_idle_fp_per_min" in enriched["summary_metrics"]


def test_aggregate_recipe_rows_keeps_missing_real_idle_as_null() -> None:
    base_row = _aggregate_test_row(
        subject="S1",
        recipe_id="win1_me1",
        async_macro_f1_5class=0.5,
        async_acc_5class=0.6,
        idle_fp_per_min=1.2,
        control_recall=0.7,
        detection_latency_s=2.1,
    )
    base_metrics = dict(base_row["summary_metrics"])
    base_metrics.update(
        {
            "real_idle_fp_per_min": None,
            "approx_idle_fp_per_min": 1.2,
            "mixed_idle_fp_per_min": 1.2,
        }
    )
    rows = [
        {
            **base_row,
            "dataset": "beta",
            "summary_metrics": base_metrics,
        }
    ]

    summary = bench.aggregate_recipe_rows(rows, expected_subject_count=1)[0]

    assert summary["mean_real_idle_fp_per_min"] != summary["mean_real_idle_fp_per_min"]
    assert bench.json_safe(summary)["mean_real_idle_fp_per_min"] is None
    assert abs(float(summary["mean_approx_idle_fp_per_min"]) - 1.2) < 1e-9
    assert abs(float(summary["mean_mixed_idle_fp_per_min"]) - 1.2) < 1e-9


def test_aggregate_recipe_rows_keeps_frequency_sets_separate() -> None:
    rows = [
        {
            **_aggregate_test_row(
                subject="S1",
                recipe_id="win1p5_me2",
                async_macro_f1_5class=0.8,
                async_acc_5class=0.9,
                idle_fp_per_min=0.5,
                control_recall=0.85,
                detection_latency_s=1.5,
            ),
            "selected_freqs": [8.0, 9.6, 10.0, 12.0],
            "frequency_set_id": "shared_fixed4_8_9p6_10_12",
            "frequency_selection_mode": "shared_fixed4",
        },
        {
            **_aggregate_test_row(
                subject="S1",
                recipe_id="win1p5_me2",
                async_macro_f1_5class=0.7,
                async_acc_5class=0.88,
                idle_fp_per_min=0.4,
                control_recall=0.75,
                detection_latency_s=1.7,
            ),
            "selected_freqs": [8.0, 10.0, 12.0, 15.0],
            "frequency_set_id": "shared_fixed4_8_10_12_15",
            "frequency_selection_mode": "shared_fixed4",
        },
    ]

    summaries = bench.aggregate_recipe_rows(rows, expected_subject_count=1)

    assert len(summaries) == 2
    assert {tuple(summary["selected_freqs"]) for summary in summaries} == {
        (8.0, 9.6, 10.0, 12.0),
        (8.0, 10.0, 12.0, 15.0),
    }
    assert all(summary["shared_eligible"] for summary in summaries)


def test_personalized_frequency_policy_can_be_shared_eligible_with_per_subject_freqs() -> None:
    rows = [
        {
            **_aggregate_test_row(
                subject="S1",
                recipe_id="win1p5_me2",
                async_macro_f1_5class=0.8,
                async_acc_5class=0.9,
                idle_fp_per_min=0.5,
                control_recall=0.85,
                detection_latency_s=1.5,
            ),
            "selected_freqs": [8.0, 9.6, 10.0, 12.0],
            "frequency_set_id": "personalized_upper_bound_calibration_only_c8",
            "frequency_selection_mode": "personalized_upper_bound",
        },
        {
            **_aggregate_test_row(
                subject="S16",
                recipe_id="win1p5_me2",
                async_macro_f1_5class=0.82,
                async_acc_5class=0.91,
                idle_fp_per_min=0.6,
                control_recall=0.86,
                detection_latency_s=1.6,
            ),
            "selected_freqs": [8.2, 9.8, 12.0, 15.0],
            "frequency_set_id": "personalized_upper_bound_calibration_only_c8",
            "frequency_selection_mode": "personalized_upper_bound",
        },
    ]

    summary = bench.aggregate_recipe_rows(rows, expected_subject_count=2)[0]

    assert summary["shared_eligible"] is True
    assert summary["coverage_subject_count"] == 2
    assert summary["selected_freqs"] == []
    assert summary["per_subject_selected_freqs"]["beta:S1"] == [8.0, 9.6, 10.0, 12.0]
    assert summary["per_subject_selected_freqs"]["beta:S16"] == [8.2, 9.8, 12.0, 15.0]


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
    assert "| Rank | Deployable | Profile | Method | Recipe | Freqs | Coverage |" in markdown
    assert "| 1 | yes |" in markdown
    assert "| fbcca_ridge5 | `win2_me1` | `` | 2/2 |" in markdown
    assert "detection_latency_s is stimulus onset to first correct control output" in markdown


def test_run_metadata_and_evaluation_contract_schema_paths(tmp_path: Path) -> None:
    artifact_paths = bench._artifact_manifest_paths(
        report_root=tmp_path / "reports",
        run_id="external_short_pretrain_beta_profile_smoke_deploy_20260511_120000",
        log_path=tmp_path / "reports" / "benchmark.log",
        failed_cases_path=tmp_path / "reports" / "failed_cases.json",
        coverage_report_path=tmp_path / "reports" / "coverage_report.json",
    )
    metadata = bench._run_metadata_payload(
        run_id="external_short_pretrain_beta_profile_smoke_deploy_20260511_120000",
        datasets=("beta",),
        freqs=(8.0, 10.0, 12.0, 15.0),
        methods=("fbcca_ridge5",),
        subjects_expected=8,
        calibration_blocks=(1, 2),
        window_lengths=(1.0, 1.25, 1.5),
        score_bank_mode="full_reference_bank",
        idle_eval_mode="both",
        timeout_sec=3600.0,
        artifact_paths=artifact_paths,
    )
    contract = bench._evaluation_contract_payload(
        datasets=("beta",),
        freqs=(8.0, 10.0, 12.0, 15.0),
        methods=("fbcca_ridge5",),
        subjects_expected=8,
        subjects_completed=8,
        calibration_blocks=(1, 2),
        window_lengths=(1.0, 1.25, 1.5),
        step_sec=0.25,
        decision_start_sec=0.5,
        decision_deadline_sec=2.5,
        min_release_windows=2,
        reject_gate="lrt_multiwindow_reject_gate:full_reference_bank",
        artifact_paths=artifact_paths,
        implementation_level="paper-faithful",
    )

    assert metadata["frequency_profile"] == "deploy_current_profile"
    assert metadata["server_writable_root"] == "/data1/zkx/brain/ssvep"
    assert metadata["server_log_contract"].endswith("/logs/external_short_pretrain_beta_profile_smoke_deploy_20260511_120000.log")
    assert metadata["candidate_artifacts_only"] is True
    assert contract["subjects_expected"] == 8
    assert contract["subjects_completed"] == 8
    assert contract["step_size_s"] == 0.25
    assert contract["decision_start_s"] == 0.5
    assert contract["decision_deadline_s"] == 2.5
    assert contract["reject_gate"] == "lrt_multiwindow_reject_gate:full_reference_bank"
    assert "no-control is produced only by the reject gate" in contract["no_control_policy"]
    assert set(artifact_paths) >= {
        "summary_json",
        "summary_md",
        "partial_summary_json",
        "failed_cases_json",
        "coverage_report_json",
        "logistic_trace_windows_csv",
        "logistic_trace_trial_summary_csv",
        "logistic_transition_counts_by_subject_csv",
        "logistic_transition_counts_by_frequency_csv",
        "logistic_feature_summary_tp_fp_csv",
        "local_log",
        "server_log_contract",
    }


def test_parser_exposes_pseudo_online_timing_and_timeout_options() -> None:
    parser = bench.build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "explicit_run",
            "--output-root",
            "out",
            "--dataset-root",
            "data",
            "--wang-raw-dir",
            "wang",
            "--wang-channels-loc",
            "loc",
            "--beta-raw-dir",
            "beta",
            "--decision-start-sec",
            "0.5",
            "--decision-deadline-sec",
            "2.5",
            "--min-release-windows",
            "4",
            "--timeout-sec",
            "60",
            "--case-limit",
            "4",
        ]
    )

    assert args.run_id == "explicit_run"
    assert float(args.decision_start_sec) == 0.5
    assert float(args.decision_deadline_sec) == 2.5
    assert int(args.min_release_windows) == 4
    assert float(args.timeout_sec) == 60.0
    assert int(args.case_limit) == 4


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

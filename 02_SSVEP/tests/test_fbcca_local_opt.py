from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.training_evaluation_ui import TrainingEvaluationWindow
from apps.training_evaluation_ui import _parse_task as ui_parse_task
from apps.training_evaluation_ui import build_parser as ui_build_parser
from ssvep_core.async_fbcca_idle_standalone import ThresholdProfile, TrialSpec
from ssvep_core.fbcca_local_opt import (
    FBCCALocalOptConfig,
    RepeatedGroupSplit,
    _fbcca_meets_promotion_thresholds,
    _fbcca_variant_priority,
    _resolve_search_plan,
    run_fbcca_local_opt,
)
from tools.training_evaluation_cli import _parse_task as cli_parse_task
from tools.training_evaluation_cli import build_parser as cli_build_parser


def _get_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_profile(*, variant: str = "fbcca_fixed_all8") -> ThresholdProfile:
    return ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=2.5,
        step_sec=0.25,
        enter_score_th=0.5,
        enter_ratio_th=1.4,
        enter_margin_th=0.2,
        exit_score_th=0.4,
        exit_ratio_th=1.1,
        min_enter_windows=1,
        min_exit_windows=1,
        model_name="fbcca",
        model_params={"Nh": 3, "fbcca_variant": variant},
        enter_p_th=0.65,
        exit_p_th=0.30,
        confidence_variant="global_correctness_logistic",
        training_window_policy="last_window_only",
        enter_log_lr_th=0.5,
        exit_log_lr_th=0.0,
        min_switch_windows=1,
        switch_enter_score_th=0.55,
        switch_enter_ratio_th=1.4,
        switch_enter_margin_th=0.2,
        control_state_mode="frequency-specific-logistic",
        frequency_specific_thresholds={
            "8": {
                "enter_p_th": 0.65,
                "exit_p_th": 0.30,
                "min_enter_windows": 1,
                "min_exit_windows": 1,
                "min_switch_windows": 1,
            }
        },
    )


def _make_scored_rows() -> list[dict[str, object]]:
    return [
        {
            "label": "8Hz",
            "trial_role": "control",
            "trial_id": 1,
            "window_index": 0,
            "pred_freq": 8.0,
            "expected_freq": 8.0,
            "top1_score": 0.9,
            "top2_score": 0.2,
            "ratio": 2.0,
            "margin": 0.7,
            "gap_12": 0.7,
            "gap_13": 0.75,
            "gap_14": 0.8,
            "gate_score": 1.0,
            "control_log_lr": 1.0,
            "p_control": 0.85,
            "p_correct": 0.82,
            "correctness_logit": 1.52,
            "normalized_top1": 0.9,
            "score_entropy": 0.2,
            "consistency": 1.0,
        },
        {
            "label": "idle",
            "trial_role": "clean_idle",
            "trial_id": 2,
            "window_index": 0,
            "pred_freq": 8.0,
            "expected_freq": None,
            "top1_score": 0.15,
            "top2_score": 0.14,
            "ratio": 1.05,
            "margin": 0.01,
            "gap_12": 0.01,
            "gap_13": 0.02,
            "gap_14": 0.03,
            "gate_score": -0.8,
            "control_log_lr": -0.8,
            "p_control": 0.1,
            "p_correct": 0.12,
            "correctness_logit": -1.99,
            "normalized_top1": 0.15,
            "score_entropy": 0.9,
            "consistency": 0.0,
        },
    ]


def _bundle_for_variant(variant: str, confidence: str, refractory_sec: float) -> dict[str, object]:
    variant_order = {
        "fbcca_sw_all8": 0,
        "fbcca_fixed_all8": 1,
        "fbcca_cw_all8": 2,
        "fbcca_cw_sw_all8": 3,
        "fbcca_cw_sw_trca_shared": 4,
    }
    base = float(variant_order[variant])
    confidence_bonus = 0.0 if confidence == "global_correctness_logistic" else 0.15
    release_latency = 2.0 + 0.2 * base + float(refractory_sec)
    switch_latency = 2.1 + 0.15 * base + 0.5 * float(refractory_sec) + confidence_bonus
    control_recall = max(0.35, 0.72 - 0.05 * base - 0.03 * confidence_bonus)
    return {
        "async_metrics": {
            "idle_fp_per_min": 0.2 + 0.05 * base + 0.05 * confidence_bonus,
            "control_recall": float(control_recall),
            "control_recall_at_3s": float(max(control_recall - 0.05, 0.2)),
            "switch_latency_s": float(switch_latency),
            "release_latency_s": float(release_latency),
            "inference_ms": float(10.0 + base),
        },
        "metrics_4class": {"acc": 0.8 - 0.04 * base, "macro_f1": 0.75 - 0.03 * base},
        "metrics_2class": {"acc": 0.88 - 0.02 * base, "macro_f1": 0.86 - 0.02 * base},
        "trial_events": [
            {
                "label": "8Hz",
                "trial_id": 1,
                "expected_freq": 8.0,
                "switch_trial": True,
                "release_trial": False,
                "first_correct_latency_s": None if variant != "fbcca_fixed_all8" else 1.0,
                "first_release_latency_s": None,
                "raw_correct_seen": True,
                "gate_pass_correct_seen": True,
                "first_gate_pass_latency_s": 0.5,
                "max_p_correct": 0.82,
                "max_decision_evidence": 1.2,
                "commit_seen": variant == "fbcca_fixed_all8",
            },
            {
                "label": "idle",
                "trial_id": 2,
                "expected_freq": None,
                "switch_trial": False,
                "release_trial": True,
                "first_release_latency_s": float(release_latency),
                "raw_correct_seen": False,
                "gate_pass_correct_seen": False,
                "commit_seen": False,
            },
        ],
    }


def test_fbcca_cli_and_ui_accept_local_opt_task() -> None:
    ui_args = ui_build_parser().parse_args(["--task", "fbcca-local-opt", "--search-preset", "smoke20"])
    cli_args = cli_build_parser().parse_args(["--task", "fbcca-local-opt", "--search-preset", "reduced40"])

    assert ui_args.task == "fbcca-local-opt"
    assert ui_args.search_preset == "smoke20"
    assert cli_args.task == "fbcca-local-opt"
    assert cli_args.search_preset == "reduced40"
    assert ui_parse_task("fbcca_local_opt") == "fbcca-local-opt"
    assert cli_parse_task("fbcca_local_opt") == "fbcca-local-opt"


def test_fbcca_search_preset_counts_and_variant_tiebreak() -> None:
    base = FBCCALocalOptConfig(
        dataset_manifest_session1=Path("session_manifest.json"),
        output_profile_path=Path("profile.json"),
        report_path=Path("report.json"),
    )
    smoke = _resolve_search_plan(
        FBCCALocalOptConfig(
            dataset_manifest_session1=base.dataset_manifest_session1,
            output_profile_path=base.output_profile_path,
            report_path=base.report_path,
            search_preset="smoke20",
        )
    )
    reduced = _resolve_search_plan(base)

    assert len(reduced["candidate_grid"]) == 40
    assert len(smoke["candidate_grid"]) == 20
    assert _fbcca_variant_priority("fbcca_fixed_all8") < _fbcca_variant_priority("fbcca_cw_sw_trca_shared")
    assert reduced["search_preset"] == "reduced40"
    assert smoke["search_preset"] == "smoke20"


def test_fbcca_ui_config_does_not_append_baseline_models() -> None:
    _ = _get_qapp()
    window = TrainingEvaluationWindow()
    manifest_dir = PROJECT_DIR / ".tmp_test_artifacts" / f"fbcca_ui_{uuid.uuid4().hex}"
    manifest_path = manifest_dir / "session_manifest.json"
    try:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        window._dataset_scan_rows = []
        window.simple_mode_check.setChecked(False)
        window.configure_fbcca_local_opt_mode(auto_start=False)
        window.session1_edit.setText(str(manifest_path))
        window.session2_edit.setText("")

        cfg = window._read_config()

        assert cfg.task == "fbcca-local-opt"
        assert cfg.model_names == ("fbcca",)
        assert cfg.channel_modes == ("all8",)
    finally:
        shutil.rmtree(manifest_dir, ignore_errors=True)
        window.close()


def test_run_fbcca_local_opt_smoke(monkeypatch, tmp_path: Path) -> None:
    import ssvep_core.fbcca_local_opt as module

    manifest_path = tmp_path / "session_manifest.json"
    manifest_path.write_text(json.dumps({"trials": [], "quality_summary": {}}), encoding="utf-8")

    merged = module.MergedLocalDataset(
        manifest_paths=(manifest_path,),
        datasets=(),
        trial_segments=(
            (TrialSpec(label="8Hz", expected_freq=8.0, trial_id=0, block_index=0), np.zeros((8, 8), dtype=np.float64)),
            (TrialSpec(label="10Hz", expected_freq=10.0, trial_id=1, block_index=1), np.zeros((8, 8), dtype=np.float64)),
            (TrialSpec(label="idle", expected_freq=None, trial_id=2, block_index=2), np.zeros((8, 8), dtype=np.float64)),
            (TrialSpec(label="switch_to_8Hz", expected_freq=None, trial_id=3, block_index=3), np.zeros((8, 8), dtype=np.float64)),
        ),
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        subject_id="subject_smoke",
        session_ids=("session_smoke",),
        trial_role_counts={"control": 2, "clean_idle": 1, "hard_idle": 1},
        quality_rows=(),
    )
    split = RepeatedGroupSplit(repeat_index=0, train_indices=(0, 1), gate_indices=(2,), holdout_indices=(3,), fingerprint="split_smoke")
    scored_rows = _make_scored_rows()
    publish_calls: list[dict[str, object]] = []

    class _FakeGateModel:
        feature_names = ("gate_score", "margin")

        def to_payload(self):
            return {"per_freq": {"8": {"enter_p_th": 0.65, "exit_p_th": 0.3}}}

    class _ProfileV2:
        def to_payload(self):
            return {"profile_v2": True}

    def fake_candidate_context(*, fbcca_variant: str, confidence_variant: str, win_sec: float, **_kwargs):
        bundle = _bundle_for_variant(str(fbcca_variant), str(confidence_variant), 0.0)
        variant_rows = [{**dict(row), "fbcca_variant": str(fbcca_variant), "confidence_variant": str(confidence_variant)} for row in scored_rows]
        return {
            "default_holdout_bundle": bundle,
            "inference_ms": float(bundle["async_metrics"]["inference_ms"]),
            "gate_calibration_summary": {"gate_calibration_valid": True, "positive_windows": 40, "negative_windows": 60},
            "tune_summary": {"valid": True, "rows_total": 64},
            "training_window_policy": "last_window_only",
            "training_latency_sec": 0.0,
            "analysis_latency_sec": 0.0,
            "effective_raw_window_sec": float(win_sec),
            "confidence_training_scheme": "oof_gate_logreg_on_train_split",
            "oof_group_key": "trial_id",
            "oof_group_count": 4,
            "sample_weight_mode": "per_trial_equal",
            "positive_trials": 2,
            "negative_trials": 2,
            "frontend_optimization_summary": {
                "fbcca_variant": str(fbcca_variant),
                "dynamic_stop_enabled": False,
                "train_only": True,
                "status": "optimized" if str(fbcca_variant) != "fbcca_fixed_all8" else "fixed_default",
            },
            "gate_search_board": [{"enter_p_th": 0.65, "exit_p_th": 0.3, "n_control_rows": 12, "n_idle_rows": 24}],
            "gate_exit_threshold_board": [{"enter_p_th": 0.65, "exit_p_th": 0.3, "enter_logit_th": 0.2, "exit_logit_th": -0.3, "n_control_rows": 12, "n_idle_rows": 24}],
            "scored_tune_rows": variant_rows,
            "scored_holdout_rows": variant_rows,
            "gate_profile": _make_profile(variant=str(fbcca_variant)),
            "gate_model": _FakeGateModel(),
            "state_payload": {"fbcca_variant": str(fbcca_variant)},
            "model_params": {"Nh": 3, "fbcca_variant": str(fbcca_variant)},
        }

    def fake_eval_structured_rows(*, scored_rows, decision_params, **_kwargs):
        sample = dict(scored_rows[0])
        return _bundle_for_variant(
            str(sample.get("fbcca_variant", sample.get("label_variant", "fbcca_fixed_all8")) or "fbcca_fixed_all8"),
            str(sample.get("confidence_variant", "global_correctness_logistic")),
            float(decision_params.get("refractory_sec", 0.0)),
        )

    def fake_baselines(*_args, **_kwargs):
        fixed_bundle = _bundle_for_variant("fbcca_fixed_all8", "global_correctness_logistic", 0.0)
        return [
            {
                "baseline_name": "legacy_fbcca_202603 @ win=3.0",
                "model_name": "legacy_fbcca_202603",
                "fbcca_variant": "legacy_fbcca_202603",
                "metrics_median": {"idle_fp_per_min": 0.35, "control_recall": 0.7, "control_recall_at_3s": 0.65, "release_latency_s": 2.6, "switch_latency_s": 2.8, "inference_ms": 11.0},
                "rank_key": [0.35, 2.6, 2.8, -0.65, -0.7, 11.0],
                "_holdout_bundles": [],
            },
            {
                "baseline_name": "fbcca_fixed_all8 @ win=3.0",
                "model_name": "fbcca",
                "fbcca_variant": "fbcca_fixed_all8",
                "metrics_median": {"idle_fp_per_min": 0.18, "control_recall": 0.82, "control_recall_at_3s": 0.77, "release_latency_s": 2.1, "switch_latency_s": 2.2, "inference_ms": 10.0},
                "rank_key": [0.18, 2.1, 2.2, -0.77, -0.82, 10.0],
                "_holdout_bundles": [fixed_bundle],
            },
            {
                "baseline_name": "trca_r @ win=3.0",
                "model_name": "trca_r",
                "fbcca_variant": "n/a",
                "metrics_median": {"idle_fp_per_min": 0.25, "control_recall": 0.8, "control_recall_at_3s": 0.75, "release_latency_s": 2.3, "switch_latency_s": 2.4, "inference_ms": 12.0},
                "rank_key": [0.25, 2.3, 2.4, -0.75, -0.8, 12.0],
                "_holdout_bundles": [],
            },
        ]

    monkeypatch.setattr(module._tdca, "_load_merged_dataset", lambda _cfg: merged)
    monkeypatch.setattr(module, "build_repeated_group_splits", lambda *_args, **_kwargs: [split])
    monkeypatch.setattr(module, "preflight_fbcca_local_env", lambda **_kwargs: {"effective_backend": "cpu", "gpu_replay_speedup": 0.0})
    monkeypatch.setattr(module._tdca, "_split_replay_policy", lambda **_kwargs: {"effective_replay_backend": "cpu", "gpu_replay_speedup": 0.0, "gpu_replay_eligible": False, "gpu_replay_reason": "batched_replay_not_implemented"})
    monkeypatch.setattr(module, "_run_baseline_suite", fake_baselines)
    monkeypatch.setattr(module, "_build_fbcca_candidate_context", fake_candidate_context)
    monkeypatch.setattr(module._tdca, "_evaluate_structured_rows", fake_eval_structured_rows)
    monkeypatch.setattr(module, "save_profile", lambda _profile, path: Path(path).write_text("{}", encoding="utf-8"))
    monkeypatch.setattr(module, "build_profile_v2", lambda **_kwargs: _ProfileV2())
    monkeypatch.setattr(module, "publish_deployed_profile", lambda **kwargs: publish_calls.append(kwargs))

    config = FBCCALocalOptConfig(
        dataset_manifest_session1=manifest_path,
        dataset_manifests=(manifest_path,),
        output_profile_path=tmp_path / "profile_fbcca_local.json",
        report_path=tmp_path / "report_fbcca_local.json",
        organize_report_dir=False,
        compute_backend="cpu",
        search_preset="smoke20",
    )
    payload = run_fbcca_local_opt(config, log_fn=lambda _msg: None)

    assert payload["task"] == "fbcca-local-opt"
    assert payload["chosen_model"] == "fbcca"
    assert payload["fbcca_search_board"]
    assert payload["decision_search_board"]
    assert payload["holdout_selection_board"]
    assert payload["variant_summary"]
    assert payload["decision_search_target"] == "tune_split"
    assert payload["final_selection_target"] == "holdout_split"
    assert payload["decision_evidence_variant"] == "centered_logit_over_enter_threshold"
    assert payload["dynamic_stop_enabled"] is False
    assert {row["fbcca_variant"] for row in payload["fbcca_search_board"]} == {
        "fbcca_fixed_all8",
        "fbcca_cw_all8",
        "fbcca_sw_all8",
        "fbcca_cw_sw_all8",
        "fbcca_cw_sw_trca_shared",
    }
    assert {row["confidence_variant"] for row in payload["fbcca_search_board"]} == {
        "global_correctness_logistic",
        "bayesian_gap_gmm",
    }
    assert all("frontend_optimization_summary" in row for row in payload["fbcca_search_board"])
    assert payload["confidence_diagnostics_board"]
    assert payload["decision_bottleneck_summary"]
    assert payload["error_attribution_board"]
    assert payload["contrast_error_board"]
    assert payload["chosen_model_rationale"] == "fbcca_not_clearly_improved"
    assert payload["run_valid_for_deployment"] is False
    assert payload["deployment_grade"] == "provisional_single_session"
    assert payload["profile_saved"] is True
    assert payload["profile_v2_saved"] is True
    assert publish_calls == []
    assert Path(payload["report_path"]).exists()
    assert Path(payload["chosen_profile_path"]).exists()
    assert Path(payload["profile_v2_path"]).exists()
    assert _fbcca_meets_promotion_thresholds(payload["chosen_async_metrics"]) is False

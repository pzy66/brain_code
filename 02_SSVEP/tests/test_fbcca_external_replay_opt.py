from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.training_evaluation_ui import (
    TrainingEvaluationWindow,
    _apply_fbcca_external_replay_args as ui_apply_external_args,
    _parse_task as ui_parse_task,
    build_parser as ui_build_parser,
)
from ssvep_core.external_replay_dataset import EXTERNAL_LED_CHANNELS, EXTERNAL_LED_FREQS, ExternalReplayDataset, ExternalReplaySession
from ssvep_core.fbcca_external_replay_opt import (
    FBCCAExternalReplayOptConfig,
    _build_external_decision_bottleneck_summary,
    _build_external_error_attribution_board,
    _build_tune_frequency_breakdown,
    _compute_per_frequency_reference_overrides,
    _diagnostic_best_row,
    _build_replay_frequency_breakdown,
    _external_decision_param_grid,
    _external_gate_grid,
    _external_rank_metrics_key,
    _is_diagnostic_only_variant,
    _outer_folds,
    _resolve_search_plan,
    _selection_validity_summary,
    _strict_selection_rows,
)
from ssvep_core.gating.correctness_calibrator import GLOBAL_CORRECTNESS_LOGISTIC
from tools.training_evaluation_cli import _parse_task as cli_parse_task
from tools.training_evaluation_cli import _apply_fbcca_external_replay_args as cli_apply_external_args
from tools.training_evaluation_cli import build_parser as cli_build_parser


def _get_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_dataset() -> ExternalReplayDataset:
    sessions = []
    for session_index in range(4):
        data = np.zeros((256, len(EXTERNAL_LED_CHANNELS)), dtype=np.float64)
        sessions.append(
            ExternalReplaySession(
                subject_id="Subject2",
                session_id=f"s{session_index + 1}",
                session_index=session_index,
                file_path=Path(f"s{session_index + 1}.gdf"),
                sampling_rate=256,
                channel_names=tuple(EXTERNAL_LED_CHANNELS),
                data=data,
                trials=tuple(),
            )
        )
    return ExternalReplayDataset(
        dataset_root=Path("dataset"),
        subject_id="Subject2",
        sampling_rate=256,
        channel_names=tuple(EXTERNAL_LED_CHANNELS),
        freqs=tuple(EXTERNAL_LED_FREQS),
        sessions=tuple(sessions),
    )


def test_external_replay_cli_and_ui_accept_task_and_args() -> None:
    ui_args = ui_build_parser().parse_args(
        [
            "--task",
            "fbcca-external-replay-opt",
            "--subject",
            "Subject2",
            "--search-preset",
            "smoke8",
            "--outer-eval",
            "loso4",
            "--replay-speed",
            "2x",
        ]
    )
    cli_args = cli_build_parser().parse_args(
        [
            "--task",
            "fbcca-external-replay-opt",
            "--subject",
            "Subject2",
            "--search-preset",
            "reduced24",
            "--outer-eval",
            "chronological-last",
            "--replay-speed",
            "max",
        ]
    )

    assert ui_args.task == "fbcca-external-replay-opt"
    assert ui_args.search_preset == "smoke8"
    assert ui_args.outer_eval == "loso4"
    assert ui_args.replay_speed == "2x"
    assert cli_args.task == "fbcca-external-replay-opt"
    assert cli_args.search_preset == "reduced24"
    assert cli_args.outer_eval == "chronological-last"
    assert cli_args.replay_speed == "max"
    assert ui_parse_task("external_fbcca") == "fbcca-external-replay-opt"
    assert cli_parse_task("external_fbcca") == "fbcca-external-replay-opt"


def test_external_replay_defaults_profile_name_to_profile_json() -> None:
    cli_args = cli_build_parser().parse_args(["--task", "fbcca-external-replay-opt", "--subject", "Subject2"])
    cli_apply_external_args(cli_args, {"--task", "--subject"})
    assert Path(cli_args.output_profile).name == "profile.json"

    ui_args = ui_build_parser().parse_args(["--task", "fbcca-external-replay-opt", "--subject", "Subject2"])
    ui_apply_external_args(ui_args, ("--task", "fbcca-external-replay-opt", "--subject", "Subject2"))
    assert Path(ui_args.output_profile).name == "profile.json"


def test_external_replay_search_presets_and_outer_folds() -> None:
    base = FBCCAExternalReplayOptConfig(
        external_dataset_root=Path("dataset"),
        subject="Subject2",
        output_profile_path=Path("profile.json"),
        report_path=Path("report.json"),
    )
    smoke = _resolve_search_plan(
        FBCCAExternalReplayOptConfig(
            external_dataset_root=base.external_dataset_root,
            subject=base.subject,
            output_profile_path=base.output_profile_path,
            report_path=base.report_path,
            search_preset="smoke8",
        )
    )
    reduced = _resolve_search_plan(base)
    dataset = _make_dataset()
    loso = _outer_folds(dataset, mode="loso4")
    chrono = _outer_folds(dataset, mode="chronological-last")

    assert len(smoke["candidate_grid"]) == 8
    assert len(reduced["candidate_grid"]) == 24
    assert {item["decoder_variant"] for item in smoke["candidate_grid"]} == {
        "fbcca_fixed_all8",
        "fbcca_sw_all8",
        "itcca_all8",
        "ecca_all8",
    }
    assert len(_external_gate_grid("smoke8")) == 8
    assert len(_external_decision_param_grid("smoke8")) == 144
    assert len(loso) == 4
    assert loso[0].holdout_session_index == 0
    assert len(chrono) == 1
    assert chrono[0].holdout_session_index == 3


def test_external_rank_key_excludes_switch_axis() -> None:
    key = _external_rank_metrics_key(
        {
            "idle_fp_per_min": 0.1,
            "release_latency_s": 2.0,
            "control_recall_at_3s": 0.8,
            "control_recall": 0.9,
            "inference_ms": 11.0,
            "switch_latency_s": 999.0,
        }
    )
    assert len(key) == 5
    assert key[0] == 0.1
    assert key[1] == 2.0


def test_per_frequency_reference_overrides_adapt_and_fallback() -> None:
    rows = []
    for trial_id in range(1, 9):
        for _ in range(8):
            rows.append(
                {
                    "trial_id": trial_id,
                    "trial_role": "control",
                    "expected_freq": 13.0,
                    "pred_freq": 13.0,
                    "p_correct": 0.58,
                    "correctness_logit": 0.32,
                    "tune_origin": "train_oof",
                }
            )
        for _ in range(8):
            rows.append(
                {
                    "trial_id": 100 + trial_id,
                    "trial_role": "clean_idle",
                    "expected_freq": None,
                    "pred_freq": 13.0,
                    "p_correct": 0.18,
                    "correctness_logit": -1.52,
                    "tune_origin": "train_oof",
                }
            )
    rows.extend(
        [
            {
                "trial_id": 999,
                "trial_role": "control",
                "expected_freq": 17.0,
                "pred_freq": 17.0,
                "p_correct": 0.75,
                "correctness_logit": 1.1,
                "tune_origin": "train_oof",
            },
            {
                "trial_id": 1999,
                "trial_role": "clean_idle",
                "expected_freq": None,
                "pred_freq": 17.0,
                "p_correct": 0.35,
                "correctness_logit": -0.6,
                "tune_origin": "train_oof",
            },
        ]
    )

    payloads, board = _compute_per_frequency_reference_overrides(
        scored_rows=rows,
        freqs=(13.0, 17.0, 21.0),
        global_enter_p_th=0.65,
        global_exit_p_th=0.30,
        base_payloads=None,
    )

    assert payloads["13"]["per_frequency_reference_valid"] is True
    assert payloads["13"]["enter_p_th"] != 0.65
    assert payloads["13"]["enter_p_th"] <= 0.58 + 1e-9
    assert payloads["13"]["enter_p_th"] >= 0.55 - 1e-9
    assert payloads["13"]["enter_p_th"] <= float(payloads["13"]["positive_trial_max_p75"]) + 1e-9
    assert payloads["17"]["per_frequency_reference_valid"] is False
    assert payloads["17"]["enter_p_th"] == 0.65
    assert payloads["21"]["exit_p_th"] == 0.30
    board_13 = next(row for row in board if row["freq"] == 13.0)
    board_17 = next(row for row in board if row["freq"] == 17.0)
    assert board_13["adaptation_mode"] == "shrunk_freq_specific_bounded"
    assert board_13["reference_bound_applied"] is True
    assert board_17["adaptation_mode"] == "global_fallback"


def test_tune_rows_reject_train_full_origin() -> None:
    rows = [
        {
            "trial_id": 1,
            "trial_role": "control",
            "expected_freq": 13.0,
            "pred_freq": 13.0,
            "p_correct": 0.7,
            "correctness_logit": 0.9,
            "tune_origin": "train_full",
        }
    ]

    with pytest.raises(ValueError, match="OOF-only"):
        _build_tune_frequency_breakdown(rows, freqs=(13.0, 17.0, 21.0))

    with pytest.raises(ValueError, match="OOF-only"):
        _compute_per_frequency_reference_overrides(
            scored_rows=rows,
            freqs=(13.0, 17.0, 21.0),
            global_enter_p_th=0.65,
            global_exit_p_th=0.30,
        )


def test_selection_validity_summary_blocks_frequency_imbalance_and_confidence_dominance() -> None:
    invalid = _selection_validity_summary(
        replay_frequency_breakdown=[
            {"freq": 13.0, "raw_correct_rate": 1.0, "gate_pass_rate": 0.0},
            {"freq": 17.0, "raw_correct_rate": 0.9, "gate_pass_rate": 0.5},
            {"freq": 21.0, "raw_correct_rate": 0.9, "gate_pass_rate": 0.6},
        ],
        decision_bottleneck_summary={
            "failure_breakdown": {
                "decoder_miss": 2,
                "confidence_reject_miss": 10,
                "decision_miss": 1,
            }
        },
    )
    valid = _selection_validity_summary(
        replay_frequency_breakdown=[
            {"freq": 13.0, "raw_correct_rate": 0.9, "gate_pass_rate": 0.3},
            {"freq": 17.0, "raw_correct_rate": 0.9, "gate_pass_rate": 0.4},
            {"freq": 21.0, "raw_correct_rate": 0.95, "gate_pass_rate": 0.5},
        ],
        decision_bottleneck_summary={
            "failure_breakdown": {
                "decoder_miss": 2,
                "confidence_reject_miss": 3,
                "decision_miss": 3,
            }
        },
    )

    assert invalid["frequency_balance_valid"] is False
    assert invalid["confidence_dominance_valid"] is False
    assert invalid["selection_eligible"] is False
    assert "frequency_balance_invalid" in invalid["invalid_reasons"]
    assert "confidence_dominance_invalid" in invalid["invalid_reasons"]
    assert valid["frequency_balance_valid"] is True
    assert valid["confidence_dominance_valid"] is True
    assert valid["selection_eligible"] is True


def test_diagnostic_selection_keeps_mainline_selection_and_prefers_higher_min_gate_pass_rate() -> None:
    rows = [
        {
            "candidate_key": "diag-itcca",
            "fbcca_variant": "itcca_all8",
            "diagnostic_only": _is_diagnostic_only_variant("itcca_all8"),
            "selection_eligible": True,
            "gate_calibration_valid": True,
            "selection_invalid_reasons": [],
            "min_gate_pass_rate_by_freq": 0.95,
            "metrics_median": {
                "idle_fp_per_min": 0.4,
                "release_latency_s": 2.0,
                "control_recall_at_3s": 0.5,
                "control_recall": 0.6,
                "inference_ms": 20.0,
            },
            "confidence_variant": GLOBAL_CORRECTNESS_LOGISTIC,
        },
        {
            "candidate_key": "mainline-strict",
            "fbcca_variant": "fbcca_fixed_all8",
            "diagnostic_only": _is_diagnostic_only_variant("fbcca_fixed_all8"),
            "selection_eligible": True,
            "gate_calibration_valid": True,
            "selection_invalid_reasons": [],
            "min_gate_pass_rate_by_freq": 0.35,
            "metrics_median": {
                "idle_fp_per_min": 0.8,
                "release_latency_s": 2.8,
                "control_recall_at_3s": 0.4,
                "control_recall": 0.5,
                "inference_ms": 30.0,
            },
            "confidence_variant": GLOBAL_CORRECTNESS_LOGISTIC,
        },
        {
            "candidate_key": "near-miss-a",
            "fbcca_variant": "fbcca_fixed_all8",
            "diagnostic_only": False,
            "selection_eligible": False,
            "gate_calibration_valid": True,
            "selection_invalid_reasons": ["confidence_dominance_invalid"],
            "min_gate_pass_rate_by_freq": 0.35,
            "metrics_median": {
                "idle_fp_per_min": 1.0,
                "release_latency_s": 3.2,
                "control_recall_at_3s": 0.3,
                "control_recall": 0.45,
                "inference_ms": 25.0,
            },
            "confidence_variant": GLOBAL_CORRECTNESS_LOGISTIC,
        },
        {
            "candidate_key": "near-miss-b",
            "fbcca_variant": "fbcca_sw_all8",
            "diagnostic_only": False,
            "selection_eligible": False,
            "gate_calibration_valid": True,
            "selection_invalid_reasons": ["confidence_dominance_invalid"],
            "min_gate_pass_rate_by_freq": 0.10,
            "metrics_median": {
                "idle_fp_per_min": 1.0,
                "release_latency_s": 3.2,
                "control_recall_at_3s": 0.3,
                "control_recall": 0.45,
                "inference_ms": 25.0,
            },
            "confidence_variant": GLOBAL_CORRECTNESS_LOGISTIC,
        },
    ]

    strict_rows = _strict_selection_rows(rows)
    diagnostic_row = _diagnostic_best_row(rows[2:])

    assert [row["candidate_key"] for row in strict_rows] == ["mainline-strict"]
    assert diagnostic_row is not None
    assert diagnostic_row["candidate_key"] == "near-miss-a"


def test_ui_can_read_external_replay_config(tmp_path: Path) -> None:
    _ = _get_qapp()
    window = TrainingEvaluationWindow()
    try:
        window._tdca_search_preset = "smoke8"
        window.configure_fbcca_external_replay_mode(auto_start=False)
        window.external_dataset_root_edit.setText(str(tmp_path))
        window.external_subject_edit.setText("Subject2")
        window.models_edit.setText("fbcca")
        window.channel_modes_edit.setText("all8")

        cfg = window._read_config()

        assert cfg.task == "fbcca-external-replay-opt"
        assert cfg.external_subject == "Subject2"
        assert cfg.external_dataset_root == tmp_path.resolve()
        assert cfg.dynamic_stop_enabled is False
        assert Path(window.output_profile_edit.text()).name == "profile.json"
        assert window._tdca_search_preset == "smoke8"
    finally:
        window.close()


def test_ui_preserves_external_replay_overrides_when_switching_mode(tmp_path: Path) -> None:
    _ = _get_qapp()
    window = TrainingEvaluationWindow()
    try:
        custom_dataset_root = tmp_path / "external_dataset"
        custom_output_profile = tmp_path / "custom_profile.json"
        window._tdca_search_preset = "smoke8"
        window.external_dataset_root_edit.setText(str(custom_dataset_root))
        window.external_outer_eval_combo.setCurrentText("chronological-last")
        window.external_replay_speed_combo.setCurrentText("max")
        window.output_profile_edit.setText(str(custom_output_profile))

        window.configure_fbcca_external_replay_mode(auto_start=False)

        assert window.external_dataset_root_edit.text() == str(custom_dataset_root)
        assert window.external_outer_eval_combo.currentText() == "chronological-last"
        assert window.external_replay_speed_combo.currentText() == "max"
        assert window.output_profile_edit.text() == str(custom_output_profile)
    finally:
        window.close()


def test_external_replay_diagnostics_and_frequency_breakdown_use_release_events() -> None:
    candidate_row = {
        "candidate_key": "variant=fbcca_fixed_all8|win=2.5|confidence=global_correctness_logistic",
        "fbcca_variant": "fbcca_fixed_all8",
        "confidence_variant": GLOBAL_CORRECTNESS_LOGISTIC,
    }
    replay_trial_events = [
        {
            "label": "13Hz",
            "trial_id": 1,
            "expected_freq": 13.0,
            "raw_correct_seen": True,
            "gate_pass_correct_seen": False,
            "commit_seen": False,
            "max_p_correct": 0.42,
            "max_decision_evidence": -0.35,
            "release_trial": True,
            "first_release_latency_s": None,
            "switch_trial": False,
        },
        {
            "label": "17Hz",
            "trial_id": 2,
            "expected_freq": 17.0,
            "raw_correct_seen": True,
            "gate_pass_correct_seen": True,
            "commit_seen": True,
            "first_gate_pass_latency_s": 1.0,
            "first_correct_latency_s": 1.25,
            "max_p_correct": 0.91,
            "max_decision_evidence": 1.2,
            "release_trial": True,
            "first_release_latency_s": 0.5,
            "switch_trial": False,
        },
        {
            "label": "21Hz",
            "trial_id": 3,
            "expected_freq": 21.0,
            "raw_correct_seen": False,
            "gate_pass_correct_seen": False,
            "commit_seen": False,
            "max_p_correct": 0.12,
            "max_decision_evidence": -0.8,
            "release_trial": True,
            "first_release_latency_s": None,
            "switch_trial": False,
        },
    ]
    holdout_bundles = [{"replay_trial_events": replay_trial_events}]

    summary = _build_external_decision_bottleneck_summary(
        candidate_row=candidate_row,
        holdout_bundles=holdout_bundles,
    )
    board = _build_external_error_attribution_board(
        candidate_row=candidate_row,
        holdout_bundles=holdout_bundles,
    )
    frequency_breakdown = _build_replay_frequency_breakdown(replay_trial_events)

    assert summary["control_trials"] == 3
    assert summary["release_trials"] == 3
    assert summary["release_seen_count"] == 1
    assert summary["failure_breakdown"] == {
        "decoder_miss": 1,
        "confidence_reject_miss": 1,
        "decision_miss": 0,
    }

    release_row = next(row for row in board if row["event_type"] == "release")
    control_row = next(row for row in board if row["event_type"] == "control")
    assert release_row["total"] == 3
    assert release_row["success"] == 1
    assert release_row["decision_miss"] == 2
    assert control_row["total"] == 3
    assert control_row["success"] == 1
    assert control_row["decoder_miss"] == 1
    assert control_row["confidence_reject_miss"] == 1

    assert [row["freq"] for row in frequency_breakdown] == [13.0, 17.0, 21.0]
    assert [row["freq_label"] for row in frequency_breakdown] == ["13", "17", "21"]
    breakdown_13 = next(row for row in frequency_breakdown if row["freq"] == 13.0)
    breakdown_17 = next(row for row in frequency_breakdown if row["freq"] == 17.0)
    assert breakdown_13["gate_pass_rate"] == 0.0
    assert breakdown_13["commit_rate"] == 0.0
    assert breakdown_17["commit_rate"] == 1.0
    assert breakdown_17["release_seen_rate"] == 1.0

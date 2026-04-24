from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.training_evaluation_ui import TrainingEvaluationWindow
from ssvep_core.async_fbcca_idle_standalone import ThresholdProfile, TrialSpec, build_feature_rows_with_decoder, create_decoder
from ssvep_core.dataset import LoadedDataset
from ssvep_core.gating import CorrectnessCalibrator, CorrectnessCalibratorConfig
from ssvep_core.gating.per_freq_logreg_gate import LogRegFitConfig
from ssvep_core.tdca_local_opt import (
    TDCALocalOptConfig,
    backfill_manifest_trial_roles,
    build_repeated_group_splits,
    preflight_tdca_local_env,
    run_tdca_local_opt,
)
from ssvep_core.trial_roles import resolve_trial_role
from tools.training_evaluation_cli import _parse_task, build_parser


def _get_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_trial_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    next_id = 0
    for freq in (8.0, 10.0, 12.0, 15.0):
        for _ in range(3):
            rows.append(
                {
                    "label": f"{freq:g}Hz",
                    "expected_freq": freq,
                    "trial_id": next_id,
                    "block_index": next_id,
                }
            )
            next_id += 1
    for _ in range(3):
        rows.append(
            {
                "label": "idle",
                "expected_freq": None,
                "trial_id": next_id,
                "block_index": next_id,
            }
        )
        next_id += 1
    for freq in (8.0, 10.0, 12.0):
        rows.append(
            {
                "label": f"switch_to_{freq:g}Hz",
                "expected_freq": None,
                "trial_id": next_id,
                "block_index": next_id,
            }
        )
        next_id += 1
    return rows


def _make_profile() -> ThresholdProfile:
    return ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=2.0,
        step_sec=0.25,
        enter_score_th=0.5,
        enter_ratio_th=1.4,
        enter_margin_th=0.2,
        exit_score_th=0.4,
        exit_ratio_th=1.1,
        min_enter_windows=1,
        min_exit_windows=1,
        model_name="tdca",
        model_params={"Nh": 3, "delay_steps": 2, "n_components": 2, "decoder_variant": "tdca_like_legacy"},
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
                "enter_score_th": 0.5,
                "enter_ratio_th": 1.4,
                "enter_margin_th": 0.2,
                "exit_score_th": 0.4,
                "exit_ratio_th": 1.1,
                "enter_log_lr_th": 0.5,
                "exit_log_lr_th": 0.0,
                "enter_p_th": 0.65,
                "exit_p_th": 0.30,
                "min_enter_windows": 1,
                "min_exit_windows": 1,
                "min_switch_windows": 1,
                "switch_enter_score_th": 0.55,
                "switch_enter_ratio_th": 1.4,
                "switch_enter_margin_th": 0.2,
            },
            "10": {
                "enter_score_th": 0.5,
                "enter_ratio_th": 1.4,
                "enter_margin_th": 0.2,
                "exit_score_th": 0.4,
                "exit_ratio_th": 1.1,
                "enter_log_lr_th": 0.5,
                "exit_log_lr_th": 0.0,
                "enter_p_th": 0.65,
                "exit_p_th": 0.30,
                "min_enter_windows": 1,
                "min_exit_windows": 1,
                "min_switch_windows": 1,
                "switch_enter_score_th": 0.55,
                "switch_enter_ratio_th": 1.4,
                "switch_enter_margin_th": 0.2,
            },
        },
    )


def _make_feature_row(
    *,
    label: str,
    trial_role: str,
    trial_id: int,
    window_index: int,
    pred_freq: Optional[float],
    expected_freq: Optional[float],
    top1_score: float,
    ratio: float,
    margin: float,
    gate_score: float,
) -> dict[str, object]:
    top2 = float(max(top1_score - margin, 0.01))
    top3 = float(max(top2 - 0.05, 0.01))
    top4 = float(max(top3 - 0.05, 0.01))
    p_control = float(0.8 if gate_score > 0.0 else 0.2)
    p_correct = p_control if expected_freq is not None and pred_freq is not None else float(0.1 if gate_score <= 0.0 else 0.7)
    return {
        "label": label,
        "trial_role": trial_role,
        "trial_id": int(trial_id),
        "window_index": int(window_index),
        "pred_freq": pred_freq,
        "expected_freq": expected_freq,
        "top1_score": float(top1_score),
        "top2_score": top2,
        "ratio": float(ratio),
        "margin": float(margin),
        "gap_12": float(top1_score - top2),
        "gap_13": float(top1_score - top3),
        "gap_14": float(top1_score - top4),
        "gate_score": float(gate_score),
        "control_log_lr": float(gate_score),
        "p_control": p_control,
        "p_correct": p_correct,
        "correctness_logit": float(np.log(max(p_correct, 1e-6) / max(1.0 - p_correct, 1e-6))),
        "normalized_top1": float(min(top1_score, 1.0)),
        "score_entropy": float(0.2 if gate_score > 0.0 else 0.8),
    }


def _relax_correctness_minima(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module,
        "DEFAULT_CORRECTNESS_CALIBRATOR_CONFIG",
        CorrectnessCalibratorConfig(learning_rate=0.08, epochs=64, l2=1e-3, min_positive_windows=1, min_negative_windows=1),
    )
    monkeypatch.setattr(module, "DEFAULT_TUNE_MIN_CONTROL_TRIALS_PER_FREQ", 1)
    monkeypatch.setattr(module, "DEFAULT_TUNE_MIN_IDLE_TRIALS", 1)


def _make_release_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window_index in range(3):
        rows.append(
            _make_feature_row(
                label="8Hz",
                trial_role="control",
                trial_id=0,
                window_index=window_index,
                pred_freq=8.0,
                expected_freq=8.0,
                top1_score=0.92,
                ratio=2.0,
                margin=0.7,
                gate_score=0.9,
            )
        )
    for window_index in range(4):
        rows.append(
            _make_feature_row(
                label="idle",
                trial_role="clean_idle",
                trial_id=1,
                window_index=window_index,
                pred_freq=8.0,
                expected_freq=None,
                top1_score=0.10,
                ratio=1.0,
                margin=0.02,
                gate_score=-0.6,
            )
        )
    return rows


def _make_switch_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window_index in range(3):
        rows.append(
            _make_feature_row(
                label="8Hz",
                trial_role="control",
                trial_id=0,
                window_index=window_index,
                pred_freq=8.0,
                expected_freq=8.0,
                top1_score=0.92,
                ratio=2.0,
                margin=0.7,
                gate_score=0.9,
            )
        )
    for window_index in range(5):
        rows.append(
            _make_feature_row(
                label="10Hz",
                trial_role="control",
                trial_id=1,
                window_index=window_index,
                pred_freq=10.0,
                expected_freq=10.0,
                top1_score=0.92,
                ratio=2.0,
                margin=0.7,
                gate_score=0.9,
            )
        )
    return rows


def _make_idle_fp_rows() -> list[dict[str, object]]:
    return [
        _make_feature_row(
            label="idle",
            trial_role="clean_idle",
            trial_id=0,
            window_index=window_index,
            pred_freq=8.0,
            expected_freq=None,
            top1_score=0.92,
            ratio=2.0,
            margin=0.7,
            gate_score=0.9,
        )
        for window_index in range(4)
    ]


def _make_tdca_training_segments(
    *,
    sampling_rate: int = 250,
    win_sec: float = 1.0,
    repeats: int = 2,
) -> list[tuple[TrialSpec, np.ndarray]]:
    win_samples = int(round(float(sampling_rate) * float(win_sec)))
    t = np.arange(win_samples, dtype=np.float64) / float(sampling_rate)
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    channel_phase = np.linspace(0.0, np.pi / 6.0, 8, dtype=np.float64)
    for freq in (8.0, 10.0, 12.0, 15.0):
        for repeat in range(int(repeats)):
            channels = []
            for phase in channel_phase:
                signal = np.sin(2.0 * np.pi * float(freq) * t + phase + 0.1 * repeat)
                signal += 0.15 * np.cos(2.0 * np.pi * float(freq) * t + 0.5 * phase)
                channels.append(signal)
            matrix = np.stack(channels, axis=1).astype(np.float64)
            segments.append(
                (
                    TrialSpec(
                        label=f"{freq:g}Hz",
                        expected_freq=float(freq),
                        trial_id=trial_id,
                        block_index=trial_id,
                    ),
                    matrix,
                )
            )
            trial_id += 1
    return segments


def test_resolve_trial_role_matches_expected_categories() -> None:
    assert resolve_trial_role({"label": "8Hz", "expected_freq": 8.0}) == "control"
    assert resolve_trial_role({"label": "idle", "expected_freq": None}) == "clean_idle"
    assert resolve_trial_role({"label": "switch_to_10Hz", "expected_freq": None}) == "hard_idle"
    assert resolve_trial_role({"label": "8Hz", "expected_freq": 8.0, "trial_role": "control"}) == "control"


def test_build_feature_rows_with_decoder_backfills_trial_metadata() -> None:
    class _Decoder:
        def iter_window_features(self, _segment, *, expected_freq, label, trial_id, block_index):
            assert expected_freq == 8.0
            assert label == "8Hz"
            assert trial_id == 3
            assert block_index == 9
            yield {
                "window_index": 0,
                "pred_freq": 8.0,
                "top1_score": 0.9,
                "top2_score": 0.2,
                "ratio": 1.8,
                "margin": 0.7,
            }

    rows = build_feature_rows_with_decoder(
        _Decoder(),
        [
            (
                TrialSpec(label="8Hz", expected_freq=8.0, trial_id=3, block_index=9),
                np.zeros((32, 8), dtype=np.float64),
            )
        ],
    )

    assert len(rows) == 1
    assert rows[0]["trial_role"] == "control"
    assert rows[0]["expected_freq"] == 8.0
    assert rows[0]["label"] == "8Hz"
    assert rows[0]["trial_id"] == 3
    assert rows[0]["block_index"] == 9


def test_correctness_calibrator_fit_predict_and_roundtrip() -> None:
    rows = [
        _make_feature_row(
            label="8Hz",
            trial_role="control",
            trial_id=index,
            window_index=0,
            pred_freq=8.0,
            expected_freq=8.0,
            top1_score=0.9,
            ratio=2.0,
            margin=0.7,
            gate_score=0.9,
        )
        for index in range(24)
    ]
    rows.extend(
        _make_feature_row(
            label="idle",
            trial_role="clean_idle",
            trial_id=100 + index,
            window_index=0,
            pred_freq=8.0,
            expected_freq=None,
            top1_score=0.15,
            ratio=1.05,
            margin=0.02,
            gate_score=-0.6,
        )
        for index in range(24)
    )
    calibrator = CorrectnessCalibrator()
    summary = calibrator.fit(
        rows=rows,
        freqs=(8.0, 10.0, 12.0, 15.0),
        config=CorrectnessCalibratorConfig(min_positive_windows=4, min_negative_windows=4, epochs=64),
    )
    prediction = calibrator.predict(rows[0])
    restored = CorrectnessCalibrator.from_payload(calibrator.to_payload())
    restored_prediction = restored.predict(rows[-1])

    assert summary["valid"] is True
    assert summary["sample_weight_mode"] == "per_trial_equal"
    assert summary["positive_trials"] == 24
    assert summary["negative_trials"] == 24
    assert {"gap_12", "gap_13", "gap_14"}.issubset(set(summary["feature_names"]))
    assert np.isfinite(float(prediction["p_correct"]))
    assert np.isfinite(float(prediction["correctness_logit"]))
    assert np.isfinite(float(restored_prediction["p_correct"]))
    assert np.isfinite(float(restored_prediction["correctness_logit"]))


def test_decision_evidence_centering_tracks_enter_threshold() -> None:
    import ssvep_core.tdca_local_opt as module

    profile = _make_profile()
    row = _make_feature_row(
        label="8Hz",
        trial_role="control",
        trial_id=0,
        window_index=0,
        pred_freq=8.0,
        expected_freq=8.0,
        top1_score=0.9,
        ratio=2.0,
        margin=0.7,
        gate_score=0.9,
    )
    enter_p = float(profile.enter_p_th)
    row_at_enter = dict(row)
    row_at_enter["p_correct"] = enter_p
    row_at_enter["correctness_logit"] = float(np.log(enter_p / (1.0 - enter_p)))
    centered = module._decision_evidence_row(row=row_at_enter, profile=profile)

    row_high = dict(row_at_enter)
    row_high["p_correct"] = 0.8
    row_high["correctness_logit"] = float(np.log(0.8 / 0.2))
    high = module._decision_evidence_row(row=row_high, profile=profile)

    row_low = dict(row_at_enter)
    row_low["p_correct"] = 0.5
    row_low["correctness_logit"] = 0.0
    low = module._decision_evidence_row(row=row_low, profile=profile)

    assert centered["decision_evidence_variant"] == "centered_logit_over_enter_threshold"
    assert abs(float(centered["decision_evidence_centered"])) < 1e-9
    assert float(high["decision_evidence_centered"]) > 0.0
    assert float(low["decision_evidence_centered"]) < 0.0


def test_resolve_oof_group_key_prefers_block_index_when_available() -> None:
    import ssvep_core.tdca_local_opt as module

    rows = [
        {"trial_id": 0, "block_index": 0},
        {"trial_id": 1, "block_index": 1},
        {"trial_id": 2, "block_index": 2},
        {"trial_id": 3, "block_index": 0},
    ]

    assert module._resolve_oof_group_key(rows) == "block_index"


def test_resolve_oof_group_key_falls_back_to_trial_id_when_blocks_are_few() -> None:
    import ssvep_core.tdca_local_opt as module

    rows = [
        {"trial_id": 0, "block_index": 0},
        {"trial_id": 1, "block_index": 1},
    ]

    assert module._resolve_oof_group_key(rows) == "trial_id"


def test_build_oof_train_scored_rows_keeps_held_out_block_out_of_fold(monkeypatch) -> None:
    import ssvep_core.tdca_local_opt as module

    fit_block_sets: list[set[int]] = []

    class _FakeGate:
        def fit(self, *, rows, freqs, fit_config):
            _ = freqs
            _ = fit_config
            self.fit_blocks = {
                int(row.get("block_index", -1))
                for row in rows
            }
            fit_block_sets.append(set(self.fit_blocks))
            return {"ok": True}

        def predict(self, feature_row, pred_freq):
            assert int(feature_row.get("block_index", -1)) not in self.fit_blocks
            return type(
                "GateOut",
                (),
                {
                    "p_control": 0.8 if pred_freq is not None else 0.0,
                    "gate_score": 1.0 if pred_freq is not None else 0.0,
                },
            )()

    monkeypatch.setattr(module, "PerFrequencyLogRegGate", _FakeGate)
    rows = []
    for block_index in range(3):
        for trial_id in range(2):
            rows.append(
                {
                    "trial_id": int(block_index * 10 + trial_id),
                    "block_index": int(block_index),
                    "pred_freq": 8.0,
                    "expected_freq": 8.0,
                    "trial_role": "control",
                    "top1_score": 0.9,
                    "top2_score": 0.2,
                    "margin": 0.7,
                    "ratio": 2.0,
                    "normalized_top1": 0.8,
                    "score_entropy": 0.2,
                    "consistency": 1.0,
                }
            )

    scored_rows, summary = module._build_oof_train_scored_rows(
        train_rows=rows,
        freqs=(8.0, 10.0, 12.0, 15.0),
        fit_config=LogRegFitConfig(min_samples=1, epochs=2),
    )

    assert len(scored_rows) == len(rows)
    assert summary["confidence_training_scheme"] == "oof_gate_logreg_on_train_split"
    assert summary["oof_group_key"] == "block_index"
    assert summary["oof_group_count"] == 3
    assert fit_block_sets


def test_extract_training_window_bank_uses_latency_trimmed_uniform_coverage() -> None:
    import ssvep_core.async_fbcca_idle_standalone as module

    segment = np.arange(0, 100, dtype=np.float64).reshape(100, 1)
    windows = module._extract_training_window_bank(
        segment,
        win_samples=20,
        step_samples=10,
        latency_samples=14,
        max_windows=4,
    )
    starts = [int(window[0, 0]) for window in windows]

    assert len(windows) == 4
    assert starts[0] >= 14
    assert starts != [14, 24, 34, 44]
    assert starts[-1] > starts[1]


def test_gate_score_partitions_falls_back_to_resolved_trial_role() -> None:
    import ssvep_core.tdca_local_opt as module

    control_scores, idle_scores = module._gate_score_partitions(
        [
            {
                "label": "8Hz",
                "expected_freq": 8.0,
                "pred_freq": 8.0,
                "gate_score": 0.9,
            },
            {
                "label": "idle",
                "expected_freq": None,
                "pred_freq": 8.0,
                "gate_score": -0.4,
            },
        ],
        freq=8.0,
    )

    assert control_scores.tolist() == [0.9]
    assert idle_scores.tolist() == [-0.4]


def test_select_enter_exit_logit_avoids_extreme_fallbacks() -> None:
    import ssvep_core.tdca_local_opt as module

    control_scores = np.asarray([-1.2, -0.3, 0.2, 0.8], dtype=float)
    idle_scores = np.asarray([-1.5, -0.9, -0.4, 0.6], dtype=float)

    enter_th, exit_th, detail = module._select_enter_exit_logit(
        control_scores=control_scores,
        idle_scores=idle_scores,
        enter_fallback=20.0,
        exit_fallback=-8.0,
    )

    assert float(enter_th) < 20.0
    assert float(exit_th) > -8.0
    assert float(np.mean(control_scores >= float(enter_th))) > 0.0
    assert float(np.mean(idle_scores < float(exit_th))) > 0.0
    assert detail["enter_selection_mode"] == "positive_control_recall"
    assert detail["exit_selection_mode"] == "positive_idle_clear"


def test_render_markdown_marks_empty_decision_board_as_ineffective() -> None:
    import ssvep_core.tdca_local_opt as module

    markdown = module._render_markdown(
        {
            "generated_at": "2026-04-15T00:00:00",
            "task": "tdca-local-opt",
            "chosen_model": "tdca",
            "decoder_variant": "tdca_like_legacy",
            "profile_saved": False,
            "chosen_profile_path": "",
            "status": "invalid",
            "status_reasons": ["gate_calibration_invalid_all_candidates"],
            "chosen_model_rationale": "invalid_run_not_comparable",
            "gate_calibration_valid": False,
            "decision_search_board": [],
            "chosen_async_metrics": {},
            "chosen_metrics_4class": {},
        }
    )

    assert "- Decision search: `ineffective`" in markdown


def test_preflight_explicit_cuda_fails_fast_without_runtime(monkeypatch) -> None:
    import ssvep_core.tdca_local_opt as module

    def _fake_resolve(requested: str, *, gpu_device: int, precision: str):
        if requested == "cuda":
            raise RuntimeError("cuda runtime missing")

        class _Backend:
            backend_name = "cpu"

            def microbenchmark_transfer(self, **_kwargs):
                return {"backend_name": "cpu"}

            def describe(self):
                return {"backend_name": "cpu", "uses_cuda": False}

        return _Backend()

    monkeypatch.setattr(module, "resolve_compute_backend", _fake_resolve)
    try:
        preflight_tdca_local_env(compute_backend="cuda", gpu_device=0, gpu_precision="float32")
        assert False, "expected explicit cuda preflight to fail"
    except RuntimeError as exc:
        assert "preflight" in str(exc)


def test_build_repeated_group_splits_produces_multiple_fingerprints() -> None:
    segments = [
        (
            TrialSpec(
                label=str(row["label"]),
                expected_freq=row["expected_freq"],
                trial_id=int(row["trial_id"]),
                block_index=int(row["block_index"]),
            ),
            np.zeros((12, 8), dtype=np.float64),
        )
        for row in _make_trial_rows()
    ]
    splits = build_repeated_group_splits(segments, repeats=5, seed=20260410)
    assert len(splits) == 5
    assert len({split.fingerprint for split in splits}) >= 2
    assert all(len(split.train_indices) > 0 for split in splits)
    assert all(len(split.holdout_indices) > 0 for split in splits)


def test_backfill_manifest_trial_roles_updates_trials_and_summary(tmp_path: Path) -> None:
    manifest_path = tmp_path / "session_manifest.json"
    manifest = {
        "trials": [
            {"label": "8Hz", "expected_freq": 8.0, "trial_id": 0, "block_index": 0},
            {"label": "switch_to_10Hz", "expected_freq": None, "trial_id": 1, "block_index": 1},
            {"label": "idle", "expected_freq": None, "trial_id": 2, "block_index": 2},
        ],
        "quality_summary": {},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = backfill_manifest_trial_roles(manifest_path)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["changed"] is True
    assert saved["trials"][0]["trial_role"] == "control"
    assert saved["trials"][1]["trial_role"] == "hard_idle"
    assert saved["quality_summary"]["trial_role_counts"]["control"] == 1
    assert saved["quality_summary"]["trial_role_counts"]["clean_idle"] == 1
    assert saved["quality_summary"]["trial_role_counts"]["hard_idle"] == 1


def test_cli_supports_tdca_local_opt_task() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", "tdca-local-opt", "--search-preset", "smoke4"])
    assert args.task == "tdca-local-opt"
    assert args.search_preset == "smoke4"
    assert str(args.tdca_delay_steps) == "2,3,4,5"
    assert str(args.tdca_n_components) == "2,3,4"
    assert _parse_task("tdca_local_opt") == "tdca-local-opt"


def test_run_tdca_local_opt_smoke(monkeypatch, tmp_path: Path) -> None:
    import ssvep_core.tdca_local_opt as module

    trial_rows = _make_trial_rows()
    manifest_path = tmp_path / "session_manifest.json"
    manifest_path.write_text(json.dumps({"trials": trial_rows, "quality_summary": {}}), encoding="utf-8")

    segments: list[tuple[TrialSpec, np.ndarray]] = []
    for row in trial_rows:
        segments.append(
            (
                TrialSpec(
                    label=str(row["label"]),
                    expected_freq=row["expected_freq"],
                    trial_id=int(row["trial_id"]),
                    block_index=int(row["block_index"]),
                ),
                np.zeros((12, 8), dtype=np.float64),
            )
        )
    dataset = LoadedDataset(
        manifest_path=manifest_path,
        npz_path=tmp_path / "raw_trials.npz",
        session_id="session_smoke",
        subject_id="subject_smoke",
        sampling_rate=4,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        protocol_config={"step_sec": 0.25},
        trial_segments=segments,
        manifest={"trials": trial_rows, "quality_summary": {}},
    )

    class _FakeDecoder:
        requires_fit = True

        def __init__(
            self,
            *,
            sampling_rate: int,
            freqs: tuple[float, ...],
            win_sec: float,
            step_sec: float,
            model_params: dict[str, object],
            **_kwargs,
        ):
            self.fs = int(sampling_rate)
            self.freqs = tuple(float(item) for item in freqs)
            self.win_sec = float(win_sec)
            self.step_sec = float(step_sec)
            self.model_params = dict(model_params)
            self.win_samples = 4
            self.step_samples = 1

        def fit(self, trial_segments):
            self._fit_count = len(trial_segments)

        def iter_window_features(self, segment, *, expected_freq, label, trial_id, block_index):
            role = "control" if expected_freq is not None and not str(label).startswith("switch_to_") else (
                "hard_idle" if "switch" in str(label) else "clean_idle"
            )
            total_windows = 4 if expected_freq is None else 3
            for index in range(total_windows):
                if expected_freq is None:
                    pred_freq = (8.0, 10.0, 12.0, 15.0)[index % 4]
                    top1_score = 0.12
                    top2_score = 0.10
                    ratio = 1.05
                    margin = 0.02
                    normalized_top1 = 0.20
                    score_entropy = 0.85
                else:
                    pred_freq = float(expected_freq)
                    top1_score = 0.92 - 0.02 * index
                    top2_score = 0.10
                    ratio = 2.2
                    margin = 0.72
                    normalized_top1 = 0.82
                    score_entropy = 0.18
                yield {
                    "trial_role": role,
                    "label": str(label),
                    "expected_freq": expected_freq,
                    "trial_id": int(trial_id),
                    "block_index": int(block_index),
                    "window_index": int(index),
                    "pred_freq": pred_freq,
                    "top1_score": top1_score,
                    "top2_score": top2_score,
                    "ratio": ratio,
                    "margin": margin,
                    "normalized_top1": normalized_top1,
                    "score_entropy": score_entropy,
                }

        def analyze_window(self, _window):
            return {
                "pred_freq": 8.0,
                "top1_score": 0.5,
                "top2_score": 0.3,
                "ratio": 1.2,
                "margin": 0.2,
                "normalized_top1": 0.4,
                "score_entropy": 0.4,
            }

        def get_state(self):
            return {"fitted": True, **self.model_params}

    monkeypatch.setattr(module, "load_collection_dataset", lambda _path: dataset)
    monkeypatch.setattr(module, "create_decoder", lambda _model_name, **kwargs: _FakeDecoder(**kwargs))
    monkeypatch.setattr(module, "DEFAULT_LOGREG_FIT_CONFIG", LogRegFitConfig(epochs=4, min_samples=1))
    _relax_correctness_minima(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "_decision_param_grid",
        lambda: [
            {
                "candidate_min_windows": 1,
                "armed_min_windows": 2,
                "lambda_decay": 0.85,
                "upper_commit_th": 2.2,
                "lower_idle_th": 0.4,
                "refractory_sec": 0.25,
            },
            {
                "candidate_min_windows": 1,
                "armed_min_windows": 2,
                "lambda_decay": 0.85,
                "upper_commit_th": 2.2,
                "lower_idle_th": 0.4,
                "refractory_sec": 1.0,
            }
        ],
    )

    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile_tdca_local.json"
    config = TDCALocalOptConfig(
        dataset_manifest_session1=manifest_path,
        output_profile_path=profile_path,
        report_path=report_path,
        dataset_manifests=(manifest_path,),
        organize_report_dir=False,
        win_candidates=(2.0,),
        tdca_delay_steps=(2,),
        tdca_n_components=(2,),
        multi_seed_count=2,
        compute_backend="cpu",
    )
    payload = run_tdca_local_opt(config, log_fn=lambda _msg: None)

    assert payload["task"] == "tdca-local-opt"
    assert payload["chosen_model"] == "tdca"
    assert "env_preflight" in payload
    assert "split_fingerprints" in payload
    assert "tdca_search_board" in payload
    assert "gate_exit_search_board" in payload
    assert "decision_search_board" in payload
    assert "holdout_selection_board" in payload
    assert "variant_summary" in payload
    assert payload["decision_search_target"] == "tune_split"
    assert payload["final_selection_target"] == "holdout_split"
    assert payload["confidence_training_scheme"] == "oof_gate_logreg_on_train_split"
    assert payload["decision_evidence_variant"] == "centered_logit_over_enter_threshold"
    assert payload["decision_evidence_raw"] == "correctness_logit"
    assert payload["decision_evidence_reference"] == "logit(enter_p_th_for_pred_freq)"
    assert payload["sample_weight_mode"] == "per_trial_equal"
    assert payload["confidence_variant"] in {"global_correctness_logistic", "bayesian_gap_gmm"}
    assert payload["tune_summary"]
    assert payload["confidence_diagnostics_board"]
    assert payload["decision_bottleneck_summary"]
    assert "error_attribution_board" in payload
    assert "contrast_error_board" in payload
    assert all("gate_calibration_valid" in row for row in payload["tdca_search_board"])
    assert payload["chosen_model_rationale"] in {
        "tdca_superior_on_primary_ranking",
        "tdca_not_clearly_superior",
        "invalid_run_not_comparable",
    }
    assert payload["baseline_opening"]
    assert payload["baseline_seal"]
    assert {row["decoder_variant"] for row in payload["tdca_search_board"]} == {
        "tdca_like_legacy",
        "tdca_paper_aligned",
    }
    assert payload["decoder_variant"] in {"tdca_like_legacy", "tdca_paper_aligned"}
    assert payload["search_preset"] == "custom"
    assert payload["ranking_boards"]["end_to_end"]
    assert payload["holdout_selection_board"]
    assert all("confidence_diagnostics_board" in row for row in payload["holdout_selection_board"])
    assert all("decision_bottleneck_summary" in row for row in payload["holdout_selection_board"])
    assert payload["decoder_variant"] == payload["ranking_boards"]["end_to_end"][0]["decoder_variant"]
    assert report_path.exists()
    assert profile_path.exists()
    assert Path(payload["profile_v2_path"]).exists()


def test_gate_calibration_invalid_run_blocks_profile_and_marks_rationale(monkeypatch, tmp_path: Path) -> None:
    import ssvep_core.tdca_local_opt as module

    trial_rows = _make_trial_rows()
    manifest_path = tmp_path / "session_manifest_invalid.json"
    manifest_path.write_text(json.dumps({"trials": trial_rows, "quality_summary": {}}), encoding="utf-8")

    segments: list[tuple[TrialSpec, np.ndarray]] = []
    for row in trial_rows:
        segments.append(
            (
                TrialSpec(
                    label=str(row["label"]),
                    expected_freq=row["expected_freq"],
                    trial_id=int(row["trial_id"]),
                    block_index=int(row["block_index"]),
                ),
                np.zeros((12, 8), dtype=np.float64),
            )
        )
    dataset = LoadedDataset(
        manifest_path=manifest_path,
        npz_path=tmp_path / "raw_trials_invalid.npz",
        session_id="session_invalid",
        subject_id="subject_invalid",
        sampling_rate=4,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        protocol_config={"step_sec": 0.25},
        trial_segments=segments,
        manifest={"trials": trial_rows, "quality_summary": {}},
    )

    class _CollapsedDecoder:
        requires_fit = True

        def __init__(self, **kwargs):
            self.fs = int(kwargs["sampling_rate"])
            self.freqs = tuple(float(item) for item in kwargs["freqs"])
            self.win_sec = float(kwargs["win_sec"])
            self.step_sec = float(kwargs["step_sec"])
            self.model_params = dict(kwargs["model_params"])
            self.win_samples = 4
            self.step_samples = 1

        def fit(self, _trial_segments):
            return None

        def iter_window_features(self, _segment, *, expected_freq, label, trial_id, block_index):
            for index in range(2):
                if expected_freq is None:
                    yield {
                        "label": str(label),
                        "expected_freq": expected_freq,
                        "trial_id": int(trial_id),
                        "block_index": int(block_index),
                        "window_index": int(index),
                        "pred_freq": 8.0,
                        "top1_score": 0.1,
                        "top2_score": 0.09,
                        "ratio": 1.02,
                        "margin": 0.01,
                        "normalized_top1": 0.15,
                        "score_entropy": 0.9,
                    }
                else:
                    yield {
                        "label": str(label),
                        "expected_freq": expected_freq,
                        "trial_id": int(trial_id),
                        "block_index": int(block_index),
                        "window_index": int(index),
                        "pred_freq": 8.0,
                        "top1_score": 0.85,
                        "top2_score": 0.2,
                        "ratio": 1.9,
                        "margin": 0.65,
                        "normalized_top1": 0.8,
                        "score_entropy": 0.2,
                    }

        def analyze_window(self, _window):
            return {
                "pred_freq": 8.0,
                "top1_score": 0.5,
                "top2_score": 0.3,
                "ratio": 1.2,
                "margin": 0.2,
                "normalized_top1": 0.4,
                "score_entropy": 0.4,
            }

        def get_state(self):
            return {"decoder_variant": self.model_params.get("decoder_variant", "tdca_like_legacy")}

    monkeypatch.setattr(module, "load_collection_dataset", lambda _path: dataset)
    monkeypatch.setattr(module, "create_decoder", lambda _model_name, **kwargs: _CollapsedDecoder(**kwargs))
    monkeypatch.setattr(module, "DEFAULT_LOGREG_FIT_CONFIG", LogRegFitConfig(epochs=4, min_samples=1))
    monkeypatch.setattr(
        module,
        "_decision_param_grid",
        lambda: [
            {
                "candidate_min_windows": 1,
                "armed_min_windows": 2,
                "lambda_decay": 0.85,
                "upper_commit_th": 2.2,
                "lower_idle_th": 0.4,
                "refractory_sec": 0.25,
            }
        ],
    )

    payload = run_tdca_local_opt(
        TDCALocalOptConfig(
            dataset_manifest_session1=manifest_path,
            output_profile_path=tmp_path / "profile_invalid.json",
            report_path=tmp_path / "report_invalid.json",
            dataset_manifests=(manifest_path,),
            organize_report_dir=False,
            win_candidates=(2.0,),
            tdca_delay_steps=(2,),
            tdca_n_components=(2,),
            multi_seed_count=1,
            compute_backend="cpu",
        ),
        log_fn=lambda _msg: None,
    )

    assert payload["status"] == "invalid"
    assert "gate_calibration_invalid_all_candidates" in payload["status_reasons"]
    assert payload["chosen_model_rationale"] == "invalid_run_not_comparable"
    assert payload["profile_saved"] is False
    assert payload["gate_calibration_valid"] is False
    assert payload["run_valid_for_deployment"] is False
    assert payload["async_metrics"] == payload["chosen_async_metrics"]
    assert set(payload["gate_calibration_summary"]["invalid_reasons"]) & {
        "positive_windows_below_min",
        "tune_rows_insufficient",
    }


def test_decision_search_board_is_not_truncated(monkeypatch, tmp_path: Path) -> None:
    import ssvep_core.tdca_local_opt as module

    trial_rows = _make_trial_rows()
    manifest_path = tmp_path / "session_manifest_full_board.json"
    manifest_path.write_text(json.dumps({"trials": trial_rows, "quality_summary": {}}), encoding="utf-8")

    segments: list[tuple[TrialSpec, np.ndarray]] = []
    for row in trial_rows:
        segments.append(
            (
                TrialSpec(
                    label=str(row["label"]),
                    expected_freq=row["expected_freq"],
                    trial_id=int(row["trial_id"]),
                    block_index=int(row["block_index"]),
                ),
                np.zeros((12, 8), dtype=np.float64),
            )
        )
    dataset = LoadedDataset(
        manifest_path=manifest_path,
        npz_path=tmp_path / "raw_trials_full_board.npz",
        session_id="session_full_board",
        subject_id="subject_full_board",
        sampling_rate=4,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        protocol_config={"step_sec": 0.25},
        trial_segments=segments,
        manifest={"trials": trial_rows, "quality_summary": {}},
    )

    class _BoardDecoder:
        requires_fit = True

        def __init__(self, **kwargs):
            self.fs = int(kwargs["sampling_rate"])
            self.freqs = tuple(float(item) for item in kwargs["freqs"])
            self.win_sec = float(kwargs["win_sec"])
            self.step_sec = float(kwargs["step_sec"])
            self.model_params = dict(kwargs["model_params"])
            self.win_samples = 4
            self.step_samples = 1

        def fit(self, _trial_segments):
            return None

        def iter_window_features(self, _segment, *, expected_freq, label, trial_id, block_index):
            total_windows = 4 if expected_freq is None else 3
            for index in range(total_windows):
                if expected_freq is None:
                    pred_freq = (8.0, 10.0, 12.0, 15.0)[index % 4]
                    top1_score = 0.12
                    top2_score = 0.10
                    ratio = 1.05
                    margin = 0.02
                    normalized_top1 = 0.20
                    score_entropy = 0.85
                else:
                    pred_freq = float(expected_freq)
                    top1_score = 0.92 - 0.02 * index
                    top2_score = 0.10
                    ratio = 2.2
                    margin = 0.72
                    normalized_top1 = 0.82
                    score_entropy = 0.18
                yield {
                    "label": str(label),
                    "expected_freq": expected_freq,
                    "trial_id": int(trial_id),
                    "block_index": int(block_index),
                    "window_index": int(index),
                    "pred_freq": pred_freq,
                    "top1_score": top1_score,
                    "top2_score": top2_score,
                    "ratio": ratio,
                    "margin": margin,
                    "normalized_top1": normalized_top1,
                    "score_entropy": score_entropy,
                }

        def analyze_window(self, _window):
            return {
                "pred_freq": 8.0,
                "top1_score": 0.5,
                "top2_score": 0.3,
                "ratio": 1.2,
                "margin": 0.2,
                "normalized_top1": 0.4,
                "score_entropy": 0.4,
            }

        def get_state(self):
            return {"decoder_variant": self.model_params.get("decoder_variant", "tdca_like_legacy")}

    monkeypatch.setattr(module, "load_collection_dataset", lambda _path: dataset)
    monkeypatch.setattr(module, "create_decoder", lambda _model_name, **kwargs: _BoardDecoder(**kwargs))
    monkeypatch.setattr(module, "DEFAULT_LOGREG_FIT_CONFIG", LogRegFitConfig(epochs=4, min_samples=1))
    _relax_correctness_minima(monkeypatch, module)
    decision_grid = [
        {
            "candidate_min_windows": 1,
            "armed_min_windows": 2,
            "lambda_decay": 0.85,
            "upper_commit_th": 2.2,
            "lower_idle_th": 0.4,
            "refractory_sec": float(index) / 100.0,
        }
        for index in range(130)
    ]
    monkeypatch.setattr(
        module,
        "_decision_param_grid",
        lambda: list(decision_grid),
    )

    payload = run_tdca_local_opt(
        TDCALocalOptConfig(
            dataset_manifest_session1=manifest_path,
            output_profile_path=tmp_path / "profile_full_board.json",
            report_path=tmp_path / "report_full_board.json",
            dataset_manifests=(manifest_path,),
            organize_report_dir=False,
            win_candidates=(2.0,),
            tdca_delay_steps=(2,),
            tdca_n_components=(2,),
            multi_seed_count=1,
            compute_backend="cpu",
        ),
        log_fn=lambda _msg: None,
    )

    selected_candidate_keys = {row["candidate_key"] for row in payload["decision_search_board"]}
    expected_rows = len({module._make_decision_params_key(item) for item in decision_grid}) * len(selected_candidate_keys)
    assert len(payload["decision_search_board"]) == expected_rows
    assert {row["decoder_variant"] for row in payload["decision_search_board"]} == {
        "tdca_like_legacy",
        "tdca_paper_aligned",
    }
    assert payload["decision_search_target"] == "tune_split"
    assert payload["final_selection_target"] == "holdout_split"
    assert payload["holdout_selection_board"]


def test_tdca_board_tie_break_prefers_paper_aligned_variant() -> None:
    import ssvep_core.tdca_local_opt as module

    rank_key = [0.0, 2.0, 2.5, -1.0, -1.0, 0.5]
    legacy = {
        "decoder_variant": "tdca_like_legacy",
        "rank_key": list(rank_key),
    }
    paper = {
        "decoder_variant": "tdca_paper_aligned",
        "rank_key": list(rank_key),
    }

    ordered = sorted([legacy, paper], key=module._tdca_board_sort_key)
    assert ordered[0]["decoder_variant"] == "tdca_paper_aligned"


def test_tdca_decoder_variants_roundtrip_and_paper_scores_differ() -> None:
    segments = _make_tdca_training_segments()
    sampling_rate = 250
    freqs = (8.0, 10.0, 12.0, 15.0)
    window = segments[0][1]
    legacy = create_decoder(
        "tdca",
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=1.0,
        step_sec=0.25,
        model_params={"Nh": 3, "delay_steps": 2, "n_components": 2, "decoder_variant": "tdca_like_legacy"},
        decoder_compute_backend="cpu",
    )
    legacy.fit(segments)
    legacy_scores = legacy.score_window(window)
    legacy_state = legacy.get_state()

    paper = create_decoder(
        "tdca",
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=1.0,
        step_sec=0.25,
        model_params={"Nh": 3, "delay_steps": 2, "n_components": 2, "decoder_variant": "tdca_paper_aligned"},
        decoder_compute_backend="cpu",
    )
    paper.fit(segments)
    paper_scores = paper.score_window(window)
    paper_state = paper.get_state()

    reloaded = create_decoder(
        "tdca",
        sampling_rate=sampling_rate,
        freqs=freqs,
        win_sec=1.0,
        step_sec=0.25,
        model_params={"Nh": 3, "delay_steps": 2, "n_components": 2, "decoder_variant": "tdca_paper_aligned"},
        decoder_compute_backend="cpu",
    )
    reloaded.set_state(paper_state)
    reloaded_scores = reloaded.score_window(window)

    assert paper_state["decoder_variant"] == "tdca_paper_aligned"
    assert paper_state["training_window_policy"] == "banked_latency_trimmed_uniform_trial_equal"
    assert paper_state["paper_alignment_level"] == "near-paper"
    assert float(paper_state["training_latency_sec"]) == 0.14
    assert paper_state["reference_q_factors"]
    assert np.all(np.isfinite(legacy_scores))
    assert np.all(np.isfinite(paper_scores))
    assert not np.allclose(legacy_scores, paper_scores)
    assert np.allclose(paper_scores, reloaded_scores)
    assert legacy_state["decoder_variant"] == "tdca_like_legacy"
    assert legacy_state["training_window_policy"] == "last_window_only"
    assert float(legacy_state["training_latency_sec"]) == 0.0


def test_gate_replay_direct_switch_without_idle_gap() -> None:
    import ssvep_core.tdca_local_opt as module

    gate = module.GateReplayState(_make_profile())
    gate.reset()
    first = gate.update(
        _make_feature_row(
            label="8Hz",
            trial_role="control",
            trial_id=0,
            window_index=0,
            pred_freq=8.0,
            expected_freq=8.0,
            top1_score=0.92,
            ratio=2.0,
            margin=0.7,
            gate_score=0.9,
        )
    )
    switched = gate.update(
        _make_feature_row(
            label="10Hz",
            trial_role="control",
            trial_id=0,
            window_index=1,
            pred_freq=10.0,
            expected_freq=10.0,
            top1_score=0.92,
            ratio=2.0,
            margin=0.7,
            gate_score=0.9,
        )
    )
    assert first["gate_open_freq"] == 8.0
    assert switched["gate_open_freq"] == 10.0
    assert switched["gate_event"] == "switch"
    assert switched["gate_exit_windows"] == 0


def test_gate_replay_switch_fallback_counts_exit() -> None:
    import ssvep_core.tdca_local_opt as module

    profile = _make_profile()
    gate = module.GateReplayState(profile)
    gate.reset()
    gate.update(
        _make_feature_row(
            label="8Hz",
            trial_role="control",
            trial_id=0,
            window_index=0,
            pred_freq=8.0,
            expected_freq=8.0,
            top1_score=0.92,
            ratio=2.0,
            margin=0.7,
            gate_score=0.9,
        )
    )
    fallback = gate.update(
        _make_feature_row(
            label="10Hz",
            trial_role="control",
            trial_id=0,
            window_index=1,
            pred_freq=10.0,
            expected_freq=10.0,
            top1_score=0.2,
            ratio=1.05,
            margin=0.02,
            gate_score=-0.4,
        )
    )
    assert fallback["switch_pass"] is False
    assert fallback["exit_fail"] is True
    assert fallback["gate_event"] == "exit"
    assert fallback["gate_open_freq"] is None


def test_decision_engine_separates_tracked_commit_and_selected() -> None:
    from ssvep_core.decision.engine import DecisionEngine, DecisionEngineConfig
    from ssvep_core.decision.accumulator import EvidenceAccumulatorConfig
    from ssvep_core.decision.state_machine import StateMachineConfig

    engine = DecisionEngine(
        DecisionEngineConfig(
            evidence=EvidenceAccumulatorConfig(
                lambda_decay=0.85,
                beta_consistency=0.5,
                upper_commit_th=0.1,
                lower_idle_th=-1.0,
            ),
            state=StateMachineConfig(
                candidate_min_windows=1,
                armed_min_windows=2,
                commit_consistency_th=0.0,
                enter_gate_th=0.0,
                exit_gate_th=-1.0,
                refractory_sec=0.5,
            ),
        )
    )
    engine.reset()
    first = engine.step(8.0, 1.0, 1.0, gate_open_freq=8.0, timestamp_s=0.0)
    second = engine.step(8.0, 1.0, 1.0, gate_open_freq=8.0, timestamp_s=0.25)
    third = engine.step(8.0, 1.0, 1.0, gate_open_freq=8.0, timestamp_s=0.50)
    release = engine.step(None, -2.0, -2.0, gate_open_freq=None, timestamp_s=1.25)

    assert first["tracked_freq"] == 8.0
    assert first["selected_freq"] is None
    assert first["commit_freq"] is None
    assert third["commit"] is True
    assert third["commit_freq"] == 8.0
    assert third["selected_freq"] == 8.0
    assert release["selected_freq"] is None


def test_decision_engine_holds_selected_output_while_new_freq_is_only_tracked() -> None:
    from ssvep_core.decision.engine import DecisionEngine, DecisionEngineConfig
    from ssvep_core.decision.accumulator import EvidenceAccumulatorConfig
    from ssvep_core.decision.state_machine import StateMachineConfig

    engine = DecisionEngine(
        DecisionEngineConfig(
            evidence=EvidenceAccumulatorConfig(
                lambda_decay=0.85,
                beta_consistency=0.5,
                upper_commit_th=0.1,
                lower_idle_th=-1.0,
            ),
            state=StateMachineConfig(
                candidate_min_windows=2,
                armed_min_windows=2,
                commit_consistency_th=0.0,
                enter_gate_th=0.0,
                exit_gate_th=-1.0,
                refractory_sec=0.5,
            ),
        )
    )
    engine.reset()
    engine.step(8.0, 1.0, 1.0, gate_open_freq=8.0, timestamp_s=0.00)
    engine.step(8.0, 1.0, 1.0, gate_open_freq=8.0, timestamp_s=0.25)
    engine.step(8.0, 1.0, 1.0, gate_open_freq=8.0, timestamp_s=0.50)
    committed = engine.step(8.0, 1.0, 1.0, gate_open_freq=8.0, timestamp_s=0.75)
    during_refractory = engine.step(8.0, 1.0, 1.0, gate_open_freq=8.0, timestamp_s=1.00)
    tracked_new = engine.step(10.0, 1.0, 1.0, gate_open_freq=10.0, timestamp_s=1.30)

    assert committed["commit"] is True
    assert committed["selected_freq"] == 8.0
    assert during_refractory["selected_freq"] == 8.0
    assert tracked_new["commit"] is False
    assert tracked_new["tracked_freq"] == 10.0
    assert tracked_new["commit_freq"] is None
    assert tracked_new["selected_freq"] == 8.0
    assert tracked_new["release"] is False


def test_structured_evaluator_does_not_use_async_decision_gate(monkeypatch) -> None:
    import ssvep_core.tdca_local_opt as module

    class _Boom:
        @classmethod
        def from_profile(cls, _profile):
            raise AssertionError("legacy AsyncDecisionGate should not be used")

    monkeypatch.setattr(module, "AsyncDecisionGate", _Boom, raising=False)
    bundle = module._evaluate_structured_rows(
        scored_rows=_make_release_rows(),
        profile=_make_profile(),
        freqs=(8.0, 10.0, 12.0, 15.0),
        decision_params={
            "candidate_min_windows": 1,
            "armed_min_windows": 1,
            "lambda_decay": 0.85,
            "upper_commit_th": 0.1,
            "lower_idle_th": 0.4,
            "refractory_sec": 0.25,
        },
        inference_ms=4.0,
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
    )
    assert "async_metrics" in bundle


def test_release_latency_is_controlled_by_engine_params() -> None:
    import ssvep_core.tdca_local_opt as module

    rows = _make_release_rows()
    profile = _make_profile()
    fast_release = module._evaluate_structured_rows(
        scored_rows=rows,
        profile=profile,
        freqs=profile.freqs,
        decision_params={
            "candidate_min_windows": 1,
            "armed_min_windows": 1,
            "lambda_decay": 0.85,
            "upper_commit_th": 0.1,
            "lower_idle_th": 0.4,
            "refractory_sec": 0.25,
        },
        inference_ms=4.0,
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
    )
    slow_release = module._evaluate_structured_rows(
        scored_rows=rows,
        profile=profile,
        freqs=profile.freqs,
        decision_params={
            "candidate_min_windows": 1,
            "armed_min_windows": 1,
            "lambda_decay": 0.85,
            "upper_commit_th": 0.1,
            "lower_idle_th": 0.4,
            "refractory_sec": 1.0,
        },
        inference_ms=4.0,
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
    )
    assert (
        float(fast_release["async_metrics"]["release_latency_s"])
        < float(slow_release["async_metrics"]["release_latency_s"])
    )


def test_switch_latency_is_controlled_by_engine_params() -> None:
    import ssvep_core.tdca_local_opt as module

    rows = _make_switch_rows()
    profile = _make_profile()
    faster = module._evaluate_structured_rows(
        scored_rows=rows,
        profile=profile,
        freqs=profile.freqs,
        decision_params={
            "candidate_min_windows": 1,
            "armed_min_windows": 1,
            "lambda_decay": 0.85,
            "upper_commit_th": 0.1,
            "lower_idle_th": 0.4,
            "refractory_sec": 0.0,
        },
        inference_ms=4.0,
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
    )
    slower = module._evaluate_structured_rows(
        scored_rows=rows,
        profile=profile,
        freqs=profile.freqs,
        decision_params={
            "candidate_min_windows": 1,
            "armed_min_windows": 1,
            "lambda_decay": 0.85,
            "upper_commit_th": 0.1,
            "lower_idle_th": 0.4,
            "refractory_sec": 1.0,
        },
        inference_ms=4.0,
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
    )
    assert float(faster["async_metrics"]["switch_latency_s"]) < float(slower["async_metrics"]["switch_latency_s"])


def test_idle_false_positive_counts_only_engine_commit() -> None:
    import ssvep_core.tdca_local_opt as module

    rows = _make_idle_fp_rows()
    profile = _make_profile()
    no_commit = module._evaluate_structured_rows(
        scored_rows=rows,
        profile=profile,
        freqs=profile.freqs,
        decision_params={
            "candidate_min_windows": 1,
            "armed_min_windows": 1,
            "lambda_decay": 0.85,
            "upper_commit_th": 10.0,
            "lower_idle_th": 0.4,
            "refractory_sec": 0.0,
        },
        inference_ms=4.0,
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
    )
    with_commit = module._evaluate_structured_rows(
        scored_rows=rows,
        profile=profile,
        freqs=profile.freqs,
        decision_params={
            "candidate_min_windows": 1,
            "armed_min_windows": 1,
            "lambda_decay": 0.85,
            "upper_commit_th": 0.1,
            "lower_idle_th": 0.4,
            "refractory_sec": 0.0,
        },
        inference_ms=4.0,
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
    )
    assert float(no_commit["async_metrics"]["idle_fp_event_count"]) == 0.0
    assert float(with_commit["async_metrics"]["idle_fp_event_count"]) >= 1.0


def test_invalid_models_and_channel_modes_fail_fast(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    profile_path = tmp_path / "profile_tdca_local.json"
    config = TDCALocalOptConfig(
        dataset_manifest_session1=tmp_path / "missing_manifest.json",
        output_profile_path=profile_path,
        report_path=report_path,
        model_names=("fbcca",),
    )
    try:
        run_tdca_local_opt(config, log_fn=lambda _msg: None)
        assert False, "expected invalid model_names to fail fast"
    except ValueError as exc:
        assert "model_names" in str(exc)

    config = TDCALocalOptConfig(
        dataset_manifest_session1=tmp_path / "missing_manifest.json",
        output_profile_path=profile_path,
        report_path=report_path,
        channel_modes=("occipital3",),
    )
    try:
        run_tdca_local_opt(config, log_fn=lambda _msg: None)
        assert False, "expected invalid channel_modes to fail fast"
    except ValueError as exc:
        assert "channel_modes" in str(exc)


def test_candidate_context_uses_decoder_backend_not_replay_backend(monkeypatch, tmp_path: Path) -> None:
    import ssvep_core.tdca_local_opt as module

    trial_rows = _make_trial_rows()
    manifest_path = tmp_path / "session_manifest.json"
    manifest_path.write_text(json.dumps({"trials": trial_rows, "quality_summary": {}}), encoding="utf-8")

    segments: list[tuple[TrialSpec, np.ndarray]] = []
    for row in trial_rows:
        segments.append(
            (
                TrialSpec(
                    label=str(row["label"]),
                    expected_freq=row["expected_freq"],
                    trial_id=int(row["trial_id"]),
                    block_index=int(row["block_index"]),
                ),
                np.zeros((12, 8), dtype=np.float64),
            )
        )

    dataset = LoadedDataset(
        manifest_path=manifest_path,
        npz_path=tmp_path / "raw_trials.npz",
        session_id="session_backend",
        subject_id="subject_backend",
        sampling_rate=4,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        protocol_config={"step_sec": 0.25},
        trial_segments=segments,
        manifest={"trials": trial_rows, "quality_summary": {}},
    )
    merged = module.MergedLocalDataset(
        manifest_paths=(manifest_path,),
        datasets=(dataset,),
        trial_segments=tuple(segments),
        sampling_rate=4,
        freqs=(8.0, 10.0, 12.0, 15.0),
        board_eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        subject_id="subject_backend",
        session_ids=("session_backend",),
        trial_role_counts={"control": 12, "clean_idle": 3, "hard_idle": 3},
        quality_rows=(),
    )

    captured: dict[str, object] = {}

    class _FakeDecoder:
        requires_fit = True

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.fs = int(kwargs["sampling_rate"])
            self.freqs = tuple(float(item) for item in kwargs["freqs"])
            self.win_sec = float(kwargs["win_sec"])
            self.step_sec = float(kwargs["step_sec"])
            self.model_params = dict(kwargs["model_params"])
            self.win_samples = 4
            self.step_samples = 1

        def fit(self, _trial_segments):
            return None

        def iter_window_features(self, _segment, *, expected_freq, label, trial_id, block_index):
            role = "control" if expected_freq is not None and not str(label).startswith("switch_to_") else (
                "hard_idle" if "switch" in str(label) else "clean_idle"
            )
            for index in range(2):
                if expected_freq is None:
                    pred_freq = 8.0
                    top1_score = 0.12
                    top2_score = 0.10
                    ratio = 1.05
                    margin = 0.02
                    normalized_top1 = 0.20
                    score_entropy = 0.85
                else:
                    pred_freq = float(expected_freq)
                    top1_score = 0.92
                    top2_score = 0.10
                    ratio = 2.2
                    margin = 0.72
                    normalized_top1 = 0.82
                    score_entropy = 0.18
                yield {
                    "trial_role": role,
                    "label": str(label),
                    "expected_freq": expected_freq,
                    "trial_id": int(trial_id),
                    "block_index": int(block_index),
                    "window_index": int(index),
                    "pred_freq": pred_freq,
                    "top1_score": top1_score,
                    "top2_score": top2_score,
                    "ratio": ratio,
                    "margin": margin,
                    "normalized_top1": normalized_top1,
                    "score_entropy": score_entropy,
                }

        def analyze_window(self, _window):
            return {
                "pred_freq": 8.0,
                "top1_score": 0.5,
                "top2_score": 0.3,
                "ratio": 1.2,
                "margin": 0.2,
                "normalized_top1": 0.4,
                "score_entropy": 0.4,
            }

        def get_state(self):
            return {"decoder_variant": self.model_params.get("decoder_variant", "tdca_like_legacy")}

    monkeypatch.setattr(module, "create_decoder", lambda _model_name, **kwargs: _FakeDecoder(**kwargs))
    monkeypatch.setattr(module, "DEFAULT_LOGREG_FIT_CONFIG", LogRegFitConfig(epochs=4, min_samples=1))

    split = module.RepeatedGroupSplit(
        repeat_index=0,
        train_indices=tuple(range(15)),
        gate_indices=tuple(range(15, len(segments))),
        holdout_indices=tuple(range(12)),
        fingerprint="backend-test",
    )
    context = module._build_candidate_context(
        merged_dataset=merged,
        split=split,
        model_name="tdca",
        win_sec=2.0,
        model_params=module._default_model_params(
            model_name="tdca",
            Nh=3,
            delay_steps=2,
            n_components=2,
            decoder_variant="tdca_like_legacy",
        ),
        confidence_variant="global_correctness_logistic",
        decoder_compute_backend="cuda",
        gpu_device=0,
        gpu_precision="float32",
        gpu_warmup=False,
        gpu_cache_policy="off",
        control_state_mode="frequency-specific-logistic",
        decision_time_mode="fixed-window",
        async_decision_time_mode="first-correct",
        replay_policy={"effective_replay_backend": "cpu", "gpu_replay_reason": "batched_replay_not_implemented"},
    )

    assert captured["decoder_compute_backend"] == "cuda"
    assert context["replay_backend_policy"]["effective_replay_backend"] == "cpu"


def test_tdca_ui_config_does_not_append_baseline_models() -> None:
    _ = _get_qapp()
    window = TrainingEvaluationWindow()
    manifest_dir = PROJECT_DIR / ".tmp_test_artifacts" / f"tdca_ui_{uuid.uuid4().hex}"
    manifest_path = manifest_dir / "session_manifest.json"
    try:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        window._dataset_scan_rows = []
        window.simple_mode_check.setChecked(False)
        window.configure_tdca_local_opt_mode(auto_start=False)
        window.session1_edit.setText(str(manifest_path))
        window.session2_edit.setText("")

        cfg = window._read_config()

        assert cfg.task == "tdca-local-opt"
        assert cfg.model_names == ("tdca",)
        assert cfg.channel_modes == ("all8",)
    finally:
        shutil.rmtree(manifest_dir, ignore_errors=True)
        window.close()


def test_tdca_ui_marks_invalid_local_run() -> None:
    _ = _get_qapp()
    window = TrainingEvaluationWindow()
    try:
        payload = {
            "task": "tdca-local-opt",
            "report_path": str(PROJECT_DIR / "artifacts" / "runs" / "local" / "dummy" / "report.json"),
            "report_dir": str(PROJECT_DIR / "artifacts" / "runs" / "local" / "dummy"),
            "status": "invalid",
            "status_reasons": ["gate_calibration_invalid_all_candidates"],
            "chosen_model_rationale": "invalid_run_not_comparable",
            "profile_saved": False,
            "chosen_async_metrics": {
                "idle_fp_per_min": 0.0,
                "control_recall": 0.0,
                "switch_latency_s": 7.0,
                "release_latency_s": 7.0,
            },
            "chosen_metrics_4class": {
                "acc": 0.25,
                "macro_f1": 0.2,
            },
            "quality_kept_trials_session1": 74,
            "quality_total_trials_session1": 74,
        }

        window._on_done(payload)

        assert "无效" in window.status_label.text()
        assert "完成" in window.progress_detail_label.text()
    finally:
        window.close()


def test_gpu_policy_falls_back_to_cpu_when_speedup_is_too_small(monkeypatch) -> None:
    import ssvep_core.tdca_local_opt as module

    class _Backend:
        def __init__(self, name: str) -> None:
            self.backend_name = name

        def microbenchmark_transfer(self, **_kwargs):
            return {"backend_name": self.backend_name}

        def describe(self):
            return {"backend_name": self.backend_name, "uses_cuda": self.backend_name == "cuda"}

    monkeypatch.setattr(
        module,
        "resolve_compute_backend",
        lambda requested, **_kwargs: _Backend("cuda" if requested == "cuda" else "cpu"),
    )
    monkeypatch.setattr(
        module,
        "_microbenchmark_replay_backend",
        lambda backend, **_kwargs: {"backend_name": backend.backend_name, "total_ms": 8.0 if backend.backend_name == "cuda" else 10.0},
    )
    env = preflight_tdca_local_env(compute_backend="auto", gpu_device=0, gpu_precision="float32")
    policy = module._resolve_replay_backend_policy(env_preflight=env, estimated_window_count=512)
    assert policy["effective_replay_backend"] == "cpu"

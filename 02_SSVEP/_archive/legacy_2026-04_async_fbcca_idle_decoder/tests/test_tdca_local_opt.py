from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from async_fbcca_idle_standalone import ThresholdProfile, TrialSpec, create_decoder
from ssvep_core.dataset import LoadedDataset
from ssvep_core.gating.per_freq_logreg_gate import LogRegFitConfig
from ssvep_core.tdca_local_opt import (
    TDCALocalOptConfig,
    backfill_manifest_trial_roles,
    build_repeated_group_splits,
    preflight_tdca_local_env,
    run_tdca_local_opt,
)
from ssvep_training_evaluation_cli import _parse_task, build_parser


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
    return {
        "label": label,
        "trial_role": trial_role,
        "trial_id": int(trial_id),
        "window_index": int(window_index),
        "pred_freq": pred_freq,
        "expected_freq": expected_freq,
        "top1_score": float(top1_score),
        "top2_score": float(max(top1_score - margin, 0.01)),
        "ratio": float(ratio),
        "margin": float(margin),
        "gate_score": float(gate_score),
        "control_log_lr": float(gate_score),
        "p_control": float(0.8 if gate_score > 0.0 else 0.2),
        "normalized_top1": float(min(top1_score, 1.0)),
        "score_entropy": float(0.2 if gate_score > 0.0 else 0.8),
    }


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
    args = parser.parse_args(["--task", "tdca-local-opt"])
    assert args.task == "tdca-local-opt"
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
            for index in range(3):
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

    report_path = tmp_path / "offline_train_eval.json"
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
    assert "variant_summary" in payload
    assert payload["baseline_opening"]
    assert payload["baseline_seal"]
    assert {row["decoder_variant"] for row in payload["tdca_search_board"]} == {
        "tdca_like_legacy",
        "tdca_paper_aligned",
    }
    assert payload["decoder_variant"] == "tdca_paper_aligned"
    assert report_path.exists()
    assert profile_path.exists()
    assert Path(payload["profile_v2_path"]).exists()


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
    assert paper_state["reference_q_factors"]
    assert np.all(np.isfinite(legacy_scores))
    assert np.all(np.isfinite(paper_scores))
    assert not np.allclose(legacy_scores, paper_scores)
    assert np.allclose(paper_scores, reloaded_scores)
    assert legacy_state["decoder_variant"] == "tdca_like_legacy"


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
    report_path = tmp_path / "offline_train_eval.json"
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

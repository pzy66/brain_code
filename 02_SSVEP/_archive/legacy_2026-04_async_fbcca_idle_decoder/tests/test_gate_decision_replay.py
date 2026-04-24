from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.decision import DecisionEngine, DecisionEngineConfig, EvidenceAccumulatorConfig, StateMachineConfig
from ssvep_core.evaluation import ReplayEvaluator, ReplayEvaluatorConfig
from ssvep_core.gating import GlobalThresholdGate, PerFrequencyLogRegGate


def test_global_gate_predict_returns_probability_and_score() -> None:
    gate = GlobalThresholdGate()
    out = gate.predict(
        {
            "top1_score": 0.7,
            "top2_score": 0.2,
            "ratio": 1.6,
            "margin": 0.5,
            "normalized_top1": 0.6,
            "score_entropy": 0.2,
        },
        10.0,
    )
    assert 0.0 < float(out.p_control) < 1.0
    assert np.isfinite(float(out.gate_score))
    assert float(out.pred_freq or 0.0) == 10.0


def test_per_freq_logreg_gate_fit_and_predict() -> None:
    rows: list[dict[str, float | str | None]] = []
    for _ in range(80):
        rows.append(
            {
                "pred_freq": 8.0,
                "expected_freq": 8.0,
                "label": "8Hz",
                "top1_score": 0.9,
                "top2_score": 0.1,
                "margin": 0.7,
                "ratio": 2.2,
                "score_entropy": 0.1,
                "normalized_top1": 0.8,
                "consistency": 0.9,
                "margin_mean_k": 0.6,
                "ratio_mean_k": 1.8,
            }
        )
        rows.append(
            {
                "pred_freq": 8.0,
                "expected_freq": None,
                "label": "switch_to_10",
                "top1_score": 0.2,
                "top2_score": 0.18,
                "margin": 0.02,
                "ratio": 1.05,
                "score_entropy": 0.8,
                "normalized_top1": 0.2,
                "consistency": 0.25,
                "margin_mean_k": 0.08,
                "ratio_mean_k": 1.1,
            }
        )
    gate = PerFrequencyLogRegGate()
    fit_summary = gate.fit(rows=rows, freqs=(8.0,))
    assert fit_summary["per_freq"]["8"]["status"] == "ok"
    positive = gate.predict(rows[0], 8.0)
    negative = gate.predict(rows[1], 8.0)
    assert float(positive.p_control) > float(negative.p_control)
    restored = PerFrequencyLogRegGate.from_payload(payload=gate.to_payload())
    restored_positive = restored.predict(rows[0], 8.0)
    assert float(restored_positive.p_control) > 0.5


def test_decision_engine_commit_and_refractory() -> None:
    engine = DecisionEngine(
        DecisionEngineConfig(
            evidence=EvidenceAccumulatorConfig(
                lambda_decay=0.8,
                beta_consistency=0.5,
                upper_commit_th=0.2,
                lower_idle_th=-2.0,
            ),
            state=StateMachineConfig(
                candidate_min_windows=2,
                armed_min_windows=3,
                commit_consistency_th=0.5,
                enter_gate_th=-1.0,
                exit_gate_th=-5.0,
                refractory_sec=0.4,
            ),
        )
    )
    t0 = 100.0
    out1 = engine.step(8.0, 0.4, 1.0, timestamp_s=t0)
    out2 = engine.step(8.0, 0.4, 1.0, timestamp_s=t0 + 0.25)
    out3 = engine.step(8.0, 0.4, 1.0, timestamp_s=t0 + 0.50)
    out4 = engine.step(8.0, 0.4, 1.0, timestamp_s=t0 + 0.75)
    assert str(out1["state"]) in {"Idle", "Candidate"}
    assert str(out2["state"]) in {"Candidate", "Armed"}
    assert bool(out4["commit"]) is True
    out5 = engine.step(8.0, 0.4, 1.0, timestamp_s=t0 + 0.80)
    assert str(out5["state"]) == "Refractory"
    assert bool(out5["commit"]) is False
    out6 = engine.step(8.0, -1.0, 0.0, timestamp_s=t0 + 1.30)
    assert str(out6["state"]) == "Idle"


def test_replay_evaluator_produces_primary_metrics() -> None:
    gate = GlobalThresholdGate()
    engine = DecisionEngine(
        DecisionEngineConfig(
            evidence=EvidenceAccumulatorConfig(upper_commit_th=0.4, lower_idle_th=-1.0),
            state=StateMachineConfig(candidate_min_windows=2, armed_min_windows=3, refractory_sec=0.5),
        )
    )
    evaluator = ReplayEvaluator(
        gate=gate,
        decision_engine=engine,
        config=ReplayEvaluatorConfig(step_sec=0.25),
    )
    stream: list[dict[str, float]] = []
    labels: list[dict[str, float | str | None]] = []
    for _ in range(20):
        stream.append(
            {
                "pred_freq": 8.0,
                "top1_score": 0.12,
                "top2_score": 0.10,
                "margin": 0.02,
                "ratio": 1.1,
                "normalized_top1": 0.2,
                "score_entropy": 0.9,
            }
        )
        labels.append({"role": "clean_idle", "expected_freq": None, "label": "idle"})
    for _ in range(30):
        stream.append(
            {
                "pred_freq": 8.0,
                "top1_score": 0.9,
                "top2_score": 0.1,
                "margin": 0.65,
                "ratio": 2.4,
                "normalized_top1": 0.8,
                "score_entropy": 0.2,
            }
        )
        labels.append({"role": "control", "expected_freq": 8.0, "label": "8Hz"})
    report = evaluator.run(stream, labels)
    metrics = dict(report.get("metrics", {}))
    assert int(report.get("window_count", 0)) == 50
    assert "idle_false_trigger_per_min" in metrics
    assert "wrong_action_rate" in metrics
    assert "median_commit_latency" in metrics
    assert int(metrics.get("commit_count", 0)) >= 1


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .decision import DecisionEngine, DecisionEngineConfig, EvidenceAccumulatorConfig, StateMachineConfig
from .gating import GlobalThresholdGate, PerFrequencyLogRegGate, RollingFeatureHistory


@dataclass
class ShadowRuntimeChain:
    gate: Any
    decision: DecisionEngine
    history: RollingFeatureHistory

    def reset(self) -> None:
        if hasattr(self.gate, "reset"):
            self.gate.reset()
        self.decision.reset()
        self.history.reset()

    def update(self, analysis: dict[str, Any], *, timestamp_s: Optional[float] = None) -> dict[str, Any]:
        pred_freq = analysis.get("pred_freq")
        pred_freq = None if pred_freq is None else float(pred_freq)
        margin = float(analysis.get("margin", 0.0) or 0.0)
        ratio = float(analysis.get("ratio", 1.0) or 1.0)
        hist = self.history.update(pred_freq=pred_freq, margin=margin, ratio=ratio)
        feature_row = dict(analysis)
        feature_row.setdefault("consistency", float(hist["consistency"]))
        feature_row.setdefault("margin_mean_k", float(hist["margin_mean_k"]))
        feature_row.setdefault("ratio_mean_k", float(hist["ratio_mean_k"]))
        gate_out = self.gate.predict(feature_row, pred_freq)
        decision = self.decision.step(
            pred_freq,
            float(gate_out.gate_score),
            float(hist["consistency"]),
            timestamp_s=timestamp_s,
        )
        payload = {
            **analysis,
            "p_control": float(gate_out.p_control),
            "gate_score": float(gate_out.gate_score),
            "consistency": float(hist["consistency"]),
            "margin_mean_k": float(hist["margin_mean_k"]),
            "ratio_mean_k": float(hist["ratio_mean_k"]),
            "state": str(decision.get("state", "Idle")),
            "selected_freq": decision.get("selected_freq"),
            "commit": bool(decision.get("commit", False)),
            "stable_windows": int(decision.get("stable_windows", 0) or 0),
            "evidence_score": float(decision.get("evidence_score", 0.0) or 0.0),
        }
        return payload


def _load_profile_v2_payload(profile_path: Path) -> Optional[dict[str, Any]]:
    path = Path(profile_path).expanduser().resolve()
    candidates = [
        path,
        path.with_name(f"{path.stem}_v2.json"),
    ]
    for candidate in candidates:
        try:
            data = candidate.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            import json

            payload = dict(json.loads(data))
        except Exception:
            continue
        if str(payload.get("version", "")).strip() == "2.0":
            return payload
    return None


def build_shadow_runtime_chain(*, profile: Any, profile_path: Path) -> tuple[ShadowRuntimeChain, dict[str, Any]]:
    payload_v2 = _load_profile_v2_payload(profile_path)
    evidence_cfg = EvidenceAccumulatorConfig()
    state_cfg = StateMachineConfig(refractory_sec=0.8)
    gate: Any
    mode = "global_gate"
    if payload_v2 is not None:
        gate_payload = dict(payload_v2.get("gate", {}))
        gate_type = str(gate_payload.get("type", "")).strip().lower()
        if gate_type == "frequency_specific_logreg":
            gate = PerFrequencyLogRegGate.from_payload(payload=gate_payload)
            mode = "per_frequency_logreg"
        else:
            gate = GlobalThresholdGate.from_profile(profile)
        evidence = dict(payload_v2.get("evidence", {}))
        evidence_cfg = EvidenceAccumulatorConfig(
            lambda_decay=float(evidence.get("lambda", evidence.get("lambda_decay", 0.85))),
            beta_consistency=float(evidence.get("beta_consistency", 0.5)),
            upper_commit_th=float(evidence.get("upper_commit_th", 2.2)),
            lower_idle_th=float(evidence.get("lower_idle_th", 0.4)),
        )
        runtime = dict(payload_v2.get("runtime", {}))
        state_cfg = StateMachineConfig(
            candidate_min_windows=2,
            armed_min_windows=3,
            refractory_sec=float(runtime.get("refractory_sec", 0.8)),
        )
    else:
        gate = GlobalThresholdGate.from_profile(profile)

    chain = ShadowRuntimeChain(
        gate=gate,
        decision=DecisionEngine(DecisionEngineConfig(evidence=evidence_cfg, state=state_cfg)),
        history=RollingFeatureHistory(window_size=4),
    )
    summary = {
        "shadow_mode": "full_chain",
        "gate_mode": mode,
        "profile_v2_loaded": bool(payload_v2 is not None),
    }
    return chain, summary


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..decision import DecisionEngine, DecisionEngineConfig
from ..gating import BaseGate, RollingFeatureHistory
from ..dataset import infer_trial_role

IDLE_ROLES = {"idle", "clean_idle", "hard_idle", "no_control"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(output):
        return float(default)
    return float(output)


def _resolve_role_from_label(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"control", "clean_idle", "hard_idle"}:
        return text
    if text.startswith("switch") or "transition" in text or "scan" in text or "hard" in text:
        return "hard_idle"
    if "idle" in text:
        return "clean_idle"
    if text:
        return "control"
    return "clean_idle"


@dataclass(frozen=True)
class ReplayEvaluatorConfig:
    step_sec: float = 0.25
    consistency_window: int = 4
    include_trajectory: bool = True


class ReplayEvaluator:
    def __init__(
        self,
        *,
        gate: BaseGate,
        decision_engine: DecisionEngine,
        config: Optional[ReplayEvaluatorConfig] = None,
    ) -> None:
        self.gate = gate
        self.decision = decision_engine
        self.config = config or ReplayEvaluatorConfig()

    def _label_at(
        self,
        *,
        index: int,
        timestamp_s: float,
        labels: Optional[Sequence[Any]],
    ) -> dict[str, Any]:
        if not labels:
            return {"role": "clean_idle", "expected_freq": None}
        if index < len(labels):
            item = labels[index]
            if isinstance(item, Mapping):
                role = _resolve_role_from_label(item.get("role", item.get("label", "")))
                expected_freq = item.get("expected_freq")
                return {
                    "role": role,
                    "expected_freq": None if expected_freq is None else _safe_float(expected_freq, float("nan")),
                    "label": str(item.get("label", role)),
                }
            role = _resolve_role_from_label(item)
            return {"role": role, "expected_freq": None, "label": str(item)}

        for item in labels:
            if not isinstance(item, Mapping):
                continue
            start = _safe_float(item.get("start_s", 0.0), 0.0)
            end = _safe_float(item.get("end_s", -1.0), -1.0)
            if start <= timestamp_s <= end:
                role = _resolve_role_from_label(item.get("role", item.get("label", "")))
                expected_freq = item.get("expected_freq")
                return {
                    "role": role,
                    "expected_freq": None if expected_freq is None else _safe_float(expected_freq, float("nan")),
                    "label": str(item.get("label", role)),
                }
        return {"role": "clean_idle", "expected_freq": None}

    def run(
        self,
        stream: Sequence[Mapping[str, Any]],
        labels: Optional[Sequence[Any]] = None,
        config: Optional[ReplayEvaluatorConfig] = None,
    ) -> dict[str, Any]:
        cfg = config or self.config
        step_sec = max(_safe_float(cfg.step_sec, 0.25), 1e-3)
        history = RollingFeatureHistory(window_size=int(cfg.consistency_window))

        self.gate.reset()
        self.decision.reset()

        trajectory: list[dict[str, Any]] = []
        commits: list[dict[str, Any]] = []
        control_segments: list[dict[str, Any]] = []
        current_control: Optional[dict[str, Any]] = None

        for index, row in enumerate(stream):
            timestamp_s = _safe_float(row.get("timestamp_s"), index * step_sec)
            pred_freq = row.get("pred_freq")
            pred_freq_f = None if pred_freq is None else _safe_float(pred_freq, 0.0)
            margin = _safe_float(row.get("margin", 0.0), 0.0)
            ratio = _safe_float(row.get("ratio", 1.0), 1.0)
            hist = history.update(pred_freq=pred_freq_f, margin=margin, ratio=ratio)
            feature_row = dict(row)
            feature_row.setdefault("consistency", float(hist["consistency"]))
            feature_row.setdefault("margin_mean_k", float(hist["margin_mean_k"]))
            feature_row.setdefault("ratio_mean_k", float(hist["ratio_mean_k"]))

            gate_score = row.get("gate_score")
            gate_p = row.get("p_control")
            if gate_score is None or gate_p is None:
                gate_out = self.gate.predict(feature_row, pred_freq_f)
                gate_score_f = float(gate_out.gate_score)
                gate_p_f = float(gate_out.p_control)
            else:
                gate_score_f = _safe_float(gate_score, 0.0)
                gate_p_f = _safe_float(gate_p, 0.5)

            decision_out = self.decision.step(
                pred_freq_f,
                gate_score_f,
                float(hist["consistency"]),
                timestamp_s=timestamp_s,
            )

            label_info = self._label_at(index=index, timestamp_s=timestamp_s, labels=labels)
            role = _resolve_role_from_label(label_info.get("role", ""))
            expected_freq = label_info.get("expected_freq")
            if expected_freq is not None and not np.isfinite(_safe_float(expected_freq, float("nan"))):
                expected_freq = None

            if role == "control":
                if current_control is None:
                    current_control = {
                        "start_s": float(timestamp_s),
                        "expected_freq": None if expected_freq is None else _safe_float(expected_freq, 0.0),
                    }
            elif current_control is not None:
                current_control["end_s"] = float(timestamp_s)
                control_segments.append(dict(current_control))
                current_control = None

            commit = bool(decision_out.get("commit", False))
            selected_freq = decision_out.get("selected_freq")
            if commit:
                commit_row = {
                    "index": int(index),
                    "timestamp_s": float(timestamp_s),
                    "selected_freq": None if selected_freq is None else _safe_float(selected_freq, 0.0),
                    "pred_freq": pred_freq_f,
                    "role": str(role),
                    "expected_freq": None if expected_freq is None else _safe_float(expected_freq, 0.0),
                    "state": str(decision_out.get("state", "")),
                }
                commits.append(commit_row)

            if bool(cfg.include_trajectory):
                trajectory.append(
                    {
                        "index": int(index),
                        "timestamp_s": float(timestamp_s),
                        "pred_freq": pred_freq_f,
                        "p_control": float(gate_p_f),
                        "gate_score": float(gate_score_f),
                        "consistency": float(hist["consistency"]),
                        "state": str(decision_out.get("state", "")),
                        "commit": bool(commit),
                        "selected_freq": None if selected_freq is None else _safe_float(selected_freq, 0.0),
                        "role": str(role),
                        "expected_freq": None if expected_freq is None else _safe_float(expected_freq, 0.0),
                    }
                )

        if current_control is not None:
            current_control["end_s"] = float(len(stream) * step_sec)
            control_segments.append(dict(current_control))

        total_duration_sec = float(max(len(stream), 1) * step_sec)
        idle_windows = 0
        for index in range(len(stream)):
            label_info = self._label_at(index=index, timestamp_s=index * step_sec, labels=labels)
            if _resolve_role_from_label(label_info.get("role", "")) in IDLE_ROLES:
                idle_windows += 1
        idle_duration_min = float(max(idle_windows * step_sec / 60.0, 1e-6))

        idle_false_triggers = [row for row in commits if str(row.get("role", "")) in IDLE_ROLES]
        control_commits = [row for row in commits if str(row.get("role", "")) == "control"]
        wrong_commits = [
            row
            for row in control_commits
            if row.get("expected_freq") is not None
            and row.get("selected_freq") is not None
            and abs(float(row["expected_freq"]) - float(row["selected_freq"])) > 1e-8
        ]

        latencies: list[float] = []
        for segment in control_segments:
            start_s = _safe_float(segment.get("start_s", 0.0), 0.0)
            end_s = _safe_float(segment.get("end_s", start_s), start_s)
            expected_freq = segment.get("expected_freq")
            segment_commits = [
                row
                for row in commits
                if start_s <= _safe_float(row.get("timestamp_s", -1.0), -1.0) <= end_s
            ]
            if expected_freq is not None:
                segment_commits = [
                    row
                    for row in segment_commits
                    if row.get("selected_freq") is not None
                    and abs(float(row["selected_freq"]) - float(expected_freq)) <= 1e-8
                ]
            if segment_commits:
                first = min(segment_commits, key=lambda row: _safe_float(row.get("timestamp_s", start_s), start_s))
                latencies.append(max(_safe_float(first.get("timestamp_s", start_s), start_s) - start_s, 0.0))

        burst_count = 0
        for idx in range(1, len(commits)):
            if _safe_float(commits[idx]["timestamp_s"], 0.0) - _safe_float(commits[idx - 1]["timestamp_s"], 0.0) <= step_sec * 2.0:
                burst_count += 1

        metrics = {
            "idle_false_trigger_per_min": float(len(idle_false_triggers) / idle_duration_min),
            "wrong_action_rate": (
                float(len(wrong_commits) / len(control_commits)) if control_commits else 0.0
            ),
            "median_commit_latency": float(np.median(np.asarray(latencies, dtype=float))) if latencies else float("inf"),
            "commit_count": int(len(commits)),
            "control_commit_count": int(len(control_commits)),
            "idle_commit_count": int(len(idle_false_triggers)),
            "burst_commit_count": int(burst_count),
            "duration_sec": float(total_duration_sec),
        }

        return {
            "metrics": metrics,
            "commits": commits,
            "trajectory": trajectory if bool(cfg.include_trajectory) else [],
            "control_segments": control_segments,
            "window_count": int(len(stream)),
            "step_sec": float(step_sec),
            "duration_sec": float(total_duration_sec),
        }


def build_replay_stream_from_trials(
    *,
    decoder: Any,
    trial_segments: Sequence[tuple[Any, np.ndarray]],
    step_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    timestamp_s = 0.0
    for trial, segment in trial_segments:
        expected_freq = None if getattr(trial, "expected_freq", None) is None else float(getattr(trial, "expected_freq"))
        label_text = str(getattr(trial, "label", ""))
        trial_role = infer_trial_role(label=label_text, expected_freq=expected_freq)
        feature_rows = decoder.iter_window_features(
            segment,
            expected_freq=expected_freq,
            label=label_text,
            trial_id=int(getattr(trial, "trial_id", -1)),
            block_index=int(getattr(trial, "block_index", -1)),
        )
        for row in feature_rows:
            item = dict(row)
            item["timestamp_s"] = float(timestamp_s)
            item["trial_role"] = str(trial_role)
            rows.append(item)
            labels.append(
                {
                    "role": str(trial_role),
                    "label": str(label_text),
                    "expected_freq": expected_freq,
                }
            )
            timestamp_s += float(step_sec)
    return rows, labels

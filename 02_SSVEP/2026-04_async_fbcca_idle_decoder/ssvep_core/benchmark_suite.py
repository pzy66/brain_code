from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .registry import ModelRegistry


@dataclass(frozen=True)
class BenchmarkReport:
    requested_models: tuple[str, ...]
    results: tuple[dict[str, Any], ...]
    ranking: tuple[str, ...]


def enrich_primary_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    payload = dict(metrics or {})
    false_trigger = float(payload.get("idle_false_trigger_per_min", payload.get("idle_fp_per_min", 0.0)))
    wrong_action = payload.get("wrong_action_rate")
    if wrong_action is None:
        control_precision = float(payload.get("control_precision", 1.0))
        wrong_action = max(0.0, 1.0 - control_precision)
    commit_latency = payload.get("median_commit_latency")
    if commit_latency is None:
        commit_latency = payload.get("median_detection_latency_sec", 0.0)
    return {
        "false_trigger_per_min": float(false_trigger),
        "wrong_action_rate": float(wrong_action),
        "median_commit_latency": float(commit_latency),
    }


def evaluate_models(
    *,
    model_names: Sequence[str],
    evaluate_fn: Callable[[str], dict[str, Any]],
) -> BenchmarkReport:
    requested = ModelRegistry.resolve_many(model_names, task="benchmark")
    results: list[dict[str, Any]] = []
    for model_name in requested:
        try:
            row = dict(evaluate_fn(model_name))
            row.setdefault("model_name", model_name)
            if isinstance(row.get("metrics"), dict):
                row["primary_metrics"] = enrich_primary_metrics(dict(row["metrics"]))
            results.append(row)
        except Exception as exc:  # pragma: no cover - safety path
            results.append({"model_name": model_name, "error": str(exc), "meets_acceptance": False})

    successful = [row for row in results if isinstance(row.get("primary_metrics"), dict)]
    successful.sort(
        key=lambda row: (
            float(dict(row["primary_metrics"]).get("false_trigger_per_min", float("inf"))),
            float(dict(row["primary_metrics"]).get("wrong_action_rate", float("inf"))),
            float(dict(row["primary_metrics"]).get("median_commit_latency", float("inf"))),
        )
    )
    ranking = tuple(str(item.get("model_name", "")) for item in successful)
    return BenchmarkReport(requested_models=requested, results=tuple(results), ranking=ranking)


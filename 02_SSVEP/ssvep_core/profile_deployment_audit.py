from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional


DEFAULT_FAST_CONTROL_PRETRAIN_TASK = "fast-control-pretrain-v1"
FAST_CONTROL_RELEASE_THRESHOLDS = {
    "idle_fp_per_min_max": 0.0,
    "control_recall_min": 0.95,
    "control_recall_at_3s_min": 0.80,
    "switch_detect_rate_min": 0.90,
    "release_latency_s_max": 2.0,
    "switch_latency_s_max": 2.5,
}
ASYNC_METRIC_KEYS = (
    "idle_fp_per_min",
    "idle_selected_windows_per_min",
    "control_recall",
    "control_recall_at_2s",
    "control_recall_at_2.5s",
    "control_recall_at_3s",
    "switch_detect_rate",
    "switch_detect_rate_at_2.8s",
    "release_detect_rate",
    "switch_latency_s",
    "release_latency_s",
    "detection_latency_s",
    "control_trials",
    "switch_trials",
    "release_trials",
)


def _safe_float(value: Any, default: float) -> float:
    try:
        output = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(output):
        return float(default)
    return float(output)


def _json_number(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def fast_control_release_failures(metrics: Mapping[str, Any]) -> list[str]:
    values = dict(metrics or {})
    thresholds = FAST_CONTROL_RELEASE_THRESHOLDS
    failures: list[str] = []
    if _safe_float(values.get("idle_fp_per_min"), float("inf")) > float(thresholds["idle_fp_per_min_max"]) + 1e-12:
        failures.append("idle_fp_per_min must be 0 for fast-control deployment")
    if _safe_float(values.get("control_recall"), 0.0) < float(thresholds["control_recall_min"]):
        failures.append(f"control_recall must be >= {thresholds['control_recall_min']:g}")
    if _safe_float(values.get("control_recall_at_3s"), 0.0) < float(thresholds["control_recall_at_3s_min"]):
        failures.append(f"control_recall_at_3s must be >= {thresholds['control_recall_at_3s_min']:g}")
    if _safe_float(values.get("switch_detect_rate"), 0.0) < float(thresholds["switch_detect_rate_min"]):
        failures.append(f"switch_detect_rate must be >= {thresholds['switch_detect_rate_min']:g}")
    if _safe_float(values.get("release_latency_s"), float("inf")) > float(thresholds["release_latency_s_max"]):
        failures.append(f"release_latency_s must be <= {thresholds['release_latency_s_max']:g}")
    if _safe_float(values.get("switch_latency_s"), float("inf")) > float(thresholds["switch_latency_s_max"]):
        failures.append(f"switch_latency_s must be <= {thresholds['switch_latency_s_max']:g}")
    return failures


def logreg_coefficients_all_zero(per_freq_gate: Mapping[str, Any], *, tol: float = 1e-12) -> bool:
    if not per_freq_gate:
        return True
    for payload in dict(per_freq_gate).values():
        if not isinstance(payload, Mapping):
            return False
        item = dict(payload)
        try:
            coef = [float(value) for value in item.get("coef", [])]
            intercept = float(item.get("intercept", 0.0))
        except Exception:
            return False
        if abs(intercept) > float(tol):
            return False
        if any(abs(value) > float(tol) for value in coef):
            return False
    return True


def effective_profile_v2_gate_type(gate_payload: Mapping[str, Any]) -> str:
    gate = dict(gate_payload or {})
    raw_type = str(gate.get("type", "")).strip().lower()
    if raw_type == "frequency_specific_logreg" and logreg_coefficients_all_zero(dict(gate.get("per_freq", {}) or {})):
        return "threshold_only_global_gate"
    return raw_type or "unknown"


def extract_async_metrics(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(profile_payload or {})
    candidates: list[Any] = [
        payload.get("benchmark_metrics"),
        dict(payload.get("metadata", {}) or {}).get("async_metrics"),
        dict(payload.get("profile_validation_status", {}) or {}).get("async_metrics"),
        dict(dict(payload.get("profile_validation_status", {}) or {}).get("chosen_candidate", {}) or {}).get(
            "async_metrics"
        ),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate:
            metrics = dict(candidate)
            output: dict[str, Any] = {}
            for key in ASYNC_METRIC_KEYS:
                if key in metrics:
                    output[key] = _json_number(metrics[key])
            for key in ("per_frequency_recall", "per_frequency_gate_pass_rate", "reference_headroom_p50"):
                if isinstance(metrics.get(key), Mapping):
                    output[key] = {str(child): _json_number(value) for child, value in dict(metrics[key]).items()}
            return output
    return {}


def _read_json_if_present(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        resolved = Path(path).expanduser().resolve()
    except Exception:
        return {}
    if not resolved.exists():
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def infer_profile_v2_path(profile_path: Path) -> Path:
    path = Path(profile_path).expanduser().resolve()
    return path.with_name(f"{path.stem}_v2.json")


def audit_profile_payload(
    profile_payload: Mapping[str, Any],
    *,
    profile_v2_payload: Optional[Mapping[str, Any]] = None,
    profile_index_payload: Optional[Mapping[str, Any]] = None,
    report_payload: Optional[Mapping[str, Any]] = None,
    profile_path: Optional[Path] = None,
    profile_v2_path: Optional[Path] = None,
) -> dict[str, Any]:
    payload = dict(profile_payload or {})
    profile_metadata = dict(payload.get("metadata", {}) or {})
    validation = dict(payload.get("profile_validation_status", {}) or {})
    metrics = extract_async_metrics(payload)
    warnings: list[str] = []
    if metrics:
        release_failures = fast_control_release_failures(metrics)
    else:
        release_failures = ["async benchmark metrics are missing; run fast-control threshold pretrain before deployment"]
    warnings.extend(release_failures)

    training_window_policy = str(payload.get("training_window_policy", "") or "").strip()
    if training_window_policy != DEFAULT_FAST_CONTROL_PRETRAIN_TASK:
        warnings.append(
            f"training_window_policy={training_window_policy or 'missing'} is not {DEFAULT_FAST_CONTROL_PRETRAIN_TASK}"
        )

    profile_recommended = payload.get("recommended_for_realtime")
    if isinstance(profile_recommended, bool) and profile_recommended and release_failures:
        warnings.append("profile is marked recommended_for_realtime but fails the current fast-control release gate")

    report = dict(report_payload or {})
    report_run_valid = report.get("run_valid_for_deployment")
    if isinstance(report_run_valid, bool) and report_run_valid is False:
        warnings.append("linked report has run_valid_for_deployment=false")

    v2_gate_type = ""
    v2_effective_gate_type = ""
    profile_v2_semantics_valid = True
    if profile_v2_payload:
        v2_payload = dict(profile_v2_payload)
        gate_payload = dict(v2_payload.get("gate", {}) or {})
        v2_gate_type = str(gate_payload.get("type", "") or "").strip()
        v2_effective_gate_type = effective_profile_v2_gate_type(gate_payload)
        if v2_gate_type.strip().lower() == "frequency_specific_logreg" and v2_effective_gate_type != "frequency_specific_logreg":
            profile_v2_semantics_valid = False
            warnings.append(
                "profile_v2 gate is labelled frequency_specific_logreg but has all-zero coefficients; "
                "runtime should treat it as threshold_only_global_gate"
            )

    release_gate_valid = not release_failures
    policy_valid = training_window_policy == DEFAULT_FAST_CONTROL_PRETRAIN_TASK
    run_valid_for_deployment = bool(release_gate_valid and policy_valid)
    profile_index = dict(profile_index_payload or {})
    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "target": DEFAULT_FAST_CONTROL_PRETRAIN_TASK,
        "status": "pass" if run_valid_for_deployment else "fail",
        "run_valid_for_deployment": bool(run_valid_for_deployment),
        "fast_control_release_valid": bool(release_gate_valid),
        "policy_valid": bool(policy_valid),
        "profile_v2_semantics_valid": bool(profile_v2_semantics_valid),
        "release_failures": list(release_failures),
        "warnings": list(dict.fromkeys(warnings)),
        "metrics": metrics,
        "release_thresholds": dict(FAST_CONTROL_RELEASE_THRESHOLDS),
        "profile_path": "" if profile_path is None else str(Path(profile_path).expanduser().resolve()),
        "profile_v2_path": "" if profile_v2_path is None else str(Path(profile_v2_path).expanduser().resolve()),
        "model_name": str(payload.get("model_name", "") or ""),
        "win_sec": _safe_float(payload.get("win_sec"), float("nan")),
        "step_sec": _safe_float(payload.get("step_sec"), float("nan")),
        "gate_policy": str(payload.get("gate_policy", "") or ""),
        "training_window_policy": training_window_policy,
        "metadata_source": str(profile_metadata.get("source", "") or ""),
        "recommended_for_realtime": profile_recommended if isinstance(profile_recommended, bool) else None,
        "profile_validation_passed": validation.get("passed") if isinstance(validation.get("passed"), bool) else None,
        "profile_v2_gate_type": v2_gate_type,
        "profile_v2_effective_gate_type": v2_effective_gate_type,
        "profile_index": {
            "updated_at": str(profile_index.get("updated_at", "") or ""),
            "task": str(profile_index.get("task", "") or ""),
            "run_tag": str(profile_index.get("run_tag", "") or ""),
            "report_json": str(profile_index.get("report_json", "") or ""),
        },
        "report_run_valid_for_deployment": report_run_valid if isinstance(report_run_valid, bool) else None,
    }


def audit_profile_files(
    profile_path: Path,
    *,
    profile_v2_path: Optional[Path] = None,
    profile_index_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> dict[str, Any]:
    profile = Path(profile_path).expanduser().resolve()
    if not profile.exists():
        raise FileNotFoundError(f"profile not found: {profile}")
    profile_payload = json.loads(profile.read_text(encoding="utf-8-sig"))
    if not isinstance(profile_payload, Mapping):
        raise ValueError(f"profile payload is not a JSON object: {profile}")
    v2_path = Path(profile_v2_path).expanduser().resolve() if profile_v2_path is not None else infer_profile_v2_path(profile)
    index_path = (
        Path(profile_index_path).expanduser().resolve()
        if profile_index_path is not None
        else profile.parent / "profile_index.json"
    )
    profile_index_payload = _read_json_if_present(index_path)
    resolved_report_path = report_path
    if resolved_report_path is None and profile_index_payload.get("report_json"):
        resolved_report_path = Path(str(profile_index_payload.get("report_json")))
    return audit_profile_payload(
        dict(profile_payload),
        profile_v2_payload=_read_json_if_present(v2_path),
        profile_index_payload=profile_index_payload,
        report_payload=_read_json_if_present(resolved_report_path),
        profile_path=profile,
        profile_v2_path=v2_path,
    )


def format_profile_audit_summary(audit: Mapping[str, Any], *, max_warnings: int = 3) -> str:
    payload = dict(audit or {})
    metrics = dict(payload.get("metrics", {}) or {})
    warnings = [str(item) for item in payload.get("warnings", [])]
    head = (
        f"profile_audit={str(payload.get('status', 'unknown')).upper()} "
        f"target={payload.get('target', DEFAULT_FAST_CONTROL_PRETRAIN_TASK)} "
        f"policy={payload.get('training_window_policy', 'missing')} "
        f"win_sec={payload.get('win_sec', 'n/a')}"
    )
    metric_text = (
        "metrics: "
        f"idle_fp={metrics.get('idle_fp_per_min', 'n/a')} "
        f"recall={metrics.get('control_recall', 'n/a')} "
        f"recall3={metrics.get('control_recall_at_3s', 'n/a')} "
        f"switch={metrics.get('switch_latency_s', 'n/a')}s "
        f"release={metrics.get('release_latency_s', 'n/a')}s"
    )
    if not warnings:
        return f"{head} | {metric_text}"
    preview = "; ".join(warnings[: max(1, int(max_warnings))])
    if len(warnings) > int(max_warnings):
        preview += f"; +{len(warnings) - int(max_warnings)} more"
    return f"{head} | {metric_text} | warnings={len(warnings)}: {preview}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an SSVEP realtime profile against fast-control deployment gates.")
    parser.add_argument("profile", type=Path, help="Path to the profile JSON to audit.")
    parser.add_argument("--profile-v2", type=Path, default=None, help="Optional explicit profile_v2 JSON path.")
    parser.add_argument("--profile-index", type=Path, default=None, help="Optional deployed profile_index.json path.")
    parser.add_argument("--report", type=Path, default=None, help="Optional linked report.json path.")
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable audit payload.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_profile_files(
        args.profile,
        profile_v2_path=args.profile_v2,
        profile_index_path=args.profile_index,
        report_path=args.report,
    )
    if bool(args.json):
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print(format_profile_audit_summary(audit))
        for warning in audit.get("warnings", []):
            print(f"- {warning}")
    return 0 if bool(audit.get("run_valid_for_deployment", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())

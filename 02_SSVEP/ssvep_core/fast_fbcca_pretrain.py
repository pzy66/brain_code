from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from brain_workspace.paths import SSVEP_PROFILE_DIR

from .async_fbcca_idle_standalone import (
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_NH,
    ThresholdProfile,
    TrialSpec,
    atomic_write_text,
    build_feature_rows_with_decoder,
    default_profile,
    evaluate_profile_on_feature_rows,
    fit_threshold_profile,
    json_dumps,
    json_safe,
    load_decoder_from_profile,
    load_profile,
    normalize_model_name,
    save_profile,
    summarize_profile_quality,
)
from .fbcca_base_profile_opt import DEFAULT_FBCCA_BASE_PROFILE_PATH, EXPECTED_FBCCA_BASE_FREQS
from .profile_v2 import DEFAULT_GATE_FEATURES, PROFILE_V2_VERSION


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FAST_FBCCA_SESSION_PROFILE_PATH = SSVEP_PROFILE_DIR / "fbcca_profile.json"
DEFAULT_FAST_FBCCA_HISTORY_DIR = SSVEP_PROFILE_DIR / "pretrain_history"
DEFAULT_FAST_FBCCA_TARGET_REPEATS = 2
DEFAULT_FAST_FBCCA_IDLE_REPEATS = 4
DEFAULT_FAST_FBCCA_PREPARE_SEC = 0.75
DEFAULT_FAST_FBCCA_ACTIVE_SEC = 3.0
DEFAULT_FAST_FBCCA_REST_SEC = 0.5
DEFAULT_FAST_FBCCA_WIN_SEC = 2.0
DEFAULT_FAST_FBCCA_STEP_SEC = 0.25
DEFAULT_FAST_FBCCA_TEMPLATE_WEIGHT = 0.25
DEFAULT_FAST_FBCCA_TEMPLATE_WIN_SEC = 2.0
MAX_FAST_FBCCA_TEMPLATE_WEIGHT = 0.4


@dataclass(frozen=True)
class FastFBCCAPretrainConfig:
    base_profile_path: Path = DEFAULT_FBCCA_BASE_PROFILE_PATH
    fallback_profile_path: Path = DEFAULT_FAST_FBCCA_SESSION_PROFILE_PATH
    output_profile_path: Path = DEFAULT_FAST_FBCCA_SESSION_PROFILE_PATH
    history_profile_path: Optional[Path] = None
    freqs: tuple[float, float, float, float] = EXPECTED_FBCCA_BASE_FREQS
    win_sec: float = DEFAULT_FAST_FBCCA_WIN_SEC
    step_sec: float = DEFAULT_FAST_FBCCA_STEP_SEC
    template_weight: float = DEFAULT_FAST_FBCCA_TEMPLATE_WEIGHT
    template_win_sec: float = DEFAULT_FAST_FBCCA_TEMPLATE_WIN_SEC
    seed: int = DEFAULT_CALIBRATION_SEED
    compute_backend: str = "cpu"
    gpu_device: int = DEFAULT_GPU_DEVICE_ID
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME
    gpu_warmup: bool = False
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE
    fallback_to_base_on_low_quality: bool = True


def fast_fbcca_trial_count(*, target_repeats: int, idle_repeats: int) -> int:
    return 4 * int(target_repeats) + int(idle_repeats)


def fast_fbcca_estimated_collection_seconds(
    *,
    target_repeats: int = DEFAULT_FAST_FBCCA_TARGET_REPEATS,
    idle_repeats: int = DEFAULT_FAST_FBCCA_IDLE_REPEATS,
    prepare_sec: float = DEFAULT_FAST_FBCCA_PREPARE_SEC,
    active_sec: float = DEFAULT_FAST_FBCCA_ACTIVE_SEC,
    rest_sec: float = DEFAULT_FAST_FBCCA_REST_SEC,
) -> float:
    trial_sec = float(prepare_sec) + float(active_sec) + float(rest_sec)
    return float(fast_fbcca_trial_count(target_repeats=target_repeats, idle_repeats=idle_repeats)) * trial_sec


def build_fast_fbcca_history_profile_path(*, timestamp: float | None = None) -> Path:
    stamp_source = datetime.fromtimestamp(timestamp) if timestamp is not None else datetime.now()
    stamp = stamp_source.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_FAST_FBCCA_HISTORY_DIR / f"ssvep_fast_fbcca_session_profile_{stamp}.json"


def _freq_tuple(values: Sequence[float]) -> tuple[float, float, float, float]:
    freqs = tuple(float(item) for item in values)
    if len(freqs) != 4:
        raise ValueError("FBCCA fast pretrain requires exactly 4 frequencies")
    return freqs  # type: ignore[return-value]


def _validate_demo_freqs(freqs: Sequence[float]) -> tuple[float, float, float, float]:
    return _freq_tuple(freqs)


def load_fast_fbcca_base_profile(config: FastFBCCAPretrainConfig) -> tuple[ThresholdProfile, str, str]:
    freqs = _validate_demo_freqs(config.freqs)
    candidates = (
        ("fbcca_base_profile", Path(config.base_profile_path)),
        ("fbcca_profile", Path(config.fallback_profile_path)),
    )
    for label, path in candidates:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            continue
        profile = load_profile(resolved, fallback_freqs=freqs, require_exists=True)
        try:
            _validate_fast_base_profile(profile, source=str(resolved), freqs=freqs)
        except ValueError:
            continue
        return profile, str(resolved), label
    profile = default_profile(freqs)
    _validate_fast_base_profile(profile, source="default_fbcca_profile", freqs=freqs)
    return profile, "default_fbcca_profile", "default"


def _validate_fast_base_profile(profile: ThresholdProfile, *, source: str, freqs: Sequence[float]) -> None:
    model_name = normalize_model_name(str(profile.model_name))
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"fast pretrain base profile must use model_name='fbcca'; got {model_name} from {source}")
    expected = _freq_tuple(freqs)
    actual = _freq_tuple(profile.freqs)
    if actual != expected:
        raise ValueError(f"fast pretrain base profile freqs mismatch: {source} has {actual}, expected {expected}")


def _selected_channel_positions(
    *,
    profile: ThresholdProfile,
    available_board_channels: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    available = tuple(int(item) for item in available_board_channels)
    if not available:
        raise ValueError("available_board_channels must not be empty")
    selected = tuple(int(item) for item in (profile.eeg_channels or available))
    positions: list[int] = []
    missing: list[int] = []
    for channel in selected:
        try:
            positions.append(available.index(int(channel)))
        except ValueError:
            missing.append(int(channel))
    if missing:
        raise ValueError(f"base profile eeg_channels not present on current board: {missing}; available={list(available)}")
    return selected, tuple(positions)


def _subset_segments_by_positions(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    positions: Sequence[int],
) -> list[tuple[TrialSpec, np.ndarray]]:
    selected: list[tuple[TrialSpec, np.ndarray]] = []
    for trial, segment in trial_segments:
        matrix = np.asarray(segment, dtype=np.float64)
        if matrix.ndim != 2:
            continue
        if matrix.shape[1] <= max(int(pos) for pos in positions):
            continue
        selected.append((trial, np.ascontiguousarray(matrix[:, list(positions)], dtype=np.float64)))
    return selected


def _trial_counts(trial_segments: Sequence[tuple[TrialSpec, np.ndarray]], freqs: Sequence[float]) -> dict[str, Any]:
    per_freq = {f"{float(freq):g}": 0 for freq in freqs}
    idle = 0
    for trial, segment in trial_segments:
        matrix = np.asarray(segment)
        if matrix.ndim != 2 or matrix.shape[0] <= 0:
            continue
        if trial.expected_freq is None:
            idle += 1
            continue
        key = f"{float(trial.expected_freq):g}"
        if key in per_freq:
            per_freq[key] += 1
    return {
        "per_freq": per_freq,
        "idle": int(idle),
        "control": int(sum(per_freq.values())),
        "total": int(sum(per_freq.values()) + idle),
    }


def _strip_fast_personalization(model_params: Optional[dict[str, Any]]) -> dict[str, Any]:
    params = dict(model_params or {})
    params.pop("fast_personalization", None)
    return params


def _profile_for_fast_window(profile: ThresholdProfile, config: FastFBCCAPretrainConfig) -> ThresholdProfile:
    model_params = _strip_fast_personalization(profile.model_params)
    model_params.setdefault("Nh", DEFAULT_NH)
    return replace(
        profile,
        model_name=DEFAULT_MODEL_NAME,
        model_params=json_safe(model_params),
        win_sec=float(config.win_sec),
        step_sec=float(config.step_sec),
    )


def _build_template_payload(
    *,
    decoder: Any,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    freqs: Sequence[float],
    sampling_rate: int,
    template_win_sec: float,
    template_weight: float,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    win_samples = int(round(float(template_win_sec) * float(sampling_rate)))
    if win_samples <= 0:
        raise ValueError("template_win_sec produced no samples")
    if int(getattr(decoder, "win_samples", win_samples)) != int(win_samples):
        return None, {
            "enabled": False,
            "warning": "template_win_sec does not match decoder win_sec",
            "template_win_samples": int(win_samples),
            "decoder_win_samples": int(getattr(decoder, "win_samples", 0) or 0),
        }
    grouped: dict[str, list[np.ndarray]] = {f"{float(freq):g}": [] for freq in freqs}
    for trial, segment in trial_segments:
        if trial.expected_freq is None:
            continue
        key = f"{float(trial.expected_freq):g}"
        if key not in grouped:
            continue
        matrix = np.asarray(segment, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < win_samples:
            continue
        window = np.ascontiguousarray(matrix[-win_samples:, :], dtype=np.float64)
        try:
            front = decoder._apply_frontend(window) if hasattr(decoder, "_apply_frontend") else window
            if hasattr(decoder, "engine") and hasattr(decoder.engine, "preprocess_window"):
                prepared = decoder.engine.preprocess_window(front)
            else:
                prepared = front - np.mean(front, axis=0, keepdims=True)
        except Exception:
            continue
        prepared = np.ascontiguousarray(np.asarray(prepared, dtype=np.float64))
        if prepared.ndim == 2 and prepared.shape[0] == win_samples and np.all(np.isfinite(prepared)):
            grouped[key].append(prepared)
    missing = [key for key, values in grouped.items() if not values]
    if missing:
        return None, {
            "enabled": False,
            "warning": f"missing templates for frequencies: {missing}",
            "missing_freqs": missing,
            "per_freq_counts": {key: len(values) for key, values in grouped.items()},
        }
    templates = {
        key: np.ascontiguousarray(np.mean(np.stack(values, axis=0), axis=0), dtype=np.float64)
        for key, values in grouped.items()
    }
    weight = min(max(float(template_weight), 0.0), MAX_FAST_FBCCA_TEMPLATE_WEIGHT)
    payload = {
        "version": 1,
        "template_weight": float(weight),
        "template_win_sec": float(template_win_sec),
        "template_format": "preprocessed_frontend_window_v1",
        "sampling_rate": int(sampling_rate),
        "templates": {key: matrix.tolist() for key, matrix in templates.items()},
    }
    summary = {
        "enabled": True,
        "template_weight": float(weight),
        "template_win_samples": int(win_samples),
        "per_freq_counts": {key: len(values) for key, values in grouped.items()},
        "template_shapes": {key: [int(dim) for dim in matrix.shape] for key, matrix in templates.items()},
    }
    return payload, summary


def _profile_v2_payload(profile: ThresholdProfile, metrics: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(profile.metadata or {})
    metadata.setdefault("profile_v2_note", "fast_pretrain_uses_profile_global_gate")
    return {
        "version": PROFILE_V2_VERSION,
        "freqs": [float(item) for item in profile.freqs],
        "decoder": {
            "name": str(profile.model_name),
            "params": dict(profile.model_params or {}),
            "channels": [int(item) for item in (profile.eeg_channels or ())],
            "templates_path": "",
        },
        "gate": {
            "type": "global_threshold",
            "feature_names": [str(item) for item in DEFAULT_GATE_FEATURES],
            "per_freq": {},
        },
        "evidence": {
            "lambda": 0.85,
            "beta_consistency": 0.5,
            "upper_commit_th": 2.2,
            "lower_idle_th": 0.4,
        },
        "runtime": {
            "win_sec": float(profile.win_sec),
            "step_sec": float(profile.step_sec),
            "refractory_sec": 0.8,
        },
        "metrics": dict(metrics),
        "metadata": metadata,
    }


def _load_fast_pretrain_decoder(
    profile: ThresholdProfile,
    config: FastFBCCAPretrainConfig,
    *,
    sampling_rate: int,
) -> Any:
    return load_decoder_from_profile(
        profile,
        sampling_rate=int(sampling_rate),
        compute_backend=str(config.compute_backend),
        gpu_device=int(config.gpu_device),
        gpu_precision=str(config.gpu_precision),
        gpu_warmup=bool(config.gpu_warmup),
        gpu_cache_policy=str(config.gpu_cache_policy),
    )


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            output[str(key)] = float(value)
    return output


def _should_fallback_to_base(*, quality: dict[str, Any], gate_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    raw_accuracy = float(quality.get("raw_accuracy", 0.0) or 0.0)
    control_recall = float(gate_metrics.get("control_recall", 0.0) or 0.0)
    idle_fp = float(gate_metrics.get("idle_fp_per_min", 0.0) or 0.0)
    if raw_accuracy < 0.25:
        reasons.append("raw 4-class accuracy below chance-level guidance")
    if control_recall <= 0.0:
        reasons.append("control recall is zero")
    if idle_fp > 30.0:
        reasons.append("idle false positive rate is too high")
    return bool(reasons), reasons


def save_fast_fbcca_profile_bundle(
    profile: ThresholdProfile,
    path: Path,
    metrics: Optional[dict[str, Any]] = None,
) -> tuple[Path, Path]:
    resolved_path = Path(path).expanduser().resolve()
    save_profile(profile, resolved_path)
    v2_path = resolved_path.with_name(f"{resolved_path.stem}_v2.json")
    atomic_write_text(v2_path, json_dumps(json_safe(_profile_v2_payload(profile, dict(metrics or {})))) + "\n")
    return resolved_path, v2_path


def run_fast_fbcca_personalization(
    config: FastFBCCAPretrainConfig,
    *,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    sampling_rate: int,
    available_board_channels: Sequence[int],
    collection_duration_sec: float = 0.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[ThresholdProfile, dict[str, Any]]:
    freqs = _validate_demo_freqs(config.freqs)
    base_profile, source_profile, source_kind = load_fast_fbcca_base_profile(config)
    selected_channels, positions = _selected_channel_positions(
        profile=base_profile,
        available_board_channels=available_board_channels,
    )
    selected_segments = _subset_segments_by_positions(trial_segments, positions)
    counts = _trial_counts(selected_segments, freqs)
    missing_freqs = [key for key, value in dict(counts["per_freq"]).items() if int(value) <= 0]
    if missing_freqs:
        raise ValueError(f"fast pretrain requires at least one valid trial per frequency; missing={missing_freqs}")
    if not selected_segments:
        raise ValueError("fast pretrain has no usable trial segments")

    if log_fn is not None:
        log_fn(f"fast FBCCA personalization | source={source_profile} | trials={counts['total']}")

    fast_base = _profile_for_fast_window(base_profile, config)
    decoder = _load_fast_pretrain_decoder(fast_base, config, sampling_rate=int(sampling_rate))
    base_feature_rows = build_feature_rows_with_decoder(decoder, selected_segments)
    if not base_feature_rows:
        raise RuntimeError("fast pretrain produced no FBCCA feature rows")

    template_payload: Optional[dict[str, Any]] = None
    template_summary: dict[str, Any] = {}
    try:
        template_payload, template_summary = _build_template_payload(
            decoder=decoder,
            trial_segments=selected_segments,
            freqs=freqs,
            sampling_rate=int(sampling_rate),
            template_win_sec=float(config.template_win_sec),
            template_weight=float(config.template_weight),
        )
    except Exception as exc:
        template_summary = {"enabled": False, "warning": str(exc)}

    template_enabled = template_payload is not None and bool(template_summary.get("enabled", False))
    feature_rows = list(base_feature_rows)
    gate_feature_source = "fbcca"
    if template_enabled:
        try:
            personalized_params = _strip_fast_personalization(fast_base.model_params)
            personalized_params["fast_personalization"] = template_payload
            personalized_profile_for_rows = replace(
                fast_base,
                model_params=json_safe(personalized_params),
            )
            personalized_decoder = _load_fast_pretrain_decoder(
                personalized_profile_for_rows,
                config,
                sampling_rate=int(sampling_rate),
            )
            personalized_rows = build_feature_rows_with_decoder(personalized_decoder, selected_segments)
            if not personalized_rows:
                raise RuntimeError("personalized decoder produced no feature rows")
            feature_rows = personalized_rows
            gate_feature_source = "fbcca_template_fused"
        except Exception as exc:
            template_payload = None
            template_enabled = False
            template_summary = {
                **dict(template_summary),
                "enabled": False,
                "warning": f"template feature rebuild failed: {exc}",
            }

    gate_profile: Optional[ThresholdProfile] = None
    gate_quality: dict[str, Any] = {}
    gate_metrics: dict[str, Any] = {}
    gate_error = ""
    try:
        gate_profile = fit_threshold_profile(
            feature_rows,
            freqs=freqs,
            win_sec=float(config.win_sec),
            step_sec=float(config.step_sec),
            min_enter_windows=max(1, min(int(fast_base.min_enter_windows), 2)),
            min_exit_windows=max(1, min(int(fast_base.min_exit_windows), 2)),
            gate_policy=str(fast_base.gate_policy),
            evaluation_rows=feature_rows,
            dynamic_stop_enabled=False,
            control_state_mode=str(fast_base.control_state_mode),
        )
        gate_quality = summarize_profile_quality(feature_rows, gate_profile)
        gate_metrics = evaluate_profile_on_feature_rows(feature_rows, gate_profile)
    except Exception as exc:
        gate_error = str(exc)

    gate_enabled = gate_profile is not None
    if not template_enabled and not gate_enabled:
        raise RuntimeError(f"fast pretrain failed to build templates and gate calibration failed: {gate_error}")

    fallback_candidate, fallback_reasons = (False, [])
    fallback = False
    if gate_enabled:
        fallback_candidate, fallback_reasons = _should_fallback_to_base(quality=gate_quality, gate_metrics=gate_metrics)
        fallback = bool(config.fallback_to_base_on_low_quality) and bool(fallback_candidate)

    if fallback:
        final_metrics = _finite_metrics(gate_metrics)
        model_params = _strip_fast_personalization(base_profile.model_params)
        metadata = dict(base_profile.metadata or {})
        metadata["fast_pretrain"] = {
            "status": "fallback_to_base",
            "source_profile": source_profile,
            "source_profile_kind": source_kind,
            "collection_duration_sec": float(collection_duration_sec),
            "trial_counts": counts,
            "quality_metrics": _finite_metrics(gate_metrics),
            "quality_summary": _finite_metrics(gate_quality),
            "template_enabled": False,
            "gate_calibration_enabled": False,
            "gate_feature_source": gate_feature_source,
            "fallback_reasons": fallback_reasons,
            "fallback_to_base_on_low_quality": bool(config.fallback_to_base_on_low_quality),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        final_profile = replace(
            fast_base,
            model_name=DEFAULT_MODEL_NAME,
            model_params=json_safe(model_params),
            eeg_channels=selected_channels,
            calibration_split_seed=int(config.seed),
            benchmark_metrics=final_metrics,
            metadata=metadata,
        )
    else:
        source_for_thresholds = gate_profile if gate_profile is not None else fast_base
        model_params = _strip_fast_personalization(fast_base.model_params)
        if template_payload is not None:
            model_params["fast_personalization"] = template_payload
        metadata = dict(fast_base.metadata or {})
        metadata["fast_pretrain"] = {
            "status": "ok",
            "source_profile": source_profile,
            "source_profile_kind": source_kind,
            "collection_duration_sec": float(collection_duration_sec),
            "trial_counts": counts,
            "quality_metrics": _finite_metrics(gate_metrics),
            "quality_summary": _finite_metrics(gate_quality),
            "template_summary": template_summary,
            "template_enabled": bool(template_enabled),
            "gate_calibration_enabled": bool(gate_enabled),
            "gate_feature_source": gate_feature_source,
            "gate_calibration_error": gate_error,
            "fallback_to_base_on_low_quality": bool(config.fallback_to_base_on_low_quality),
            "release_fallback_candidate": bool(fallback_candidate),
            "release_fallback_reasons": list(fallback_reasons),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        final_profile = replace(
            source_for_thresholds,
            model_name=DEFAULT_MODEL_NAME,
            model_params=json_safe(model_params),
            eeg_channels=selected_channels,
            calibration_split_seed=int(config.seed),
            benchmark_metrics=_finite_metrics(gate_metrics),
            metadata=metadata,
            recommended_for_realtime=not bool(fallback_candidate),
        )
        final_metrics = _finite_metrics(gate_metrics)

    output_path = Path(config.output_profile_path).expanduser().resolve()
    save_fast_fbcca_profile_bundle(final_profile, output_path, final_metrics)
    history_path = Path(config.history_profile_path).expanduser().resolve() if config.history_profile_path is not None else None
    if history_path is not None:
        save_fast_fbcca_profile_bundle(final_profile, history_path, final_metrics)

    result = {
        "task": "fast-fbcca-pretrain",
        "status": dict((final_profile.metadata or {}).get("fast_pretrain", {})).get("status", "ok"),
        "profile_path": str(output_path),
        "profile_v2_path": str(output_path.with_name(f"{output_path.stem}_v2.json")),
        "history_profile_path": "" if history_path is None else str(history_path),
        "history_profile_v2_path": "" if history_path is None else str(history_path.with_name(f"{history_path.stem}_v2.json")),
        "model_name": str(final_profile.model_name),
        "source_profile": source_profile,
        "source_profile_kind": source_kind,
        "selected_eeg_channels": [int(item) for item in selected_channels],
        "trial_counts": counts,
        "template_enabled": bool(template_enabled and not fallback),
        "gate_calibration_enabled": bool(gate_enabled and not fallback),
        "gate_feature_source": gate_feature_source,
        "quality_metrics": _finite_metrics(gate_metrics),
        "quality_summary": _finite_metrics(gate_quality),
        "fallback_reasons": list(fallback_reasons) if fallback else [],
        "fallback_to_base_on_low_quality": bool(config.fallback_to_base_on_low_quality),
        "release_fallback_candidate": bool(fallback_candidate),
        "release_fallback_reasons": list(fallback_reasons),
        "recommended_for_realtime": bool(final_profile.recommended_for_realtime),
        "config": asdict(config),
    }
    return final_profile, json_safe(result)

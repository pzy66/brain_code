from __future__ import annotations

import hashlib
import csv
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .async_fbcca_idle_standalone import (
    DEFAULT_CALIBRATION_SEED,
    TrialSpec,
    build_benchmark_eval_trials,
    json_dumps,
)
from .trial_roles import (
    TRIAL_ROLE_CLEAN_IDLE,
    TRIAL_ROLE_CONTROL,
    TRIAL_ROLE_HARD_IDLE,
    infer_trial_role,
    resolve_trial_role,
)

COLLECTION_DATA_SCHEMA_VERSION = "2.0"
QUALITY_ROW_PASSTHROUGH_KEYS = (
    "protocol_name",
    "block_id",
    "split_role",
    "state_type",
    "nc_subtype",
    "target_freq",
    "target_position",
    "target_position_label",
    "stimulus_active",
    "all_targets_flickering",
    "cue_sec",
    "rest_sec",
    "trial_start_sample",
    "cue_on_sample",
    "active_window_start_sample",
    "active_window_end_sample",
    "stim_on_sample",
    "stim_off_sample",
    "trial_end_sample",
    "valid",
    "reject_reason",
    "event_codes",
    "events",
)
COLLECTION_TIMESTAMP_PASSTHROUGH_KEYS = (
    "trial_prepare_started_at",
    "active_start_tone_started_at",
    "active_window_started_at",
    "active_window_ended_at",
    "segment_captured_at",
    "active_end_tone_started_at",
    "stimulus_phase_apply_requested_at",
    "stimulus_first_frame_presented_at",
    "stimulus_first_frame_ack_received_at",
    "stimulus_first_frame_mode",
    "board_buffer_cleared_at",
)
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def sanitize_collection_token(value: str | Any, *, default: str = "session") -> str:
    """Return one portable path segment for collection subject/session names."""
    token = str(value if value is not None else "").strip()
    token = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", token)
    token = re.sub(r"\s+", "_", token)
    token = token.strip("._ ")
    if not token:
        token = str(default or "session").strip() or "session"
    if token.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        token = f"{token}_"
    return token


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class CollectionProtocol:
    name: str
    prepare_sec: float
    active_sec: float
    rest_sec: float
    target_repeats: int
    idle_repeats: int
    switch_trials: int
    long_idle_sec: float = 0.0


ENHANCED_45M_PROTOCOL = CollectionProtocol(
    name="enhanced_45m",
    prepare_sec=1.0,
    active_sec=4.0,
    rest_sec=1.0,
    target_repeats=24,
    idle_repeats=48,
    switch_trials=32,
    long_idle_sec=0.0,
)


def summarize_trial_roles(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    summary = {
        TRIAL_ROLE_CONTROL: 0,
        TRIAL_ROLE_CLEAN_IDLE: 0,
        TRIAL_ROLE_HARD_IDLE: 0,
    }
    for row in records:
        role = resolve_trial_role(row)
        summary[role] = int(summary.get(role, 0) + 1)
    return summary


def _now_iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _replace_with_windows_retry(source: Path, target: Path) -> None:
    source_text = str(source)
    target_text = str(target)
    replace_error: OSError | None = None
    for delay_sec in (0.0, 0.02, 0.05, 0.1, 0.2):
        if delay_sec > 0.0:
            time.sleep(delay_sec)
        try:
            os.replace(source_text, target_text)
            replace_error = None
            break
        except PermissionError as error:
            replace_error = error
            if os.name != "nt":
                raise
        except OSError as error:
            replace_error = error
            if os.name != "nt" or getattr(error, "winerror", None) != 5:
                raise
    if replace_error is not None:
        raise replace_error


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".atomic_{uuid.uuid4().hex[:8]}.tmp"
    try:
        tmp_path.write_text(str(text), encoding=encoding)
        _replace_with_windows_retry(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".atomic_{uuid.uuid4().hex[:8]}.csv"
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(_jsonable(row.get(key)), ensure_ascii=False)
                        if isinstance(row.get(key), (dict, list, tuple))
                        else row.get(key)
                        for key in fieldnames
                    }
                )
        _replace_with_windows_retry(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".atomic_{uuid.uuid4().hex[:8]}.npz"
    try:
        np.savez_compressed(tmp_path, **arrays)
        _replace_with_windows_retry(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".atomic_{uuid.uuid4().hex[:8]}.npy"
    try:
        with tmp_path.open("wb") as file:
            np.save(file, array, allow_pickle=False)
        _replace_with_windows_retry(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class LoadedDataset:
    manifest_path: Path
    npz_path: Path
    session_id: str
    subject_id: str
    sampling_rate: int
    freqs: tuple[float, float, float, float]
    board_eeg_channels: tuple[int, ...]
    protocol_config: dict[str, Any]
    trial_segments: list[tuple[TrialSpec, np.ndarray]]
    manifest: dict[str, Any]


def build_collection_trials(
    freqs: Sequence[float],
    *,
    protocol: CollectionProtocol = ENHANCED_45M_PROTOCOL,
    seed: int = DEFAULT_CALIBRATION_SEED,
    session_index: int = 1,
) -> list[TrialSpec]:
    trials = build_benchmark_eval_trials(
        freqs,
        target_repeats=int(protocol.target_repeats),
        idle_repeats=int(protocol.idle_repeats),
        switch_trials=int(protocol.switch_trials),
        seed=int(seed) + int(max(session_index, 1) - 1) * 1009,
    )
    long_idle_sec = float(getattr(protocol, "long_idle_sec", 0.0) or 0.0)
    if long_idle_sec > 0.0:
        next_trial_id = max((trial.trial_id for trial in trials), default=-1) + 1
        next_block_index = max((trial.block_index for trial in trials), default=-1) + 1
        trials.append(
            TrialSpec(
                label="long_idle",
                expected_freq=None,
                trial_id=next_trial_id,
                block_index=next_block_index,
            )
        )
    return trials


def _build_collection_records(
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    npz_arrays: dict[str, np.ndarray],
    default_target_samples: int,
    sampling_rate: int,
    quality_rows: Optional[Sequence[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    quality_by_order: dict[int, dict[str, Any]] = {}
    if quality_rows is not None:
        for entry in quality_rows:
            if not isinstance(entry, dict):
                continue
            try:
                order_index = int(entry.get("order_index", -1))
            except Exception:
                continue
            if order_index >= 0:
                quality_by_order[order_index] = dict(entry)
    for order_index, (trial, segment) in enumerate(segments):
        base_key = f"trial_collection_{int(trial.trial_id)}"
        npz_key = base_key
        suffix = 1
        while npz_key in npz_arrays:
            npz_key = f"{base_key}_{suffix}"
            suffix += 1
        matrix = np.ascontiguousarray(np.asarray(segment, dtype=np.float32))
        npz_arrays[npz_key] = matrix
        quality = quality_by_order.get(int(order_index), {})
        if quality:
            effective_target = int(quality.get("target_samples", default_target_samples))
        else:
            # Without per-trial quality metadata, treat the saved segment itself as
            # the nominal target so heterogeneous trial lengths remain readable.
            effective_target = int(matrix.shape[0])
        effective_target = max(1, int(effective_target))
        used_samples = int(matrix.shape[0])
        retry_count = max(0, int(quality.get("retry_count", 0)))
        shortfall_ratio = float(max(effective_target - used_samples, 0) / effective_target)
        sample_ratio = _safe_float(quality.get("sample_ratio", used_samples / max(effective_target, 1)), 0.0)
        label_text = str(trial.label)
        label_lower = label_text.strip().lower()
        stage_name = "long_idle" if ("long_idle" in label_lower or "long idle" in label_lower) else "collection"
        trial_role = str(quality.get("trial_role", "")).strip().lower() or infer_trial_role(
            label=label_text,
            expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
        )
        stage_name = str(quality.get("stage", "")).strip() or stage_name
        record = {
            "stage": stage_name,
            "trial_role": trial_role,
            "label": label_text,
            "expected_freq": None if trial.expected_freq is None else float(trial.expected_freq),
            "trial_id": int(trial.trial_id),
            "block_index": int(trial.block_index),
            "order_index": int(order_index),
            "used_samples": used_samples,
            "target_samples": effective_target,
            "sample_ratio": float(sample_ratio),
            "shortfall_ratio": shortfall_ratio,
            "retry_count": retry_count,
            "channels": int(matrix.shape[1]),
            "npz_key": npz_key,
        }
        for key in QUALITY_ROW_PASSTHROUGH_KEYS:
            if key in quality:
                record[key] = _jsonable(quality.get(key))
        if "active_sec" in quality:
            record["active_sec"] = _safe_float(quality.get("active_sec", 0.0), 0.0)
        elif int(sampling_rate) > 0:
            record["active_sec"] = float(used_samples / float(sampling_rate))
        if "available_samples" in quality:
            record["available_samples"] = _safe_int(quality.get("available_samples", 0), 0)
        for key in COLLECTION_TIMESTAMP_PASSTHROUGH_KEYS:
            value = str(quality.get(key, "")).strip()
            if value:
                record[key] = value
        for key in (
            "stimulus_first_frame_presented_t_sec",
            "stimulus_first_frame_cue_freq",
            "stimulus_first_frame_ack_latency_sec",
        ):
            if key in quality and quality.get(key) is not None:
                record[key] = _safe_float(quality.get(key), 0.0)
        for key in ("stimulus_first_frame_frame_index", "board_buffer_clear_samples"):
            if key in quality and quality.get(key) is not None:
                record[key] = _safe_int(quality.get(key), 0)
        if "stimulus_first_frame_ack_timed_out" in quality:
            record["stimulus_first_frame_ack_timed_out"] = bool(quality.get("stimulus_first_frame_ack_timed_out"))
        for key in ("stimulus_profile_id",):
            value = str(quality.get(key, "")).strip()
            if value:
                record[key] = value
        for key in ("stim_mean", "stim_amp", "ramp_sec"):
            if key in quality and quality.get(key) is not None:
                record[key] = _safe_float(quality.get(key), 0.0)
        frame_stats = quality.get("stimulus_frame_interval_stats", {})
        if isinstance(frame_stats, dict):
            record["stimulus_frame_interval_stats"] = _jsonable(frame_stats)
        records.append(record)
    return records


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(converted):
        return float(default)
    return float(converted)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _signature_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _signature_value(value.item())
    if isinstance(value, dict):
        return {str(key): _signature_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_signature_value(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if np.isfinite(numeric):
            return round(numeric, 6)
        return str(value)
    if value is None:
        return None
    return str(value)


def _protocol_signature_payload(
    *,
    sampling_rate: int,
    protocol_config: dict[str, Any],
    freqs: Sequence[float],
    board_eeg_channels: Sequence[int],
) -> dict[str, Any]:
    cfg = dict(protocol_config or {})
    return {
        "sampling_rate": int(sampling_rate),
        "prepare_sec": round(_safe_float(cfg.get("prepare_sec", 0.0), 0.0), 6),
        "active_sec": round(_safe_float(cfg.get("active_sec", 0.0), 0.0), 6),
        "rest_sec": round(_safe_float(cfg.get("rest_sec", 0.0), 0.0), 6),
        "long_idle_sec": round(_safe_float(cfg.get("long_idle_sec", 0.0), 0.0), 6),
        "target_repeats": _safe_int(cfg.get("target_repeats", 0), 0),
        "idle_repeats": _safe_int(cfg.get("idle_repeats", 0), 0),
        "switch_trials": _safe_int(cfg.get("switch_trials", 0), 0),
        "stimulus_profile_id": str(cfg.get("stimulus_profile_id", "")),
        "stimulus_mode": str(cfg.get("stimulus_mode", "")),
        "stimulus_backend": str(cfg.get("stimulus_backend", "")),
        "stim_refresh_rate_hz": round(_safe_float(cfg.get("stim_refresh_rate_hz", 0.0), 0.0), 6),
        "stim_mean": round(_safe_float(cfg.get("stim_mean", 0.0), 0.0), 6),
        "stim_amp": round(_safe_float(cfg.get("stim_amp", 0.0), 0.0), 6),
        "stim_luminance_min": round(_safe_float(cfg.get("stim_luminance_min", 0.0), 0.0), 6),
        "stim_luminance_max": round(_safe_float(cfg.get("stim_luminance_max", 0.0), 0.0), 6),
        "stim_michelson_contrast": round(_safe_float(cfg.get("stim_michelson_contrast", 0.0), 0.0), 6),
        "ramp_sec": round(_safe_float(cfg.get("ramp_sec", 0.0), 0.0), 6),
        "frame_interval_stats": _signature_value(cfg.get("frame_interval_stats", {})),
        "frame_lock_frequency_report": _signature_value(cfg.get("frame_lock_frequency_report", {})),
        "comfort_rating": _signature_value(cfg.get("comfort_rating", None)),
        "screen_brightness_note": str(cfg.get("screen_brightness_note", "")),
        "active_start_cue_sec": round(_safe_float(cfg.get("active_start_cue_sec", 0.0), 0.0), 6),
        "active_start_buffer_clear_timing": str(cfg.get("active_start_buffer_clear_timing", "")),
        "active_saved_window": str(cfg.get("active_saved_window", "")),
        "active_end_cue_timing": str(cfg.get("active_end_cue_timing", "")),
        "freqs": [round(float(freq), 6) for freq in freqs],
        "board_eeg_channels": [int(channel) for channel in board_eeg_channels],
    }


def build_protocol_signature(
    *,
    sampling_rate: int,
    protocol_config: dict[str, Any],
    freqs: Sequence[float],
    board_eeg_channels: Sequence[int],
) -> str:
    payload = _protocol_signature_payload(
        sampling_rate=sampling_rate,
        protocol_config=protocol_config,
        freqs=freqs,
        board_eeg_channels=board_eeg_channels,
    )
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


def _aggregate_frame_interval_stats(trial_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stats_rows = [
        dict(row.get("stimulus_frame_interval_stats", {}))
        for row in trial_records
        if isinstance(row.get("stimulus_frame_interval_stats", {}), dict)
    ]
    nonempty = [row for row in stats_rows if _safe_int(row.get("count", 0), 0) > 0]
    if not stats_rows:
        return {}
    counts = np.asarray([_safe_int(row.get("count", 0), 0) for row in stats_rows], dtype=int)
    p95_values = np.asarray([_safe_float(row.get("p95_ms", 0.0), 0.0) for row in nonempty], dtype=float)
    max_values = np.asarray([_safe_float(row.get("max_ms", 0.0), 0.0) for row in nonempty], dtype=float)
    mean_values = np.asarray([_safe_float(row.get("mean_ms", 0.0), 0.0) for row in nonempty], dtype=float)
    refresh_values = np.asarray(
        [_safe_float(row.get("refresh_rate_hz_estimate", 0.0), 0.0) for row in nonempty],
        dtype=float,
    )
    return {
        "trial_count": int(len(stats_rows)),
        "nonempty_trial_count": int(len(nonempty)),
        "sample_count_total": int(np.sum(counts)) if counts.size else 0,
        "mean_ms_mean": float(np.mean(mean_values)) if mean_values.size else 0.0,
        "p95_ms_max": float(np.max(p95_values)) if p95_values.size else 0.0,
        "max_ms_max": float(np.max(max_values)) if max_values.size else 0.0,
        "refresh_rate_hz_estimate_mean": float(np.mean(refresh_values)) if refresh_values.size else 0.0,
    }


def _build_quality_summary(trial_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_trials = int(len(trial_records))
    shortfalls = [_safe_float(row.get("shortfall_ratio", 0.0), 0.0) for row in trial_records]
    retries = [_safe_int(row.get("retry_count", 0), 0) for row in trial_records]
    ack_timeouts = [
        bool(row.get("stimulus_first_frame_ack_timed_out", False))
        for row in trial_records
    ]
    role_counts = summarize_trial_roles(trial_records)
    valid_trials = [row for row in trial_records if bool(row.get("valid", True))]
    invalid_trials = [row for row in trial_records if not bool(row.get("valid", True))]
    calibration_trials = [row for row in trial_records if str(row.get("split_role", "")).strip().lower() == "calibration"]
    test_trials = [row for row in trial_records if str(row.get("split_role", "")).strip().lower() == "test"]
    valid_command_trials = [
        row
        for row in valid_trials
        if str(row.get("state_type", "")).strip().lower() == "command"
        or row.get("expected_freq") is not None
    ]
    valid_nc_segments = [
        row
        for row in valid_trials
        if str(row.get("state_type", "")).strip().lower() == "no_control"
    ]
    subtype_counts: dict[str, int] = {}
    for row in trial_records:
        subtype = str(row.get("nc_subtype", "")).strip().lower()
        if subtype:
            subtype_counts[subtype] = int(subtype_counts.get(subtype, 0) + 1)
    return {
        "valid_trial_count": int(len(valid_trials)),
        "kept_trial_count": total_trials,
        "short_segment_excluded": 0,
        "retry_total": int(np.sum(np.asarray(retries, dtype=int))) if retries else 0,
        "retry_max": int(np.max(np.asarray(retries, dtype=int))) if retries else 0,
        "shortfall_ratio_mean": float(np.mean(np.asarray(shortfalls, dtype=float))) if shortfalls else 0.0,
        "shortfall_ratio_max": float(np.max(np.asarray(shortfalls, dtype=float))) if shortfalls else 0.0,
        "stimulus_first_frame_ack_timeout_count": int(sum(1 for item in ack_timeouts if item)),
        "trial_role_counts": role_counts,
        "invalid_trial_count": int(len(invalid_trials)),
        "valid_command_trials": int(len(valid_command_trials)),
        "valid_no_control_segments": int(len(valid_nc_segments)),
        "calibration_split_trial_count": int(len(calibration_trials)),
        "test_split_trial_count": int(len(test_trials)),
        "no_control_subtype_counts": subtype_counts,
    }


def _events_from_trials(trial_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in trial_records:
        raw_events = row.get("events", [])
        if isinstance(raw_events, list):
            for item in raw_events:
                if isinstance(item, dict):
                    event = {
                        "sample_index": item.get("sample_index"),
                        "event_code": item.get("event_code"),
                        "event_name": item.get("event_name"),
                        "event_value": item.get("event_value"),
                        "perf_time": item.get("perf_time"),
                        "trial_id": item.get("trial_id", row.get("trial_id")),
                    }
                    events.append(event)
    return events


def _snr_by_freq_from_trials(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    sampling_rate: int,
    freqs: Sequence[float],
) -> dict[str, float]:
    result = {f"command_{float(freq):g}_snr": 0.0 for freq in freqs}
    fs = int(sampling_rate)
    if fs <= 0:
        return result
    for freq in freqs:
        rows: list[float] = []
        for trial, segment in trial_segments:
            if trial.expected_freq is None or abs(float(trial.expected_freq) - float(freq)) > 1e-8:
                continue
            matrix = np.asarray(segment, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] < 4:
                continue
            signal = np.mean(matrix, axis=1)
            signal = signal - float(np.mean(signal))
            spectrum = np.abs(np.fft.rfft(signal))
            freqs_fft = np.fft.rfftfreq(int(signal.shape[0]), d=1.0 / float(fs))
            if spectrum.size <= 1:
                continue
            target_index = int(np.argmin(np.abs(freqs_fft - float(freq))))
            band = np.where((freqs_fft >= max(0.0, float(freq) - 2.0)) & (freqs_fft <= float(freq) + 2.0))[0]
            band = band[band != target_index]
            noise = float(np.median(spectrum[band])) if band.size else float(np.median(spectrum))
            rows.append(float(spectrum[target_index] / max(noise, 1e-12)))
        result[f"command_{float(freq):g}_snr"] = float(np.median(np.asarray(rows, dtype=float))) if rows else 0.0
    return result


def _line_noise_50hz_from_trials(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    sampling_rate: int,
) -> dict[str, Any]:
    fs = int(sampling_rate)
    if fs <= 100:
        return {"ratio": 0.0, "db": 0.0, "status": "unavailable_sampling_rate"}
    rows: list[float] = []
    for _trial, segment in trial_segments:
        matrix = np.asarray(segment, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 8:
            continue
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centered, axis=0))
        freqs_fft = np.fft.rfftfreq(int(centered.shape[0]), d=1.0 / float(fs))
        if spectrum.shape[0] <= 1:
            continue
        target_index = int(np.argmin(np.abs(freqs_fft - 50.0)))
        if abs(float(freqs_fft[target_index]) - 50.0) > 2.0:
            continue
        band = np.where((freqs_fft >= 45.0) & (freqs_fft <= 55.0))[0]
        band = band[band != target_index]
        if band.size <= 0:
            continue
        noise = np.median(spectrum[band, :], axis=0)
        ratio = spectrum[target_index, :] / np.maximum(noise, 1e-12)
        rows.extend(float(item) for item in ratio.reshape(-1) if np.isfinite(item))
    if not rows:
        return {"ratio": 0.0, "db": 0.0, "status": "unavailable_no_valid_segments"}
    value = float(np.median(np.asarray(rows, dtype=np.float64)))
    return {
        "ratio": value,
        "db": float(20.0 * np.log10(max(value, 1e-12))),
        "status": "estimated_from_saved_trial_windows",
    }


def _build_quality_report(
    *,
    trial_records: Sequence[dict[str, Any]],
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    sampling_rate: int,
    freqs: Sequence[float],
    continuous_board_data: Optional[np.ndarray],
    quality_summary: Mapping[str, Any],
) -> dict[str, Any]:
    matrices = [np.asarray(segment, dtype=np.float64) for _trial, segment in trial_segments if np.asarray(segment).ndim == 2]
    if matrices:
        try:
            stacked = np.vstack(matrices)
        except Exception:
            stacked = np.empty((0, 0), dtype=np.float64)
    else:
        stacked = np.empty((0, 0), dtype=np.float64)
    channel_variance: list[float] = []
    bad_channels: list[int] = []
    saturation = False
    if stacked.ndim == 2 and stacked.shape[0] > 0:
        variance = np.var(stacked, axis=0)
        channel_variance = [float(item) for item in variance.tolist()]
        finite = variance[np.isfinite(variance)]
        if finite.size:
            median = float(np.median(finite))
            for index, value in enumerate(variance):
                if not np.isfinite(value) or value <= 1e-12 or (median > 0.0 and value > median * 20.0):
                    bad_channels.append(int(index))
        saturation = bool(np.any(np.abs(stacked) > 1e6))
    dropped_frame_count = 0
    for row in trial_records:
        stats = row.get("stimulus_frame_interval_stats", {})
        if isinstance(stats, dict):
            p95 = _safe_float(stats.get("p95_ms", 0.0), 0.0)
            mean = _safe_float(stats.get("mean_ms", 0.0), 0.0)
            if mean > 0.0 and p95 > mean * 1.5:
                dropped_frame_count += 1
    report = {
        "generated_at": _now_iso_timestamp(),
        "channel_variance": channel_variance,
        "bad_channel_candidates": bad_channels,
        "saturation_or_dropout": bool(saturation),
        "line_noise_50hz": _line_noise_50hz_from_trials(trial_segments, sampling_rate=int(sampling_rate)),
        "dropped_frame_count": int(dropped_frame_count),
        "invalid_trial_count": int(quality_summary.get("invalid_trial_count", 0) or 0),
        "retry_total": int(quality_summary.get("retry_total", 0) or 0),
        "marker_count_check": {
            "events_in_trials": int(len(_events_from_trials(trial_records))),
            "trial_count": int(len(trial_records)),
        },
        "valid_command_trials": int(quality_summary.get("valid_command_trials", 0) or 0),
        "valid_no_control_segments": int(quality_summary.get("valid_no_control_segments", 0) or 0),
        "test_split_trial_count": int(quality_summary.get("test_split_trial_count", 0) or 0),
    }
    report.update(_snr_by_freq_from_trials(trial_segments, sampling_rate=int(sampling_rate), freqs=freqs))
    return _jsonable(report)


def save_collection_dataset_bundle(
    *,
    dataset_root: Path,
    session_id: str,
    subject_id: str,
    serial_port: str,
    board_id: int,
    sampling_rate: int,
    freqs: Sequence[float],
    board_eeg_channels: Sequence[int],
    protocol_config: dict[str, Any],
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    quality_rows: Optional[Sequence[dict[str, Any]]] = None,
    continuous_board_data: Optional[np.ndarray] = None,
    continuous_board_info: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    requested_session_id = str(session_id if session_id is not None else "").strip()
    requested_subject_id = str(subject_id if subject_id is not None else "").strip()
    safe_session_id = sanitize_collection_token(requested_session_id, default="session")
    safe_subject_id = sanitize_collection_token(requested_subject_id, default="subject")
    session_dir = (dataset_root / safe_session_id).resolve()
    if not _path_is_relative_to(session_dir, dataset_root):
        raise ValueError(f"collection session directory must stay inside dataset_root: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)

    target_samples = int(round(float(protocol_config.get("active_sec", 0.0)) * float(sampling_rate)))
    target_samples = max(1, target_samples)
    npz_arrays: dict[str, np.ndarray] = {}
    trial_records = _build_collection_records(
        trial_segments,
        npz_arrays=npz_arrays,
        default_target_samples=target_samples,
        sampling_rate=int(sampling_rate),
        quality_rows=quality_rows,
    )
    npz_path = session_dir / "raw_trials.npz"
    _atomic_save_npz(npz_path, npz_arrays)
    files_payload: dict[str, str] = {"raw_trials_npz": str(npz_path)}
    continuous_payload: dict[str, Any] = {}
    continuous_save_error = ""
    if continuous_board_data is not None:
        try:
            continuous_matrix = np.ascontiguousarray(np.asarray(continuous_board_data, dtype=np.float64))
            if continuous_matrix.ndim != 2:
                raise ValueError("continuous_board_data must be a 2-D BrainFlow matrix")
            continuous_payload = {
                "saved": True,
                "shape": [int(continuous_matrix.shape[0]), int(continuous_matrix.shape[1])],
                "dtype": str(continuous_matrix.dtype),
                **dict(continuous_board_info or {}),
            }
            continuous_path = session_dir / "continuous_board.npz"
            try:
                _atomic_save_npz(continuous_path, {"board_data": continuous_matrix})
                files_payload["continuous_board_npz"] = str(continuous_path)
                continuous_payload["format"] = "npz_compressed"
            except Exception as npz_error:
                fallback_path = session_dir / "continuous_board.npy"
                try:
                    _atomic_save_npy(fallback_path, continuous_matrix)
                except Exception as fallback_error:
                    continuous_save_error = (
                        f"continuous_board npz save failed: {npz_error}; "
                        f"npy fallback failed: {fallback_error}"
                    )
                    continuous_payload["saved"] = False
                    continuous_payload["save_error"] = continuous_save_error
                else:
                    files_payload["continuous_board_npy"] = str(fallback_path)
                    continuous_payload["format"] = "npy"
                    continuous_payload["compressed_npz_save_error"] = str(npz_error)
        except Exception as error:
            continuous_save_error = str(error)
            continuous_payload = {
                "saved": False,
                "save_error": continuous_save_error,
                **dict(continuous_board_info or {}),
            }

    protocol_payload = dict(protocol_config)
    protocol_payload.setdefault("frame_interval_stats", _aggregate_frame_interval_stats(trial_records))
    protocol_signature = build_protocol_signature(
        sampling_rate=int(sampling_rate),
        protocol_config=protocol_payload,
        freqs=freqs,
        board_eeg_channels=board_eeg_channels,
    )
    protocol_payload.setdefault("requested_session_id", requested_session_id or safe_session_id)
    protocol_payload["saved_session_id"] = safe_session_id
    if requested_subject_id and requested_subject_id != safe_subject_id:
        protocol_payload.setdefault("requested_subject_id", requested_subject_id)
        protocol_payload["saved_subject_id"] = safe_subject_id
    if continuous_save_error:
        protocol_payload["continuous_board_save_error"] = continuous_save_error
    protocol_payload["protocol_signature"] = str(protocol_signature)
    quality_summary = _build_quality_summary(trial_records)
    quality_summary.update(
        {
            "collection_aborted": bool(protocol_config.get("collection_aborted", False)),
            "planned_trial_count": _safe_int(
                protocol_config.get("planned_total_trials", len(trial_records)),
                len(trial_records),
            ),
            "saved_trial_count": _safe_int(
                protocol_config.get("saved_trial_count", len(trial_records)),
                len(trial_records),
            ),
        }
    )
    manifest_payload = {
        "data_schema_version": COLLECTION_DATA_SCHEMA_VERSION,
        "session_id": safe_session_id,
        "subject_id": safe_subject_id,
        "generated_at": _now_iso_timestamp(),
        "serial_port": str(serial_port),
        "board_id": int(board_id),
        "sampling_rate": int(sampling_rate),
        "freqs": [float(freq) for freq in freqs],
        "board_eeg_channels": [int(channel) for channel in board_eeg_channels],
        "protocol_signature": str(protocol_signature),
        "protocol_config": protocol_payload,
        "quality_summary": quality_summary,
        "trials": trial_records,
        "splits": {"train": [], "gate": [], "holdout": []},
        "files": files_payload,
    }
    if continuous_payload:
        manifest_payload["continuous_board"] = _jsonable(continuous_payload)
    manifest_path = session_dir / "session_manifest.json"
    _atomic_write_text(manifest_path, json_dumps(_jsonable(manifest_payload)) + "\n", encoding="utf-8")
    events_rows = _events_from_trials(trial_records)
    if events_rows:
        events_path = session_dir / "events.csv"
        _atomic_write_csv(
            events_path,
            events_rows,
            ("sample_index", "event_code", "event_name", "event_value", "perf_time", "trial_id"),
        )
        files_payload["events_csv"] = str(events_path)
        manifest_payload["files"] = files_payload
        _atomic_write_text(manifest_path, json_dumps(_jsonable(manifest_payload)) + "\n", encoding="utf-8")
    quality_report = _build_quality_report(
        trial_records=trial_records,
        trial_segments=trial_segments,
        sampling_rate=int(sampling_rate),
        freqs=freqs,
        continuous_board_data=continuous_board_data,
        quality_summary=quality_summary,
    )
    quality_report_path = session_dir / "quality_report.json"
    _atomic_write_text(quality_report_path, json_dumps(_jsonable(quality_report)) + "\n", encoding="utf-8")
    files_payload["quality_report_json"] = str(quality_report_path)
    manifest_payload["files"] = files_payload
    manifest_payload["quality_report"] = quality_report
    _atomic_write_text(manifest_path, json_dumps(_jsonable(manifest_payload)) + "\n", encoding="utf-8")
    result = {
        "dataset_dir": str(session_dir),
        "dataset_manifest": str(manifest_path),
        "dataset_npz": str(npz_path),
        "events_csv": str(session_dir / "events.csv") if events_rows else "",
        "quality_report_json": str(quality_report_path),
        "data_schema_version": COLLECTION_DATA_SCHEMA_VERSION,
    }
    if "continuous_board_npz" in files_payload:
        result["dataset_continuous_board_npz"] = files_payload["continuous_board_npz"]
    if "continuous_board_npy" in files_payload:
        result["dataset_continuous_board_npy"] = files_payload["continuous_board_npy"]
    return result


def load_collection_dataset(manifest_path: Path) -> LoadedDataset:
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = dict(manifest.get("files", {}))
    raw_npz_value = str(files.get("raw_trials_npz", "")).strip()
    if raw_npz_value:
        npz_path = Path(raw_npz_value).expanduser().resolve()
    else:
        npz_path = path.parent / "raw_trials.npz"
    if not npz_path.exists():
        # Dataset bundles are often moved from the acquisition PC to a server.
        # In that case an absolute Windows path in the manifest is no longer valid,
        # but raw_trials.npz should still live beside session_manifest.json.
        sibling_npz = path.parent / "raw_trials.npz"
        if sibling_npz.exists():
            npz_path = sibling_npz.resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    try:
        trial_rows = list(manifest.get("trials", []))
        trial_rows.sort(key=lambda row: int(row.get("order_index", 0)))
        segments: list[tuple[TrialSpec, np.ndarray]] = []
        for row in trial_rows:
            key = str(row.get("npz_key", ""))
            if key not in data.files:
                raise KeyError(f"npz key missing: {key}")
            matrix = np.ascontiguousarray(np.asarray(data[key], dtype=np.float64))
            trial = TrialSpec(
                label=str(row.get("label", "")),
                expected_freq=None if row.get("expected_freq") is None else float(row.get("expected_freq")),
                trial_id=int(row.get("trial_id", -1)),
                block_index=int(row.get("block_index", -1)),
                metadata=dict(row),
            )
            segments.append((trial, matrix))
    finally:
        data.close()

    freqs = tuple(float(value) for value in manifest.get("freqs", []))
    if len(freqs) != 4:
        raise ValueError("manifest freqs must contain 4 frequencies")
    board_channels = tuple(int(value) for value in manifest.get("board_eeg_channels", []))
    return LoadedDataset(
        manifest_path=path,
        npz_path=npz_path,
        session_id=str(manifest.get("session_id", "")),
        subject_id=str(manifest.get("subject_id", "")),
        sampling_rate=int(manifest.get("sampling_rate", 0)),
        freqs=freqs,  # type: ignore[arg-type]
        board_eeg_channels=board_channels,
        protocol_config=dict(manifest.get("protocol_config", {})),
        trial_segments=segments,
        manifest=dict(manifest),
    )


def summarize_collection_manifest(manifest_path: Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol_config = dict(payload.get("protocol_config", {}))
    quality_summary = dict(payload.get("quality_summary", {}))
    trials = [row for row in list(payload.get("trials", [])) if isinstance(row, dict)]
    shortfalls = [_safe_float(row.get("shortfall_ratio", 0.0), 0.0) for row in trials]
    retry_counts = [_safe_int(row.get("retry_count", 0), 0) for row in trials]
    stage_values = sorted(
        {
            str(row.get("stage", "")).strip().lower()
            for row in trials
            if str(row.get("stage", "")).strip() != ""
        }
    )
    long_idle_trials = sum(
        1
        for row in trials
        if "long_idle" in str(row.get("label", "")).strip().lower()
        or "long idle" in str(row.get("label", "")).strip().lower()
        or "long_idle" in str(row.get("stage", "")).strip().lower()
        or "long idle" in str(row.get("stage", "")).strip().lower()
    )
    target_trials = sum(
        1
        for row in trials
        if row.get("expected_freq") is not None and not str(row.get("label", "")).startswith("switch_to_")
    )
    idle_trials = sum(
        1
        for row in trials
        if row.get("expected_freq") is None
        and "long_idle" not in str(row.get("label", "")).strip().lower()
        and "long idle" not in str(row.get("label", "")).strip().lower()
        and "long_idle" not in str(row.get("stage", "")).strip().lower()
        and "long idle" not in str(row.get("stage", "")).strip().lower()
    )
    switch_trials = sum(1 for row in trials if str(row.get("label", "")).startswith("switch_to_"))
    trial_role_counts = summarize_trial_roles(trials)
    generated_at = str(payload.get("generated_at", ""))
    return {
        "manifest_path": str(path),
        "session_id": str(payload.get("session_id", "")),
        "subject_id": str(payload.get("subject_id", "")),
        "generated_at": generated_at,
        "sampling_rate": _safe_int(payload.get("sampling_rate", 0), 0),
        "freqs": [float(value) for value in payload.get("freqs", [])],
        "board_eeg_channels": [int(value) for value in payload.get("board_eeg_channels", [])],
        "trial_count": int(len(trials)),
        "target_trial_count": int(target_trials),
        "idle_trial_count": int(idle_trials),
        "long_idle_trial_count": int(long_idle_trials),
        "switch_trial_count": int(switch_trials),
        "trial_role_counts": trial_role_counts,
        "shortfall_ratio_mean": float(np.mean(np.asarray(shortfalls, dtype=float))) if shortfalls else 0.0,
        "shortfall_ratio_max": float(np.max(np.asarray(shortfalls, dtype=float))) if shortfalls else 0.0,
        "retry_count_total": int(np.sum(np.asarray(retry_counts, dtype=int))) if retry_counts else 0,
        "retry_count_max": int(np.max(np.asarray(retry_counts, dtype=int))) if retry_counts else 0,
        "stage_values": stage_values,
        "preset_name": str(protocol_config.get("preset_name", "")),
        "round_index": _safe_int(protocol_config.get("round_index", 0), 0),
        "rounds_planned": _safe_int(protocol_config.get("rounds_planned", 0), 0),
        "protocol_signature": str(payload.get("protocol_signature", "")),
        "protocol_config": protocol_config,
        "quality_summary": quality_summary,
        "data_schema_version": str(payload.get("data_schema_version", "")),
    }


def discover_collection_manifests(dataset_root: Path) -> list[dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        return []
    dedup_rows: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}

    def _identity_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
        session_id = str(row.get("session_id", "")).strip()
        subject_id = str(row.get("subject_id", "")).strip()
        generated_at = str(row.get("generated_at", "")).strip()
        protocol_signature = str(row.get("protocol_signature", "")).strip()
        trial_count = _safe_int(row.get("trial_count", 0), 0)
        manifest_path = str(row.get("manifest_path", "")).strip()
        if session_id:
            return (session_id, subject_id, generated_at, protocol_signature, trial_count)
        return (manifest_path, subject_id, generated_at, protocol_signature, trial_count)

    def _path_rank(path_text: str) -> tuple[int, int, str]:
        path = Path(path_text).expanduser().resolve()
        try:
            rel = path.relative_to(root)
            depth = len(rel.parts)
        except Exception:
            depth = len(path.parts)
        return (depth, len(str(path)), str(path))

    for path in sorted(root.rglob("session_manifest.json")):
        try:
            row = summarize_collection_manifest(path)
        except Exception as exc:
            row = {
                "manifest_path": str(path),
                "session_id": "",
                "subject_id": "",
                "generated_at": "",
                "trial_count": 0,
                "target_trial_count": 0,
                "idle_trial_count": 0,
                "long_idle_trial_count": 0,
                "switch_trial_count": 0,
                "trial_role_counts": {},
                "shortfall_ratio_mean": 0.0,
                "shortfall_ratio_max": 0.0,
                "retry_count_total": 0,
                "retry_count_max": 0,
                "stage_values": [],
                "preset_name": "",
                "round_index": 0,
                "rounds_planned": 0,
                "protocol_signature": "",
                "protocol_config": {},
                "quality_summary": {},
                "data_schema_version": "",
                "error": str(exc),
            }
        key = _identity_key(row)
        existing = dedup_rows.get(key)
        if existing is None:
            dedup_rows[key] = row
            continue
        if _path_rank(str(row.get("manifest_path", ""))) < _path_rank(str(existing.get("manifest_path", ""))):
            dedup_rows[key] = row
    rows = list(dedup_rows.values())
    rows.sort(
        key=lambda item: (
            str(item.get("generated_at", "")),
            str(item.get("session_id", "")),
            str(item.get("manifest_path", "")),
        ),
        reverse=True,
    )
    return rows

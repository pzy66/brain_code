from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from async_fbcca_idle_standalone import (
    DEFAULT_CALIBRATION_SEED,
    TrialSpec,
    build_benchmark_eval_trials,
    json_dumps,
)

COLLECTION_DATA_SCHEMA_VERSION = "2.0"


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


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".atomic_{uuid.uuid4().hex[:8]}.tmp"
    try:
        tmp_path.write_text(str(text), encoding=encoding)
        os.replace(str(tmp_path), str(target))
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
        os.replace(str(tmp_path), str(target))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


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
    target_samples: int,
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
        effective_target = int(quality.get("target_samples", target_samples))
        effective_target = max(1, int(effective_target))
        used_samples = int(matrix.shape[0])
        retry_count = max(0, int(quality.get("retry_count", 0)))
        shortfall_ratio = float(max(effective_target - used_samples, 0) / effective_target)
        label_text = str(trial.label)
        label_lower = label_text.strip().lower()
        stage_name = "long_idle" if ("long_idle" in label_lower or "long idle" in label_lower) else "collection"
        records.append(
            {
                "stage": stage_name,
                "label": label_text,
                "expected_freq": None if trial.expected_freq is None else float(trial.expected_freq),
                "trial_id": int(trial.trial_id),
                "block_index": int(trial.block_index),
                "order_index": int(order_index),
                "used_samples": used_samples,
                "target_samples": effective_target,
                "shortfall_ratio": shortfall_ratio,
                "retry_count": retry_count,
                "channels": int(matrix.shape[1]),
                "npz_key": npz_key,
            }
        )
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


def _build_quality_summary(trial_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_trials = int(len(trial_records))
    shortfalls = [_safe_float(row.get("shortfall_ratio", 0.0), 0.0) for row in trial_records]
    retries = [_safe_int(row.get("retry_count", 0), 0) for row in trial_records]
    return {
        "valid_trial_count": total_trials,
        "kept_trial_count": total_trials,
        "short_segment_excluded": 0,
        "retry_total": int(np.sum(np.asarray(retries, dtype=int))) if retries else 0,
        "retry_max": int(np.max(np.asarray(retries, dtype=int))) if retries else 0,
        "shortfall_ratio_mean": float(np.mean(np.asarray(shortfalls, dtype=float))) if shortfalls else 0.0,
        "shortfall_ratio_max": float(np.max(np.asarray(shortfalls, dtype=float))) if shortfalls else 0.0,
    }


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
) -> dict[str, str]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    session_dir = dataset_root / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    target_samples = int(round(float(protocol_config.get("active_sec", 0.0)) * float(sampling_rate)))
    target_samples = max(1, target_samples)
    npz_arrays: dict[str, np.ndarray] = {}
    trial_records = _build_collection_records(
        trial_segments,
        npz_arrays=npz_arrays,
        target_samples=target_samples,
        quality_rows=quality_rows,
    )
    npz_path = session_dir / "raw_trials.npz"
    _atomic_save_npz(npz_path, npz_arrays)

    protocol_signature = build_protocol_signature(
        sampling_rate=int(sampling_rate),
        protocol_config=dict(protocol_config),
        freqs=freqs,
        board_eeg_channels=board_eeg_channels,
    )
    protocol_payload = dict(protocol_config)
    protocol_payload["protocol_signature"] = str(protocol_signature)
    quality_summary = _build_quality_summary(trial_records)
    manifest_payload = {
        "data_schema_version": COLLECTION_DATA_SCHEMA_VERSION,
        "session_id": str(session_id),
        "subject_id": str(subject_id),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
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
        "files": {"raw_trials_npz": str(npz_path)},
    }
    manifest_path = session_dir / "session_manifest.json"
    _atomic_write_text(manifest_path, json_dumps(manifest_payload) + "\n", encoding="utf-8")
    return {
        "dataset_dir": str(session_dir),
        "dataset_manifest": str(manifest_path),
        "dataset_npz": str(npz_path),
        "data_schema_version": COLLECTION_DATA_SCHEMA_VERSION,
    }


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
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("session_manifest.json")):
        try:
            rows.append(summarize_collection_manifest(path))
        except Exception as exc:
            rows.append(
                {
                    "manifest_path": str(path),
                    "session_id": "",
                    "subject_id": "",
                    "generated_at": "",
                    "trial_count": 0,
                    "target_trial_count": 0,
                    "idle_trial_count": 0,
                    "long_idle_trial_count": 0,
                    "switch_trial_count": 0,
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
            )
    rows.sort(
        key=lambda item: (
            str(item.get("generated_at", "")),
            str(item.get("session_id", "")),
            str(item.get("manifest_path", "")),
        ),
        reverse=True,
    )
    return rows

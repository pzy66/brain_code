from __future__ import annotations

import json
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


ENHANCED_45M_PROTOCOL = CollectionProtocol(
    name="enhanced_45m",
    prepare_sec=1.0,
    active_sec=4.0,
    rest_sec=1.0,
    target_repeats=24,
    idle_repeats=48,
    switch_trials=32,
)


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
    return build_benchmark_eval_trials(
        freqs,
        target_repeats=int(protocol.target_repeats),
        idle_repeats=int(protocol.idle_repeats),
        switch_trials=int(protocol.switch_trials),
        seed=int(seed) + int(max(session_index, 1) - 1) * 1009,
    )


def _build_collection_records(
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    npz_arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for order_index, (trial, segment) in enumerate(segments):
        base_key = f"trial_collection_{int(trial.trial_id)}"
        npz_key = base_key
        suffix = 1
        while npz_key in npz_arrays:
            npz_key = f"{base_key}_{suffix}"
            suffix += 1
        matrix = np.ascontiguousarray(np.asarray(segment, dtype=np.float32))
        npz_arrays[npz_key] = matrix
        records.append(
            {
                "stage": "collection",
                "label": str(trial.label),
                "expected_freq": None if trial.expected_freq is None else float(trial.expected_freq),
                "trial_id": int(trial.trial_id),
                "block_index": int(trial.block_index),
                "order_index": int(order_index),
                "used_samples": int(matrix.shape[0]),
                "channels": int(matrix.shape[1]),
                "npz_key": npz_key,
            }
        )
    return records


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
) -> dict[str, str]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    session_dir = dataset_root / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    npz_arrays: dict[str, np.ndarray] = {}
    trial_records = _build_collection_records(trial_segments, npz_arrays=npz_arrays)
    npz_path = session_dir / "raw_trials.npz"
    np.savez_compressed(npz_path, **npz_arrays)

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
        "protocol_config": dict(protocol_config),
        "trials": trial_records,
        "splits": {"train": [], "gate": [], "holdout": []},
        "files": {"raw_trials_npz": str(npz_path)},
    }
    manifest_path = session_dir / "session_manifest.json"
    manifest_path.write_text(json_dumps(manifest_payload) + "\n", encoding="utf-8")
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
    npz_path = Path(files.get("raw_trials_npz", "")).expanduser().resolve()
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


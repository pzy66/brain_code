from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import scipy.io as sio

from .async_fbcca_idle_standalone import TrialSpec, json_dumps
from .dataset import save_collection_dataset_bundle


WANG2016_ZENODO_RECORD = "https://zenodo.org/records/14865172"
WANG2016_REQUIRED_CHANNELS = ("Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4")
WANG2016_FREQS = (8.0, 10.0, 12.0, 15.0)
WANG2016_TARGET_FREQUENCIES = (
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    8.2,
    9.2,
    10.2,
    11.2,
    12.2,
    13.2,
    14.2,
    15.2,
    8.4,
    9.4,
    10.4,
    11.4,
    12.4,
    13.4,
    14.4,
    15.4,
    8.6,
    9.6,
    10.6,
    11.6,
    12.6,
    13.6,
    14.6,
    15.6,
    8.8,
    9.8,
    10.8,
    11.8,
    12.8,
    13.8,
    14.8,
    15.8,
)
WANG2016_TARGET_INDEX_BY_FREQ = {
    8.0: 1,
    10.0: 3,
    12.0: 5,
    15.0: 8,
}
WANG2016_ALL_TARGET_INDEX_BY_FREQ = {
    float(freq): int(index + 1) for index, freq in enumerate(WANG2016_TARGET_FREQUENCIES)
}
WANG2016_SAMPLING_RATE = 250
WANG2016_STIMULUS_OFFSET_SEC = 0.5
WANG2016_STIMULUS_SEC = 5.0
WANG2016_PRE_STIMULUS_IDLE_SEC = 0.5
WANG2016_BLOCKS = 6
WANG2016_TARGET_COUNT = 40


@dataclass(frozen=True)
class Wang2016LoadedSubject:
    subject: str
    mat_path: Path
    eeg: np.ndarray
    channel_names: tuple[str, ...]
    selected_channel_names: tuple[str, ...]
    selected_channel_indices: tuple[int, ...]


def _normalize_channel_name(value: str) -> str:
    return str(value).strip().lower()


def parse_wang2016_channel_loc(path: Path) -> tuple[str, ...]:
    names: list[str] = []
    for line in Path(path).expanduser().resolve().read_text(encoding="utf-8-sig").splitlines():
        parts = str(line).strip().split()
        if len(parts) < 4:
            continue
        names.append(str(parts[-1]).strip())
    if len(names) != 64:
        raise ValueError(f"Wang2016 channel loc must contain 64 channels; got {len(names)} from {path}")
    return tuple(names)


def selected_channel_indices(
    channel_names: Sequence[str],
    *,
    required_channels: Sequence[str] = WANG2016_REQUIRED_CHANNELS,
) -> tuple[int, ...]:
    normalized_to_index = {
        _normalize_channel_name(name): int(index) for index, name in enumerate(tuple(str(item) for item in channel_names))
    }
    indices: list[int] = []
    missing: list[str] = []
    for name in tuple(required_channels):
        key = _normalize_channel_name(str(name))
        if key not in normalized_to_index:
            missing.append(str(name))
            continue
        indices.append(int(normalized_to_index[key]))
    if missing:
        raise ValueError(f"Wang2016 subject is missing required channels: {missing}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Wang2016 required channels resolved to duplicate indices: {indices}")
    return tuple(indices)


def load_wang2016_subject(mat_path: Path, channel_loc_path: Path) -> Wang2016LoadedSubject:
    resolved_mat = Path(mat_path).expanduser().resolve()
    payload = sio.loadmat(resolved_mat, squeeze_me=True, struct_as_record=False)
    if "data" not in payload:
        raise ValueError(f"Wang2016 subject file missing variable 'data': {resolved_mat}")
    eeg = np.asarray(payload["data"], dtype=np.float64)
    if eeg.shape != (64, 1500, WANG2016_TARGET_COUNT, WANG2016_BLOCKS):
        raise ValueError(f"Wang2016 data must have shape (64, 1500, 40, 6); got {eeg.shape}")
    channel_names = parse_wang2016_channel_loc(channel_loc_path)
    indices = selected_channel_indices(channel_names)
    selected_names = tuple(channel_names[index] for index in indices)
    if tuple(selected_names) != tuple(WANG2016_REQUIRED_CHANNELS):
        raise ValueError(f"selected channel order mismatch: {selected_names} != {WANG2016_REQUIRED_CHANNELS}")
    return Wang2016LoadedSubject(
        subject=resolved_mat.stem.upper(),
        mat_path=resolved_mat,
        eeg=np.ascontiguousarray(eeg, dtype=np.float64),
        channel_names=channel_names,
        selected_channel_names=selected_names,
        selected_channel_indices=indices,
    )


def _stimulus_segment(eeg: np.ndarray, *, target_index_1based: int, block_index: int, channel_indices: Sequence[int]) -> np.ndarray:
    target_index = int(target_index_1based) - 1
    start = int(round(WANG2016_STIMULUS_OFFSET_SEC * WANG2016_SAMPLING_RATE))
    stop = start + int(round(WANG2016_STIMULUS_SEC * WANG2016_SAMPLING_RATE))
    matrix = eeg[np.asarray(channel_indices, dtype=int), start:stop, target_index, int(block_index)]
    return np.ascontiguousarray(matrix.T, dtype=np.float64)


def _pre_stimulus_idle_segment(
    eeg: np.ndarray,
    *,
    target_index_1based: int,
    block_index: int,
    channel_indices: Sequence[int],
) -> np.ndarray:
    target_index = int(target_index_1based) - 1
    stop = int(round(WANG2016_PRE_STIMULUS_IDLE_SEC * WANG2016_SAMPLING_RATE))
    matrix = eeg[np.asarray(channel_indices, dtype=int), 0:stop, target_index, int(block_index)]
    return np.ascontiguousarray(matrix.T, dtype=np.float64)


def resolve_wang2016_command_frequencies(freqs: Sequence[float] = WANG2016_FREQS) -> tuple[tuple[float, ...], dict[float, int]]:
    resolved: list[float] = []
    target_map: dict[float, int] = {}
    for value in tuple(freqs or ()):
        freq = round(float(value), 10)
        if freq not in WANG2016_ALL_TARGET_INDEX_BY_FREQ:
            raise ValueError(f"Wang2016 target frequency is unavailable: {float(value):g}Hz")
        if any(abs(freq - existing) < 1e-9 for existing in resolved):
            raise ValueError(f"duplicate Wang2016 command frequency: {float(value):g}Hz")
        resolved.append(float(freq))
        target_map[float(freq)] = int(WANG2016_ALL_TARGET_INDEX_BY_FREQ[float(freq)])
    if len(resolved) != 4:
        raise ValueError(f"Wang2016 command frequency set must contain exactly 4 frequencies; got {len(resolved)}")
    return tuple(resolved), target_map


def build_wang2016_segments(
    subject: Wang2016LoadedSubject,
    *,
    freqs: Sequence[float] = WANG2016_FREQS,
    include_hard_idle: bool = True,
    include_pre_stim_idle: bool = False,
) -> list[tuple[TrialSpec, np.ndarray]]:
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    command_freqs, target_index_by_freq = resolve_wang2016_command_frequencies(freqs)
    control_target_indices = {int(index) for index in target_index_by_freq.values()}
    for block_index in range(WANG2016_BLOCKS):
        for freq in command_freqs:
            target_index = int(target_index_by_freq[float(freq)])
            segments.append(
                (
                    TrialSpec(
                        label=f"{float(freq):g}Hz",
                        expected_freq=float(freq),
                        trial_id=int(trial_id),
                        block_index=int(block_index),
                    ),
                    _stimulus_segment(
                        subject.eeg,
                        target_index_1based=target_index,
                        block_index=block_index,
                        channel_indices=subject.selected_channel_indices,
                    ),
                )
            )
            trial_id += 1
            if bool(include_pre_stim_idle):
                segments.append(
                    (
                        TrialSpec(
                            label=f"pre_stim_idle_{float(freq):g}Hz",
                            expected_freq=None,
                            trial_id=int(trial_id),
                            block_index=int(block_index),
                        ),
                        _pre_stimulus_idle_segment(
                            subject.eeg,
                            target_index_1based=target_index,
                            block_index=block_index,
                            channel_indices=subject.selected_channel_indices,
                        ),
                    )
                )
                trial_id += 1
        if bool(include_hard_idle):
            for target_index in range(1, WANG2016_TARGET_COUNT + 1):
                if int(target_index) in control_target_indices:
                    continue
                target_freq = float(WANG2016_TARGET_FREQUENCIES[int(target_index) - 1])
                segments.append(
                    (
                        TrialSpec(
                            label=f"hard_idle_wang2016_target{int(target_index):02d}_{target_freq:g}Hz",
                            expected_freq=None,
                            trial_id=int(trial_id),
                            block_index=int(block_index),
                        ),
                        _stimulus_segment(
                            subject.eeg,
                            target_index_1based=int(target_index),
                            block_index=block_index,
                            channel_indices=subject.selected_channel_indices,
                        ),
                    )
                )
                trial_id += 1
    return segments


def wang2016_protocol_config(
    subject: Wang2016LoadedSubject,
    *,
    freqs: Sequence[float] = WANG2016_FREQS,
    saved_trial_count: int,
    include_hard_idle: bool,
    include_pre_stim_idle: bool,
) -> dict[str, Any]:
    command_freqs, target_index_by_freq = resolve_wang2016_command_frequencies(freqs)
    control_count = int(len(command_freqs) * WANG2016_BLOCKS)
    hard_idle_count = int((WANG2016_TARGET_COUNT - len(command_freqs)) * WANG2016_BLOCKS) if include_hard_idle else 0
    clean_idle_count = int(len(command_freqs) * WANG2016_BLOCKS) if include_pre_stim_idle else 0
    freq_token = "_".join(f"{float(freq):g}".replace(".", "p") for freq in command_freqs)
    return {
        "protocol_name": "external-wang2016-fourfreq-v1",
        "source_dataset": "Wang2016 SSVEP Benchmark",
        "source_record": WANG2016_ZENODO_RECORD,
        "subject_file": str(subject.mat_path),
        "sampling_rate_original_hz": 250,
        "sampling_rate": WANG2016_SAMPLING_RATE,
        "active_sec": WANG2016_STIMULUS_SEC,
        "pre_stimulus_idle_sec": WANG2016_PRE_STIMULUS_IDLE_SEC,
        "stimulus_offset_sec": WANG2016_STIMULUS_OFFSET_SEC,
        "target_repeats": WANG2016_BLOCKS,
        "idle_repeats": int(clean_idle_count + hard_idle_count),
        "switch_trials": 0,
        "planned_total_trials": int(saved_trial_count),
        "saved_trial_count": int(saved_trial_count),
        "control_trial_count": int(control_count),
        "clean_idle_trial_count": int(clean_idle_count),
        "hard_idle_trial_count": int(hard_idle_count),
        "hard_idle_definition": "Wang2016 stimulus trials whose target frequency is outside the four command frequencies",
        "include_hard_idle_non_command_targets": bool(include_hard_idle),
        "include_pre_stimulus_clean_idle": bool(include_pre_stim_idle),
        "freqs": [float(freq) for freq in command_freqs],
        "freq_token": str(freq_token),
        "all_target_frequencies": [float(freq) for freq in WANG2016_TARGET_FREQUENCIES],
        "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index_by_freq.items()},
        "all_channel_count": int(len(subject.channel_names)),
        "all_channel_names": [str(name) for name in subject.channel_names],
        "selected_channel_policy": "strict_required_8_channels_only",
        "selected_channel_names": [str(name) for name in subject.selected_channel_names],
        "selected_channel_indices_zero_based": [int(index) for index in subject.selected_channel_indices],
        "selected_channel_indices_one_based": [int(index) + 1 for index in subject.selected_channel_indices],
        "excluded_channel_count": int(len(subject.channel_names) - len(subject.selected_channel_names)),
        "dataset_contains_required_channels": True,
        "only_required_channels_saved": True,
        "stimulus_profile_id": "external_wang2016_original",
        "stimulus_mode": "external_original_jfpm",
        "stimulus_backend": "external_dataset",
        "stim_refresh_rate_hz": 60.0,
        "stim_mean": 0.0,
        "stim_amp": 0.0,
        "stim_luminance_min": 0.0,
        "stim_luminance_max": 0.0,
        "stim_michelson_contrast": 0.0,
        "ramp_sec": 0.0,
        "comfort_rating": None,
        "screen_brightness_note": "external Wang2016 original stimulus; not generated by this repo",
    }


def convert_wang2016_subject_to_collection(
    *,
    mat_path: Path,
    channel_loc_path: Path,
    dataset_root: Path,
    session_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    freqs: Sequence[float] = WANG2016_FREQS,
    include_hard_idle: bool = True,
    include_pre_stim_idle: bool = False,
) -> dict[str, Any]:
    subject = load_wang2016_subject(mat_path, channel_loc_path)
    command_freqs, target_index_by_freq = resolve_wang2016_command_frequencies(freqs)
    segments = build_wang2016_segments(
        subject,
        freqs=command_freqs,
        include_hard_idle=bool(include_hard_idle),
        include_pre_stim_idle=bool(include_pre_stim_idle),
    )
    if not segments:
        raise RuntimeError(f"no Wang2016 segments built for {mat_path}")
    if any(segment.shape[1] != len(WANG2016_REQUIRED_CHANNELS) for _trial, segment in segments):
        raise RuntimeError("Wang2016 conversion produced a segment that is not exactly 8 channels")
    protocol = wang2016_protocol_config(
        subject,
        freqs=command_freqs,
        saved_trial_count=len(segments),
        include_hard_idle=bool(include_hard_idle),
        include_pre_stim_idle=bool(include_pre_stim_idle),
    )
    resolved_subject = subject_id or f"wang2016_{subject.subject.lower()}"
    freq_token = str(protocol.get("freq_token", "fourfreq"))
    resolved_session = session_id or f"{resolved_subject}_{freq_token}_strict8"
    payload = save_collection_dataset_bundle(
        dataset_root=Path(dataset_root),
        session_id=resolved_session,
        subject_id=resolved_subject,
        serial_port="external_wang2016",
        board_id=-1,
        sampling_rate=WANG2016_SAMPLING_RATE,
        freqs=command_freqs,
        board_eeg_channels=tuple(range(len(WANG2016_REQUIRED_CHANNELS))),
        protocol_config=protocol,
        trial_segments=segments,
    )
    manifest_path = Path(str(payload["dataset_manifest"])).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_dataset_validation"] = {
        "dataset": "Wang2016",
        "required_channel_names": [str(name) for name in WANG2016_REQUIRED_CHANNELS],
        "selected_channel_names": [str(name) for name in subject.selected_channel_names],
        "selected_channel_indices_zero_based": [int(index) for index in subject.selected_channel_indices],
        "source_shape": [int(item) for item in subject.eeg.shape],
        "used_shape_per_control_trial": [int(WANG2016_STIMULUS_SEC * WANG2016_SAMPLING_RATE), 8],
        "used_shape_per_hard_idle_trial": [int(WANG2016_STIMULUS_SEC * WANG2016_SAMPLING_RATE), 8],
        "only_required_channels_saved": True,
        "excluded_channel_count": int(len(subject.channel_names) - len(subject.selected_channel_names)),
        "include_hard_idle_non_command_targets": bool(include_hard_idle),
        "include_pre_stimulus_clean_idle": bool(include_pre_stim_idle),
        "freqs": [float(freq) for freq in command_freqs],
        "all_target_frequencies": [float(freq) for freq in WANG2016_TARGET_FREQUENCIES],
        "target_index_by_freq": {f"{float(freq):g}": int(index) for freq, index in target_index_by_freq.items()},
    }
    manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")
    return {
        **payload,
        "subject": subject.subject,
        "selected_channel_names": list(subject.selected_channel_names),
        "selected_channel_indices_zero_based": list(subject.selected_channel_indices),
        "segment_count": int(len(segments)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Wang2016 benchmark subject data into this repo's strict 8-channel collection format."
    )
    parser.add_argument("--mat", type=Path, required=True, help="Path to Wang2016 S*.mat subject file.")
    parser.add_argument("--channels-loc", type=Path, required=True, help="Path to 64-channels.loc.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Output collection dataset root.")
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument(
        "--freqs",
        type=str,
        default=",".join(f"{float(freq):g}" for freq in WANG2016_FREQS),
        help="Comma-separated four Wang2016 target frequencies to mark as commands.",
    )
    parser.add_argument(
        "--include-hard-idle",
        type=int,
        default=1,
        help="Include non-command target stimulus windows as 5s hard idle segments.",
    )
    parser.add_argument(
        "--include-pre-stim-idle",
        type=int,
        default=0,
        help="Also include 0.5s pre-stimulus clean idle segments. Off by default to preserve 1.5-3s window search.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = convert_wang2016_subject_to_collection(
        mat_path=args.mat,
        channel_loc_path=args.channels_loc,
        dataset_root=args.dataset_root,
        session_id=args.session_id,
        subject_id=args.subject_id,
        freqs=tuple(float(item.strip()) for item in str(args.freqs).split(",") if item.strip()),
        include_hard_idle=bool(int(args.include_hard_idle)),
        include_pre_stim_idle=bool(int(args.include_pre_stim_idle)),
    )
    print(json_dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

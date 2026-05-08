from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import scipy.io as sio

from brain_workspace.paths import SSVEP_DATASET_DIR

from .async_fbcca_idle_standalone import TrialSpec, json_dumps
from .dataset import save_collection_dataset_bundle


BETA_FIGSHARE_RECORD = "https://doi.org/10.6084/m9.figshare.12264401"
BETA_FIGSHARE_API = "https://api.figshare.com/v2/articles/12264401"
BETA_REQUIRED_CHANNELS = ("Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4")
BETA_FREQS = (9.8, 12.0, 14.8, 15.8)
BETA_SAMPLING_RATE = 250
BETA_PRE_STIMULUS_SEC = 0.5
BETA_POST_STIMULUS_SEC = 0.5
BETA_BLOCKS = 4
BETA_TARGET_COUNT = 40
BETA_SOURCE_ROOT = SSVEP_DATASET_DIR / "external" / "beta" / "raw"


@dataclass(frozen=True)
class BetaLoadedSubject:
    subject: str
    mat_path: Path
    eeg: np.ndarray
    channel_names: tuple[str, ...]
    selected_channel_names: tuple[str, ...]
    selected_channel_indices: tuple[int, ...]
    target_frequencies: tuple[float, ...]
    target_phases: tuple[float, ...]
    sampling_rate: int
    stimulus_sec: float
    trial_sec: float


def _normalize_channel_name(value: str) -> str:
    return str(value).strip().lower()


def _subject_number(subject: str) -> int:
    raw = str(subject).strip().upper().lstrip("S")
    try:
        return int(raw)
    except Exception:
        return 0


def _extract_channel_names(chan_matrix: Any) -> tuple[str, ...]:
    matrix = np.asarray(chan_matrix, dtype=object)
    if matrix.shape != (64, 4):
        raise ValueError(f"BETA channel matrix must have shape (64, 4); got {matrix.shape}")
    names = tuple(str(row[3]).strip() for row in matrix)
    if len(names) != 64 or any(not name for name in names):
        raise ValueError("BETA channel matrix did not yield 64 nonempty channel names")
    return names


def selected_channel_indices(
    channel_names: Sequence[str],
    *,
    required_channels: Sequence[str] = BETA_REQUIRED_CHANNELS,
) -> tuple[int, ...]:
    normalized_to_index = {
        _normalize_channel_name(name): int(index)
        for index, name in enumerate(tuple(str(item) for item in channel_names))
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
        raise ValueError(f"BETA subject is missing required channels: {missing}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"BETA required channels resolved to duplicate indices: {indices}")
    return tuple(indices)


def load_beta_subject(mat_path: Path) -> BetaLoadedSubject:
    resolved = Path(mat_path).expanduser().resolve()
    payload = sio.loadmat(resolved, squeeze_me=True, struct_as_record=False)
    if "data" not in payload:
        raise ValueError(f"BETA subject file missing variable 'data': {resolved}")
    data = payload["data"]
    if not hasattr(data, "EEG") or not hasattr(data, "suppl_info"):
        raise ValueError(f"BETA subject data must contain EEG and suppl_info: {resolved}")
    eeg = np.asarray(data.EEG, dtype=np.float64)
    if eeg.ndim != 4 or eeg.shape[0] != 64 or eeg.shape[2] != BETA_BLOCKS or eeg.shape[3] != BETA_TARGET_COUNT:
        raise ValueError(f"BETA EEG must have shape (64, time, 4, 40); got {eeg.shape}")
    info = data.suppl_info
    channel_names = _extract_channel_names(info.chan)
    indices = selected_channel_indices(channel_names)
    selected_names = tuple(channel_names[index] for index in indices)
    if tuple(_normalize_channel_name(name) for name in selected_names) != tuple(
        _normalize_channel_name(name) for name in BETA_REQUIRED_CHANNELS
    ):
        raise ValueError(f"selected channel order mismatch: {selected_names} != {BETA_REQUIRED_CHANNELS}")
    target_frequencies = tuple(float(freq) for freq in np.ravel(np.asarray(info.freqs, dtype=np.float64)))
    target_phases = tuple(float(phase) for phase in np.ravel(np.asarray(info.phases, dtype=np.float64)))
    if len(target_frequencies) != BETA_TARGET_COUNT:
        raise ValueError(f"BETA target frequency count must be 40; got {len(target_frequencies)}")
    sampling_rate = int(getattr(info, "srate", BETA_SAMPLING_RATE))
    if sampling_rate != BETA_SAMPLING_RATE:
        raise ValueError(f"BETA sampling rate must be 250Hz for this converter; got {sampling_rate}")
    trial_sec = float(eeg.shape[1]) / float(sampling_rate)
    stimulus_sec = max(0.0, trial_sec - BETA_PRE_STIMULUS_SEC - BETA_POST_STIMULUS_SEC)
    return BetaLoadedSubject(
        subject=str(getattr(info, "sub", resolved.stem)).strip() or resolved.stem,
        mat_path=resolved,
        eeg=np.ascontiguousarray(eeg, dtype=np.float64),
        channel_names=channel_names,
        selected_channel_names=selected_names,
        selected_channel_indices=indices,
        target_frequencies=target_frequencies,
        target_phases=target_phases,
        sampling_rate=sampling_rate,
        stimulus_sec=float(stimulus_sec),
        trial_sec=float(trial_sec),
    )


def resolve_beta_command_frequencies(
    subject: BetaLoadedSubject,
    freqs: Sequence[float] = BETA_FREQS,
) -> tuple[tuple[float, ...], dict[float, int]]:
    available = {round(float(freq), 10): int(index + 1) for index, freq in enumerate(subject.target_frequencies)}
    resolved: list[float] = []
    target_map: dict[float, int] = {}
    for value in tuple(freqs or ()):
        freq = round(float(value), 10)
        if freq not in available:
            raise ValueError(f"BETA target frequency is unavailable: {float(value):g}Hz")
        if any(abs(freq - existing) < 1e-9 for existing in resolved):
            raise ValueError(f"duplicate BETA command frequency: {float(value):g}Hz")
        resolved.append(float(freq))
        target_map[float(freq)] = int(available[float(freq)])
    if len(resolved) != 4:
        raise ValueError(f"BETA command frequency set must contain exactly 4 frequencies; got {len(resolved)}")
    return tuple(resolved), target_map


def _stimulus_segment(
    subject: BetaLoadedSubject,
    *,
    target_index_1based: int,
    block_index: int,
    channel_indices: Sequence[int],
) -> np.ndarray:
    target_index = int(target_index_1based) - 1
    start = int(round(BETA_PRE_STIMULUS_SEC * subject.sampling_rate))
    stop = start + int(round(subject.stimulus_sec * subject.sampling_rate))
    matrix = subject.eeg[np.asarray(channel_indices, dtype=int), start:stop, int(block_index), target_index]
    return np.ascontiguousarray(matrix.T, dtype=np.float64)


def _pre_stimulus_idle_segment(
    subject: BetaLoadedSubject,
    *,
    target_index_1based: int,
    block_index: int,
    channel_indices: Sequence[int],
) -> np.ndarray:
    target_index = int(target_index_1based) - 1
    stop = int(round(BETA_PRE_STIMULUS_SEC * subject.sampling_rate))
    matrix = subject.eeg[np.asarray(channel_indices, dtype=int), 0:stop, int(block_index), target_index]
    return np.ascontiguousarray(matrix.T, dtype=np.float64)


def build_beta_segments(
    subject: BetaLoadedSubject,
    *,
    freqs: Sequence[float] = BETA_FREQS,
    include_hard_idle: bool = True,
    include_pre_stim_idle: bool = False,
) -> list[tuple[TrialSpec, np.ndarray]]:
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    command_freqs, target_index_by_freq = resolve_beta_command_frequencies(subject, freqs)
    control_target_indices = {int(index) for index in target_index_by_freq.values()}
    for block_index in range(BETA_BLOCKS):
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
                        subject,
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
                            subject,
                            target_index_1based=target_index,
                            block_index=block_index,
                            channel_indices=subject.selected_channel_indices,
                        ),
                    )
                )
                trial_id += 1
        if bool(include_hard_idle):
            for target_index in range(1, BETA_TARGET_COUNT + 1):
                if int(target_index) in control_target_indices:
                    continue
                target_freq = float(subject.target_frequencies[int(target_index) - 1])
                segments.append(
                    (
                        TrialSpec(
                            label=f"hard_idle_beta_target{int(target_index):02d}_{target_freq:g}Hz",
                            expected_freq=None,
                            trial_id=int(trial_id),
                            block_index=int(block_index),
                        ),
                        _stimulus_segment(
                            subject,
                            target_index_1based=int(target_index),
                            block_index=block_index,
                            channel_indices=subject.selected_channel_indices,
                        ),
                    )
                )
                trial_id += 1
    return segments


def beta_protocol_config(
    subject: BetaLoadedSubject,
    *,
    freqs: Sequence[float] = BETA_FREQS,
    saved_trial_count: int,
    include_hard_idle: bool,
    include_pre_stim_idle: bool,
) -> dict[str, Any]:
    command_freqs, target_index_by_freq = resolve_beta_command_frequencies(subject, freqs)
    control_count = int(len(command_freqs) * BETA_BLOCKS)
    hard_idle_count = int((BETA_TARGET_COUNT - len(command_freqs)) * BETA_BLOCKS) if include_hard_idle else 0
    clean_idle_count = int(len(command_freqs) * BETA_BLOCKS) if include_pre_stim_idle else 0
    freq_token = "_".join(f"{float(freq):g}".replace(".", "p") for freq in command_freqs)
    return {
        "protocol_name": "external-beta-fourfreq-v1",
        "source_dataset": "BETA SSVEP database",
        "source_record": BETA_FIGSHARE_RECORD,
        "source_api": BETA_FIGSHARE_API,
        "subject_file": str(subject.mat_path),
        "subject_number": int(_subject_number(subject.subject)),
        "sampling_rate_original_hz": int(subject.sampling_rate),
        "sampling_rate": int(subject.sampling_rate),
        "active_sec": float(subject.stimulus_sec),
        "trial_sec": float(subject.trial_sec),
        "pre_stimulus_idle_sec": float(BETA_PRE_STIMULUS_SEC),
        "post_stimulus_sec": float(BETA_POST_STIMULUS_SEC),
        "stimulus_offset_sec": float(BETA_PRE_STIMULUS_SEC),
        "target_repeats": int(BETA_BLOCKS),
        "idle_repeats": int(clean_idle_count + hard_idle_count),
        "switch_trials": 0,
        "planned_total_trials": int(saved_trial_count),
        "saved_trial_count": int(saved_trial_count),
        "control_trial_count": int(control_count),
        "clean_idle_trial_count": int(clean_idle_count),
        "hard_idle_trial_count": int(hard_idle_count),
        "hard_idle_definition": "BETA stimulus trials whose target frequency is outside the four command frequencies",
        "include_hard_idle_non_command_targets": bool(include_hard_idle),
        "include_pre_stimulus_clean_idle": bool(include_pre_stim_idle),
        "freqs": [float(freq) for freq in command_freqs],
        "freq_token": str(freq_token),
        "all_target_frequencies": [float(freq) for freq in subject.target_frequencies],
        "target_phase_by_freq": {
            f"{float(freq):g}": float(subject.target_phases[int(index) - 1])
            for freq, index in target_index_by_freq.items()
        },
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
        "stimulus_profile_id": "external_beta_original",
        "stimulus_mode": "external_original_beta",
        "stimulus_backend": "external_dataset",
        "stim_refresh_rate_hz": 60.0,
        "stim_mean": 0.0,
        "stim_amp": 0.0,
        "stim_luminance_min": 0.0,
        "stim_luminance_max": 0.0,
        "stim_michelson_contrast": 0.0,
        "ramp_sec": 0.0,
        "comfort_rating": None,
        "screen_brightness_note": "external BETA original stimulus; not generated by this repo",
    }


def convert_beta_subject_to_collection(
    *,
    mat_path: Path,
    dataset_root: Path,
    session_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    freqs: Sequence[float] = BETA_FREQS,
    include_hard_idle: bool = True,
    include_pre_stim_idle: bool = False,
) -> dict[str, Any]:
    subject = load_beta_subject(mat_path)
    command_freqs, target_index_by_freq = resolve_beta_command_frequencies(subject, freqs)
    segments = build_beta_segments(
        subject,
        freqs=command_freqs,
        include_hard_idle=bool(include_hard_idle),
        include_pre_stim_idle=bool(include_pre_stim_idle),
    )
    if not segments:
        raise RuntimeError(f"no BETA segments built for {mat_path}")
    if any(segment.shape[1] != len(BETA_REQUIRED_CHANNELS) for _trial, segment in segments):
        raise RuntimeError("BETA conversion produced a segment that is not exactly 8 channels")
    protocol = beta_protocol_config(
        subject,
        freqs=command_freqs,
        saved_trial_count=len(segments),
        include_hard_idle=bool(include_hard_idle),
        include_pre_stim_idle=bool(include_pre_stim_idle),
    )
    resolved_subject = subject_id or f"beta_{subject.subject.lower()}"
    freq_token = str(protocol.get("freq_token", "fourfreq"))
    resolved_session = session_id or f"{resolved_subject}_{freq_token}_strict8"
    payload = save_collection_dataset_bundle(
        dataset_root=Path(dataset_root),
        session_id=resolved_session,
        subject_id=resolved_subject,
        serial_port="external_beta",
        board_id=-1,
        sampling_rate=int(subject.sampling_rate),
        freqs=command_freqs,
        board_eeg_channels=tuple(range(len(BETA_REQUIRED_CHANNELS))),
        protocol_config=protocol,
        trial_segments=segments,
    )
    manifest_path = Path(str(payload["dataset_manifest"])).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_dataset_validation"] = {
        "dataset": "BETA",
        "required_channel_names": [str(name) for name in BETA_REQUIRED_CHANNELS],
        "selected_channel_names": [str(name) for name in subject.selected_channel_names],
        "selected_channel_indices_zero_based": [int(index) for index in subject.selected_channel_indices],
        "source_shape": [int(item) for item in subject.eeg.shape],
        "used_shape_per_control_trial": [int(subject.stimulus_sec * subject.sampling_rate), 8],
        "used_shape_per_hard_idle_trial": [int(subject.stimulus_sec * subject.sampling_rate), 8],
        "only_required_channels_saved": True,
        "excluded_channel_count": int(len(subject.channel_names) - len(subject.selected_channel_names)),
        "include_hard_idle_non_command_targets": bool(include_hard_idle),
        "include_pre_stimulus_clean_idle": bool(include_pre_stim_idle),
        "freqs": [float(freq) for freq in command_freqs],
        "all_target_frequencies": [float(freq) for freq in subject.target_frequencies],
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
        description="Convert BETA benchmark subject data into this repo's strict 8-channel collection format."
    )
    parser.add_argument("--mat", type=Path, required=True, help="Path to BETA S*.mat subject file.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Output collection dataset root.")
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument(
        "--freqs",
        type=str,
        default=",".join(f"{float(freq):g}" for freq in BETA_FREQS),
        help="Comma-separated four BETA target frequencies to mark as commands.",
    )
    parser.add_argument(
        "--include-hard-idle",
        type=int,
        default=1,
        help="Include non-command target stimulus windows as hard idle segments.",
    )
    parser.add_argument(
        "--include-pre-stim-idle",
        type=int,
        default=0,
        help="Also include 0.5s pre-stimulus clean idle segments.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = convert_beta_subject_to_collection(
        mat_path=args.mat,
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

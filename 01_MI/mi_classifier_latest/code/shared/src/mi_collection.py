"""Utilities for collecting and saving custom motor-imagery EEG sessions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
MNE_HOME_DIR = RUNTIME_ROOT / "mne_home"
MNE_HOME_DIR.mkdir(parents=True, exist_ok=True)
# Avoid writing MNE config under user home (can fail due lock/permission in some environments).
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(MNE_HOME_DIR))

import mne
import numpy as np


MI_CLASSES = [
    {
        "key": "left_hand",
        "display_name": "左手想象",
        "short_name": "左手",
        "color": "#2E86DE",
        "cue_marker": 301,
        "imagery_marker": 311,
    },
    {
        "key": "right_hand",
        "display_name": "右手想象",
        "short_name": "右手",
        "color": "#E67E22",
        "cue_marker": 302,
        "imagery_marker": 312,
    },
    {
        "key": "feet",
        "display_name": "双脚想象",
        "short_name": "双脚",
        "color": "#16A085",
        "cue_marker": 303,
        "imagery_marker": 313,
    },
    {
        "key": "tongue",
        "display_name": "舌头想象",
        "short_name": "舌头",
        "color": "#C0392B",
        "cue_marker": 304,
        "imagery_marker": 314,
    },
]

CLASS_LOOKUP = {item["key"]: item for item in MI_CLASSES}
CLASS_NAME_TO_LABEL = {item["key"]: index for index, item in enumerate(MI_CLASSES)}
CLASS_MARKER_LOOKUP = {
    item["cue_marker"]: item["key"] for item in MI_CLASSES
} | {item["imagery_marker"]: item["key"] for item in MI_CLASSES}

CONTINUOUS_PROMPT_LABELS = ["left_hand", "right_hand", "feet", "tongue", "no_control"]
CONTINUOUS_PROMPT_EVENT_NAMES = {
    "left_hand": "continuous_command_left_hand",
    "right_hand": "continuous_command_right_hand",
    "feet": "continuous_command_feet",
    "tongue": "continuous_command_tongue",
    "no_control": "continuous_command_no_control",
}

DEFAULT_ARTIFACT_TYPES = ["eye_movement", "blink", "swallow", "jaw", "head_motion"]
ARTIFACT_EVENT_INTERVALS = {
    "eye_movement": ("eye_movement_block_start", "eye_movement_block_end"),
    "blink": ("blink_block_start", "blink_block_end"),
    "swallow": ("swallow_block_start", "swallow_block_end"),
    "jaw": ("jaw_block_start", "jaw_block_end"),
    "head_motion": ("head_motion_block_start", "head_motion_block_end"),
}
ARTIFACT_EVENT_NAMES = sorted({name for pair in ARTIFACT_EVENT_INTERVALS.values() for name in pair})

EVENT_CODE_MAP = {
    100: "session_start",
    101: "session_end",
    110: "baseline_start",
    111: "baseline_end",
    112: "fixation_start",
    120: "cue_start",
    122: "cue_end",
    121: "imagery_start",
    130: "mi_run_start",
    131: "mi_run_end",
    140: "quality_check_start",
    141: "quality_check_end",
    150: "calibration_start",
    151: "calibration_end",
    152: "practice_start",
    153: "practice_end",
    160: "eyes_open_rest_start",
    161: "eyes_open_rest_end",
    162: "eyes_closed_rest_start",
    163: "eyes_closed_rest_end",
    164: "eye_movement_block_start",
    165: "eye_movement_block_end",
    166: "blink_block_start",
    167: "blink_block_end",
    168: "swallow_block_start",
    169: "swallow_block_end",
    170: "jaw_block_start",
    171: "jaw_block_end",
    172: "head_motion_block_start",
    173: "head_motion_block_end",
    180: "run_rest_start",
    181: "run_rest_end",
    200: "trial_start",
    210: "trial_end",
    320: "imagery_end",
    330: "iti_start",
    400: "idle_block_start",
    401: "idle_block_end",
    402: "idle_prepare_start",
    403: "idle_prepare_end",
    500: "continuous_block_start",
    501: "continuous_block_end",
    510: "continuous_command_left_hand",
    511: "continuous_command_right_hand",
    512: "continuous_command_feet",
    513: "continuous_command_tongue",
    514: "continuous_command_no_control",
    515: "continuous_command_end",
    901: "bad_trial_marked",
    950: "pause",
    951: "resume",
}
EVENT_CODE_MAP.update({item["cue_marker"]: f"cue_{item['key']}" for item in MI_CLASSES})
EVENT_CODE_MAP.update({item["imagery_marker"]: f"imagery_{item['key']}" for item in MI_CLASSES})

EVENT_NAME_TO_CODE = {name: code for code, name in EVENT_CODE_MAP.items()}
GENERIC_CHANNEL_NAMES = [f"EEG{i + 1}" for i in range(32)]
COLLECTION_SCHEMA_VERSION = 2
COLLECTION_EXPORTER_NAME = "mi_collection"
COLLECTION_MANIFEST_NAME = "collection_manifest.csv"
COLLECTION_MANIFEST_FIELDS = [
    "saved_at",
    "schema_version",
    "subject_id",
    "session_id",
    "protocol_mode",
    "save_index",
    "run_stem",
    "trials_per_class",
    "mi_run_count",
    "trials_per_run",
    "trial_count",
    "accepted_trial_count",
    "rejected_trial_count",
    "sampling_rate_hz",
    "channel_names",
    "class_names",
    "board_data_npy",
    "board_map_json",
    "mi_epochs_npz",
    "mi_epochs_meta_json",
    "gate_epochs_npz",
    "gate_epochs_meta_json",
    "artifact_epochs_npz",
    "artifact_epochs_meta_json",
    "continuous_npz",
    "continuous_meta_json",
    "session_raw_fif",
    "events_csv",
    "trials_csv",
    "segments_csv",
    "session_meta_json",
    "quality_report_json",
]
PAUSE_REJECTED_TRIAL_NOTE = "会话中发生暂停，自动标记为坏试次"


@dataclass
class SessionSettings:
    """Serializable session configuration used by the collection UI."""

    subject_id: str
    session_id: str
    output_root: str
    board_id: int
    serial_port: str
    channel_names: list[str]
    channel_positions: list[int]
    trials_per_class: int
    baseline_sec: float
    cue_sec: float
    imagery_sec: float
    iti_sec: float
    random_seed: int
    protocol_mode: str = "full"
    run_count: int = 3
    max_consecutive_same_class: int = 2
    run_rest_sec: float = 60.0
    long_run_rest_every: int = 2
    long_run_rest_sec: float = 120.0
    quality_check_sec: float = 0.0
    practice_sec: float = 180.0
    calibration_open_sec: float = 60.0
    calibration_closed_sec: float = 60.0
    calibration_eye_sec: float = 30.0
    calibration_blink_sec: float = 20.0
    calibration_swallow_sec: float = 20.0
    calibration_jaw_sec: float = 20.0
    calibration_head_sec: float = 20.0
    idle_block_count: int = 2
    idle_block_sec: float = 60.0
    idle_prepare_block_count: int = 0
    idle_prepare_sec: float = 0.0
    continuous_block_count: int = 2
    continuous_block_sec: float = 240.0
    continuous_command_min_sec: float = 4.0
    continuous_command_max_sec: float = 5.0
    continuous_gap_min_sec: float = 2.0
    continuous_gap_max_sec: float = 3.0
    include_eyes_closed_rest_in_gate_neg: bool = False
    artifact_types: list[str] = field(default_factory=lambda: list(DEFAULT_ARTIFACT_TYPES))
    reference_mode: str = ""
    participant_state: str = ""
    caffeine_intake: str = ""
    recent_exercise: str = ""
    sleep_note: str = ""
    operator: str = ""
    notes: str = ""
    board_name: str = ""


@dataclass
class TrialRecord:
    """Runtime metadata for one MI trial."""

    trial_id: int
    class_name: str
    display_name: str
    run_index: int = 1
    run_trial_index: int = 1
    accepted: bool = True
    note: str = ""
    cue_onset_sample: int | None = None
    imagery_onset_sample: int | None = None
    imagery_offset_sample: int | None = None
    trial_end_sample: int | None = None

    def to_row(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "run_index": self.run_index,
            "run_trial_index": self.run_trial_index,
            "class_name": self.class_name,
            "display_name": self.display_name,
            "accepted": int(self.accepted),
            "cue_onset_sample": self.cue_onset_sample,
            "imagery_onset_sample": self.imagery_onset_sample,
            "imagery_offset_sample": self.imagery_offset_sample,
            "trial_end_sample": self.trial_end_sample,
            "note": self.note,
        }


def default_channel_names(expected_count: int) -> list[str]:
    """Return a generic channel-name list when the user leaves it blank."""
    if expected_count <= len(GENERIC_CHANNEL_NAMES):
        return GENERIC_CHANNEL_NAMES[:expected_count]
    return [f"EEG{i + 1}" for i in range(expected_count)]


def parse_channel_names(raw_value: str | list[str] | None, expected_count: int | None = None) -> list[str]:
    """Parse a comma-separated channel-name list."""
    if raw_value is None:
        return default_channel_names(expected_count or 8)
    if isinstance(raw_value, list):
        parsed = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        parsed = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not parsed and expected_count is not None:
        return default_channel_names(expected_count)
    if expected_count is not None and len(parsed) != expected_count:
        raise ValueError(f"通道名称数量应为 {expected_count}，当前为 {len(parsed)}。")
    if len(set(parsed)) != len(parsed):
        raise ValueError("通道名称中存在重复项。")
    return parsed


def parse_channel_positions(raw_value: str | list[int] | tuple[int, ...] | None, expected_count: int) -> list[int]:
    """Parse a channel-position list from the UI."""
    if raw_value is None or str(raw_value).strip() == "":
        return list(range(expected_count))
    if isinstance(raw_value, (list, tuple)):
        parsed = [int(item) for item in raw_value]
    else:
        tokens = [item.strip() for item in str(raw_value).split(",") if item.strip()]
        parsed = [int(item) for item in tokens]
    if len(parsed) != expected_count:
        raise ValueError(f"通道位置数量应为 {expected_count}，当前为 {len(parsed)}。")
    if min(parsed) < 0:
        raise ValueError("通道位置不能为负数。")
    if len(set(parsed)) != len(parsed):
        raise ValueError("通道位置中存在重复索引。")
    return parsed


def normalize_artifact_type_name(raw_name: str) -> str:
    """Normalize artifact type aliases to canonical names."""
    token = str(raw_name).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "eye": "eye_movement",
        "eye_move": "eye_movement",
        "eye_movement": "eye_movement",
        "blink": "blink",
        "swallow": "swallow",
        "jaw": "jaw",
        "jaw_clench": "jaw",
        "head": "head_motion",
        "head_move": "head_motion",
        "head_motion": "head_motion",
    }
    return aliases.get(token, token)


def normalize_artifact_types(raw_value: str | list[str] | None) -> list[str]:
    """Parse and normalize selected artifact types."""
    if raw_value is None:
        return list(DEFAULT_ARTIFACT_TYPES)
    if isinstance(raw_value, list):
        tokens = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        tokens = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not tokens:
        return list(DEFAULT_ARTIFACT_TYPES)
    normalized: list[str] = []
    for token in tokens:
        name = normalize_artifact_type_name(token)
        if name in ARTIFACT_EVENT_INTERVALS and name not in normalized:
            normalized.append(name)
    return normalized or list(DEFAULT_ARTIFACT_TYPES)


def build_balanced_trial_sequence(
    trials_per_class: int,
    random_seed: int,
    max_consecutive_same_class: int = 2,
) -> list[str]:
    """Create a balanced class order with optional max-consecutive constraint."""
    if trials_per_class <= 0:
        raise ValueError("每类试次数必须大于 0。")

    rng = np.random.default_rng(int(random_seed))
    class_names = [item["key"] for item in MI_CLASSES]
    total_count = int(trials_per_class) * len(class_names)
    consecutive_limit = max(0, int(max_consecutive_same_class))

    counts = {name: int(trials_per_class) for name in class_names}
    sequence: list[str] = []
    last_name = ""
    streak_length = 0
    while len(sequence) < total_count:
        candidates = [name for name in class_names if counts[name] > 0]
        if consecutive_limit > 0 and streak_length >= consecutive_limit and last_name:
            candidates = [name for name in candidates if name != last_name]
        if not candidates:
            break

        max_count = max(counts[name] for name in candidates)
        preferred = [name for name in candidates if counts[name] == max_count]
        chosen_index = int(rng.integers(0, len(preferred)))
        chosen = str(preferred[chosen_index])
        sequence.append(chosen)
        counts[chosen] -= 1
        if chosen == last_name:
            streak_length += 1
        else:
            last_name = chosen
            streak_length = 1

    if len(sequence) == total_count:
        return sequence

    # Fallback to legacy block-wise shuffle.
    sequence = []
    for _ in range(int(trials_per_class)):
        block = class_names.copy()
        rng.shuffle(block)
        sequence.extend(block)
    return sequence


def sanitize_session_token(value: str) -> str:
    """Convert a free-form subject/session token into a filesystem-safe name.

    Keep non-illegal Windows filename characters (including Chinese names).
    """
    token = str(value).strip()
    token = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", token)
    token = re.sub(r"\s+", "_", token)
    token = token.strip("._ ")
    return token or "unknown"


def create_session_folder(output_root: str | Path, subject_id: str, session_id: str) -> Path:
    """Build the output directory for one recording session."""
    subject_token = sanitize_session_token(subject_id)
    session_token = sanitize_session_token(session_id)
    folder = Path(output_root) / f"sub-{subject_token}" / f"ses-{session_token}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _next_save_index(session_dir: Path) -> int:
    """Return next save index under one session directory.

    Scan every run-scoped artifact, not only metadata, so a failed partial save
    still reserves its save index and cannot be overwritten by a retry.
    """
    run_pattern = re.compile(r"_run-(\d{3})_")
    max_index = 0
    for path in session_dir.glob("*"):
        if not path.is_file():
            continue
        matched = run_pattern.search(path.name)
        if matched:
            max_index = max(max_index, int(matched.group(1)))
        elif path.name == "session_meta.json":
            # Legacy naming without run suffix should still reserve save index 1.
            max_index = max(max_index, 1)
    return max_index + 1


def _build_run_stem(
    *,
    settings: SessionSettings,
    save_index: int,
    trial_count: int,
    accepted_count: int,
) -> str:
    """Build one informative filename stem for all saved artifacts."""
    subject_token = sanitize_session_token(settings.subject_id)
    session_token = sanitize_session_token(settings.session_id)
    return (
        f"sub-{subject_token}"
        f"_ses-{session_token}"
        f"_run-{int(save_index):03d}"
        f"_tpc-{int(settings.trials_per_class):02d}"
        f"_n-{int(trial_count):03d}"
        f"_ok-{int(accepted_count):03d}"
    )


def _detect_session_bounds(marker_channel: np.ndarray) -> tuple[int, int]:
    """Find session-start/session-end markers and return a crop range."""
    marker_channel = np.asarray(marker_channel, dtype=np.float64)
    start_matches = np.flatnonzero(np.isclose(marker_channel, 100.0))
    end_matches = np.flatnonzero(np.isclose(marker_channel, 101.0))
    start_index = int(start_matches[0]) if start_matches.size else 0
    end_index = int(end_matches[-1]) if end_matches.size else marker_channel.size - 1
    if end_index < start_index:
        end_index = marker_channel.size - 1
    return start_index, end_index


def _extract_marker_occurrences(marker_channel: np.ndarray, crop_start: int) -> list[dict[str, object]]:
    """Return all non-zero markers within the cropped session window."""
    occurrences: list[dict[str, object]] = []
    marker_values = np.asarray(marker_channel, dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(marker_values) > 1e-9)
    for index in nonzero:
        marker_code = int(round(float(marker_values[index])))
        occurrences.append(
            {
                "marker_code": marker_code,
                "event_name": EVENT_CODE_MAP.get(marker_code, f"marker_{marker_code}"),
                "sample_index": int(index),
                "absolute_sample_index": int(index + crop_start),
            }
        )
    return occurrences


def _attach_sample_indices(
    event_log: list[dict[str, object]],
    recorded_markers: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Match runtime events to the recorded BrainFlow marker channel in order.

    The runtime event log and marker channel must be a strict 1:1 match. If a
    marker is missing, duplicated, or reordered, abort saving instead of
    producing silently corrupted labels.
    """
    expected_codes = [int(event["marker_code"]) for event in event_log]
    recorded_codes = [int(marker["marker_code"]) for marker in recorded_markers]
    if len(expected_codes) != len(recorded_codes):
        raise ValueError(
            "Marker sequence length mismatch: "
            f"expected {len(expected_codes)} events, recorded {len(recorded_codes)} markers."
        )

    first_mismatch_index = next(
        (
            index
            for index, (expected_code, recorded_code) in enumerate(zip(expected_codes, recorded_codes))
            if int(expected_code) != int(recorded_code)
        ),
        None,
    )
    if first_mismatch_index is not None:
        expected_event = event_log[first_mismatch_index]
        recorded_marker = recorded_markers[first_mismatch_index]
        raise ValueError(
            "Marker sequence mismatch at position "
            f"{first_mismatch_index}: expected code {int(expected_event['marker_code'])} "
            f"({expected_event.get('event_name', '')}), recorded code {int(recorded_marker['marker_code'])} "
            f"({recorded_marker.get('event_name', '')})."
        )

    enriched: list[dict[str, object]] = []
    for event, matched_marker in zip(event_log, recorded_markers):
        merged = dict(event)
        merged["sample_index"] = int(matched_marker["sample_index"])
        merged["absolute_sample_index"] = int(matched_marker["absolute_sample_index"])
        enriched.append(merged)

    return enriched


def _update_trials_from_events(
    trial_records: list[TrialRecord],
    events: list[dict[str, object]],
) -> list[TrialRecord]:
    """Populate sample indices in TrialRecord objects from the event log."""
    records = [TrialRecord(**asdict(trial)) if isinstance(trial, TrialRecord) else TrialRecord(**trial) for trial in trial_records]
    by_id = {int(trial.trial_id): trial for trial in records}

    def _set_earliest_sample(trial: TrialRecord, field_name: str, sample_index: int) -> None:
        current_value = getattr(trial, field_name)
        new_value = int(sample_index)
        if current_value is None or new_value < int(current_value):
            setattr(trial, field_name, new_value)

    for event in events:
        trial_id = event.get("trial_id")
        if trial_id is None:
            continue
        trial_key = int(trial_id)
        if trial_key not in by_id:
            continue
        sample_index = event.get("sample_index")
        if sample_index is None:
            continue
        event_name = str(event.get("event_name"))
        trial = by_id[trial_key]
        if (event_name.startswith("cue_") and event_name != "cue_end") or event_name == "cue_start":
            _set_earliest_sample(trial, "cue_onset_sample", int(sample_index))
        elif event_name == "imagery_end":
            trial.imagery_offset_sample = int(sample_index)
        elif event_name.startswith("imagery_") or event_name == "imagery_start":
            _set_earliest_sample(trial, "imagery_onset_sample", int(sample_index))
        elif event_name == "trial_end":
            trial.trial_end_sample = int(sample_index)
        elif event_name == "bad_trial_marked":
            trial.accepted = False
        run_index = event.get("run_index")
        if run_index is not None:
            trial.run_index = int(run_index)
        run_trial_index = event.get("run_trial_index")
        if run_trial_index is not None:
            trial.run_trial_index = int(run_trial_index)

    return records


def _make_annotations(
    events: list[dict[str, object]],
    sfreq: float,
    segment_rows: list[dict[str, object]] | None = None,
) -> mne.Annotations:
    """Convert enriched events into MNE annotations."""
    onsets = []
    durations = []
    descriptions = []
    for event in events:
        sample_index = event.get("sample_index")
        if sample_index is None:
            continue
        onsets.append(float(sample_index) / float(sfreq))
        durations.append(0.0)
        descriptions.append(str(event["event_name"]))
    for row in segment_rows or []:
        start_sample = row.get("start_sample")
        end_sample = row.get("end_sample")
        if start_sample is None or end_sample is None:
            continue
        start_index = int(start_sample)
        stop_index = max(int(end_sample), start_index + 1)
        label = str(row.get("label", "")).strip()
        description = str(row.get("segment_type", "segment")).strip() or "segment"
        if label:
            description = f"{description}:{label}"
        onsets.append(float(start_index) / float(sfreq))
        durations.append(float((stop_index - start_index) / float(sfreq)))
        descriptions.append(description)
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def _maybe_set_standard_montage(raw: mne.io.BaseRaw, channel_names: list[str]) -> None:
    """Attach the standard_1020 montage when channel names look compatible."""
    if not channel_names:
        return
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        if all(name in montage.ch_names for name in channel_names):
            raw.set_montage(montage, on_missing="warn")
    except Exception:
        pass


def _build_quality_summary(eeg_uvolts: np.ndarray, channel_names: list[str]) -> dict[str, object]:
    """Compute simple post-session quality statistics."""
    def _stats(values: np.ndarray) -> dict[str, object]:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite_values = array[np.isfinite(array)]
        non_finite_count = int(array.size - finite_values.size)
        if finite_values.size == 0:
            return {
                "std_uV": 0.0,
                "peak_to_peak_uV": 0.0,
                "rms_uV": 0.0,
                "sample_count": int(array.size),
                "finite_sample_count": 0,
                "non_finite_sample_count": non_finite_count,
            }
        return {
            "std_uV": float(np.std(finite_values)),
            "peak_to_peak_uV": float(np.ptp(finite_values)),
            "rms_uV": float(np.sqrt(np.mean(np.square(finite_values)))),
            "sample_count": int(array.size),
            "finite_sample_count": int(finite_values.size),
            "non_finite_sample_count": non_finite_count,
        }

    overall = _stats(eeg_uvolts)
    summary = {
        "overall_std_uV": float(overall["std_uV"]),
        "overall_peak_to_peak_uV": float(overall["peak_to_peak_uV"]),
        "overall_rms_uV": float(overall["rms_uV"]),
        "sample_count": int(overall["sample_count"]),
        "finite_sample_count": int(overall["finite_sample_count"]),
        "non_finite_sample_count": int(overall["non_finite_sample_count"]),
        "channels": [],
    }
    for index, channel_name in enumerate(channel_names):
        channel_data = eeg_uvolts[index]
        channel_stats = _stats(channel_data)
        summary["channels"].append(
            {
                "channel_name": channel_name,
                "std_uV": float(channel_stats["std_uV"]),
                "peak_to_peak_uV": float(channel_stats["peak_to_peak_uV"]),
                "rms_uV": float(channel_stats["rms_uV"]),
                "sample_count": int(channel_stats["sample_count"]),
                "finite_sample_count": int(channel_stats["finite_sample_count"]),
                "non_finite_sample_count": int(channel_stats["non_finite_sample_count"]),
            }
        )
    return summary


def _build_data_flow_summary(
    *,
    cropped: np.ndarray,
    marker_channel: np.ndarray,
    timestamp_row: int | None,
    package_num_row: int | None,
    sampling_rate: float,
    event_count: int,
) -> dict[str, object]:
    """Summarize hardware data-flow rows preserved in the raw BrainFlow matrix."""
    sample_count = int(cropped.shape[1])
    row_count = int(cropped.shape[0])
    expected_sampling_rate = float(sampling_rate)
    marker_values = np.asarray(marker_channel, dtype=np.float64).reshape(-1)
    finite_markers = marker_values[np.isfinite(marker_values)]
    marker_count = int(np.count_nonzero(np.abs(finite_markers) > 1e-9))
    summary: dict[str, object] = {
        "row_count": row_count,
        "sample_count": sample_count,
        "expected_sampling_rate_hz": expected_sampling_rate,
        "duration_sec_from_samples": (
            float(sample_count / expected_sampling_rate) if expected_sampling_rate > 0 else 0.0
        ),
        "marker_count": marker_count,
        "event_count": int(event_count),
        "marker_event_count_match": bool(marker_count == int(event_count)),
        "timestamp_available": False,
        "timestamp_monotonic": None,
        "timestamp_non_finite_count": None,
        "timestamp_span_sec": None,
        "effective_sampling_rate_hz": None,
        "package_num_available": False,
        "package_non_finite_count": None,
        "package_first": None,
        "package_last": None,
        "package_jump_count": None,
    }

    if timestamp_row is not None and 0 <= int(timestamp_row) < row_count:
        timestamps = np.asarray(cropped[int(timestamp_row)], dtype=np.float64).reshape(-1)
        finite_timestamps = timestamps[np.isfinite(timestamps)]
        summary["timestamp_available"] = True
        summary["timestamp_non_finite_count"] = int(timestamps.size - finite_timestamps.size)
        if finite_timestamps.size >= 2:
            diffs = np.diff(finite_timestamps)
            summary["timestamp_monotonic"] = bool(np.all(diffs >= 0.0))
            timestamp_span_sec = float(finite_timestamps[-1] - finite_timestamps[0])
            summary["timestamp_span_sec"] = timestamp_span_sec
            summary["effective_sampling_rate_hz"] = (
                float((finite_timestamps.size - 1) / timestamp_span_sec)
                if timestamp_span_sec > 0.0
                else None
            )
        elif finite_timestamps.size == 1:
            summary["timestamp_monotonic"] = True
            summary["timestamp_span_sec"] = 0.0
        else:
            summary["timestamp_monotonic"] = False

    if package_num_row is not None and 0 <= int(package_num_row) < row_count:
        packages = np.asarray(cropped[int(package_num_row)], dtype=np.float64).reshape(-1)
        finite_packages = packages[np.isfinite(packages)]
        summary["package_num_available"] = True
        summary["package_non_finite_count"] = int(packages.size - finite_packages.size)
        if finite_packages.size:
            package_ints = np.asarray(np.rint(finite_packages), dtype=np.int64) % 256
            summary["package_first"] = int(package_ints[0])
            summary["package_last"] = int(package_ints[-1])
            if package_ints.size >= 2:
                expected_next = (package_ints[:-1] + 1) % 256
                summary["package_jump_count"] = int(np.count_nonzero(package_ints[1:] != expected_next))
            else:
                summary["package_jump_count"] = 0

    return summary


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write a UTF-8 CSV file from dictionaries."""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a UTF-8 JSON file with stable formatting."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _coerce_manifest_save_index(raw_row: dict[str, object]) -> int | str:
    """Best-effort conversion from legacy/new manifest rows to save_index."""
    for field_name in ("save_index", "run_index"):
        raw_value = str(raw_row.get(field_name, "")).strip()
        if not raw_value:
            continue
        try:
            return int(float(raw_value))
        except Exception:
            continue
    run_stem = str(raw_row.get("run_stem", "")).strip()
    matched = re.search(r"_run-(\d{3})_", run_stem)
    if matched:
        return int(matched.group(1))
    return ""


def _normalize_manifest_row(raw_row: dict[str, object]) -> dict[str, object]:
    """Normalize legacy/new manifest rows into the current schema."""
    row = {field: raw_row.get(field, "") for field in COLLECTION_MANIFEST_FIELDS}
    schema_version = str(raw_row.get("schema_version", "")).strip()
    if not schema_version:
        schema_version = "1" if str(raw_row.get("run_index", "")).strip() else ""
    row["schema_version"] = schema_version
    row["save_index"] = _coerce_manifest_save_index(raw_row)
    if not str(row.get("protocol_mode", "")).strip():
        row["protocol_mode"] = "full"
    if not row.get("artifact_epochs_npz"):
        row["artifact_epochs_npz"] = raw_row.get("artifact_npz", "")
    if not row.get("mi_epochs_npz"):
        row["mi_epochs_npz"] = raw_row.get("epochs_npz", raw_row.get("legacy_epochs_npz", ""))
    return row


def _ensure_manifest_schema(manifest_path: Path) -> None:
    """Rewrite manifest in-place when the header schema does not match current fields."""
    if not manifest_path.exists():
        return

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        existing_fields = list(reader.fieldnames or [])
        if existing_fields == COLLECTION_MANIFEST_FIELDS:
            return
        existing_rows = [dict(row) for row in reader]

    backup_path = manifest_path.with_name(
        f"{manifest_path.stem}_legacy_schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}{manifest_path.suffix}"
    )
    manifest_path.replace(backup_path)

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLLECTION_MANIFEST_FIELDS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(_normalize_manifest_row(row))


def _string_array(values: str | list[object] | tuple[object, ...] | np.ndarray) -> np.ndarray:
    """Store text arrays as numpy unicode instead of object arrays."""
    if isinstance(values, str):
        return np.asarray([values], dtype=np.str_)
    return np.asarray([str(item) for item in list(values)], dtype=np.str_)


def _relative_path(path: Path, root: str | Path) -> str:
    """Return a portable relative path when possible."""
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except Exception:
        return path.name


def _compute_sha256(path: Path) -> str:
    """Compute a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _append_collection_manifest(output_root: str | Path, record: dict[str, object]) -> Path:
    """Append one run-level row to dataset manifest CSV for easier training traceability."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / COLLECTION_MANIFEST_NAME
    _ensure_manifest_schema(manifest_path)
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLLECTION_MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(_normalize_manifest_row(record))
    return manifest_path


def _first_matching_event(
    events: list[dict[str, object]],
    *,
    trial_id: int,
    event_names: list[str],
) -> dict[str, object] | None:
    for event in events:
        event_trial_id = event.get("trial_id")
        if event_trial_id is None:
            continue
        if int(event_trial_id) != int(trial_id):
            continue
        if str(event.get("event_name", "")) not in event_names:
            continue
        sample_index = event.get("sample_index")
        if sample_index is not None:
            return event
    return None


def _first_event_sample(
    events: list[dict[str, object]],
    *,
    trial_id: int,
    event_names: list[str],
) -> int | None:
    matched_event = _first_matching_event(events, trial_id=trial_id, event_names=event_names)
    if matched_event is not None:
        return int(matched_event["sample_index"])
    return None


def _extract_intervals(events: list[dict[str, object]], start_name: str, end_name: str) -> list[tuple[int, int]]:
    """Extract ordered [start, end) intervals from event start/end pairs."""
    intervals: list[tuple[int, int]] = []
    pending_starts: list[int] = []
    for event in events:
        name = str(event.get("event_name", ""))
        sample = event.get("sample_index")
        if sample is None:
            continue
        sample_index = int(sample)
        if name == start_name:
            pending_starts.append(sample_index)
        elif name == end_name and pending_starts:
            start_index = pending_starts.pop(0)
            stop_index = max(sample_index, start_index + 1)
            intervals.append((start_index, stop_index))
    return intervals


def _normalize_intervals(intervals: list[tuple[int, int]], *, total_samples: int) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    total = max(0, int(total_samples))
    for start, stop in intervals:
        s = max(0, int(start))
        t = min(total, int(stop))
        if t > s:
            cleaned.append((s, t))
    if not cleaned:
        return []
    cleaned.sort()
    merged: list[tuple[int, int]] = [cleaned[0]]
    for start, stop in cleaned[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _extract_pause_intervals(events: list[dict[str, object]], *, total_samples: int) -> list[tuple[int, int]]:
    pause_intervals: list[tuple[int, int]] = []
    pause_start: int | None = None
    for event in events:
        sample = event.get("sample_index")
        if sample is None:
            continue
        sample_index = int(sample)
        event_name = str(event.get("event_name", ""))
        if event_name == "pause":
            if pause_start is None:
                pause_start = sample_index
            continue
        if event_name == "resume" and pause_start is not None:
            pause_intervals.append((pause_start, max(pause_start + 1, sample_index)))
            pause_start = None
    if pause_start is not None:
        pause_intervals.append((pause_start, max(pause_start + 1, int(total_samples))))
    return _normalize_intervals(pause_intervals, total_samples=total_samples)


def _interval_overlaps_any(start: int, stop: int, intervals: list[tuple[int, int]]) -> bool:
    s = int(start)
    t = int(stop)
    if t <= s:
        return False
    for excluded_start, excluded_stop in intervals:
        if int(excluded_start) < t and s < int(excluded_stop):
            return True
    return False


def _subtract_intervals(interval: tuple[int, int], excluded_intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    start, stop = int(interval[0]), int(interval[1])
    if stop <= start:
        return []
    if not excluded_intervals:
        return [(start, stop)]
    fragments: list[tuple[int, int]] = []
    cursor = start
    for excluded_start, excluded_stop in excluded_intervals:
        excluded_s = int(excluded_start)
        excluded_t = int(excluded_stop)
        if excluded_t <= cursor:
            continue
        if excluded_s >= stop:
            break
        if excluded_s > cursor:
            fragments.append((cursor, min(excluded_s, stop)))
        cursor = max(cursor, excluded_t)
        if cursor >= stop:
            break
    if cursor < stop:
        fragments.append((cursor, stop))
    return [(s, t) for s, t in fragments if t > s]


def _append_note_once(note: str, addition: str) -> str:
    base = str(note).strip()
    extra = str(addition).strip()
    if not extra:
        return base
    if not base:
        return extra
    if extra in base:
        return base
    return f"{base}; {extra}"


def _mark_trials_overlapping_intervals(
    trials: list[TrialRecord],
    events: list[dict[str, object]],
    excluded_intervals: list[tuple[int, int]],
    *,
    note: str,
) -> set[int]:
    rejected_trial_ids: set[int] = set()
    if not excluded_intervals:
        return rejected_trial_ids
    for trial in trials:
        trial_id = int(trial.trial_id)
        trial_start = _first_event_sample(
            events,
            trial_id=trial_id,
            event_names=["trial_start", "fixation_start", "baseline_start", "cue_start", "imagery_start"],
        )
        trial_end = trial.trial_end_sample
        if trial_end is None:
            trial_end = _first_event_sample(
                events,
                trial_id=trial_id,
                event_names=["trial_end", "imagery_end", "iti_start"],
            )
        if trial_start is None or trial_end is None or int(trial_end) <= int(trial_start):
            continue
        if _interval_overlaps_any(int(trial_start), int(trial_end), excluded_intervals):
            trial.accepted = False
            trial.note = _append_note_once(trial.note, note)
            rejected_trial_ids.add(trial_id)
    return rejected_trial_ids


def _normalize_execution_success(value: object) -> int | None:
    if value is None:
        return None
    try:
        return 0 if int(value) == 0 else 1
    except Exception:
        return 1 if bool(value) else 0


def _mark_continuous_prompts_overlapping_intervals(
    prompts: list[dict[str, object]],
    excluded_intervals: list[tuple[int, int]],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for prompt in prompts:
        updated = dict(prompt)
        updated["execution_success"] = _normalize_execution_success(updated.get("execution_success"))
        start_sample = int(updated.get("start_sample") or 0)
        end_sample = int(updated.get("end_sample") or (start_sample + 1))
        if _interval_overlaps_any(start_sample, end_sample, excluded_intervals):
            updated["execution_success"] = 0
        normalized.append(updated)
    return normalized


def _append_clean_interval(
    intervals: list[tuple[int, int]],
    sources: list[str],
    *,
    interval: tuple[int, int],
    source: str,
    excluded_intervals: list[tuple[int, int]],
    base_lengths: list[int] | None = None,
) -> None:
    base_length = max(0, int(interval[1]) - int(interval[0]))
    for fragment in _subtract_intervals(interval, excluded_intervals):
        intervals.append(fragment)
        sources.append(str(source))
        if base_lengths is not None:
            base_lengths.append(base_length)


def _extract_event_pairs(
    events: list[dict[str, object]],
    start_names: str | list[str] | tuple[str, ...],
    end_names: str | list[str] | tuple[str, ...],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Return ordered start/end event pairs while preserving start metadata."""
    start_set = {str(start_names)} if isinstance(start_names, str) else {str(item) for item in start_names}
    end_set = {str(end_names)} if isinstance(end_names, str) else {str(item) for item in end_names}
    pending_starts: list[dict[str, object]] = []
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for event in events:
        name = str(event.get("event_name", ""))
        sample_index = event.get("sample_index")
        if sample_index is None:
            continue
        if name in start_set:
            pending_starts.append(event)
        elif name in end_set and pending_starts:
            start_event = pending_starts.pop(0)
            pairs.append((start_event, event))
    return pairs


def _build_segment_rows(
    events: list[dict[str, object]],
    trials: list[TrialRecord],
    *,
    sampling_rate: float,
    continuous_prompts: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Build interval-style semantic segments from atomic event markers."""
    rows: list[dict[str, object]] = []
    next_segment_id = 1

    def _append_segment(
        *,
        segment_type: str,
        start_sample: int | None,
        end_sample: int | None,
        label: str = "",
        trial_id: int | None = None,
        mi_run_index: int | None = None,
        run_trial_index: int | None = None,
        block_index: int | None = None,
        prompt_index: int | None = None,
        accepted: bool | int | None = None,
        execution_success: bool | int | None = None,
        source_start_event: str = "",
        source_end_event: str = "",
    ) -> None:
        nonlocal next_segment_id
        if start_sample is None or end_sample is None:
            return
        start_index = int(start_sample)
        stop_index = max(int(end_sample), start_index + 1)
        rows.append(
            {
                "segment_id": next_segment_id,
                "segment_type": str(segment_type),
                "label": str(label),
                "start_sample": start_index,
                "end_sample": stop_index,
                "duration_sec": float((stop_index - start_index) / float(sampling_rate)),
                "trial_id": "" if trial_id is None else int(trial_id),
                "mi_run_index": "" if mi_run_index is None else int(mi_run_index),
                "run_trial_index": "" if run_trial_index is None else int(run_trial_index),
                "block_index": "" if block_index is None else int(block_index),
                "prompt_index": "" if prompt_index is None else int(prompt_index),
                "accepted": "" if accepted is None else int(bool(accepted)),
                "execution_success": "" if execution_success is None else int(bool(execution_success)),
                "source_start_event": str(source_start_event),
                "source_end_event": str(source_end_event),
            }
        )
        next_segment_id += 1

    for trial in trials:
        trial_id = int(trial.trial_id)
        mi_run_index = int(trial.run_index)
        run_trial_index = int(trial.run_trial_index)
        class_name = str(trial.class_name)

        trial_start_event = _first_matching_event(events, trial_id=trial_id, event_names=["trial_start"])
        trial_end_event = _first_matching_event(events, trial_id=trial_id, event_names=["trial_end"])
        trial_start = None if trial_start_event is None else int(trial_start_event["sample_index"])
        trial_end = (
            int(trial_end_event["sample_index"])
            if trial_end_event is not None
            else trial.trial_end_sample
        )
        _append_segment(
            segment_type="trial",
            start_sample=trial_start,
            end_sample=trial_end,
            label=class_name,
            trial_id=trial_id,
            mi_run_index=mi_run_index,
            run_trial_index=run_trial_index,
            accepted=trial.accepted,
            source_start_event="" if trial_start_event is None else str(trial_start_event.get("event_name", "")),
            source_end_event="" if trial_end_event is None else str(trial_end_event.get("event_name", "")),
        )

        baseline_start_event = _first_matching_event(
            events,
            trial_id=trial_id,
            event_names=["fixation_start", "baseline_start"],
        )
        baseline_end_event = _first_matching_event(
            events,
            trial_id=trial_id,
            event_names=["baseline_end", "cue_start", f"cue_{class_name}"],
        )
        _append_segment(
            segment_type="baseline",
            start_sample=None if baseline_start_event is None else int(baseline_start_event["sample_index"]),
            end_sample=None if baseline_end_event is None else int(baseline_end_event["sample_index"]),
            label=class_name,
            trial_id=trial_id,
            mi_run_index=mi_run_index,
            run_trial_index=run_trial_index,
            accepted=trial.accepted,
            source_start_event="" if baseline_start_event is None else str(baseline_start_event.get("event_name", "")),
            source_end_event="" if baseline_end_event is None else str(baseline_end_event.get("event_name", "")),
        )

        cue_start_event = _first_matching_event(
            events,
            trial_id=trial_id,
            event_names=["cue_start", f"cue_{class_name}"],
        )
        cue_end_event = _first_matching_event(events, trial_id=trial_id, event_names=["cue_end"])
        imagery_start_event = _first_matching_event(
            events,
            trial_id=trial_id,
            event_names=["imagery_start", f"imagery_{class_name}"],
        )
        cue_start = None if cue_start_event is None else int(cue_start_event["sample_index"])
        cue_end = (
            int(cue_end_event["sample_index"])
            if cue_end_event is not None
            else (None if imagery_start_event is None else int(imagery_start_event["sample_index"]))
        )
        imagery_start = None if imagery_start_event is None else int(imagery_start_event["sample_index"])
        _append_segment(
            segment_type="cue",
            start_sample=cue_start,
            end_sample=cue_end,
            label=class_name,
            trial_id=trial_id,
            mi_run_index=mi_run_index,
            run_trial_index=run_trial_index,
            accepted=trial.accepted,
            source_start_event="" if cue_start_event is None else str(cue_start_event.get("event_name", "")),
            source_end_event=(
                str(cue_end_event.get("event_name", ""))
                if cue_end_event is not None
                else ("" if imagery_start_event is None else str(imagery_start_event.get("event_name", "")))
            ),
        )

        imagery_end_event = _first_matching_event(events, trial_id=trial_id, event_names=["imagery_end"])
        _append_segment(
            segment_type="imagery",
            start_sample=imagery_start,
            end_sample=(
                int(imagery_end_event["sample_index"])
                if imagery_end_event is not None
                else trial.imagery_offset_sample
            ),
            label=class_name,
            trial_id=trial_id,
            mi_run_index=mi_run_index,
            run_trial_index=run_trial_index,
            accepted=trial.accepted,
            source_start_event="" if imagery_start_event is None else str(imagery_start_event.get("event_name", "")),
            source_end_event="" if imagery_end_event is None else str(imagery_end_event.get("event_name", "")),
        )

        iti_start_event = _first_matching_event(events, trial_id=trial_id, event_names=["iti_start"])
        _append_segment(
            segment_type="iti",
            start_sample=None if iti_start_event is None else int(iti_start_event["sample_index"]),
            end_sample=trial_end,
            label=class_name,
            trial_id=trial_id,
            mi_run_index=mi_run_index,
            run_trial_index=run_trial_index,
            accepted=trial.accepted,
            source_start_event="" if iti_start_event is None else str(iti_start_event.get("event_name", "")),
            source_end_event="" if trial_end_event is None else str(trial_end_event.get("event_name", "")),
        )

    interval_specs = [
        ("quality_check", "quality_check", "quality_check_start", "quality_check_end", ""),
        ("calibration", "calibration", "calibration_start", "calibration_end", ""),
        ("practice", "practice", "practice_start", "practice_end", ""),
        ("run_rest", "run_rest", "run_rest_start", "run_rest_end", ""),
        ("eyes_open_rest", "eyes_open_rest", "eyes_open_rest_start", "eyes_open_rest_end", ""),
        ("eyes_closed_rest", "eyes_closed_rest", "eyes_closed_rest_start", "eyes_closed_rest_end", ""),
        ("idle_block", "idle_block", "idle_block_start", "idle_block_end", ""),
        ("idle_prepare", "idle_prepare", "idle_prepare_start", "idle_prepare_end", ""),
        ("continuous_block", "continuous_block", "continuous_block_start", "continuous_block_end", ""),
    ]
    for segment_type, label, start_name, end_name, default_label in interval_specs:
        for start_event, end_event in _extract_event_pairs(events, start_name, end_name):
            _append_segment(
                segment_type=segment_type,
                start_sample=start_event.get("sample_index"),
                end_sample=end_event.get("sample_index"),
                label=label or default_label,
                mi_run_index=start_event.get("run_index"),
                block_index=start_event.get("block_index"),
                source_start_event=str(start_event.get("event_name", "")),
                source_end_event=str(end_event.get("event_name", "")),
            )

    for artifact_type, (start_name, end_name) in ARTIFACT_EVENT_INTERVALS.items():
        for start_event, end_event in _extract_event_pairs(events, start_name, end_name):
            _append_segment(
                segment_type="artifact_block",
                start_sample=start_event.get("sample_index"),
                end_sample=end_event.get("sample_index"),
                label=artifact_type,
                source_start_event=str(start_event.get("event_name", "")),
                source_end_event=str(end_event.get("event_name", "")),
            )

    for prompt in (continuous_prompts if continuous_prompts is not None else _extract_continuous_prompts(events)):
        _append_segment(
            segment_type="continuous_prompt",
            start_sample=prompt.get("start_sample"),
            end_sample=prompt.get("end_sample"),
            label=str(prompt.get("class_label", "")),
            block_index=prompt.get("block_index"),
            prompt_index=prompt.get("prompt_index"),
            execution_success=prompt.get("execution_success"),
            source_start_event=CONTINUOUS_PROMPT_EVENT_NAMES.get(str(prompt.get("class_label", "")), "continuous_command"),
            source_end_event="continuous_command_end",
        )

    return rows


def _split_intervals_to_windows(
    *,
    signal: np.ndarray,
    intervals: list[tuple[int, int]],
    window_samples: int,
    step_samples: int,
    source_name: str,
    allow_short_interval: bool = False,
    min_short_interval_samples: int | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Slice intervals into fixed-size channel-first windows."""
    windows: list[np.ndarray] = []
    sources: list[str] = []
    if signal.ndim != 2 or window_samples <= 0:
        return windows, sources
    step = max(1, int(step_samples))
    total_samples = int(signal.shape[1])
    for start, stop in intervals:
        s = max(0, int(start))
        t = min(total_samples, int(stop))
        if t - s < window_samples:
            minimum = int(min_short_interval_samples or window_samples)
            if allow_short_interval and t > s and t - s >= minimum:
                short_window = np.asarray(signal[:, s:t], dtype=np.float32)
                pad_samples = int(window_samples - short_window.shape[1])
                if pad_samples > 0:
                    short_window = np.pad(short_window, ((0, 0), (0, pad_samples)), mode="edge")
                windows.append(np.asarray(short_window[:, :window_samples], dtype=np.float32))
                sources.append(str(source_name))
            continue
        cursor = s
        while cursor + window_samples <= t:
            windows.append(np.asarray(signal[:, cursor : cursor + window_samples], dtype=np.float32))
            sources.append(str(source_name))
            cursor += step
    return windows, sources


def _stack_windows(
    windows: list[np.ndarray],
    *,
    channel_count: int,
    sample_count: int,
) -> np.ndarray:
    if not windows:
        return np.empty((0, int(channel_count), int(sample_count)), dtype=np.float32)
    return np.stack([np.asarray(item, dtype=np.float32) for item in windows], axis=0).astype(np.float32)


def _extract_continuous_prompts(events: list[dict[str, object]]) -> list[dict[str, object]]:
    """Pair continuous prompt starts with matching end markers."""
    by_key: dict[tuple[int, int], dict[str, object]] = {}
    ordered: list[dict[str, object]] = []
    reverse_prompt_lookup = {name: label for label, name in CONTINUOUS_PROMPT_EVENT_NAMES.items()}

    for event in events:
        name = str(event.get("event_name", ""))
        sample = event.get("sample_index")
        if sample is None:
            continue
        sample_index = int(sample)
        block_index = int(event.get("block_index") or 0)
        prompt_index = int(event.get("prompt_index") or 0)

        if name in reverse_prompt_lookup:
            label = str(event.get("class_name") or reverse_prompt_lookup[name])
            key = (block_index, prompt_index)
            payload = {
                "block_index": block_index,
                "prompt_index": prompt_index,
                "class_label": label,
                "start_sample": sample_index,
                "end_sample": None,
                "execution_success": event.get("execution_success"),
                "duration_sec": event.get("command_duration_sec"),
            }
            by_key[key] = payload
            ordered.append(payload)
            continue

        if name == "continuous_command_end":
            key = (block_index, prompt_index)
            if key not in by_key:
                continue
            by_key[key]["end_sample"] = max(int(by_key[key]["start_sample"]) + 1, sample_index)
            if event.get("execution_success") is not None:
                by_key[key]["execution_success"] = event.get("execution_success")

    for item in ordered:
        if item["end_sample"] is None:
            item["end_sample"] = int(item["start_sample"]) + 1

    return ordered


def save_mi_session(
    *,
    brainflow_data: np.ndarray,
    sampling_rate: float,
    eeg_rows: list[int],
    marker_row: int,
    timestamp_row: int | None,
    package_num_row: int | None = None,
    board_descr: dict[str, object] | None = None,
    settings: SessionSettings,
    event_log: list[dict[str, object]],
    trial_records: list[TrialRecord],
) -> dict[str, object]:
    """Persist one collected MI session to disk."""
    brainflow_data = np.asarray(brainflow_data)
    if brainflow_data.ndim != 2:
        raise ValueError("brainflow_data must have shape (rows, samples).")
    if brainflow_data.shape[1] <= 0:
        raise ValueError("brainflow_data must contain at least one sample.")
    if float(sampling_rate) <= 0:
        raise ValueError("sampling_rate must be positive.")
    if not eeg_rows:
        raise ValueError("eeg_rows cannot be empty.")
    row_count = int(brainflow_data.shape[0])
    normalized_eeg_rows = [int(row) for row in eeg_rows]
    invalid_eeg_rows = [row for row in normalized_eeg_rows if row < 0 or row >= row_count]
    if invalid_eeg_rows:
        raise ValueError(f"eeg_rows contains out-of-range rows for data with {row_count} rows: {invalid_eeg_rows}.")
    marker_row = int(marker_row)
    if marker_row < 0 or marker_row >= row_count:
        raise ValueError(f"marker_row {marker_row} is out of range for data with {row_count} rows.")
    if timestamp_row is not None:
        timestamp_row = int(timestamp_row)
        if timestamp_row < 0 or timestamp_row >= row_count:
            raise ValueError(f"timestamp_row {timestamp_row} is out of range for data with {row_count} rows.")
    if package_num_row is not None:
        package_num_row = int(package_num_row)
        if package_num_row < 0 or package_num_row >= row_count:
            raise ValueError(f"package_num_row {package_num_row} is out of range for data with {row_count} rows.")
    eeg_rows = normalized_eeg_rows

    marker_channel = np.asarray(brainflow_data[marker_row], dtype=np.float64)
    crop_start, crop_end = _detect_session_bounds(marker_channel)
    cropped = np.asarray(brainflow_data[:, crop_start : crop_end + 1], dtype=np.float64)
    cropped_marker = marker_channel[crop_start : crop_end + 1]
    cropped_timestamps = None
    if timestamp_row is not None:
        cropped_timestamps = np.asarray(brainflow_data[timestamp_row, crop_start : crop_end + 1], dtype=np.float64)

    channel_names = settings.channel_names
    eeg_uvolts = np.asarray(cropped[eeg_rows], dtype=np.float32)
    quality_summary = _build_quality_summary(eeg_uvolts, channel_names)
    if int(quality_summary.get("non_finite_sample_count", 0)) > 0:
        eeg_uvolts_for_derivatives = np.nan_to_num(eeg_uvolts, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        eeg_uvolts_for_derivatives = eeg_uvolts
    eeg_volts = eeg_uvolts_for_derivatives * 1e-6
    recorded_markers = _extract_marker_occurrences(cropped_marker, crop_start=crop_start)
    data_flow_summary = _build_data_flow_summary(
        cropped=cropped,
        marker_channel=cropped_marker,
        timestamp_row=timestamp_row,
        package_num_row=package_num_row,
        sampling_rate=float(sampling_rate),
        event_count=len(event_log),
    )
    enriched_events = _attach_sample_indices(event_log, recorded_markers)
    pause_intervals = _extract_pause_intervals(enriched_events, total_samples=int(eeg_volts.shape[1]))
    updated_trials = _update_trials_from_events(trial_records, enriched_events)
    pause_rejected_trial_ids = _mark_trials_overlapping_intervals(
        updated_trials,
        enriched_events,
        pause_intervals,
        note=PAUSE_REJECTED_TRIAL_NOTE,
    )
    continuous_prompts = _mark_continuous_prompts_overlapping_intervals(
        _extract_continuous_prompts(enriched_events),
        pause_intervals,
    )
    semantic_segment_rows = _build_segment_rows(
        enriched_events,
        updated_trials,
        sampling_rate=float(sampling_rate),
        continuous_prompts=continuous_prompts,
    )

    session_dir = create_session_folder(settings.output_root, settings.subject_id, settings.session_id)
    dataset_root = Path(settings.output_root).resolve()
    save_index = _next_save_index(session_dir)
    trial_count_total = int(len(updated_trials))
    accepted_count_total = int(sum(1 for trial in updated_trials if trial.accepted))
    rejected_count_total = int(trial_count_total - accepted_count_total)
    trials_per_run = int(settings.trials_per_class * len(MI_CLASSES))
    run_stem = _build_run_stem(
        settings=settings,
        save_index=save_index,
        trial_count=trial_count_total,
        accepted_count=accepted_count_total,
    )
    save_timestamp = datetime.now().isoformat(timespec="seconds")

    board_data_path = session_dir / f"{run_stem}_board_data.npy"
    board_map_path = session_dir / f"{run_stem}_board_map.json"
    fif_path = session_dir / f"{run_stem}_raw.fif"
    events_csv_path = session_dir / f"{run_stem}_events.csv"
    trials_csv_path = session_dir / f"{run_stem}_trials.csv"
    segments_csv_path = session_dir / f"{run_stem}_segments.csv"
    meta_json_path = session_dir / f"{run_stem}_session_meta.json"
    quality_json_path = session_dir / f"{run_stem}_quality_report.json"
    mi_epochs_path = session_dir / f"{run_stem}_mi_epochs.npz"
    gate_epochs_path = session_dir / f"{run_stem}_gate_epochs.npz"
    artifact_epochs_path = session_dir / f"{run_stem}_artifact_epochs.npz"
    continuous_npz_path = session_dir / f"{run_stem}_continuous.npz"
    mi_epochs_meta_path = mi_epochs_path.with_suffix(".meta.json")
    gate_epochs_meta_path = gate_epochs_path.with_suffix(".meta.json")
    artifact_epochs_meta_path = artifact_epochs_path.with_suffix(".meta.json")
    continuous_meta_path = continuous_npz_path.with_suffix(".meta.json")
    np.save(board_data_path, np.asarray(cropped, dtype=np.float64), allow_pickle=False)
    board_map = {
        "schema_version": int(COLLECTION_SCHEMA_VERSION),
        "exporter_name": COLLECTION_EXPORTER_NAME,
        "saved_at": save_timestamp,
        "subject_id": sanitize_session_token(settings.subject_id),
        "session_id": sanitize_session_token(settings.session_id),
        "save_index": int(save_index),
        "run_stem": run_stem,
        "board_id": int(settings.board_id),
        "board_name": str(settings.board_name or ""),
        "board_descr": json.loads(json.dumps(board_descr or {}, ensure_ascii=False, default=str)),
        "crop_start_sample": int(crop_start),
        "crop_end_sample": int(crop_end),
        "cropped_sample_count": int(cropped.shape[1]),
        "row_count": int(cropped.shape[0]),
        "selected_eeg_rows": [int(item) for item in eeg_rows],
        "marker_row": int(marker_row),
        "timestamp_row": None if timestamp_row is None else int(timestamp_row),
        "package_num_row": None if package_num_row is None else int(package_num_row),
        "selected_eeg_non_finite_sample_count": int(quality_summary.get("non_finite_sample_count", 0)),
        "derivative_non_finite_replacement": (
            "Selected EEG NaN/Inf values are preserved in board_data.npy and replaced with 0.0 only in FIF/epoch derivatives."
        ),
        "data_flow_report": data_flow_summary,
        "channel_rows": [
            {"channel_name": str(name), "board_row": int(row)}
            for name, row in zip(channel_names, eeg_rows)
        ],
    }
    _write_json(board_map_path, board_map)
    board_data_sha256 = _compute_sha256(board_data_path)

    stim_channel = cropped_marker[np.newaxis, :]
    data_for_raw = np.vstack([eeg_volts, stim_channel])
    info = mne.create_info(
        ch_names=channel_names + ["STI 014"],
        sfreq=float(sampling_rate),
        ch_types=["eeg"] * len(channel_names) + ["stim"],
    )
    raw = mne.io.RawArray(data_for_raw, info, verbose=False)
    _maybe_set_standard_montage(raw, channel_names)
    raw.set_annotations(_make_annotations(enriched_events, float(sampling_rate), semantic_segment_rows))
    raw.save(fif_path, overwrite=True, verbose=False)

    event_rows = [
        {
            "event_index": index,
            "save_index": int(save_index),
            "event_name": event["event_name"],
            "marker_code": event["marker_code"],
            "trial_id": event.get("trial_id"),
            "mi_run_index": event.get("run_index"),
            "run_trial_index": event.get("run_trial_index"),
            "block_index": event.get("block_index"),
            "prompt_index": event.get("prompt_index"),
            "class_name": event.get("class_name", ""),
            "command_duration_sec": event.get("command_duration_sec"),
            "execution_success": event.get("execution_success"),
            "sample_index": event.get("sample_index"),
            "absolute_sample_index": event.get("absolute_sample_index"),
            "elapsed_sec": event.get("elapsed_sec"),
            "iso_time": event.get("iso_time"),
        }
        for index, event in enumerate(enriched_events)
    ]
    _write_csv(
        events_csv_path,
        event_rows,
        [
            "event_index",
            "save_index",
            "event_name",
            "marker_code",
            "trial_id",
            "mi_run_index",
            "run_trial_index",
            "block_index",
            "prompt_index",
            "class_name",
            "command_duration_sec",
            "execution_success",
            "sample_index",
            "absolute_sample_index",
            "elapsed_sec",
            "iso_time",
        ],
    )

    trial_rows = [
        {
            "trial_id": int(trial.trial_id),
            "save_index": int(save_index),
            "mi_run_index": int(trial.run_index),
            "run_trial_index": int(trial.run_trial_index),
            "class_name": str(trial.class_name),
            "display_name": str(trial.display_name),
            "accepted": int(trial.accepted),
            "cue_onset_sample": trial.cue_onset_sample,
            "imagery_onset_sample": trial.imagery_onset_sample,
            "imagery_offset_sample": trial.imagery_offset_sample,
            "trial_end_sample": trial.trial_end_sample,
            "note": str(trial.note),
        }
        for trial in updated_trials
    ]
    _write_csv(
        trials_csv_path,
        trial_rows,
        [
            "trial_id",
            "save_index",
            "mi_run_index",
            "run_trial_index",
            "class_name",
            "display_name",
            "accepted",
            "cue_onset_sample",
            "imagery_onset_sample",
            "imagery_offset_sample",
            "trial_end_sample",
            "note",
        ],
    )
    segment_rows = [{"save_index": int(save_index), **row} for row in semantic_segment_rows]
    _write_csv(
        segments_csv_path,
        segment_rows,
        [
            "segment_id",
            "save_index",
            "segment_type",
            "label",
            "start_sample",
            "end_sample",
            "duration_sec",
            "trial_id",
            "mi_run_index",
            "run_trial_index",
            "block_index",
            "prompt_index",
            "accepted",
            "execution_success",
            "source_start_event",
            "source_end_event",
        ],
    )

    _write_json(quality_json_path, quality_summary)

    class_names = [item["key"] for item in MI_CLASSES]
    window_samples = max(1, int(round(float(settings.imagery_sec) * float(sampling_rate))))
    rest_step_samples = max(1, int(round(0.5 * float(sampling_rate))))

    mi_windows: list[np.ndarray] = []
    mi_labels: list[int] = []
    mi_trial_ids: list[int] = []
    baseline_windows: list[np.ndarray] = []
    iti_windows: list[np.ndarray] = []
    trial_start_lookup = {
        int(event["trial_id"]): int(event["sample_index"])
        for event in enriched_events
        if event.get("sample_index") is not None and event.get("trial_id") is not None and str(event.get("event_name")) == "trial_start"
    }
    trial_end_lookup = {
        int(event["trial_id"]): int(event["sample_index"])
        for event in enriched_events
        if event.get("sample_index") is not None and event.get("trial_id") is not None and str(event.get("event_name")) == "trial_end"
    }
    for trial in updated_trials:
        if trial.imagery_onset_sample is not None and trial.imagery_offset_sample is not None:
            start = int(trial.imagery_onset_sample)
            stop = int(trial.imagery_offset_sample)
            if stop > start and stop <= eeg_volts.shape[1] and trial.accepted:
                mi_windows.append(np.asarray(eeg_volts[:, start:stop], dtype=np.float32))
                mi_labels.append(int(CLASS_NAME_TO_LABEL[trial.class_name]))
                mi_trial_ids.append(int(trial.trial_id))

        baseline_start = _first_event_sample(enriched_events, trial_id=int(trial.trial_id), event_names=["fixation_start", "baseline_start"])
        baseline_end = _first_event_sample(enriched_events, trial_id=int(trial.trial_id), event_names=["baseline_end", "cue_start"])
        if trial.accepted and baseline_start is not None and baseline_end is not None and baseline_end > baseline_start:
            baseline_windows.append(np.asarray(eeg_volts[:, baseline_start:baseline_end], dtype=np.float32))

        iti_start = _first_event_sample(enriched_events, trial_id=int(trial.trial_id), event_names=["iti_start"])
        trial_end = trial_end_lookup.get(int(trial.trial_id))
        if trial.accepted and iti_start is not None and trial_end is not None and trial_end > iti_start:
            iti_windows.append(np.asarray(eeg_volts[:, int(iti_start):int(trial_end)], dtype=np.float32))

    mi_target_samples = min((window.shape[1] for window in mi_windows), default=window_samples)
    X_mi = _stack_windows([window[:, :mi_target_samples] for window in mi_windows if window.shape[1] >= mi_target_samples], channel_count=len(channel_names), sample_count=mi_target_samples)
    y_mi = np.asarray(mi_labels, dtype=np.int64)
    mi_trial_ids_array = np.asarray(mi_trial_ids, dtype=np.int64)
    baseline_target_samples = min((window.shape[1] for window in baseline_windows), default=window_samples)
    X_baseline = _stack_windows([window[:, :baseline_target_samples] for window in baseline_windows if window.shape[1] >= baseline_target_samples], channel_count=len(channel_names), sample_count=baseline_target_samples)
    iti_target_samples = min((window.shape[1] for window in iti_windows), default=window_samples)
    X_iti = _stack_windows([window[:, :iti_target_samples] for window in iti_windows if window.shape[1] >= iti_target_samples], channel_count=len(channel_names), sample_count=iti_target_samples)

    gate_positive = np.asarray(X_mi, dtype=np.float32)
    gate_negative_intervals: list[tuple[int, int]] = []
    gate_negative_sources: list[str] = []
    gate_negative_base_lengths: list[int] = []
    for trial in updated_trials:
        if not trial.accepted:
            continue
        baseline_start = _first_event_sample(enriched_events, trial_id=int(trial.trial_id), event_names=["fixation_start", "baseline_start"])
        baseline_end = _first_event_sample(enriched_events, trial_id=int(trial.trial_id), event_names=["baseline_end", "cue_start"])
        if baseline_start is not None and baseline_end is not None and baseline_end > baseline_start:
            _append_clean_interval(
                gate_negative_intervals,
                gate_negative_sources,
                interval=(int(baseline_start), int(baseline_end)),
                source="baseline",
                excluded_intervals=pause_intervals,
                base_lengths=gate_negative_base_lengths,
            )

        iti_start = _first_event_sample(enriched_events, trial_id=int(trial.trial_id), event_names=["iti_start"])
        trial_end = trial_end_lookup.get(int(trial.trial_id))
        if iti_start is not None and trial_end is not None and trial_end > iti_start:
            _append_clean_interval(
                gate_negative_intervals,
                gate_negative_sources,
                interval=(int(iti_start), int(trial_end)),
                source="iti",
                excluded_intervals=pause_intervals,
                base_lengths=gate_negative_base_lengths,
            )

    for interval in _extract_intervals(enriched_events, "eyes_open_rest_start", "eyes_open_rest_end"):
        _append_clean_interval(
            gate_negative_intervals,
            gate_negative_sources,
            interval=interval,
            source="eyes_open_rest",
            excluded_intervals=pause_intervals,
            base_lengths=gate_negative_base_lengths,
        )
    if bool(settings.include_eyes_closed_rest_in_gate_neg):
        for interval in _extract_intervals(enriched_events, "eyes_closed_rest_start", "eyes_closed_rest_end"):
            _append_clean_interval(
                gate_negative_intervals,
                gate_negative_sources,
                interval=interval,
                source="eyes_closed_rest",
                excluded_intervals=pause_intervals,
                base_lengths=gate_negative_base_lengths,
            )
    for interval in _extract_intervals(enriched_events, "idle_block_start", "idle_block_end"):
        _append_clean_interval(
            gate_negative_intervals,
            gate_negative_sources,
            interval=interval,
            source="idle_block",
            excluded_intervals=pause_intervals,
            base_lengths=gate_negative_base_lengths,
        )
    for interval in _extract_intervals(enriched_events, "idle_prepare_start", "idle_prepare_end"):
        _append_clean_interval(
            gate_negative_intervals,
            gate_negative_sources,
            interval=interval,
            source="idle_prepare",
            excluded_intervals=pause_intervals,
            base_lengths=gate_negative_base_lengths,
        )

    for prompt in continuous_prompts:
        if str(prompt["class_label"]) == "no_control" and _normalize_execution_success(prompt.get("execution_success")) != 0:
            start_sample = int(prompt["start_sample"])
            end_sample = int(prompt["end_sample"])
            if end_sample > start_sample:
                _append_clean_interval(
                    gate_negative_intervals,
                    gate_negative_sources,
                    interval=(start_sample, end_sample),
                    source="continuous_no_control",
                    excluded_intervals=pause_intervals,
                    base_lengths=gate_negative_base_lengths,
                )

    min_gate_negative_samples = max(1, rest_step_samples)
    gate_negative_window_samples = int(window_samples)
    eligible_gate_negative_lengths = [
        int(base_length)
        for base_length in gate_negative_base_lengths
        if int(base_length) >= min_gate_negative_samples
    ]
    if eligible_gate_negative_lengths:
        gate_negative_window_samples = min(int(window_samples), min(eligible_gate_negative_lengths))

    gate_negative_windows: list[np.ndarray] = []
    gate_negative_sources_windows: list[str] = []
    for interval, source in zip(gate_negative_intervals, gate_negative_sources):
        windows, sources = _split_intervals_to_windows(
            signal=eeg_volts,
            intervals=[interval],
            window_samples=gate_negative_window_samples,
            step_samples=rest_step_samples,
            source_name=str(source),
            allow_short_interval=True,
            min_short_interval_samples=min_gate_negative_samples,
        )
        gate_negative_windows.extend(windows)
        gate_negative_sources_windows.extend(sources)
    X_gate_neg = _stack_windows(
        gate_negative_windows,
        channel_count=len(channel_names),
        sample_count=gate_negative_window_samples,
    )
    gate_neg_sources_array = np.asarray(gate_negative_sources_windows, dtype=object)

    hard_negative_intervals: list[tuple[int, int]] = []
    hard_negative_sources: list[str] = []
    for artifact_type in normalize_artifact_types(settings.artifact_types):
        start_name, end_name = ARTIFACT_EVENT_INTERVALS[artifact_type]
        for interval in _extract_intervals(enriched_events, start_name, end_name):
            _append_clean_interval(
                hard_negative_intervals,
                hard_negative_sources,
                interval=interval,
                source=f"artifact_{artifact_type}",
                excluded_intervals=pause_intervals,
            )
    for trial in updated_trials:
        if trial.accepted:
            continue
        if int(trial.trial_id) in pause_rejected_trial_ids:
            continue
        start_sample = trial_start_lookup.get(int(trial.trial_id))
        end_sample = trial_end_lookup.get(int(trial.trial_id))
        if start_sample is None or end_sample is None or end_sample <= start_sample:
            continue
        _append_clean_interval(
            hard_negative_intervals,
            hard_negative_sources,
            interval=(int(start_sample), int(end_sample)),
            source="rejected_trial",
            excluded_intervals=pause_intervals,
        )

    hard_negative_windows: list[np.ndarray] = []
    hard_negative_sources_windows: list[str] = []
    for interval, source in zip(hard_negative_intervals, hard_negative_sources):
        is_artifact_source = str(source).startswith("artifact_")
        windows, sources = _split_intervals_to_windows(
            signal=eeg_volts,
            intervals=[interval],
            window_samples=window_samples,
            step_samples=rest_step_samples,
            source_name=str(source),
            allow_short_interval=is_artifact_source,
            min_short_interval_samples=max(1, int(round(0.8 * float(window_samples)))),
        )
        hard_negative_windows.extend(windows)
        hard_negative_sources_windows.extend(sources)
    X_gate_hard_neg = _stack_windows(hard_negative_windows, channel_count=len(channel_names), sample_count=window_samples)
    gate_hard_neg_sources_array = np.asarray(hard_negative_sources_windows, dtype=object)
    X_artifact = np.asarray(X_gate_hard_neg, dtype=np.float32)
    artifact_labels = np.asarray([str(item).replace("artifact_", "") for item in hard_negative_sources_windows], dtype=object)

    continuous_block_intervals = _extract_intervals(enriched_events, "continuous_block_start", "continuous_block_end")
    continuous_blocks: list[np.ndarray] = []
    for start_sample, end_sample in continuous_block_intervals:
        if end_sample <= start_sample:
            continue
        segment = np.asarray(eeg_volts[:, int(start_sample):int(end_sample)], dtype=np.float32)
        if segment.shape[1] > 0:
            continuous_blocks.append(segment)
    continuous_target_samples = min((block.shape[1] for block in continuous_blocks), default=0)
    X_continuous = _stack_windows([block[:, :continuous_target_samples] for block in continuous_blocks if continuous_target_samples > 0], channel_count=len(channel_names), sample_count=continuous_target_samples)
    continuous_event_labels = np.asarray([str(item["class_label"]) for item in continuous_prompts], dtype=object)
    continuous_event_samples = np.asarray([int(item["start_sample"]) for item in continuous_prompts], dtype=np.int64)
    continuous_event_end_samples = np.asarray([int(item["end_sample"]) for item in continuous_prompts], dtype=np.int64)
    continuous_block_indices = np.asarray([int(item.get("block_index", 0)) for item in continuous_prompts], dtype=np.int32)
    continuous_prompt_indices = np.asarray([int(item.get("prompt_index", 0)) for item in continuous_prompts], dtype=np.int32)
    continuous_execution_success = np.asarray(
        [
            -1 if _normalize_execution_success(item.get("execution_success")) is None else int(_normalize_execution_success(item.get("execution_success")))
            for item in continuous_prompts
        ],
        dtype=np.int8,
    )
    continuous_command_durations = np.asarray(
        [
            np.nan if item.get("duration_sec") is None else float(item.get("duration_sec"))
            for item in continuous_prompts
        ],
        dtype=np.float32,
    )
    continuous_block_start_samples = np.asarray([int(item[0]) for item in continuous_block_intervals], dtype=np.int64)
    continuous_block_end_samples = np.asarray([int(item[1]) for item in continuous_block_intervals], dtype=np.int64)

    common_npz_payload = {
        "schema_version": np.asarray([int(COLLECTION_SCHEMA_VERSION)], dtype=np.int32),
        "class_names": _string_array(class_names),
        "channel_names": _string_array(channel_names),
        "sampling_rate": np.asarray([float(sampling_rate)], dtype=np.float32),
        "signal_unit": _string_array(["volt"]),
        "subject_id": _string_array([sanitize_session_token(settings.subject_id)]),
        "session_id": _string_array([sanitize_session_token(settings.session_id)]),
        "protocol_mode": _string_array([str(settings.protocol_mode or "full")]),
        "save_index": np.asarray([int(save_index)], dtype=np.int32),
        "run_stem": _string_array([run_stem]),
        "trials_per_class": np.asarray([int(settings.trials_per_class)], dtype=np.int32),
        "mi_run_count": np.asarray([int(settings.run_count)], dtype=np.int32),
        "trials_per_run": np.asarray([int(trials_per_run)], dtype=np.int32),
        "total_trials": np.asarray([int(trial_count_total)], dtype=np.int32),
        "accepted_trials": np.asarray([int(accepted_count_total)], dtype=np.int32),
        "rejected_trials": np.asarray([int(rejected_count_total)], dtype=np.int32),
        "created_at": _string_array([save_timestamp]),
    }

    np.savez_compressed(
        mi_epochs_path,
        X_mi=np.asarray(X_mi, dtype=np.float32),
        y_mi=np.asarray(y_mi, dtype=np.int64),
        mi_trial_ids=np.asarray(mi_trial_ids_array, dtype=np.int64),
        **common_npz_payload,
    )
    np.savez_compressed(
        gate_epochs_path,
        X_gate_pos=np.asarray(gate_positive, dtype=np.float32),
        X_gate_neg=np.asarray(X_gate_neg, dtype=np.float32),
        X_gate_hard_neg=np.asarray(X_gate_hard_neg, dtype=np.float32),
        gate_neg_sources=_string_array(gate_neg_sources_array),
        gate_hard_neg_sources=_string_array(gate_hard_neg_sources_array),
        **common_npz_payload,
    )
    np.savez_compressed(
        artifact_epochs_path,
        X_artifact=np.asarray(X_artifact, dtype=np.float32),
        artifact_labels=_string_array(artifact_labels),
        **common_npz_payload,
    )
    np.savez_compressed(
        continuous_npz_path,
        X_continuous=np.asarray(X_continuous, dtype=np.float32),
        continuous_event_labels=_string_array(continuous_event_labels),
        continuous_event_samples=np.asarray(continuous_event_samples, dtype=np.int64),
        continuous_event_end_samples=np.asarray(continuous_event_end_samples, dtype=np.int64),
        continuous_block_indices=np.asarray(continuous_block_indices, dtype=np.int32),
        continuous_prompt_indices=np.asarray(continuous_prompt_indices, dtype=np.int32),
        continuous_execution_success=np.asarray(continuous_execution_success, dtype=np.int8),
        continuous_command_duration_sec=np.asarray(continuous_command_durations, dtype=np.float32),
        continuous_block_start_samples=np.asarray(continuous_block_start_samples, dtype=np.int64),
        continuous_block_end_samples=np.asarray(continuous_block_end_samples, dtype=np.int64),
        **common_npz_payload,
    )

    files_rel = {
        "board_data_npy": _relative_path(board_data_path, dataset_root),
        "board_map_json": _relative_path(board_map_path, dataset_root),
        "session_raw_fif": _relative_path(fif_path, dataset_root),
        "events_csv": _relative_path(events_csv_path, dataset_root),
        "trials_csv": _relative_path(trials_csv_path, dataset_root),
        "segments_csv": _relative_path(segments_csv_path, dataset_root),
        "session_meta_json": _relative_path(meta_json_path, dataset_root),
        "quality_report_json": _relative_path(quality_json_path, dataset_root),
        "mi_epochs_npz": _relative_path(mi_epochs_path, dataset_root),
        "mi_epochs_meta_json": _relative_path(mi_epochs_meta_path, dataset_root),
        "gate_epochs_npz": _relative_path(gate_epochs_path, dataset_root),
        "gate_epochs_meta_json": _relative_path(gate_epochs_meta_path, dataset_root),
        "artifact_epochs_npz": _relative_path(artifact_epochs_path, dataset_root),
        "artifact_epochs_meta_json": _relative_path(artifact_epochs_meta_path, dataset_root),
        "continuous_npz": _relative_path(continuous_npz_path, dataset_root),
        "continuous_meta_json": _relative_path(continuous_meta_path, dataset_root),
    }

    derivation_common = {
        "schema_version": int(COLLECTION_SCHEMA_VERSION),
        "exporter_name": COLLECTION_EXPORTER_NAME,
        "created_at": save_timestamp,
        "source_run_stem": run_stem,
        "source_save_index": int(save_index),
        "source_sha256": board_data_sha256,
        "source_files": {
            "board_data_npy": files_rel["board_data_npy"],
            "events_csv": files_rel["events_csv"],
            "trials_csv": files_rel["trials_csv"],
            "segments_csv": files_rel["segments_csv"],
        },
        "subject_id": sanitize_session_token(settings.subject_id),
        "session_id": sanitize_session_token(settings.session_id),
        "protocol_mode": str(settings.protocol_mode or "full"),
        "source_data_flow_report": data_flow_summary,
        "source_quality_report": quality_summary,
    }
    _write_json(
        mi_epochs_meta_path,
        {
            **derivation_common,
            "derivation_name": "mi_epochs",
            "derivation_policy": {
                "window_sec": float(settings.imagery_sec),
                "source_segment": "imagery",
                "accepted_trials_only": True,
            },
        },
    )
    _write_json(
        gate_epochs_meta_path,
        {
            **derivation_common,
            "derivation_name": "gate_epochs",
            "derivation_policy": {
                "positive_source": "accepted_imagery",
                "negative_sources": sorted(set(str(item) for item in gate_negative_sources)),
                "hard_negative_sources": sorted(set(str(item) for item in hard_negative_sources)),
                "window_sec": float(settings.imagery_sec),
                "step_sec": float(rest_step_samples / float(sampling_rate)),
            },
        },
    )
    _write_json(
        artifact_epochs_meta_path,
        {
            **derivation_common,
            "derivation_name": "artifact_epochs",
            "derivation_policy": {
                "artifact_types": [str(item) for item in normalize_artifact_types(settings.artifact_types)],
                "window_sec": float(settings.imagery_sec),
                "step_sec": float(rest_step_samples / float(sampling_rate)),
            },
        },
    )
    _write_json(
        continuous_meta_path,
        {
            **derivation_common,
            "derivation_name": "continuous",
            "derivation_policy": {
                "contains_blocks": True,
                "contains_prompt_metadata": True,
                "stores_execution_success": True,
            },
        },
    )

    session_payload = asdict(settings)
    session_payload.pop("operator", None)
    session_payload["output_root"] = "."
    meta = {
        "schema_version": int(COLLECTION_SCHEMA_VERSION),
        "exporter_name": COLLECTION_EXPORTER_NAME,
        "saved_at": save_timestamp,
        "task_name": "motor_imagery",
        "subject_id": sanitize_session_token(settings.subject_id),
        "session_id": sanitize_session_token(settings.session_id),
        "protocol_mode": str(settings.protocol_mode or "full"),
        "save_index": int(save_index),
        "run_stem": run_stem,
        "session": session_payload,
        "sampling_rate_hz": float(sampling_rate),
        "recording_type": "continuous",
        "power_line_frequency_hz": None,
        "eeg_reference": str(settings.reference_mode or ""),
        "eeg_ground": "",
        "preview_filter_applied_to_saved_signal": False,
        "selected_eeg_rows": [int(item) for item in eeg_rows],
        "marker_row": int(marker_row),
        "timestamp_row": None if timestamp_row is None else int(timestamp_row),
        "package_num_row": None if package_num_row is None else int(package_num_row),
        "sample_count": int(eeg_uvolts.shape[1]),
        "duration_sec": float(eeg_uvolts.shape[1] / float(sampling_rate)),
        "brainflow_eeg_unit": "microvolt",
        "saved_fif_unit": "volt",
        "epochs_unit": "volt",
        "quality_report": quality_summary,
        "data_flow_report": data_flow_summary,
        "event_count": len(enriched_events),
        "segment_count": int(len(segment_rows)),
        "trials_per_run": int(trials_per_run),
        "mi_run_count": int(settings.run_count),
        "trial_count": trial_count_total,
        "accepted_trial_count": accepted_count_total,
        "rejected_trial_count": rejected_count_total,
        "mi_epochs_saved": int(X_mi.shape[0]),
        "gate_pos_segments": int(gate_positive.shape[0]),
        "gate_neg_segments": int(X_gate_neg.shape[0]),
        "gate_hard_neg_segments": int(X_gate_hard_neg.shape[0]),
        "artifact_segments": int(X_artifact.shape[0]),
        "continuous_blocks": int(X_continuous.shape[0]),
        "continuous_prompts": int(continuous_event_labels.shape[0]),
        "raw_preservation_level": "cropped_full_board_matrix",
        "board_data_sha256": board_data_sha256,
        "source_alignment_policy": "strict_marker_sequence_1_to_1",
        "session_start_wall_time": event_log[0]["iso_time"] if event_log else "",
        "session_end_wall_time": event_log[-1]["iso_time"] if event_log else "",
        "first_board_timestamp": None if cropped_timestamps is None or cropped_timestamps.size == 0 else float(cropped_timestamps[0]),
        "last_board_timestamp": None if cropped_timestamps is None or cropped_timestamps.size == 0 else float(cropped_timestamps[-1]),
        "files": files_rel,
    }
    _write_json(meta_json_path, meta)

    # Keep one lightweight pointer for "latest run" convenience.
    latest_meta_path = session_dir / "session_meta_latest.json"
    _write_json(latest_meta_path, meta)

    manifest_path = _append_collection_manifest(
        settings.output_root,
        {
            "saved_at": save_timestamp,
            "schema_version": int(COLLECTION_SCHEMA_VERSION),
            "subject_id": sanitize_session_token(settings.subject_id),
            "session_id": sanitize_session_token(settings.session_id),
            "protocol_mode": str(settings.protocol_mode or "full"),
            "save_index": int(save_index),
            "run_stem": run_stem,
            "trials_per_class": int(settings.trials_per_class),
            "mi_run_count": int(settings.run_count),
            "trials_per_run": int(trials_per_run),
            "trial_count": int(trial_count_total),
            "accepted_trial_count": int(accepted_count_total),
            "rejected_trial_count": int(rejected_count_total),
            "sampling_rate_hz": float(sampling_rate),
            "channel_names": ",".join(channel_names),
            "class_names": ",".join(class_names),
            "board_data_npy": files_rel["board_data_npy"],
            "board_map_json": files_rel["board_map_json"],
            "mi_epochs_npz": files_rel["mi_epochs_npz"],
            "mi_epochs_meta_json": files_rel["mi_epochs_meta_json"],
            "gate_epochs_npz": files_rel["gate_epochs_npz"],
            "gate_epochs_meta_json": files_rel["gate_epochs_meta_json"],
            "artifact_epochs_npz": files_rel["artifact_epochs_npz"],
            "artifact_epochs_meta_json": files_rel["artifact_epochs_meta_json"],
            "continuous_npz": files_rel["continuous_npz"],
            "continuous_meta_json": files_rel["continuous_meta_json"],
            "session_raw_fif": files_rel["session_raw_fif"],
            "events_csv": files_rel["events_csv"],
            "trials_csv": files_rel["trials_csv"],
            "segments_csv": files_rel["segments_csv"],
            "session_meta_json": files_rel["session_meta_json"],
            "quality_report_json": files_rel["quality_report_json"],
        },
    )

    return {
        "session_dir": str(session_dir),
        "board_data_path": str(board_data_path),
        "board_map_path": str(board_map_path),
        "fif_path": str(fif_path),
        "trials_csv_path": str(trials_csv_path),
        "events_csv_path": str(events_csv_path),
        "segments_csv_path": str(segments_csv_path),
        "meta_json_path": str(meta_json_path),
        "quality_json_path": str(quality_json_path),
        "mi_epochs_path": str(mi_epochs_path),
        "gate_epochs_path": str(gate_epochs_path),
        "artifact_epochs_path": str(artifact_epochs_path),
        "continuous_path": str(continuous_npz_path),
        "save_index": int(save_index),
        "run_stem": run_stem,
        "trial_count": trial_count_total,
        "accepted_trial_count": accepted_count_total,
        "rejected_trial_count": rejected_count_total,
        "quality_report": quality_summary,
        "data_flow_report": data_flow_summary,
        "manifest_csv_path": str(manifest_path),
    }


def make_event(
    event_name: str,
    *,
    trial_id: int | None = None,
    class_name: str | None = None,
    elapsed_sec: float | None = None,
    run_index: int | None = None,
    run_trial_index: int | None = None,
    block_index: int | None = None,
    prompt_index: int | None = None,
    command_duration_sec: float | None = None,
    execution_success: int | bool | None = None,
) -> dict[str, object]:
    """Create one event-log entry that can later be matched to BrainFlow markers."""
    if event_name not in EVENT_NAME_TO_CODE:
        raise KeyError(f"Unknown event name: {event_name}")
    return {
        "event_name": event_name,
        "marker_code": int(EVENT_NAME_TO_CODE[event_name]),
        "trial_id": trial_id,
        "class_name": class_name,
        "run_index": run_index,
        "run_trial_index": run_trial_index,
        "block_index": block_index,
        "prompt_index": prompt_index,
        "command_duration_sec": None if command_duration_sec is None else float(command_duration_sec),
        "execution_success": None if execution_success is None else int(bool(execution_success)),
        "elapsed_sec": None if elapsed_sec is None else float(elapsed_sec),
        "iso_time": datetime.now().isoformat(timespec="milliseconds"),
    }

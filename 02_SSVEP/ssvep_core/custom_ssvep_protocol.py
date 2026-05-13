from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .async_fbcca_idle_standalone import DEFAULT_CALIBRATION_SEED, TrialSpec
from .trial_roles import TRIAL_ROLE_CLEAN_IDLE, TRIAL_ROLE_CONTROL, TRIAL_ROLE_HARD_IDLE


CUSTOM_SSVEP_PROTOCOL_V1 = "custom_ssvep_command_nc_pseudoonline_v1"
CUSTOM_SSVEP_SHORT_PROTOCOL_V1 = "custom_ssvep_command_nc_short_v1"
CUSTOM_SSVEP_PROTOCOL_KEYS = (CUSTOM_SSVEP_PROTOCOL_V1, CUSTOM_SSVEP_SHORT_PROTOCOL_V1)

CUSTOM_SSVEP_FREQ_POSITIONS = ("up", "left", "down", "right")
CUSTOM_SSVEP_POSITION_LABELS_CN = {
    "up": "上方",
    "left": "左侧",
    "down": "下方",
    "right": "右侧",
}

CUSTOM_SSVEP_COMMAND_CUE_SEC = 0.5
CUSTOM_SSVEP_COMMAND_ACTIVE_SEC = 3.0
CUSTOM_SSVEP_COMMAND_REST_SEC = 1.2
CUSTOM_SSVEP_SIGNAL_CHECK_SEC = 20.0
CUSTOM_SSVEP_BASELINE_SEC = 20.0
CUSTOM_SSVEP_NC_SEGMENT_SEC = 10.0
CUSTOM_SSVEP_NC_REST_SEC = 5.0

CUSTOM_SSVEP_INVALID_REASONS = (
    "blink_or_eye_movement",
    "looked_wrong_target",
    "body_movement",
    "electrode_noise",
    "stimulus_stutter",
    "other",
)

EVENT_CODES = {
    "TRIAL_START": 100,
    "CUE_ON": 110,
    "STIM_ON_COMMAND_8": 201,
    "STIM_ON_COMMAND_10": 202,
    "STIM_ON_COMMAND_10P5": 202,
    "STIM_ON_COMMAND_12": 203,
    "STIM_ON_COMMAND_15": 204,
    "STIM_ON_COMMAND_OTHER": 299,
    "STIM_OFF": 210,
    "REST_ON": 220,
    "TRIAL_END": 230,
    "NC_BLANK_CENTER_ON": 301,
    "NC_FLICKER_CENTER_ON": 302,
    "NC_FLICKER_OBJECT_ON": 303,
    "NC_FLICKER_FREE_ON": 304,
    "NC_SEGMENT_OFF": 310,
    "PAUSE_ON": 401,
    "PAUSE_OFF": 402,
    "BREAK_ON": 403,
    "BREAK_OFF": 404,
    "MARK_INVALID": 405,
    "SESSION_START": 900,
    "SESSION_END": 999,
}


@dataclass(frozen=True)
class CustomProtocolSpec:
    key: str
    command_calibration_repeats: int
    nc_calibration_repeats: int
    command_test_repeats: int
    nc_test_repeats: int


CUSTOM_PROTOCOL_SPECS = {
    CUSTOM_SSVEP_PROTOCOL_V1: CustomProtocolSpec(
        key=CUSTOM_SSVEP_PROTOCOL_V1,
        command_calibration_repeats=5,
        nc_calibration_repeats=3,
        command_test_repeats=2,
        nc_test_repeats=2,
    ),
    CUSTOM_SSVEP_SHORT_PROTOCOL_V1: CustomProtocolSpec(
        key=CUSTOM_SSVEP_SHORT_PROTOCOL_V1,
        command_calibration_repeats=4,
        nc_calibration_repeats=2,
        command_test_repeats=2,
        nc_test_repeats=1,
    ),
}


def normalize_custom_ssvep_protocol_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key in CUSTOM_PROTOCOL_SPECS:
        return key
    return ""


def is_custom_ssvep_protocol(value: Any) -> bool:
    return bool(normalize_custom_ssvep_protocol_key(value))


def _freq_key(freq: float) -> str:
    text = f"{float(freq):g}"
    return text.replace(".", "p")


def _freq_event_name(freq: float) -> str:
    token = _freq_key(float(freq)).upper()
    return f"STIM_ON_COMMAND_{token}"


def _target_metadata(freqs: Sequence[float], freq: float) -> tuple[str, str]:
    freq_values = tuple(float(item) for item in freqs)
    try:
        index = freq_values.index(float(freq))
    except ValueError:
        index = 0
    position = CUSTOM_SSVEP_FREQ_POSITIONS[min(max(index, 0), len(CUSTOM_SSVEP_FREQ_POSITIONS) - 1)]
    return position, CUSTOM_SSVEP_POSITION_LABELS_CN[position]


def _event(event_name: str, *, trial_id: int, event_value: Any = "") -> dict[str, Any]:
    return {
        "sample_index": None,
        "event_code": int(EVENT_CODES[event_name]),
        "event_name": str(event_name),
        "event_value": event_value,
        "perf_time": "",
        "trial_id": int(trial_id),
    }


def _idle_target_position(nc_subtype: str) -> str:
    subtype = str(nc_subtype or "").strip().lower()
    if subtype in {"blank_center", "flicker_center", "signal_check_center", "eyes_open_center"}:
        return "center"
    if subtype == "flicker_object":
        return "object_area"
    if subtype == "flicker_free":
        return "free_non_target"
    if subtype == "eyes_closed":
        return "eyes_closed"
    return ""


def _idle_target_position_label(nc_subtype: str) -> str:
    subtype = str(nc_subtype or "").strip().lower()
    labels = {
        "blank_center": "center fixation",
        "flicker_center": "center fixation",
        "signal_check_center": "center fixation",
        "eyes_open_center": "center fixation",
        "flicker_object": "object area",
        "flicker_free": "free non-target gaze",
        "eyes_closed": "eyes closed",
    }
    return labels.get(subtype, "")


def _command_trial_metadata(
    *,
    protocol_key: str,
    stage: str,
    split_role: str,
    trial_id: int,
    block_index: int,
    freq: float,
    freqs: Sequence[float],
) -> dict[str, Any]:
    position, position_cn = _target_metadata(freqs, freq)
    stim_event_name = _freq_event_name(float(freq))
    if stim_event_name not in EVENT_CODES:
        stim_event_name = "STIM_ON_COMMAND_OTHER"
    return {
        "protocol_name": str(protocol_key),
        "block_id": str(stage),
        "stage": str(stage),
        "split_role": str(split_role),
        "state_type": "command",
        "trial_role": TRIAL_ROLE_CONTROL,
        "target_freq": float(freq),
        "target_position": str(position),
        "target_position_label": str(position_cn),
        "stimulus_active": True,
        "all_targets_flickering": True,
        "cue_sec": float(CUSTOM_SSVEP_COMMAND_CUE_SEC),
        "active_sec": float(CUSTOM_SSVEP_COMMAND_ACTIVE_SEC),
        "rest_sec": float(CUSTOM_SSVEP_COMMAND_REST_SEC),
        "valid": True,
        "reject_reason": "",
        "event_codes": [
            int(EVENT_CODES["TRIAL_START"]),
            int(EVENT_CODES["CUE_ON"]),
            int(EVENT_CODES[stim_event_name]),
            int(EVENT_CODES["STIM_OFF"]),
            int(EVENT_CODES["REST_ON"]),
            int(EVENT_CODES["TRIAL_END"]),
        ],
        "events": [
            _event("TRIAL_START", trial_id=trial_id),
            _event("CUE_ON", trial_id=trial_id, event_value=float(freq)),
            _event(stim_event_name, trial_id=trial_id, event_value=float(freq)),
            _event("STIM_OFF", trial_id=trial_id),
            _event("REST_ON", trial_id=trial_id),
            _event("TRIAL_END", trial_id=trial_id),
        ],
    }


def _idle_metadata(
    *,
    protocol_key: str,
    stage: str,
    split_role: str,
    state_type: str,
    trial_role: str,
    nc_subtype: str,
    trial_id: int,
    active_sec: float,
    rest_sec: float,
    stimulus_active: bool,
    all_targets_flickering: bool,
    event_on_name: Optional[str],
) -> dict[str, Any]:
    events = [_event("TRIAL_START", trial_id=trial_id)]
    event_codes = [int(EVENT_CODES["TRIAL_START"])]
    if event_on_name:
        events.append(_event(event_on_name, trial_id=trial_id, event_value=nc_subtype))
        event_codes.append(int(EVENT_CODES[event_on_name]))
    events.extend(
        [
            _event("NC_SEGMENT_OFF", trial_id=trial_id, event_value=nc_subtype),
            _event("TRIAL_END", trial_id=trial_id),
        ]
    )
    event_codes.extend([int(EVENT_CODES["NC_SEGMENT_OFF"]), int(EVENT_CODES["TRIAL_END"])])
    return {
        "protocol_name": str(protocol_key),
        "block_id": str(stage),
        "stage": str(stage),
        "split_role": str(split_role),
        "state_type": str(state_type),
        "trial_role": str(trial_role),
        "nc_subtype": str(nc_subtype),
        "target_freq": None,
        "target_position": _idle_target_position(str(nc_subtype)),
        "target_position_label": _idle_target_position_label(str(nc_subtype)),
        "stimulus_active": bool(stimulus_active),
        "all_targets_flickering": bool(all_targets_flickering),
        "cue_sec": float(CUSTOM_SSVEP_COMMAND_CUE_SEC),
        "active_sec": float(active_sec),
        "rest_sec": float(rest_sec),
        "valid": True,
        "reject_reason": "",
        "event_codes": event_codes,
        "events": events,
    }


def build_custom_ssvep_collection_plan(
    freqs: Sequence[float],
    *,
    protocol_key: str = CUSTOM_SSVEP_PROTOCOL_V1,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> list[tuple[TrialSpec, dict[str, Any]]]:
    resolved_key = normalize_custom_ssvep_protocol_key(protocol_key) or CUSTOM_SSVEP_PROTOCOL_V1
    spec = CUSTOM_PROTOCOL_SPECS[resolved_key]
    freq_values = tuple(float(item) for item in freqs)
    if len(freq_values) != 4:
        raise ValueError("custom SSVEP collection protocol requires exactly 4 frequencies")
    rng = random.Random(int(seed))
    plan: list[tuple[TrialSpec, dict[str, Any]]] = []
    trial_id = 0
    block_index = 0

    def append(label: str, expected_freq: Optional[float], metadata: dict[str, Any]) -> None:
        nonlocal trial_id, block_index
        row_metadata = {**metadata, "trial_id": int(trial_id), "block_index": int(block_index)}
        plan.append(
            (
                TrialSpec(
                    label=str(label),
                    expected_freq=None if expected_freq is None else float(expected_freq),
                    trial_id=int(trial_id),
                    block_index=int(block_index),
                    metadata=dict(row_metadata),
                ),
                row_metadata,
            )
        )
        trial_id += 1
        block_index += 1

    append(
        "signal_check_center",
        None,
        _idle_metadata(
            protocol_key=resolved_key,
            stage="signal_check",
            split_role="calibration",
            state_type="baseline",
            trial_role=TRIAL_ROLE_CLEAN_IDLE,
            nc_subtype="signal_check_center",
            trial_id=trial_id,
            active_sec=CUSTOM_SSVEP_SIGNAL_CHECK_SEC,
            rest_sec=1.0,
            stimulus_active=False,
            all_targets_flickering=False,
            event_on_name=None,
        ),
    )
    for label, subtype in (("baseline_eyes_open", "eyes_open_center"), ("baseline_eyes_closed", "eyes_closed")):
        append(
            label,
            None,
            _idle_metadata(
                protocol_key=resolved_key,
                stage=label,
                split_role="calibration",
                state_type="baseline",
                trial_role=TRIAL_ROLE_CLEAN_IDLE,
                nc_subtype=subtype,
                trial_id=trial_id,
                active_sec=CUSTOM_SSVEP_BASELINE_SEC,
                rest_sec=1.0,
                stimulus_active=False,
                all_targets_flickering=False,
                event_on_name=None,
            ),
        )

    command_freqs = list(freq_values) * int(spec.command_calibration_repeats)
    rng.shuffle(command_freqs)
    # Avoid three identical command targets in a row without making the order deterministic by position.
    for _ in range(50):
        bad_index = next(
            (
                index
                for index in range(2, len(command_freqs))
                if command_freqs[index] == command_freqs[index - 1] == command_freqs[index - 2]
            ),
            -1,
        )
        if bad_index < 0:
            break
        swap_index = rng.randrange(0, len(command_freqs))
        command_freqs[bad_index], command_freqs[swap_index] = command_freqs[swap_index], command_freqs[bad_index]
    for freq in command_freqs:
        position, _position_cn = _target_metadata(freq_values, freq)
        append(
            f"command_calibration_{_freq_key(freq)}Hz_{position}",
            float(freq),
            _command_trial_metadata(
                protocol_key=resolved_key,
                stage="command_calibration",
                split_role="calibration",
                trial_id=trial_id,
                block_index=block_index,
                freq=float(freq),
                freqs=freq_values,
            ),
        )

    nc_calibration = (
        ("NC_BLANK_CENTER", "blank_center", TRIAL_ROLE_CLEAN_IDLE, False, False, "NC_BLANK_CENTER_ON"),
        ("NC_FLICKER_CENTER", "flicker_center", TRIAL_ROLE_HARD_IDLE, True, True, "NC_FLICKER_CENTER_ON"),
        ("NC_FLICKER_OBJECT", "flicker_object", TRIAL_ROLE_HARD_IDLE, True, True, "NC_FLICKER_OBJECT_ON"),
        ("NC_FLICKER_FREE", "flicker_free", TRIAL_ROLE_HARD_IDLE, True, True, "NC_FLICKER_FREE_ON"),
    )
    for repeat in range(int(spec.nc_calibration_repeats)):
        for label, subtype, role, stim_active, all_flicker, event_name in nc_calibration:
            append(
                f"{label}_calibration_r{repeat + 1}",
                None,
                _idle_metadata(
                    protocol_key=resolved_key,
                    stage="no_control_calibration",
                    split_role="calibration",
                    state_type="no_control",
                    trial_role=role,
                    nc_subtype=subtype,
                    trial_id=trial_id,
                    active_sec=CUSTOM_SSVEP_NC_SEGMENT_SEC,
                    rest_sec=CUSTOM_SSVEP_NC_REST_SEC,
                    stimulus_active=stim_active,
                    all_targets_flickering=all_flicker,
                    event_on_name=event_name,
                ),
            )

    command_test_freqs = list(freq_values) * int(spec.command_test_repeats)
    rng.shuffle(command_test_freqs)
    for freq in command_test_freqs:
        position, _position_cn = _target_metadata(freq_values, freq)
        append(
            f"command_test_{_freq_key(freq)}Hz_{position}",
            float(freq),
            _command_trial_metadata(
                protocol_key=resolved_key,
                stage="pseudo_online_command_test",
                split_role="test",
                trial_id=trial_id,
                block_index=block_index,
                freq=float(freq),
                freqs=freq_values,
            ),
        )

    nc_test = (
        ("NC_FLICKER_CENTER_TEST", "flicker_center", "NC_FLICKER_CENTER_ON"),
        ("NC_FLICKER_OBJECT_TEST", "flicker_object", "NC_FLICKER_OBJECT_ON"),
        ("NC_FLICKER_FREE_TEST", "flicker_free", "NC_FLICKER_FREE_ON"),
    )
    for repeat in range(int(spec.nc_test_repeats)):
        for label, subtype, event_name in nc_test:
            append(
                f"{label}_r{repeat + 1}",
                None,
                _idle_metadata(
                    protocol_key=resolved_key,
                    stage="pseudo_online_no_control_test",
                    split_role="test",
                    state_type="no_control",
                    trial_role=TRIAL_ROLE_HARD_IDLE,
                    nc_subtype=subtype,
                    trial_id=trial_id,
                    active_sec=CUSTOM_SSVEP_NC_SEGMENT_SEC,
                    rest_sec=CUSTOM_SSVEP_NC_REST_SEC,
                    stimulus_active=True,
                    all_targets_flickering=True,
                    event_on_name=event_name,
                ),
            )
    return plan


def build_custom_ssvep_collection_trials(
    freqs: Sequence[float],
    *,
    protocol_key: str = CUSTOM_SSVEP_PROTOCOL_V1,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> list[TrialSpec]:
    return [trial for trial, _metadata in build_custom_ssvep_collection_plan(freqs, protocol_key=protocol_key, seed=seed)]


def custom_ssvep_metadata_by_trial_id(
    freqs: Sequence[float],
    *,
    protocol_key: str = CUSTOM_SSVEP_PROTOCOL_V1,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> dict[int, dict[str, Any]]:
    return {
        int(trial.trial_id): dict(metadata)
        for trial, metadata in build_custom_ssvep_collection_plan(freqs, protocol_key=protocol_key, seed=seed)
    }


def custom_ssvep_protocol_summary(protocol_key: str, freqs: Sequence[float]) -> dict[str, Any]:
    resolved_key = normalize_custom_ssvep_protocol_key(protocol_key) or CUSTOM_SSVEP_PROTOCOL_V1
    plan = build_custom_ssvep_collection_plan(freqs, protocol_key=resolved_key)
    counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    total_sec = 0.0
    for _trial, metadata in plan:
        stage = str(metadata.get("stage", ""))
        counts[stage] = int(counts.get(stage, 0) + 1)
        split = str(metadata.get("split_role", ""))
        split_counts[split] = int(split_counts.get(split, 0) + 1)
        total_sec += float(metadata.get("cue_sec", CUSTOM_SSVEP_COMMAND_CUE_SEC))
        total_sec += float(metadata.get("active_sec", 0.0))
        total_sec += float(metadata.get("rest_sec", 0.0))
    return {
        "protocol_name": resolved_key,
        "trial_count": int(len(plan)),
        "stage_counts": counts,
        "split_counts": split_counts,
        "estimated_seconds": float(total_sec),
        "invalid_reasons": list(CUSTOM_SSVEP_INVALID_REASONS),
        "event_codes": dict(EVENT_CODES),
    }

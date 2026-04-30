from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Optional, Sequence

_MNE_HOME_DIR = Path(__file__).resolve().parents[1] / "runtime" / "mne_home"
_MNE_HOME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(_MNE_HOME_DIR))

import mne
import numpy as np

from .async_fbcca_idle_standalone import TrialSpec


EXTERNAL_LED_FREQS = (13.0, 17.0, 21.0)
EXTERNAL_LED_CHANNELS = ("Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4")

EXTERNAL_LED_REST_CODE = "33024"
EXTERNAL_LED_FREQ_CODE_MAP = {
    "33025": 13.0,
    "33027": 17.0,
    "33026": 21.0,
}
EXTERNAL_LED_LABEL_NAME_MAP = {
    EXTERNAL_LED_REST_CODE: "rest",
    **{code: f"{float(freq):g}Hz" for code, freq in EXTERNAL_LED_FREQ_CODE_MAP.items()},
}
EXTERNAL_LED_VISUAL_START = "32779"
EXTERNAL_LED_VISUAL_STOP = "32780"
EXTERNAL_LED_IGNORED_MARKERS = {"32769", "32770"}


def _round_sample(value_sec: float, sampling_rate: float) -> int:
    return int(round(float(value_sec) * float(sampling_rate)))


def _natural_token_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(text))
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


@dataclass(frozen=True)
class ExternalReplayTrial:
    subject_id: str
    session_id: str
    session_index: int
    trial_index: int
    label_code: str
    label_name: str
    expected_freq: Optional[float]
    label_sample: int
    stim_start_sample: int
    stim_stop_sample: int

    @property
    def is_rest(self) -> bool:
        return self.expected_freq is None

    @property
    def trial_role(self) -> str:
        return "control" if self.expected_freq is not None else "clean_idle"

    @property
    def duration_samples(self) -> int:
        return max(int(self.stim_stop_sample) - int(self.stim_start_sample), 0)


@dataclass(frozen=True)
class ExternalReplaySession:
    subject_id: str
    session_id: str
    session_index: int
    file_path: Path
    sampling_rate: int
    channel_names: tuple[str, ...]
    data: np.ndarray
    trials: tuple[ExternalReplayTrial, ...]

    @property
    def sample_count(self) -> int:
        return int(self.data.shape[0])

    @property
    def duration_sec(self) -> float:
        return float(self.sample_count) / float(max(self.sampling_rate, 1))


@dataclass(frozen=True)
class ExternalReplayDataset:
    dataset_root: Path
    subject_id: str
    sampling_rate: int
    channel_names: tuple[str, ...]
    freqs: tuple[float, ...]
    sessions: tuple[ExternalReplaySession, ...]


def discover_external_replay_subjects(dataset_root: Path) -> dict[str, tuple[Path, ...]]:
    root = Path(dataset_root).expanduser().resolve()
    grouped: dict[str, list[Path]] = {}
    for path in root.rglob("*.gdf"):
        subject_id = str(path.parent.name)
        grouped.setdefault(subject_id, []).append(path.resolve())
    return {
        subject_id: tuple(sorted(paths, key=lambda item: _natural_token_key(item.stem)))
        for subject_id, paths in sorted(grouped.items(), key=lambda item: _natural_token_key(item[0]))
    }


def _read_gdf_session(path: Path, *, session_index: int) -> ExternalReplaySession:
    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
    sampling_rate = int(round(float(raw.info["sfreq"])))
    available_names = tuple(str(name) for name in raw.ch_names)
    missing = [name for name in EXTERNAL_LED_CHANNELS if name not in available_names]
    if missing:
        raise ValueError(f"missing required channels in {path}: {missing}")
    channel_indices = [available_names.index(name) for name in EXTERNAL_LED_CHANNELS]
    data = np.asarray(raw.get_data(picks=channel_indices).T, dtype=np.float64)

    events = [
        {
            "description": str(raw.annotations.description[index]).strip(),
            "onset": float(raw.annotations.onset[index]),
        }
        for index in range(len(raw.annotations))
    ]
    trials: list[ExternalReplayTrial] = []
    current_label_code: Optional[str] = None
    current_label_onset_sec: Optional[float] = None
    trial_index = 0
    session_id = str(path.stem)
    subject_id = str(path.parent.name)

    for index, event in enumerate(events):
        description = str(event["description"]).strip()
        if description in EXTERNAL_LED_IGNORED_MARKERS:
            continue
        if description in EXTERNAL_LED_FREQ_CODE_MAP or description == EXTERNAL_LED_REST_CODE:
            current_label_code = str(description)
            current_label_onset_sec = float(event["onset"])
            continue
        if description != EXTERNAL_LED_VISUAL_START or current_label_code is None or current_label_onset_sec is None:
            continue
        stop_onset_sec: Optional[float] = None
        for follow in events[index + 1 :]:
            follow_desc = str(follow["description"]).strip()
            if follow_desc == EXTERNAL_LED_VISUAL_STOP:
                stop_onset_sec = float(follow["onset"])
                break
            if follow_desc in EXTERNAL_LED_FREQ_CODE_MAP or follow_desc in {
                EXTERNAL_LED_REST_CODE,
                EXTERNAL_LED_VISUAL_START,
            }:
                break
        if stop_onset_sec is None:
            stop_onset_sec = float(event["onset"]) + 5.0
        stim_start_sample = _round_sample(float(event["onset"]), sampling_rate)
        stim_stop_sample = _round_sample(float(stop_onset_sec), sampling_rate)
        label_sample = _round_sample(float(current_label_onset_sec), sampling_rate)
        if stim_stop_sample <= stim_start_sample:
            current_label_code = None
            current_label_onset_sec = None
            continue
        expected_freq = EXTERNAL_LED_FREQ_CODE_MAP.get(current_label_code)
        trials.append(
            ExternalReplayTrial(
                subject_id=subject_id,
                session_id=session_id,
                session_index=int(session_index),
                trial_index=int(trial_index),
                label_code=str(current_label_code),
                label_name=str(EXTERNAL_LED_LABEL_NAME_MAP.get(current_label_code, current_label_code)),
                expected_freq=None if expected_freq is None else float(expected_freq),
                label_sample=int(label_sample),
                stim_start_sample=int(stim_start_sample),
                stim_stop_sample=int(stim_stop_sample),
            )
        )
        trial_index += 1
        current_label_code = None
        current_label_onset_sec = None

    return ExternalReplaySession(
        subject_id=subject_id,
        session_id=session_id,
        session_index=int(session_index),
        file_path=path.resolve(),
        sampling_rate=int(sampling_rate),
        channel_names=tuple(EXTERNAL_LED_CHANNELS),
        data=np.ascontiguousarray(data, dtype=np.float64),
        trials=tuple(trials),
    )


def load_external_replay_dataset(dataset_root: Path, *, subject: str) -> ExternalReplayDataset:
    subjects = discover_external_replay_subjects(dataset_root)
    subject_key = str(subject).strip()
    if subject_key not in subjects:
        raise FileNotFoundError(
            f"subject '{subject}' not found under {Path(dataset_root).expanduser().resolve()}"
        )
    session_paths = tuple(subjects[subject_key])
    if not session_paths:
        raise FileNotFoundError(f"no sessions found for subject '{subject_key}'")

    sessions: list[ExternalReplaySession] = []
    sampling_rate: Optional[int] = None
    for session_index, path in enumerate(session_paths):
        session = _read_gdf_session(path, session_index=session_index)
        if sampling_rate is None:
            sampling_rate = int(session.sampling_rate)
        elif int(session.sampling_rate) != int(sampling_rate):
            raise ValueError(
                f"inconsistent sampling rate for {subject_key}: {sampling_rate} vs {session.sampling_rate}"
            )
        sessions.append(session)

    if sampling_rate is None:
        raise RuntimeError(f"failed to load external replay dataset for subject '{subject_key}'")

    return ExternalReplayDataset(
        dataset_root=Path(dataset_root).expanduser().resolve(),
        subject_id=subject_key,
        sampling_rate=int(sampling_rate),
        channel_names=tuple(EXTERNAL_LED_CHANNELS),
        freqs=tuple(float(freq) for freq in EXTERNAL_LED_FREQS),
        sessions=tuple(sessions),
    )


def collect_trial_segments(
    dataset: ExternalReplayDataset,
    *,
    session_indices: Sequence[int],
    include_rest: bool = True,
) -> list[tuple[TrialSpec, np.ndarray]]:
    keep = {int(index) for index in session_indices}
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    global_trial_id = 0
    for session in dataset.sessions:
        if int(session.session_index) not in keep:
            continue
        for trial in session.trials:
            if trial.expected_freq is None and not include_rest:
                continue
            segment = np.ascontiguousarray(
                session.data[trial.stim_start_sample : trial.stim_stop_sample, :],
                dtype=np.float64,
            )
            segments.append(
                (
                    TrialSpec(
                        label=str(trial.label_name),
                        expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
                        trial_id=int(global_trial_id),
                        block_index=int(session.session_index),
                    ),
                    segment,
                )
            )
            global_trial_id += 1
    return segments

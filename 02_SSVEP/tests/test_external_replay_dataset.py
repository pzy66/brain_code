from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.external_replay_dataset import (
    EXTERNAL_LED_CHANNELS,
    EXTERNAL_LED_FREQS,
    ExternalReplayDataset,
    ExternalReplaySession,
    ExternalReplayTrial,
    collect_trial_segments,
    load_external_replay_dataset,
)


def _make_session(*, subject: str, session_index: int, file_path: Path) -> ExternalReplaySession:
    data = np.arange(160 * len(EXTERNAL_LED_CHANNELS), dtype=np.float64).reshape(160, len(EXTERNAL_LED_CHANNELS))
    trials = (
        ExternalReplayTrial(
            subject_id=subject,
            session_id=f"s{session_index + 1}",
            session_index=session_index,
            trial_index=0,
            label_code="33025",
            label_name="13Hz",
            expected_freq=13.0,
            label_sample=0,
            stim_start_sample=10,
            stim_stop_sample=70,
        ),
        ExternalReplayTrial(
            subject_id=subject,
            session_id=f"s{session_index + 1}",
            session_index=session_index,
            trial_index=1,
            label_code="33024",
            label_name="rest",
            expected_freq=None,
            label_sample=70,
            stim_start_sample=80,
            stim_stop_sample=140,
        ),
    )
    return ExternalReplaySession(
        subject_id=subject,
        session_id=f"s{session_index + 1}",
        session_index=session_index,
        file_path=file_path,
        sampling_rate=256,
        channel_names=tuple(EXTERNAL_LED_CHANNELS),
        data=data,
        trials=trials,
    )


def test_collect_trial_segments_preserves_active_and_rest() -> None:
    session = _make_session(subject="Subject2", session_index=0, file_path=Path("Subject2") / "s1.gdf")
    dataset = ExternalReplayDataset(
        dataset_root=Path("dataset"),
        subject_id="Subject2",
        sampling_rate=256,
        channel_names=tuple(EXTERNAL_LED_CHANNELS),
        freqs=tuple(EXTERNAL_LED_FREQS),
        sessions=(session,),
    )

    segments = collect_trial_segments(dataset, session_indices=(0,), include_rest=True)

    assert len(segments) == 2
    assert segments[0][0].expected_freq == 13.0
    assert segments[1][0].expected_freq is None
    assert segments[0][1].shape == (60, len(EXTERNAL_LED_CHANNELS))
    assert segments[1][1].shape == (60, len(EXTERNAL_LED_CHANNELS))


def test_load_external_replay_dataset_uses_discovery_and_session_reader(monkeypatch, tmp_path: Path) -> None:
    subject_dir = tmp_path / "Subject2"
    subject_dir.mkdir(parents=True, exist_ok=True)
    session_path = subject_dir / "session_01.gdf"
    session_path.write_bytes(b"")

    fake_session = _make_session(subject="Subject2", session_index=0, file_path=session_path)

    monkeypatch.setattr(
        "ssvep_core.external_replay_dataset.discover_external_replay_subjects",
        lambda dataset_root: {"Subject2": (session_path.resolve(),)},
    )
    monkeypatch.setattr(
        "ssvep_core.external_replay_dataset._read_gdf_session",
        lambda path, session_index: fake_session,
    )

    dataset = load_external_replay_dataset(tmp_path, subject="Subject2")

    assert dataset.subject_id == "Subject2"
    assert dataset.sampling_rate == 256
    assert dataset.channel_names == tuple(EXTERNAL_LED_CHANNELS)
    assert dataset.freqs == tuple(EXTERNAL_LED_FREQS)
    assert len(dataset.sessions) == 1

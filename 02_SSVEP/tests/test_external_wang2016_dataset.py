from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core import external_wang2016_dataset as wang
from ssvep_core.dataset import load_collection_dataset
from ssvep_core.trial_roles import infer_trial_role


def _write_loc(path: Path, names: list[str]) -> None:
    path.write_text(
        "\n".join(f"{index + 1}\t0\t0\t{name}" for index, name in enumerate(names)) + "\n",
        encoding="utf-8",
    )


def _channel_names_64() -> list[str]:
    names = [f"Ch{index + 1}" for index in range(64)]
    names[52] = "PO7"
    names[54] = "PO3"
    names[55] = "POz"
    names[56] = "PO4"
    names[58] = "PO8"
    names[60] = "O1"
    names[61] = "Oz"
    names[62] = "O2"
    return names


def _small_subject(blocks: int = 2, target_count: int = 10) -> wang.Wang2016LoadedSubject:
    channel_names = tuple(_channel_names_64())
    indices = wang.selected_channel_indices(channel_names)
    eeg = np.arange(64 * 1375 * target_count * blocks, dtype=np.float32).reshape(
        64,
        1375,
        target_count,
        blocks,
    )
    return wang.Wang2016LoadedSubject(
        subject="S1",
        mat_path=Path("S1.mat"),
        eeg=eeg,
        channel_names=channel_names,
        selected_channel_names=tuple(channel_names[index] for index in indices),
        selected_channel_indices=indices,
    )


def test_parse_wang2016_channel_loc_and_required_order(tmp_path: Path) -> None:
    loc_path = tmp_path / "64-channels.loc"
    _write_loc(loc_path, _channel_names_64())

    channel_names = wang.parse_wang2016_channel_loc(loc_path)
    indices = wang.selected_channel_indices(channel_names)

    assert indices == (61, 60, 62, 54, 55, 52, 58, 56)
    assert tuple(channel_names[index] for index in indices) == wang.WANG2016_REQUIRED_CHANNELS


def test_selected_channel_indices_rejects_missing_required_channel() -> None:
    channel_names = _channel_names_64()
    channel_names[58] = "PO6"

    try:
        wang.selected_channel_indices(channel_names)
    except ValueError as error:
        assert "PO8" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing PO8 should be rejected")


def test_build_segments_uses_non_command_targets_as_hard_idle(monkeypatch) -> None:
    monkeypatch.setattr(wang, "WANG2016_BLOCKS", 2)
    monkeypatch.setattr(wang, "WANG2016_TARGET_COUNT", 10)
    monkeypatch.setattr(wang, "WANG2016_TARGET_FREQUENCIES", (8, 9, 10, 11, 12, 13, 14, 15, 8.2, 9.2))
    subject = _small_subject(blocks=2, target_count=10)

    segments = wang.build_wang2016_segments(subject)

    control = [(trial, segment) for trial, segment in segments if trial.expected_freq is not None]
    idle = [(trial, segment) for trial, segment in segments if trial.expected_freq is None]
    assert len(control) == 8
    assert len(idle) == 12
    assert all(segment.shape == (1250, 8) for _trial, segment in segments)
    assert all(infer_trial_role(label=trial.label, expected_freq=None) == "hard_idle" for trial, _segment in idle)


def test_build_segments_accepts_configurable_four_command_frequencies(monkeypatch) -> None:
    monkeypatch.setattr(wang, "WANG2016_BLOCKS", 1)
    monkeypatch.setattr(wang, "WANG2016_TARGET_COUNT", 10)
    monkeypatch.setattr(wang, "WANG2016_TARGET_FREQUENCIES", (8, 9, 10, 11, 12, 13, 14, 15, 8.2, 9.2))
    monkeypatch.setattr(
        wang,
        "WANG2016_ALL_TARGET_INDEX_BY_FREQ",
        {float(freq): int(index + 1) for index, freq in enumerate(wang.WANG2016_TARGET_FREQUENCIES)},
    )
    subject = _small_subject(blocks=1, target_count=10)

    segments = wang.build_wang2016_segments(subject, freqs=(9.0, 11.0, 13.0, 15.0))

    control_freqs = [trial.expected_freq for trial, _segment in segments if trial.expected_freq is not None]
    idle = [(trial, segment) for trial, segment in segments if trial.expected_freq is None]
    assert control_freqs == [9.0, 11.0, 13.0, 15.0]
    assert len(idle) == 6
    assert all(segment.shape == (1250, 8) for _trial, segment in segments)


def test_convert_saves_only_required_8_channels_and_manifest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(wang, "WANG2016_BLOCKS", 2)
    monkeypatch.setattr(wang, "WANG2016_TARGET_COUNT", 10)
    monkeypatch.setattr(wang, "WANG2016_TARGET_FREQUENCIES", (8, 9, 10, 11, 12, 13, 14, 15, 8.2, 9.2))
    subject = _small_subject(blocks=2, target_count=10)
    monkeypatch.setattr(wang, "load_wang2016_subject", lambda _mat, _loc: subject)

    payload = wang.convert_wang2016_subject_to_collection(
        mat_path=tmp_path / "S1.mat",
        channel_loc_path=tmp_path / "64-channels.loc",
        dataset_root=tmp_path / "datasets",
        session_id="wang2016_s1_test",
    )

    loaded = load_collection_dataset(Path(payload["dataset_manifest"]))
    assert len(loaded.trial_segments) == 20
    assert all(segment.shape[1] == 8 for _trial, segment in loaded.trial_segments)
    assert all(int(row["channels"]) == 8 for row in loaded.manifest["trials"])
    protocol = loaded.manifest["protocol_config"]
    validation = loaded.manifest["external_dataset_validation"]
    assert protocol["selected_channel_names"] == list(wang.WANG2016_REQUIRED_CHANNELS)
    assert protocol["selected_channel_indices_zero_based"] == [61, 60, 62, 54, 55, 52, 58, 56]
    assert protocol["excluded_channel_count"] == 56
    assert protocol["only_required_channels_saved"] is True
    assert protocol["control_trial_count"] == 8
    assert protocol["hard_idle_trial_count"] == 12
    assert loaded.manifest["quality_summary"]["trial_role_counts"] == {
        "control": 8,
        "clean_idle": 0,
        "hard_idle": 12,
    }
    assert validation["only_required_channels_saved"] is True


def test_convert_manifest_records_configurable_freqs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(wang, "WANG2016_BLOCKS", 1)
    monkeypatch.setattr(wang, "WANG2016_TARGET_COUNT", 10)
    monkeypatch.setattr(wang, "WANG2016_TARGET_FREQUENCIES", (8, 9, 10, 11, 12, 13, 14, 15, 8.2, 9.2))
    monkeypatch.setattr(
        wang,
        "WANG2016_ALL_TARGET_INDEX_BY_FREQ",
        {float(freq): int(index + 1) for index, freq in enumerate(wang.WANG2016_TARGET_FREQUENCIES)},
    )
    subject = _small_subject(blocks=1, target_count=10)
    monkeypatch.setattr(wang, "load_wang2016_subject", lambda _mat, _loc: subject)

    payload = wang.convert_wang2016_subject_to_collection(
        mat_path=tmp_path / "S1.mat",
        channel_loc_path=tmp_path / "64-channels.loc",
        dataset_root=tmp_path / "datasets",
        freqs=(9.0, 11.0, 13.0, 15.0),
    )

    loaded = load_collection_dataset(Path(payload["dataset_manifest"]))
    assert loaded.freqs == (9.0, 11.0, 13.0, 15.0)
    assert loaded.manifest["protocol_config"]["freqs"] == [9.0, 11.0, 13.0, 15.0]
    assert loaded.manifest["protocol_config"]["target_index_by_freq"] == {
        "9": 2,
        "11": 4,
        "13": 6,
        "15": 8,
    }
    assert loaded.manifest["external_dataset_validation"]["freqs"] == [9.0, 11.0, 13.0, 15.0]

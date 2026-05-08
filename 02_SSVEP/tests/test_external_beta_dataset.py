from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.external_beta_dataset import (
    BETA_REQUIRED_CHANNELS,
    BetaLoadedSubject,
    build_beta_segments,
    resolve_beta_command_frequencies,
    selected_channel_indices,
)


def _fake_subject() -> BetaLoadedSubject:
    names = tuple(
        [
            "FP1",
            "FPZ",
            "FP2",
            "AF3",
            "AF4",
            "F7",
            "F5",
            "F3",
            "F1",
            "FZ",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "FCZ",
            "FC2",
            "FC4",
            "FC6",
            "FT8",
            "T7",
            "C5",
            "C3",
            "C1",
            "CZ",
            "C2",
            "C4",
            "C6",
            "T8",
            "M1",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPZ",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "M2",
            "P7",
            "P5",
            "P3",
            "P1",
            "PZ",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO5",
            "PO3",
            "POZ",
            "PO4",
            "PO6",
            "PO8",
            "CB1",
            "O1",
            "OZ",
            "O2",
            "CB2",
        ]
    )
    freqs = tuple(float(item) for item in np.linspace(8.0, 15.8, 40))
    eeg = np.zeros((64, 1000, 4, 40), dtype=np.float64)
    return BetaLoadedSubject(
        subject="S16",
        mat_path=Path("S16.mat"),
        eeg=eeg,
        channel_names=names,
        selected_channel_names=("OZ", "O1", "O2", "PO3", "POZ", "PO7", "PO8", "PO4"),
        selected_channel_indices=selected_channel_indices(names),
        target_frequencies=freqs,
        target_phases=tuple(0.0 for _ in freqs),
        sampling_rate=250,
        stimulus_sec=3.0,
        trial_sec=4.0,
    )


def test_selected_channel_indices_uses_required_order_case_insensitive() -> None:
    subject = _fake_subject()
    assert subject.selected_channel_indices == (61, 60, 62, 54, 55, 52, 58, 56)
    assert tuple(name.lower() for name in BETA_REQUIRED_CHANNELS) == (
        "oz",
        "o1",
        "o2",
        "po3",
        "poz",
        "po7",
        "po8",
        "po4",
    )


def test_resolve_beta_command_frequencies_rejects_unavailable_values() -> None:
    subject = _fake_subject()
    freqs, mapping = resolve_beta_command_frequencies(subject, (8.0, 8.2, 8.4, 8.6))
    assert freqs == (8.0, 8.2, 8.4, 8.6)
    assert mapping[8.0] == 1
    with pytest.raises(ValueError, match="unavailable"):
        resolve_beta_command_frequencies(subject, (8.0, 10.1, 12.0, 15.0))


def test_build_beta_segments_marks_non_command_targets_hard_idle() -> None:
    subject = _fake_subject()
    segments = build_beta_segments(
        subject,
        freqs=(8.0, 8.2, 8.4, 8.6),
        include_hard_idle=True,
        include_pre_stim_idle=False,
    )
    assert len(segments) == 4 * 4 + 36 * 4
    assert all(segment.shape == (750, 8) for _trial, segment in segments)
    control = [trial for trial, _segment in segments if trial.expected_freq is not None]
    hard_idle = [trial for trial, _segment in segments if trial.expected_freq is None]
    assert len(control) == 16
    assert len(hard_idle) == 144
    assert hard_idle[0].label.startswith("hard_idle_beta_target")

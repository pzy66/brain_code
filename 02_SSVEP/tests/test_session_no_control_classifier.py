from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core import async_fbcca_idle_standalone as async_module
from ssvep_core import session_no_control_classifier as session_nc
from ssvep_core.async_fbcca_idle_standalone import (
    AsyncDecisionGate,
    TrialSpec,
    load_decoder_from_profile,
    load_profile,
)


class _FakeFBCCADecoder:
    def __init__(
        self,
        *,
        sampling_rate: int,
        freqs: tuple[float, ...],
        win_sec: float,
        step_sec: float,
        **_kwargs: object,
    ) -> None:
        self.sampling_rate = int(sampling_rate)
        self.freqs = tuple(float(freq) for freq in freqs)
        self.win_samples = int(round(float(win_sec) * float(sampling_rate)))
        self.step_samples = int(round(float(step_sec) * float(sampling_rate)))
        self.compute_backend_requested = "cpu"
        self.compute_backend_used = "cpu"

    def configure_runtime(self, sampling_rate: int) -> None:
        self.sampling_rate = int(sampling_rate)

    def score_windows_batch(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float64)
        codes = np.rint(values[:, 0, 0]).astype(int)
        rows: list[np.ndarray] = []
        command_by_code = {1: 8.0, 2: 10.0, 3: 12.0, 4: 15.0}
        command_freqs = set(command_by_code.values())
        for code in codes:
            row = np.full(len(self.freqs), 0.1, dtype=np.float64)
            target_freq = command_by_code.get(int(code))
            if target_freq is None:
                row[:] = 0.2
                for index, freq in enumerate(self.freqs):
                    if float(freq) not in command_freqs:
                        row[index] = 4.0
            else:
                for index, freq in enumerate(self.freqs):
                    if abs(float(freq) - float(target_freq)) <= 1e-8:
                        row[index] = 5.0
            rows.append(row)
        return np.vstack(rows)


def _fake_create_decoder(model_name: str, *args: object, **kwargs: object) -> _FakeFBCCADecoder:
    del model_name, args
    return _FakeFBCCADecoder(
        sampling_rate=int(kwargs["sampling_rate"]),
        freqs=tuple(float(freq) for freq in kwargs["freqs"]),
        win_sec=float(kwargs["win_sec"]),
        step_sec=float(kwargs["step_sec"]),
    )


def _segment_for_code(code: int, *, samples: int = 1000, channels: int = 8) -> np.ndarray:
    return np.full((int(samples), int(channels)), float(code), dtype=np.float64)


def _session_segments() -> list[tuple[TrialSpec, np.ndarray]]:
    rows: list[tuple[TrialSpec, np.ndarray]] = []
    for repeat in range(2):
        for code, freq in ((1, 8.0), (2, 10.0), (3, 12.0), (4, 15.0)):
            rows.append(
                (
                    TrialSpec(label=f"{freq:g}Hz_r{repeat}", expected_freq=float(freq), trial_id=len(rows)),
                    _segment_for_code(code),
                )
            )
    for repeat in range(8):
        rows.append((TrialSpec(label=f"idle_r{repeat}", expected_freq=None, trial_id=len(rows)), _segment_for_code(0)))
    return rows


def test_session_no_control_profile_trains_and_roundtrips_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(session_nc, "create_decoder", _fake_create_decoder)
    profile, quality = session_nc.fit_session_no_control_fbcca_ridge_profile(
        _session_segments(),
        sampling_rate=250,
        available_board_channels=tuple(range(8)),
        compute_backend="cpu",
        gpu_warmup=False,
    )

    assert profile.model_name == "fbcca_score_ridge_5class"
    assert profile.freqs == session_nc.SESSION_NC_DEFAULT_FREQS
    assert profile.gate_policy == "lrt_multiwindow_reject_gate"
    assert profile.min_enter_windows == session_nc.SESSION_NC_MIN_ENTER_WINDOWS
    assert dict(profile.model_params or {})["score_bank_mode"] == "full_reference_bank"
    state = dict(dict(profile.model_params or {})["state"])
    assert state["gate_policy"] == "lrt_multiwindow_reject_gate"
    assert state["smoothing_windows"] == session_nc.SESSION_NC_SMOOTHING_WINDOWS
    assert state["fit_summary"]["calibration_counts"]["idle_trials"] == 8
    assert quality["score_bank_mode"] == "full_reference_bank"

    saved_profile, saved_v2 = session_nc.save_session_no_control_profile_bundle(
        profile,
        tmp_path / "fbcca_ridge5_session_nc_profile.json",
        quality,
    )
    loaded = load_profile(saved_profile, require_exists=True)
    payload_v2 = json.loads(saved_v2.read_text(encoding="utf-8"))
    assert loaded.model_name == "fbcca_score_ridge_5class"
    assert payload_v2["version"] == "2.0"
    assert payload_v2["decoder"]["name"] == "fbcca_score_ridge_5class"
    assert payload_v2["gate"]["type"] == "session_no_control_lrt_multiwindow"

    original_create_decoder = async_module.create_decoder

    def runtime_create_decoder(model_name: str, *args: object, **kwargs: object):
        if str(model_name) == "fbcca_fixed_all8":
            return _fake_create_decoder(model_name, *args, **kwargs)
        return original_create_decoder(model_name, *args, **kwargs)

    monkeypatch.setattr(async_module, "create_decoder", runtime_create_decoder)
    decoder = load_decoder_from_profile(loaded, sampling_rate=250, compute_backend="cpu", gpu_warmup=False)
    gate = AsyncDecisionGate.from_profile(loaded)
    window = _segment_for_code(1, samples=decoder.win_samples)

    first = gate.update(decoder.analyze_window(window))
    second = gate.update(decoder.analyze_window(window))

    assert first["state"] == "idle"
    assert second["state"] == "selected"
    assert second["selected_freq"] == 8.0


def test_session_no_control_profile_requires_idle_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_nc, "create_decoder", _fake_create_decoder)
    command_only = [item for item in _session_segments() if item[0].expected_freq is not None]

    with pytest.raises(ValueError, match="idle"):
        session_nc.fit_session_no_control_fbcca_ridge_profile(
            command_only,
            sampling_rate=250,
            available_board_channels=tuple(range(8)),
            compute_backend="cpu",
            gpu_warmup=False,
        )


def test_session_no_control_profile_uses_only_valid_calibration_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_nc, "create_decoder", _fake_create_decoder)
    rows = list(_session_segments())
    rows.append(
        (
            TrialSpec(
                label="command_test_8Hz",
                expected_freq=8.0,
                trial_id=100,
                metadata={"split_role": "test", "state_type": "command", "valid": True},
            ),
            _segment_for_code(1),
        )
    )
    rows.append(
        (
            TrialSpec(
                label="baseline_eyes_open",
                expected_freq=None,
                trial_id=101,
                metadata={"split_role": "calibration", "state_type": "baseline", "valid": True},
            ),
            _segment_for_code(0),
        )
    )
    rows.append(
        (
            TrialSpec(
                label="invalid_idle",
                expected_freq=None,
                trial_id=102,
                metadata={"split_role": "calibration", "state_type": "no_control", "valid": False},
            ),
            _segment_for_code(0),
        )
    )

    _profile, quality = session_nc.fit_session_no_control_fbcca_ridge_profile(
        rows,
        sampling_rate=250,
        available_board_channels=tuple(range(8)),
        compute_backend="cpu",
        gpu_warmup=False,
    )

    counts = dict(quality["calibration_counts"])
    assert int(counts["control_trials"]) == 8
    assert int(counts["idle_trials"]) == 8


def test_session_no_control_profile_excludes_metadata_less_test_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_nc, "create_decoder", _fake_create_decoder)
    rows = list(_session_segments())
    rows.append(
        (
            TrialSpec(label="command_test_8Hz", expected_freq=8.0, trial_id=100),
            _segment_for_code(1),
        )
    )
    rows.append(
        (
            TrialSpec(label="NC_FLICKER_CENTER_TEST_r1", expected_freq=None, trial_id=101),
            _segment_for_code(0),
        )
    )

    _profile, quality = session_nc.fit_session_no_control_fbcca_ridge_profile(
        rows,
        sampling_rate=250,
        available_board_channels=tuple(range(8)),
        compute_backend="cpu",
        gpu_warmup=False,
    )

    counts = dict(quality["calibration_counts"])
    assert int(counts["control_trials"]) == 8
    assert int(counts["idle_trials"]) == 8


def test_session_no_control_profile_rejects_string_false_valid_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_nc, "create_decoder", _fake_create_decoder)
    rows = list(_session_segments())
    rows[0] = (
        TrialSpec(
            label="8Hz_invalid",
            expected_freq=8.0,
            trial_id=0,
            metadata={"split_role": "calibration", "state_type": "command", "valid": "false"},
        ),
        _segment_for_code(1),
    )

    _profile, quality = session_nc.fit_session_no_control_fbcca_ridge_profile(
        rows,
        sampling_rate=250,
        available_board_channels=tuple(range(8)),
        compute_backend="cpu",
        gpu_warmup=False,
    )

    counts = dict(quality["calibration_counts"])
    assert int(dict(counts["per_label_trials"])["8"]) == 1


def test_session_no_control_profile_v2_sibling_does_not_overwrite_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(session_nc, "create_decoder", _fake_create_decoder)
    profile, quality = session_nc.fit_session_no_control_fbcca_ridge_profile(
        _session_segments(),
        sampling_rate=250,
        available_board_channels=tuple(range(8)),
        compute_backend="cpu",
        gpu_warmup=False,
    )

    profile_path, profile_v2_path = session_nc.save_session_no_control_profile_bundle(
        profile,
        tmp_path / "session_nc.json",
        quality,
    )

    assert profile_path.name == "session_nc.json"
    assert profile_v2_path.name == "session_nc_v2.json"
    assert profile_path != profile_v2_path
    assert json.loads(profile_path.read_text(encoding="utf-8"))["model_name"] == "fbcca_score_ridge_5class"
    assert json.loads(profile_v2_path.read_text(encoding="utf-8"))["version"] == "2.0"


def test_session_no_control_profile_refuses_default_profile_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(session_nc, "create_decoder", _fake_create_decoder)
    monkeypatch.setattr(session_nc, "DEFAULT_PROFILE_PATH", tmp_path / "default_profile.json")
    profile, quality = session_nc.fit_session_no_control_fbcca_ridge_profile(
        _session_segments(),
        sampling_rate=250,
        available_board_channels=tuple(range(8)),
        compute_backend="cpu",
        gpu_warmup=False,
    )

    with pytest.raises(ValueError, match="default_profile"):
        session_nc.save_session_no_control_profile_bundle(
            profile,
            tmp_path / "default_profile.json",
            quality,
        )


def test_session_no_control_profile_refuses_any_default_profile_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(session_nc, "create_decoder", _fake_create_decoder)
    profile, quality = session_nc.fit_session_no_control_fbcca_ridge_profile(
        _session_segments(),
        sampling_rate=250,
        available_board_channels=tuple(range(8)),
        compute_backend="cpu",
        gpu_warmup=False,
    )

    with pytest.raises(ValueError, match="default_profile"):
        session_nc.save_session_no_control_profile_bundle(
            profile,
            tmp_path / "nested" / "default_profile.json",
            quality,
        )

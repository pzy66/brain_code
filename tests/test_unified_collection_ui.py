from __future__ import annotations

import numpy as np

from brain_workspace.bootstrap import configure_qt_offscreen

configure_qt_offscreen()

from PyQt5.QtWidgets import QApplication  # noqa: E402

from unified_collection import (  # noqa: E402
    SSVEP_MARKER_ACTIVE_WINDOW_END,
    SSVEP_MARKER_ACTIVE_WINDOW_START,
    SSVEP_MARKER_SESSION_START,
    SSVEP_MARKER_STIM_FIRST_FRAME,
    SSVEPUnifiedConfig,
    SSVEPProtocolOnlyWorker,
    UnifiedCollectionWindow,
    derive_ssvep_trial_segments_from_markers,
    _jsonable_event,
)
from ssvep_core.async_fbcca_idle_standalone import TrialSpec  # noqa: E402


def test_ssvep_marker_derivation_uses_active_window_start() -> None:
    board = np.zeros((5, 500), dtype=np.float64)
    board[0, :] = np.arange(500, dtype=np.float64)
    board[1, :] = np.arange(500, dtype=np.float64) + 1000.0
    marker_row = 4
    board[marker_row, 10] = SSVEP_MARKER_SESSION_START
    board[marker_row, 20] = SSVEP_MARKER_STIM_FIRST_FRAME
    board[marker_row, 30] = SSVEP_MARKER_ACTIVE_WINDOW_START
    board[marker_row, 130] = SSVEP_MARKER_ACTIVE_WINDOW_END

    trial = TrialSpec(label="8.0Hz", expected_freq=8.0, trial_id=1, block_index=0)
    event_log = [
        _jsonable_event(marker_code=SSVEP_MARKER_SESSION_START),
        _jsonable_event(
            marker_code=SSVEP_MARKER_STIM_FIRST_FRAME,
            trial_order=1,
            trial_id=1,
            label="8.0Hz",
            expected_freq=8.0,
        ),
        _jsonable_event(
            marker_code=SSVEP_MARKER_ACTIVE_WINDOW_START,
            trial_order=1,
            trial_id=1,
            label="8.0Hz",
            expected_freq=8.0,
        ),
        _jsonable_event(
            marker_code=SSVEP_MARKER_ACTIVE_WINDOW_END,
            trial_order=1,
            trial_id=1,
            label="8.0Hz",
            expected_freq=8.0,
        ),
    ]

    segments, quality_rows, rejected_rows = derive_ssvep_trial_segments_from_markers(
        board,
        eeg_rows=[0, 1],
        marker_row=marker_row,
        event_log=event_log,
        trials=[trial],
        sampling_rate=100,
        default_active_sec=1.0,
        trial_runtime_rows=[{"trial_order": 1, "active_sec": 1.0}],
    )

    assert not rejected_rows
    assert len(segments) == 1
    _, segment = segments[0]
    assert segment.shape == (2, 100)
    assert segment[0, 0] == 30.0
    assert segment[1, 0] == 1030.0
    assert quality_rows[0]["active_window_start_sample"] == 30
    assert quality_rows[0]["stimulus_first_frame_marker_sample"] == 20


def test_protocol_worker_does_not_write_active_marker_without_ack(tmp_path) -> None:
    written_codes: list[int] = []

    def marker_writer(code: float) -> tuple[bool, str]:
        written_codes.append(int(code))
        return True, ""

    config = SSVEPUnifiedConfig(
        serial_port="COM_TEST",
        board_id=0,
        freqs=(8.0, 10.0, 12.0, 15.0),
        subject_id="s01",
        session_id="ack_timeout",
        dataset_dir=tmp_path,
        prepare_sec=0.0,
        active_sec=1.5,
        rest_sec=0.0,
        target_repeats=1,
        idle_repeats=0,
        switch_trials=0,
        long_idle_sec=0.0,
        seed=1,
        stim_refresh_rate_hz=240.0,
        stimulus_mode="elapsed_time_sine",
        simulation_only=False,
        ack_timeout_sec=0.01,
        max_trial_retries=0,
    )
    worker = SSVEPProtocolOnlyWorker(config, marker_writer=marker_writer)
    worker.run()

    assert SSVEP_MARKER_STIM_FIRST_FRAME not in written_codes
    assert SSVEP_MARKER_ACTIVE_WINDOW_START not in written_codes


def test_unified_window_instantiates() -> None:
    _app = QApplication.instance() or QApplication([])
    window = UnifiedCollectionWindow()
    assert window.mode_tabs.count() == 2
    assert window.btn_start_ssvep.isEnabled()
    window.close()


def test_legacy_unified_collection_wrapper_still_exports_main() -> None:
    import unified_collection_ui

    assert callable(unified_collection_ui.main)

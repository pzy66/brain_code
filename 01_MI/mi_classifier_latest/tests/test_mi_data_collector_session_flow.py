import os
import sys
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = PROJECT_ROOT / "code" / "collection"
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLECTION_ROOT))
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from mi_data_collector import (
    BoardCaptureWorker,
    CLASS_LOOKUP,
    CUE_WARNING_TONE_HZ,
    CUE_WARNING_TONE_MS,
    CUE_VOICE_PROMPT_TIMEOUT_SEC,
    IMAGERY_START_POST_TONE_GUARD_MS,
    IMAGERY_START_TONE_HZ,
    IMAGERY_START_TONE_MS,
    MIDataCollectorWindow,
    MiSpeechPromptPlayer,
    MIN_MI_CUE_SEC_FOR_NATURAL_AUDIO,
    MIN_MI_CUE_SEC_FOR_STANDARD_VISUAL,
    MIN_MI_ITI_SEC_BETWEEN_TRIALS,
    MISSING_SAVE_RESULT_GRACE_MS,
    PHASE_START_POST_TONE_GUARD_MS,
    STANDARD_MI_CUE_SEC,
    SessionSaveWorker,
    VOICE_PROMPT_FINISH_GUARD_MS,
)
from src.mi_collection import PAUSE_REJECTED_TRIAL_NOTE, SessionSettings, TrialRecord, make_event


class MIDataCollectorSessionFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _build_window_with_fake_session(self) -> MIDataCollectorWindow:
        window = MIDataCollectorWindow()
        window.show_error = lambda message: None
        window.log = lambda message: None
        window.worker_thread = object()
        window.worker = None
        window.session_running = True
        window.waiting_for_save = False
        window.session_paused = False
        window.capture_on_stop = True
        window.audio_prompts_enabled = False
        window.event_log = []
        window.trial_records = []

        def fake_record_event(
            event_name: str,
            *,
            trial_id: int | None = None,
            class_name: str | None = None,
            run_index: int | None = None,
            run_trial_index: int | None = None,
            block_index: int | None = None,
            prompt_index: int | None = None,
            command_duration_sec: float | None = None,
            execution_success: int | bool | None = None,
            wait_for_marker_flush: bool = True,
        ) -> None:
            del wait_for_marker_flush
            window.event_log.append(
                make_event(
                    event_name,
                    trial_id=trial_id,
                    class_name=class_name,
                    run_index=run_index,
                    run_trial_index=run_trial_index,
                    block_index=block_index,
                    prompt_index=prompt_index,
                    command_duration_sec=command_duration_sec,
                    execution_success=execution_success,
                )
            )

        window.record_event = fake_record_event
        return window

    def _build_settings(self) -> SessionSettings:
        return SessionSettings(
            subject_id="sub-test",
            session_id="20260403_120000",
            output_root=str(PROJECT_ROOT / "runtime" / "test_output"),
            board_id=0,
            serial_port="COM4",
            channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            channel_positions=list(range(8)),
            trials_per_class=1,
            baseline_sec=1.0,
            cue_sec=2.0,
            imagery_sec=1.0,
            iti_sec=1.0,
            random_seed=1234,
            run_count=2,
            continuous_block_count=0,
            continuous_block_sec=0.0,
        )

    def _pump(self, seconds: float = 0.05) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.app.processEvents()
            time.sleep(0.01)

    @staticmethod
    def _visible_control_button_names(window: MIDataCollectorWindow) -> set[str]:
        buttons = [
            window.connect_button,
            window.start_button,
            window.start_mi_only_button,
            window.show_plan_button,
            window.show_mi_only_plan_button,
            window.pause_button,
            window.bad_trial_button,
            window.stop_button,
            window.disconnect_button,
        ]
        return {button.objectName() for button in buttons if button is not None and not button.isHidden()}

    def test_finish_session_does_not_duplicate_terminal_end_markers(self) -> None:
        phase_cases = [
            ("idle_block", "idle_block_end", {"block_index": 1}, {"idle_block_index": 1}),
            ("continuous", "continuous_block_end", {"block_index": 2}, {"continuous_block_index": 2}),
        ]

        for phase_name, end_event, end_kwargs, state_kwargs in phase_cases:
            with self.subTest(phase=phase_name):
                window = self._build_window_with_fake_session()
                try:
                    window.current_phase = phase_name
                    window.current_continuous_prompt = None
                    for attr_name, attr_value in state_kwargs.items():
                        setattr(window, attr_name, attr_value)
                    window.event_log.append(make_event(end_event, **end_kwargs))

                    window.finish_session_and_request_save(manual_stop=False)

                    end_count = sum(1 for event in window.event_log if str(event.get("event_name", "")) == end_event)
                    session_end_count = sum(
                        1 for event in window.event_log if str(event.get("event_name", "")) == "session_end"
                    )
                    self.assertEqual(end_count, 1, f"{phase_name} appended a duplicate terminal end marker")
                    self.assertEqual(session_end_count, 1, f"{phase_name} failed to append session_end exactly once")
                finally:
                    window.waiting_for_save = False
                    window.worker_thread = None
                    window.close()

    def test_formal_protocol_starts_with_calibration_even_if_quality_check_duration_is_configured(self) -> None:
        window = self._build_window_with_fake_session()
        transition_markers: list[str] = []
        try:
            settings = self._build_settings()
            settings.quality_check_sec = 45.0
            window.current_settings = settings
            window.calibration_plan = window._build_calibration_plan(settings)
            window.use_separate_participant_screen = False
            window._start_next_calibration_step = lambda: transition_markers.append("calibration_step_started")

            window._start_formal_protocol()

            self.assertEqual(
                [str(event["event_name"]) for event in window.event_log],
                ["calibration_start"],
            )
            self.assertEqual(transition_markers, ["calibration_step_started"])
        finally:
            window.session_running = False
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_formal_protocol_paints_first_calibration_phase_before_calibration_marker(self) -> None:
        window = self._build_window_with_fake_session()
        events: list[tuple[str, str, str | None]] = []
        try:
            settings = self._build_settings()
            settings.calibration_open_sec = 5.0
            settings.calibration_closed_sec = 0.0
            settings.calibration_eye_sec = 0.0
            settings.calibration_blink_sec = 0.0
            settings.calibration_swallow_sec = 0.0
            settings.calibration_jaw_sec = 0.0
            settings.calibration_head_sec = 0.0
            settings.artifact_types = []
            window.current_settings = settings
            window.calibration_plan = window._build_calibration_plan(settings)
            window.use_separate_participant_screen = True
            window._process_ui_events = lambda: events.append(("paint", window.current_phase, None))
            window._play_phase_start_prompt_blocking = (
                lambda phase, class_name=None: events.append(("prompt", window.current_phase, str(phase)))
            )
            window.record_event = lambda event_name, **kwargs: events.append(
                ("event", window.current_phase, str(event_name))
            )
            window._start_next_calibration_step = lambda: events.append(("next_step", window.current_phase, None))

            window._start_formal_protocol()

            self.assertIn(("paint", "idle", None), events)
            self.assertIn(("paint", "calibration_open", None), events)
            self.assertNotIn(("prompt", "calibration_open", "calibration"), events)
            self.assertIn(("event", "calibration_open", "calibration_start"), events)
            self.assertLess(
                events.index(("paint", "calibration_open", None)),
                events.index(("event", "calibration_open", "calibration_start")),
            )
        finally:
            window.restore_operator_window()
            window.session_running = False
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_formal_protocol_without_calibration_plan_starts_post_calibration_sequence_directly(self) -> None:
        window = self._build_window_with_fake_session()
        markers: list[str] = []
        try:
            settings = self._build_settings()
            settings.quality_check_sec = 45.0
            window.current_settings = settings
            window.calibration_plan = []
            window.use_separate_participant_screen = False
            window._start_post_calibration_sequence = lambda: markers.append("post_calibration_started")

            window._start_formal_protocol()

            self.assertEqual([str(event["event_name"]) for event in window.event_log], [])
            self.assertEqual(markers, ["post_calibration_started"])
        finally:
            window.session_running = False
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_config_section_defaults_to_device(self) -> None:
        window = MIDataCollectorWindow()
        try:
            self.assertIsNotNone(window.config_section_combo)
            self.assertIsNotNone(window.config_stack)
            self.assertEqual(window.config_section_combo.currentText(), "设备")
            self.assertEqual(window.config_stack.currentIndex(), window.config_section_combo.currentIndex())
        finally:
            window.close()

    def test_disconnected_control_panel_only_shows_connect(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window.device_info = None
            window.worker_thread = None
            window.session_running = False
            window.waiting_for_save = False
            window.update_button_states()

            self.assertEqual(self._visible_control_button_names(window), {"btnConnect"})
            self.assertEqual(window.control_layout_columns, 1)
        finally:
            window.close()

    def test_connected_idle_control_panel_shows_start_and_disconnect(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window.device_info = {
                "sampling_rate": 250.0,
                "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
                "selected_rows": list(range(8)),
            }
            window.worker_thread = None
            window.session_running = False
            window.waiting_for_save = False
            window.update_button_states()

            self.assertEqual(
                self._visible_control_button_names(window),
                {"btnStart", "btnStartMiOnly", "btnShowPlan", "btnShowMiOnlyPlan", "btnDisconnect"},
            )
            self.assertTrue(window.show_plan_button.isEnabled())
            self.assertTrue(window.show_mi_only_plan_button.isEnabled())
            self.assertEqual(window.control_layout_columns, 1)
        finally:
            window.worker_thread = None
            window.close()

    def test_operator_notice_and_control_tooltips_are_contextual(self) -> None:
        window = MIDataCollectorWindow()
        try:
            self.assertIsNotNone(window.operator_notice_label)
            self.assertIn("质量检查", window.operator_notice_label.text())
            self.assertIn("BrainFlow", window.connect_button.toolTip())

            window.session_running = True
            window.current_phase = "continuous"
            window.current_continuous_prompt = {"class_label": "no_control", "prompt_index": 1}
            window.update_button_states()

            self.assertIn("连续命令失败", window.bad_trial_button.toolTip())
        finally:
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_bad_trial_button_matches_all_mi_trial_phases(self) -> None:
        window = self._build_window_with_fake_session()
        try:
            window.sequence = ["left_hand"]
            window.current_trial_index = 0
            window.current_trial = TrialRecord(
                trial_id=1,
                class_name="left_hand",
                display_name="左手",
                run_index=1,
                run_trial_index=1,
            )
            window.trial_records = [window.current_trial]

            for phase in ["baseline", "cue", "imagery", "iti"]:
                with self.subTest(phase=phase):
                    window.current_phase = phase
                    window.current_trial.accepted = True
                    window.update_button_states()
                    self.assertTrue(window.bad_trial_button.isEnabled())

            window.current_phase = "baseline"
            window.mark_bad_trial()

            self.assertFalse(window.current_trial.accepted)
            self.assertEqual([str(event["event_name"]) for event in window.event_log], ["bad_trial_marked"])
            self.assertFalse(window.bad_trial_button.isEnabled())
            self.assertIn("不会进入训练集", window.operator_notice_label.text())
        finally:
            window.session_running = False
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_failed_continuous_prompt_disables_mark_button(self) -> None:
        window = self._build_window_with_fake_session()
        try:
            window.current_phase = "continuous"
            window.current_continuous_prompt = {
                "class_label": "left_hand",
                "prompt_index": 1,
                "execution_success": 1,
            }
            window.update_button_states()
            self.assertTrue(window.bad_trial_button.isEnabled())

            window.mark_continuous_prompt_failed()

            self.assertEqual(int(window.current_continuous_prompt["execution_success"]), 0)
            self.assertFalse(window.bad_trial_button.isEnabled())
            self.assertIn("已标记失败", window.bad_trial_button.toolTip())
            self.assertIn("不会参与连续评估", window.operator_notice_label.text())
        finally:
            window.session_running = False
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_new_connection_clears_previous_save_path_tooltips(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window.current_label.setToolTip("old-session")
            window.next_task_label.setToolTip("old-session")

            window.on_connection_ready(
                {
                    "sampling_rate": 250.0,
                    "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
                    "selected_rows": list(range(8)),
                }
            )

            self.assertEqual(window.current_label.toolTip(), "")
            self.assertEqual(window.next_task_label.toolTip(), "")
        finally:
            window.worker_thread = None
            window.close()

    def test_single_mi_run_clamps_continuous_count_control(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window.continuous_count_spin.setValue(2)
            self.assertEqual(int(window.continuous_count_spin.value()), 2)

            window.run_count_spin.setValue(1)

            self.assertEqual(int(window.continuous_count_spin.maximum()), 1)
            self.assertEqual(int(window.continuous_count_spin.value()), 1)

            window.continuous_count_spin.setValue(2)
            self.assertEqual(int(window.continuous_count_spin.value()), 1)
            self.assertIn("设为 0", window.continuous_count_spin.toolTip())
        finally:
            window.close()

    def test_mi_only_ignores_full_protocol_continuous_count_limit(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return False

            def insert_marker_sync(self, marker_code: float) -> tuple[bool, str]:
                del marker_code
                return True, ""

        window = MIDataCollectorWindow()
        errors: list[str] = []
        try:
            window.show_error = errors.append
            window.audio_prompts_enabled = False
            window.device_info = {
                "sampling_rate": 250.0,
                "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
                "selected_rows": list(range(8)),
            }
            window.worker = FakeWorker()
            window.serial_combo.setCurrentText("COM4")
            window.run_count_spin.setValue(1)
            window.continuous_count_spin.setValue(2)
            window.continuous_sec_spin.setValue(1.0)
            window.cont_cmd_min_spin.setValue(0.5)
            window.cont_cmd_max_spin.setValue(0.5)

            with mock.patch(
                "mi_data_collector.validate_serial_port_selection",
                return_value={"ok": True, "reason": "detected", "requested_port": "COM4", "detected_ports": ["COM4"]},
            ):
                window.start_mi_only_session()

            self.assertEqual(errors, [])
            self.assertTrue(window.session_running)
            self.assertIsNotNone(window.current_settings)
            self.assertEqual(window.current_settings.protocol_mode, "mi_only")
            self.assertEqual(int(window.current_settings.continuous_block_count), 0)
        finally:
            window.phase_timer.stop()
            window.session_running = False
            window.waiting_for_save = False
            window.worker = None
            window.close()

    def test_full_protocol_validation_rejects_too_many_continuous_blocks(self) -> None:
        settings = self._build_settings()
        settings.run_count = 1
        settings.continuous_block_count = 2
        settings.continuous_block_sec = 1.0
        settings.continuous_command_min_sec = 0.5
        settings.continuous_command_max_sec = 0.5

        with self.assertRaises(ValueError) as raised:
            MIDataCollectorWindow._validate_protocol_settings(settings)

        message = str(raised.exception)
        self.assertIn("连续模式段数不能大于 MI run 数", message)
        self.assertIn("设为 0 或 1", message)

    def test_start_full_session_allows_single_run_with_or_without_one_continuous_block(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return False

        cases = [(0, []), (1, [1])]
        for continuous_count, expected_schedule in cases:
            with self.subTest(continuous_count=continuous_count):
                window = MIDataCollectorWindow()
                errors: list[str] = []
                begin_calls: list[tuple[SessionSettings, list[list[str]]]] = []
                try:
                    window.show_error = errors.append
                    window.log = lambda message: None
                    window.device_info = {
                        "sampling_rate": 250.0,
                        "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
                        "selected_rows": list(range(8)),
                    }
                    window.worker = FakeWorker()
                    window.serial_combo.setCurrentText("COM4")
                    window.run_count_spin.setValue(1)
                    window.continuous_count_spin.setValue(continuous_count)
                    window.continuous_sec_spin.setValue(1.0)
                    window.cont_cmd_min_spin.setValue(0.5)
                    window.cont_cmd_max_spin.setValue(0.5)
                    window._begin_session_with_settings = lambda settings, sequence_by_run: begin_calls.append(
                        (settings, sequence_by_run)
                    )

                    with mock.patch(
                        "mi_data_collector.validate_serial_port_selection",
                        return_value={
                            "ok": True,
                            "reason": "detected",
                            "requested_port": "COM4",
                            "detected_ports": ["COM4"],
                        },
                    ):
                        window.start_session()

                    self.assertEqual(errors, [])
                    self.assertEqual(len(begin_calls), 1)
                    settings, sequence_by_run = begin_calls[0]
                    self.assertEqual(int(settings.run_count), 1)
                    self.assertEqual(int(settings.continuous_block_count), continuous_count)
                    self.assertEqual(window._build_continuous_schedule(settings), expected_schedule)
                    self.assertEqual(len(sequence_by_run), 1)
                finally:
                    window.phase_timer.stop()
                    window.session_running = False
                    window.waiting_for_save = False
                    window.worker = None
                    window.close()

    def test_pause_marks_active_mi_trial_rejected(self) -> None:
        window = self._build_window_with_fake_session()
        try:
            window.current_phase = "imagery"
            window.phase_started_perf = time.perf_counter()
            window.phase_deadline = time.perf_counter() + 2.0
            window.current_trial = TrialRecord(
                trial_id=1,
                class_name="left_hand",
                display_name="left_hand",
                run_index=1,
                run_trial_index=1,
            )
            window.trial_records = [window.current_trial]

            window.toggle_pause()

            self.assertTrue(window.session_paused)
            self.assertFalse(window.current_trial.accepted)
            self.assertIn(PAUSE_REJECTED_TRIAL_NOTE, window.current_trial.note)
            self.assertIn("不会进入训练集", window.summary_label.text())
            self.assertIn("不会进入训练集", window.operator_notice_label.text())
            self.assertEqual([str(item["event_name"]) for item in window.event_log], ["pause"])
        finally:
            window.phase_timer.stop()
            window.session_running = False
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_pause_marks_active_continuous_prompt_failed(self) -> None:
        window = self._build_window_with_fake_session()
        stop_calls: list[str] = []
        try:
            window._stop_voice_prompt = lambda: stop_calls.append("stop") or 0
            window.current_phase = "continuous"
            window.phase_started_perf = time.perf_counter()
            window.phase_deadline = time.perf_counter() + 2.0
            window.current_continuous_prompt = {
                "class_label": "no_control",
                "prompt_index": 2,
                "execution_success": 1,
            }

            window.toggle_pause()

            self.assertTrue(window.session_paused)
            self.assertEqual(stop_calls, ["stop"])
            self.assertEqual(int(window.current_continuous_prompt["execution_success"]), 0)
            self.assertIn("不会参与连续评估", window.summary_label.text())
            self.assertIn("不会参与连续评估", window.operator_notice_label.text())
            self.assertEqual([str(item["event_name"]) for item in window.event_log], ["pause"])
        finally:
            window.phase_timer.stop()
            window.session_running = False
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_save_thread_keeps_controls_locked_until_cleanup(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window.device_info = {
                "sampling_rate": 250.0,
                "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
                "selected_rows": list(range(8)),
            }
            window.worker_thread = None
            window.session_running = False
            window.waiting_for_save = False
            window.save_thread = object()

            window.update_button_states()

            self.assertFalse(window.connect_button.isEnabled())
            self.assertFalse(window.start_button.isEnabled())
            self.assertFalse(window.start_mi_only_button.isEnabled())
            self.assertFalse(window.show_plan_button.isEnabled())
            self.assertFalse(window.disconnect_button.isEnabled())

            errors: list[str] = []
            window.show_error = errors.append
            window.disconnect_device()
            self.assertTrue(errors)
            self.assertIn("保存", errors[0])
        finally:
            window.save_thread = None
            window.worker_thread = None
            window.close()

    def test_save_thread_cleanup_reenables_reconnect_after_session_save(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window.device_info = None
            window.worker_thread = None
            window.session_running = False
            window.waiting_for_save = False
            window.save_thread = object()

            window.update_button_states()
            self.assertFalse(window.connect_button.isEnabled())

            window.on_save_thread_finished()

            self.assertIsNone(window.save_thread)
            self.assertIsNone(window.save_worker)
            self.assertTrue(window.connect_button.isEnabled())
            self.assertFalse(window.start_button.isEnabled())
            self.assertTrue(window.subject_edit.isEnabled())
        finally:
            window.save_thread = None
            window.worker_thread = None
            window.close()

    def test_quality_warning_flags_flat_saved_channels(self) -> None:
        window = MIDataCollectorWindow()
        try:
            warnings = window._quality_warnings_from_report(
                {
                    "channels": [
                        {"channel_name": "C3", "std_uV": 0.0, "peak_to_peak_uV": 0.0},
                        {"channel_name": "Cz", "std_uV": 25.0, "peak_to_peak_uV": 120.0},
                        {"channel_name": "C4", "std_uV": 25.0, "peak_to_peak_uV": 120.0, "non_finite_sample_count": 2},
                    ]
                }
            )

            self.assertEqual(len(warnings), 1)
            self.assertIn("C3", warnings[0])
            self.assertIn("C4", warnings[0])
            self.assertNotIn("Cz", warnings[0])
        finally:
            window.close()

    def test_data_flow_warning_flags_packet_and_timestamp_problems(self) -> None:
        window = MIDataCollectorWindow()
        try:
            warnings = window._data_flow_warnings_from_report(
                {
                    "marker_event_count_match": False,
                    "marker_count": 3,
                    "event_count": 4,
                    "timestamp_available": True,
                    "timestamp_monotonic": False,
                    "timestamp_non_finite_count": 2,
                    "package_jump_count": 1,
                    "package_non_finite_count": 1,
                }
            )

            joined = "\n".join(warnings)
            self.assertIn("Marker", joined)
            self.assertIn("timestamp", joined)
            self.assertIn("包号跳变", joined)
            self.assertIn("package number", joined)
        finally:
            window.close()

    def test_full_session_plan_lists_calibration_continuous_and_total_duration(self) -> None:
        window = MIDataCollectorWindow()
        try:
            settings = self._build_settings()
            settings.practice_sec = 3.0
            settings.run_rest_sec = 9.0
            settings.long_run_rest_every = 0
            settings.long_run_rest_sec = 0.0
            settings.calibration_open_sec = 4.0
            settings.calibration_closed_sec = 5.0
            settings.calibration_eye_sec = 6.0
            settings.calibration_blink_sec = 0.0
            settings.calibration_swallow_sec = 0.0
            settings.calibration_jaw_sec = 0.0
            settings.calibration_head_sec = 0.0
            settings.artifact_types = ["eye_movement"]
            settings.continuous_block_count = 1
            settings.continuous_block_sec = 10.0
            settings.idle_block_count = 1
            settings.idle_block_sec = 8.0

            plan = window._build_session_plan(settings)
            plan_text = window._format_session_plan(plan)

            self.assertEqual(plan["protocol_mode"], "full")
            self.assertEqual(plan["trials_per_run"], 4)
            self.assertEqual(plan["total_trials"], 8)
            self.assertAlmostEqual(float(plan["trial_phase_duration_sec"]), 5.0)
            self.assertAlmostEqual(float(plan["trial_start_prompt_overhead_sec"]), 0.26)
            self.assertAlmostEqual(float(plan["trial_duration_sec"]), 5.26)
            self.assertAlmostEqual(float(plan["mi_duration_sec"]), 42.08)
            self.assertAlmostEqual(float(plan["total_duration_sec"]), 87.08)
            self.assertIn("总时间：87.1 秒", plan_text)
            self.assertIn("每试次开始音过渡：0.3 秒", plan_text)
            self.assertIn("睁眼静息", plan_text)
            self.assertIn("想象训练：3 秒", plan_text)
            self.assertIn("MI run 1（4 个试次）：21.0 秒", plan_text)
            self.assertIn("连续模式 1：10 秒", plan_text)
            self.assertIn("无控制 1：8 秒", plan_text)
        finally:
            window.close()

    def test_session_plan_keeps_run_rest_after_interleaved_continuous_block(self) -> None:
        window = MIDataCollectorWindow()
        try:
            settings = self._build_settings()
            settings.run_count = 2
            settings.trials_per_class = 1
            settings.practice_sec = 0.0
            settings.calibration_open_sec = 0.0
            settings.calibration_closed_sec = 0.0
            settings.artifact_types = []
            settings.run_rest_sec = 9.0
            settings.long_run_rest_every = 0
            settings.long_run_rest_sec = 0.0
            settings.continuous_block_count = 2
            settings.continuous_block_sec = 10.0
            settings.idle_block_count = 0
            settings.idle_block_sec = 0.0

            plan = window._build_session_plan(settings)
            stage_titles = [str(item["title"]) for item in plan["stages"]]

            self.assertEqual(
                stage_titles,
                [
                    "MI run 1（4 个试次）",
                    "连续模式 1",
                    "轮次间休息（run 1 后）",
                    "MI run 2（4 个试次）",
                    "连续模式 2",
                ],
            )
            self.assertAlmostEqual(float(plan["total_duration_sec"]), 71.08)
        finally:
            window.close()

    def test_continuous_block_returns_to_run_rest_before_next_mi_run(self) -> None:
        window = self._build_window_with_fake_session()
        next_run_calls: list[str] = []
        try:
            settings = self._build_settings()
            settings.run_count = 2
            settings.run_rest_sec = 3.0
            settings.long_run_rest_every = 0
            settings.long_run_rest_sec = 0.0
            window.current_settings = settings
            window.current_phase = "continuous"
            window.current_run_index = 1
            window.continuous_block_index = 1
            window.current_continuous_prompt = None
            window._start_next_mi_run = lambda: next_run_calls.append("next_run")

            window._finish_continuous_block()

            self.assertEqual([str(event["event_name"]) for event in window.event_log], ["continuous_block_end", "run_rest_start"])
            self.assertEqual(window.current_phase, "run_rest")
            self.assertEqual(next_run_calls, [])
        finally:
            window.phase_timer.stop()
            window.session_running = False
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_mi_only_session_plan_removes_calibration_rest_and_post_blocks(self) -> None:
        window = MIDataCollectorWindow()
        try:
            settings = self._build_settings()
            settings.protocol_mode = "mi_only"
            settings.practice_sec = 30.0
            settings.run_rest_sec = 9.0
            settings.calibration_open_sec = 4.0
            settings.calibration_closed_sec = 5.0
            settings.continuous_block_count = 1
            settings.continuous_block_sec = 10.0
            settings.idle_block_count = 1
            settings.idle_block_sec = 8.0

            plan = window._build_session_plan(window._apply_protocol_mode_overrides(settings))
            plan_text = window._format_session_plan(plan)

            self.assertEqual(plan["protocol_mode"], "mi_only")
            self.assertEqual(plan["trials_per_run"], 4)
            self.assertEqual(plan["total_trials"], 8)
            self.assertAlmostEqual(float(plan["mi_duration_sec"]), 42.08)
            self.assertAlmostEqual(float(plan["total_duration_sec"]), 42.08)
            self.assertIn("总时间：42.1 秒", plan_text)
            self.assertIn("每试次开始音过渡：0.3 秒", plan_text)
            self.assertNotIn("睁眼静息", plan_text)
            self.assertNotIn("想象训练", plan_text)
            self.assertNotIn("轮次间休息", plan_text)
            self.assertNotIn("连续模式", plan_text)
            self.assertNotIn("无控制", plan_text)
            self.assertIn("MI run 1（4 个试次）：21.0 秒", plan_text)
        finally:
            window.close()

    def test_preview_mode_switch_is_queued_and_locks_controls_until_completion(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return True

        window = MIDataCollectorWindow()
        window.show_error = lambda message: None
        window.log = lambda message: None
        window.device_info = {
            "sampling_rate": 250.0,
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_rows": list(range(8)),
        }
        window.worker = FakeWorker()
        window.preview_mode = "EEG"
        emitted: list[tuple[str, int, bool]] = []
        window.preview_mode_switch_requested.connect(lambda mode, channel, reset: emitted.append((mode, channel, reset)))
        try:
            window.update_button_states()
            self.assertTrue(window.start_button.isEnabled())
            self.assertTrue(window.disconnect_button.isEnabled())

            accepted = window._switch_preview_mode("IMP", channel=3)

            self.assertTrue(accepted)
            self.assertTrue(window.preview_mode_switch_pending)
            self.assertEqual(emitted, [("IMP", 3, False)])
            self.assertFalse(window.start_button.isEnabled())
            self.assertFalse(window.disconnect_button.isEnabled())
            self.assertFalse(window.preview_to_imp_button.isEnabled())

            window.on_preview_mode_switch_finished(
                {
                    "ok": True,
                    "message": "",
                    "target_mode": "IMP",
                    "target_channel": 3,
                    "reset_default": False,
                }
            )

            self.assertFalse(window.preview_mode_switch_pending)
            self.assertEqual(window.preview_mode, "IMP")
            self.assertEqual(window.preview_impedance_channel, 3)
            self.assertTrue(window.start_button.isEnabled())
            self.assertTrue(window.disconnect_button.isEnabled())
        finally:
            window.close()

    def test_start_session_waits_for_eeg_switch_completion(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return True

        window = MIDataCollectorWindow()
        window.show_error = lambda message: None
        window.log = lambda message: None
        window.device_info = {
            "sampling_rate": 250.0,
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_rows": list(range(8)),
        }
        window.worker = FakeWorker()
        window.preview_mode = "IMP"
        window.preview_impedance_channel = 4
        window.collect_settings = self._build_settings
        emitted: list[tuple[str, int, bool]] = []
        begin_calls: list[tuple[SessionSettings, list[list[str]]]] = []
        window.preview_mode_switch_requested.connect(lambda mode, channel, reset: emitted.append((mode, channel, reset)))
        window._begin_session_with_settings = lambda settings, sequence_by_run: begin_calls.append((settings, sequence_by_run))
        try:
            window.start_session()

            self.assertFalse(begin_calls)
            self.assertFalse(window.session_running)
            self.assertIsNotNone(window.pending_session_start_context)
            self.assertTrue(window.preview_mode_switch_pending)
            self.assertEqual(emitted, [("EEG", 4, True)])

            window.on_preview_mode_switch_finished(
                {
                    "ok": True,
                    "message": "",
                    "target_mode": "EEG",
                    "target_channel": 4,
                    "reset_default": True,
                }
            )

            self.assertFalse(window.preview_mode_switch_pending)
            self.assertIsNone(window.pending_session_start_context)
            self.assertEqual(len(begin_calls), 1)
            self.assertEqual(begin_calls[0][0].session_id, "20260403_120000")
            self.assertEqual(len(begin_calls[0][1]), 2)
        finally:
            window.close()

    def test_start_mi_only_session_applies_protocol_overrides_before_begin(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return False

        window = MIDataCollectorWindow()
        window.show_error = lambda message: None
        window.log = lambda message: None
        window.device_info = {
            "sampling_rate": 250.0,
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_rows": list(range(8)),
        }
        window.worker = FakeWorker()
        begin_calls: list[tuple[SessionSettings, list[list[str]]]] = []
        window.collect_settings = self._build_settings
        window._begin_session_with_settings = lambda settings, sequence_by_run: begin_calls.append((settings, sequence_by_run))
        try:
            window.start_mi_only_session()

            self.assertEqual(len(begin_calls), 1)
            settings, sequence_by_run = begin_calls[0]
            self.assertEqual(settings.protocol_mode, "mi_only")
            self.assertEqual(settings.practice_sec, 0.0)
            self.assertEqual(settings.calibration_open_sec, 0.0)
            self.assertEqual(settings.calibration_closed_sec, 0.0)
            self.assertEqual(settings.idle_block_count, 0)
            self.assertEqual(settings.idle_prepare_block_count, 0)
            self.assertEqual(settings.continuous_block_count, 0)
            self.assertEqual(settings.run_rest_sec, 0.0)
            self.assertEqual(settings.long_run_rest_every, 0)
            self.assertEqual(len(sequence_by_run), 2)
        finally:
            window.close()

    def test_start_session_validation_failure_does_not_queue_hardware_switch(self) -> None:
        class FakeWorker:
            def supports_impedance_mode(self) -> bool:
                return True

        window = MIDataCollectorWindow()
        errors: list[str] = []
        window.show_error = errors.append
        window.log = lambda message: None
        window.device_info = {
            "sampling_rate": 250.0,
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_rows": list(range(8)),
        }
        window.worker = FakeWorker()
        window.preview_mode = "IMP"
        window.collect_settings = lambda: (_ for _ in ()).throw(ValueError("bad settings"))
        emitted: list[tuple[str, int, bool]] = []
        window.preview_mode_switch_requested.connect(lambda mode, channel, reset: emitted.append((mode, channel, reset)))
        try:
            window.start_session()

            self.assertEqual(errors, ["bad settings"])
            self.assertFalse(window.preview_mode_switch_pending)
            self.assertIsNone(window.pending_session_start_context)
            self.assertEqual(emitted, [])
        finally:
            window.close()

    def test_baseline_to_cue_transition_starts_cue_timer_before_voice_prompt(self) -> None:
        window = self._build_window_with_fake_session()
        prompt_calls: list[tuple[str, str, str | None, int, str]] = []
        window.current_settings = self._build_settings()
        window.current_phase = "baseline"
        window.phase_deadline = time.perf_counter() - 0.1
        window.current_trial = mock.Mock()
        window.current_trial.trial_id = 1
        window.current_trial.class_name = "left_hand"
        window.current_trial.run_index = 1
        window.current_trial.run_trial_index = 1
        window.current_trial.display_name = "左手"
        window._play_phase_end_prompt_blocking = (
            lambda phase, class_name=None: prompt_calls.append(
                ("end", phase, class_name, len(window.event_log), window.current_phase)
            )
        )
        window._play_cue_prompt_during_phase = (
            lambda class_name: prompt_calls.append(
                ("cue_async", "cue", class_name, len(window.event_log), window.current_phase)
            )
        )
        try:
            window.on_phase_tick()

            self.assertEqual(window.current_phase, "cue")
            self.assertEqual(
                [event["event_name"] for event in window.event_log[-3:]],
                ["baseline_end", "cue_start", "cue_left_hand"],
            )
            self.assertEqual(
                prompt_calls,
                [
                    ("end", "baseline", "left_hand", 1, "baseline"),
                    ("cue_async", "cue", "left_hand", 3, "cue"),
                ],
            )
        finally:
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_standard_cue_uses_warning_tone_without_voice_prompt(self) -> None:
        window = self._build_window_with_fake_session()
        prompt_calls: list[tuple[str, int, float]] = []
        tone_calls: list[tuple[int, int]] = []
        window.audio_prompts_enabled = True
        window.speech_prompt_player.say = (
            lambda text, *, request_id, timeout_sec: prompt_calls.append(
                (str(text), int(request_id), float(timeout_sec))
            )
        )
        window._play_prompt_tone_async = lambda *, frequency, duration_ms: tone_calls.append(
            (int(frequency), int(duration_ms))
        )
        window.current_settings = self._build_settings()

        try:
            window._play_cue_prompt_during_phase("left_hand")

            self.assertEqual(tone_calls, [(CUE_WARNING_TONE_HZ, CUE_WARNING_TONE_MS)])
            self.assertEqual(prompt_calls, [])
        finally:
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_phase_start_voice_text_omits_extra_prepare_suffix(self) -> None:
        window = MIDataCollectorWindow()
        prepare_suffix = "\uff0c\u51c6\u5907"
        try:
            for phase in ("calibration", "practice", "continuous", "idle_block"):
                with self.subTest(phase=phase):
                    voice_text = window._phase_start_voice_text(phase)
                    self.assertTrue(voice_text)
                    self.assertNotIn(prepare_suffix, voice_text)

            self.assertEqual(window._phase_start_voice_text("cue", "left_hand"), "\u5de6\u624b")
            self.assertEqual(window._phase_start_voice_text("run_rest"), "\u4f11\u606f")
        finally:
            window.close()

    def test_training_cue_voice_prompt_is_short_and_timeout_capped(self) -> None:
        window = self._build_window_with_fake_session()
        prompt_calls: list[tuple[str, int, float]] = []
        tone_calls: list[tuple[int, int]] = []
        window.audio_prompts_enabled = True
        window.cue_voice_check.setChecked(True)
        window.speech_prompt_player.say = (
            lambda text, *, request_id, timeout_sec: prompt_calls.append(
                (str(text), int(request_id), float(timeout_sec))
            )
        )
        window._play_prompt_tone_async = lambda *, frequency, duration_ms: tone_calls.append(
            (int(frequency), int(duration_ms))
        )
        window.current_settings = self._build_settings()

        try:
            window._play_cue_prompt_during_phase("left_hand")

            self.assertEqual(tone_calls, [(CUE_WARNING_TONE_HZ, CUE_WARNING_TONE_MS)])
            self.assertEqual(prompt_calls, [("左手", 1, CUE_VOICE_PROMPT_TIMEOUT_SEC)])
        finally:
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_stale_windows_fallback_voice_process_is_cancelled_if_stopped_during_launch(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.terminated = False
                self.wait_called = False

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout: float) -> None:
                del timeout
                self.wait_called = True

        player = MiSpeechPromptPlayer()
        player._engine = None
        process = FakeProcess()

        def fake_popen(*args, **kwargs) -> FakeProcess:
            del args, kwargs
            player.stop()
            return process

        try:
            player._pending_request_id = 1
            player._awaiting_completion = True
            with player._fallback_lock:
                generation = int(player._fallback_generation)
            with mock.patch("mi_data_collector.sys.platform", "win32"):
                with mock.patch("mi_data_collector.subprocess.Popen", side_effect=fake_popen):
                    player._run_windows_fallback("左手", 1, generation, 1.0)

            self.assertTrue(process.terminated)
            self.assertFalse(process.wait_called)
            self.assertIsNone(player._fallback_process)
        finally:
            player.stop()
            player.deleteLater()

    def test_voice_prompt_without_tts_completes_without_blocking(self) -> None:
        player = MiSpeechPromptPlayer()
        player._engine = None
        finished: list[int] = []
        player.playback_finished.connect(lambda request_id: finished.append(int(request_id)))
        try:
            with mock.patch("mi_data_collector.sys.platform", "linux"):
                player.say("左手", request_id=7, timeout_sec=1.0)
                self._pump(0.05)
            self.assertEqual(finished, [7])
        finally:
            player.stop()
            player.deleteLater()

    def test_qt_voice_prompt_timeout_stops_engine_and_finishes(self) -> None:
        class FakeStateChanged:
            def connect(self, callback) -> None:
                self.callback = callback

        class FakeEngine:
            def __init__(self) -> None:
                self.stateChanged = FakeStateChanged()
                self.spoken: list[str] = []
                self.stop_count = 0

            def say(self, text: str) -> None:
                self.spoken.append(str(text))

            def stop(self) -> None:
                self.stop_count += 1

        player = MiSpeechPromptPlayer()
        fake_engine = FakeEngine()
        player._engine = fake_engine
        finished: list[int] = []
        player.playback_finished.connect(lambda request_id: finished.append(int(request_id)))
        try:
            player.say("左手", request_id=11, timeout_sec=0.01)
            self._pump(0.18)
            self.assertEqual(fake_engine.spoken, ["左手"])
            self.assertEqual(finished, [11])
            self.assertGreaterEqual(fake_engine.stop_count, 2)
        finally:
            player.stop()
            player.deleteLater()

    def test_mi_timing_controls_keep_natural_audio_and_rest_minimums(self) -> None:
        window = MIDataCollectorWindow()
        try:
            self.assertFalse(window.cue_voice_check.isChecked())
            self.assertAlmostEqual(float(window.cue_spin.minimum()), MIN_MI_CUE_SEC_FOR_STANDARD_VISUAL)
            self.assertAlmostEqual(float(window.cue_spin.value()), STANDARD_MI_CUE_SEC)
            self.assertAlmostEqual(float(window.iti_spin.minimum()), MIN_MI_ITI_SEC_BETWEEN_TRIALS)

            window.cue_voice_check.setChecked(True)
            self.assertAlmostEqual(float(window.cue_spin.minimum()), MIN_MI_CUE_SEC_FOR_NATURAL_AUDIO)
            self.assertGreaterEqual(float(window.cue_spin.value()), MIN_MI_CUE_SEC_FOR_NATURAL_AUDIO)
        finally:
            window.close()

    def test_pre_prompt_display_does_not_start_countdown(self) -> None:
        window = MIDataCollectorWindow()
        try:
            window._show_phase_visuals_before_prompt(
                phase="calibration_open",
                duration_sec=2.0,
                class_name=None,
            )

            self.assertTrue(window.phase_prompt_pending)
            self.assertFalse(window.phase_timer.isActive())
            self.assertEqual(window._participant_countdown_text(), "听提示")
            self.assertIn("提示中", window.countdown_label.text())

            window.enter_phase(
                phase="calibration_open",
                duration_sec=2.0,
                class_name=None,
                title="",
                subtitle="",
            )

            self.assertFalse(window.phase_prompt_pending)
            self.assertTrue(window.phase_timer.isActive())
            self.assertLessEqual(window.remaining_phase_sec, 2.0)
            self.assertGreater(window.remaining_phase_sec, 1.7)
        finally:
            window.phase_timer.stop()
            window.close()

    def test_calibration_step_countdown_starts_after_prompt(self) -> None:
        window = self._build_window_with_fake_session()
        timeline: list[tuple[str, str, bool, bool, int]] = []
        window.audio_prompts_enabled = True
        window.current_settings = self._build_settings()
        window.calibration_plan = [
            {
                "phase": "calibration_open",
                "start_event": "eyes_open_rest_start",
                "end_event": "eyes_open_rest_end",
                "duration": 1.0,
            }
        ]
        window.calibration_step_index = -1

        original_show = window._show_phase_visuals_before_prompt
        original_enter = window.enter_phase
        window._show_phase_visuals_before_prompt = (
            lambda **kwargs: original_show(**kwargs)
            or timeline.append(
                (
                    "show",
                    window.current_phase,
                    bool(window.phase_prompt_pending),
                    bool(window.phase_timer.isActive()),
                    len(window.event_log),
                )
            )
        )
        window._play_phase_start_prompt_blocking = lambda phase, class_name=None: timeline.append(
            (
                "prompt",
                str(phase),
                bool(window.phase_prompt_pending),
                bool(window.phase_timer.isActive()),
                len(window.event_log),
            )
        )
        window.enter_phase = (
            lambda **kwargs: timeline.append(
                (
                    "before_enter",
                    str(kwargs["phase"]),
                    bool(window.phase_prompt_pending),
                    bool(window.phase_timer.isActive()),
                    len(window.event_log),
                )
            )
            or original_enter(**kwargs)
            or timeline.append(
                (
                    "after_enter",
                    window.current_phase,
                    bool(window.phase_prompt_pending),
                    bool(window.phase_timer.isActive()),
                    len(window.event_log),
                )
            )
        )

        try:
            window._start_next_calibration_step()

            self.assertEqual(
                timeline[:4],
                [
                    ("show", "calibration_open", True, False, 0),
                    ("prompt", "calibration_open", True, False, 0),
                    ("before_enter", "calibration_open", True, False, 1),
                    ("after_enter", "calibration_open", False, True, 1),
                ],
            )
            self.assertEqual([event["event_name"] for event in window.event_log], ["eyes_open_rest_start"])
        finally:
            window.phase_timer.stop()
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_voice_prompt_blocking_waits_for_finish_guard(self) -> None:
        window = MIDataCollectorWindow()
        sleeps: list[float] = []
        window.audio_prompts_enabled = True
        try:
            window.speech_prompt_player.say = (
                lambda text, *, request_id, timeout_sec: QTimer.singleShot(
                    0,
                    lambda request_id=request_id: window.speech_prompt_player.playback_finished.emit(request_id),
                )
            )
            with mock.patch("mi_data_collector.time.sleep", side_effect=lambda seconds: sleeps.append(float(seconds))):
                window._play_voice_prompt_blocking("睁眼静息", timeout_sec=1.0)

            self.assertEqual(sleeps, [VOICE_PROMPT_FINISH_GUARD_MS / 1000.0])
        finally:
            window.close()

    def test_phase_start_prompt_waits_after_start_tone(self) -> None:
        window = MIDataCollectorWindow()
        tone_calls: list[tuple[int, int]] = []
        sleeps: list[float] = []
        window.audio_prompts_enabled = True
        window._phase_start_voice_text = lambda phase, class_name=None: ""
        window._play_prompt_tone_blocking = lambda *, frequency, duration_ms: tone_calls.append(
            (int(frequency), int(duration_ms))
        )
        try:
            with mock.patch("mi_data_collector.time.sleep", side_effect=lambda seconds: sleeps.append(float(seconds))):
                window._play_phase_start_prompt_blocking("calibration_open")

            self.assertEqual(len(tone_calls), 1)
            self.assertEqual(sleeps, [PHASE_START_POST_TONE_GUARD_MS / 1000.0])
        finally:
            window.close()

    def test_imagery_prompt_keeps_explicit_task_text_without_class_artwork(self) -> None:
        window = self._build_window_with_fake_session()
        try:
            window.current_phase = "imagery"
            window.current_trial = mock.Mock()
            window.current_trial.class_name = "left_hand"

            title, subtitle = window._phase_texts("imagery", "left_hand")
            stage_text, participant_title, participant_subtitle, participant_class = window._participant_prompt()

            self.assertEqual(title, "想象：左手")
            self.assertIn("左手", title + subtitle)
            self.assertEqual(stage_text, "想象执行")
            self.assertEqual(participant_title, "想象：左手")
            self.assertIn("左手", participant_title + participant_subtitle)
            self.assertEqual(participant_class, "left_hand")
            self.assertEqual(window._resolve_phase_accent("imagery", "left_hand"), CLASS_LOOKUP["left_hand"]["color"])
        finally:
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_imagery_start_visual_switches_before_class_marker(self) -> None:
        window = self._build_window_with_fake_session()
        timeline: list[tuple[object, ...]] = []
        window.audio_prompts_enabled = True
        window.current_settings = self._build_settings()
        window.current_trial = mock.Mock()
        window.current_trial.trial_id = 1
        window.current_trial.class_name = "left_hand"
        window.current_trial.run_index = 1
        window.current_trial.run_trial_index = 1
        original_record_event = window.record_event
        original_refresh_phase_display = window._refresh_phase_display
        window.record_event = (
            lambda event_name, **kwargs: timeline.append(
                ("event", str(event_name), len(window.event_log), window.current_phase)
            )
            or original_record_event(event_name, **kwargs)
        )
        window._refresh_phase_display = (
            lambda: timeline.append(("display", window.current_phase, len(window.event_log)))
            or original_refresh_phase_display()
        )
        window._stop_voice_prompt = (
            lambda: timeline.append(("stop", window.current_phase, len(window.event_log))) or 0
        )
        window._play_prompt_tone_blocking = lambda **kwargs: timeline.append(
            (
                "tone",
                window.current_phase,
                len(window.event_log),
                int(kwargs["frequency"]),
                int(kwargs["duration_ms"]),
            )
        )
        try:
            window.start_imagery_phase()

            self.assertEqual(
                timeline[:6],
                [
                    ("display", "imagery_ready", 0),
                    ("stop", "imagery_ready", 0),
                    ("tone", "imagery_ready", 0, IMAGERY_START_TONE_HZ, IMAGERY_START_TONE_MS),
                    ("event", "imagery_start", 0, "imagery_ready"),
                    ("display", "imagery", 1),
                    ("event", "imagery_left_hand", 1, "imagery"),
                ],
            )
            self.assertEqual(
                [event["event_name"] for event in window.event_log[:2]],
                ["imagery_start", "imagery_left_hand"],
            )
            self.assertEqual(window.current_phase, "imagery")
        finally:
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_imagery_start_tone_waits_for_guard_after_tone(self) -> None:
        window = MIDataCollectorWindow()
        tone_calls: list[tuple[int, int]] = []
        sleeps: list[float] = []
        window.audio_prompts_enabled = True
        window._play_prompt_tone_blocking = lambda *, frequency, duration_ms: tone_calls.append(
            (int(frequency), int(duration_ms))
        )
        try:
            with mock.patch("mi_data_collector.time.sleep", side_effect=lambda seconds: sleeps.append(float(seconds))):
                window._play_imagery_start_tone_blocking()

            self.assertEqual(tone_calls, [(IMAGERY_START_TONE_HZ, IMAGERY_START_TONE_MS)])
            self.assertEqual(sleeps, [IMAGERY_START_POST_TONE_GUARD_MS / 1000.0])
        finally:
            window.close()

    def test_enter_phase_can_anchor_countdown_to_marker_time(self) -> None:
        window = MIDataCollectorWindow()
        try:
            started_perf = time.perf_counter() - 0.2
            window.enter_phase(
                phase="imagery",
                duration_sec=4.0,
                class_name="left_hand",
                title="",
                subtitle="",
                started_perf=started_perf,
            )

            self.assertAlmostEqual(window.phase_started_perf, started_perf, places=3)
            self.assertAlmostEqual(window.phase_deadline, started_perf + 4.0, places=3)
            self.assertLess(window.remaining_phase_sec, 4.0)
            self.assertGreater(window.remaining_phase_sec, 3.6)
        finally:
            window.phase_timer.stop()
            window.close()

    def test_imagery_start_stops_voice_even_when_audio_prompts_are_disabled(self) -> None:
        window = self._build_window_with_fake_session()
        stop_calls: list[str] = []
        window.audio_prompts_enabled = False
        window._stop_voice_prompt = lambda: stop_calls.append("stop") or 0
        try:
            window._play_phase_start_prompt_blocking("imagery", class_name="left_hand")

            self.assertEqual(stop_calls, ["stop"])
        finally:
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_imagery_end_prompt_starts_before_iti_marker(self) -> None:
        window = self._build_window_with_fake_session()
        prompt_calls: list[tuple[str, str, str | None, int, str]] = []
        window.current_settings = self._build_settings()
        window.current_phase = "imagery"
        window.phase_deadline = time.perf_counter() - 0.1
        window.current_trial = mock.Mock()
        window.current_trial.trial_id = 1
        window.current_trial.class_name = "left_hand"
        window.current_trial.run_index = 1
        window.current_trial.run_trial_index = 1
        window._play_phase_end_prompt_blocking = (
            lambda phase, class_name=None: prompt_calls.append(
                ("end", phase, class_name, len(window.event_log), window.current_phase)
            )
        )
        window._play_phase_start_prompt_blocking = (
            lambda phase, class_name=None: prompt_calls.append(
                ("start", phase, class_name, len(window.event_log), window.current_phase)
            )
        )
        try:
            window.on_phase_tick()

            self.assertEqual(
                [event["event_name"] for event in window.event_log[:2]],
                ["imagery_end", "iti_start"],
            )
            self.assertEqual(
                prompt_calls,
                [
                    ("end", "imagery", "left_hand", 1, "imagery"),
                    ("start", "iti", "left_hand", 1, "iti"),
                ],
            )
            self.assertEqual(window.current_phase, "iti")
        finally:
            window.waiting_for_save = False
            window.session_running = False
            window.worker_thread = None
            window.close()

    def test_marker_writes_are_spaced_by_sampling_rate(self) -> None:
        class FakeBoard:
            def __init__(self) -> None:
                self.markers: list[float] = []

            def insert_marker(self, marker_code: float) -> None:
                self.markers.append(float(marker_code))

        worker = BoardCaptureWorker(
            board_id=0,
            serial_port="",
            channel_positions=list(range(8)),
            channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
        )
        fake_board = FakeBoard()
        worker.board = fake_board
        worker.sampling_rate = 250.0
        sleep_calls: list[float] = []

        with mock.patch("mi_data_collector.time.sleep", side_effect=lambda seconds: sleep_calls.append(float(seconds))):
            self.assertEqual(worker.insert_marker_sync(200.0), (True, ""))
            self.assertEqual(worker.insert_marker_sync(210.0), (True, ""))
            self.assertEqual(worker.insert_marker_sync(220.0, wait_for_flush=False), (True, ""))

        self.assertEqual(fake_board.markers, [200.0, 210.0, 220.0])
        self.assertEqual(sleep_calls, [0.008, 0.008])

    def test_worker_finish_waits_before_declaring_missing_save_payload(self) -> None:
        window = MIDataCollectorWindow()
        scheduled: list[tuple[int, object]] = []
        try:
            window.waiting_for_save = True
            window.current_label.setText("当前任务：保存中")
            window.worker = object()
            window.worker_thread = object()

            with mock.patch(
                "mi_data_collector.QTimer.singleShot",
                side_effect=lambda delay_ms, callback: scheduled.append((int(delay_ms), callback)),
            ):
                window.on_worker_thread_finished()

            self.assertEqual(len(scheduled), 1)
            self.assertEqual(scheduled[0][0], MISSING_SAVE_RESULT_GRACE_MS)
            self.assertEqual(window.current_label.text(), "当前任务：等待保存数据")
            self.assertTrue(window.waiting_for_save)
        finally:
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_session_data_ready_saves_in_background_and_updates_ui_on_completion(self) -> None:
        window = MIDataCollectorWindow()
        logs: list[str] = []
        errors: list[str] = []
        window.log = logs.append
        window.show_error = errors.append
        window.capture_on_stop = True
        window.waiting_for_save = True
        window.current_settings = self._build_settings()
        window.event_log = [make_event("session_start")]
        window.trial_records = []
        window.use_separate_participant_screen = False

        payload = {
            "brainflow_data": np.zeros((10, 32), dtype=np.float32),
            "sampling_rate": 250.0,
            "selected_rows": list(range(8)),
            "marker_row": 8,
            "timestamp_row": None,
            "package_num_row": None,
            "board_descr": {},
        }

        save_started = threading.Event()
        release_save = threading.Event()

        def fake_save_mi_session(**kwargs):
            del kwargs
            save_started.set()
            release_save.wait(timeout=1.0)
            return {
                "trial_count": 1,
                "accepted_trial_count": 1,
                "rejected_trial_count": 0,
                "session_dir": str(PROJECT_ROOT / "runtime" / "async_save"),
                "fif_path": str(PROJECT_ROOT / "runtime" / "async_save" / "run_raw.fif"),
                "board_data_path": "",
                "segments_csv_path": "",
                "mi_epochs_path": "",
                "gate_epochs_path": "",
                "artifact_epochs_path": "",
                "continuous_path": "",
                "manifest_csv_path": "",
                "save_index": 1,
                "run_stem": "sub-test_ses-20260403_120000_run-001_tpc-01_n-001_ok-001",
            }

        try:
            started_at = time.perf_counter()
            with mock.patch("mi_data_collector.save_mi_session", side_effect=fake_save_mi_session):
                window.on_session_data_ready(payload)
                elapsed = time.perf_counter() - started_at

                self.assertLess(elapsed, 0.1)
                self.assertTrue(window.waiting_for_save)
                self.assertIsNotNone(window.save_thread)
                self.assertTrue(save_started.wait(timeout=0.5))

                release_save.set()
                deadline = time.time() + 1.5
                while window.save_thread is not None and time.time() < deadline:
                    self._pump(0.05)

                self.assertFalse(window.waiting_for_save)
                self.assertIsNone(window.save_thread)
                self.assertIsNone(window.save_worker)
                self.assertEqual(errors, [])
                self.assertEqual(window.current_label.text(), "当前任务：数据已保存")
                self.assertIn("runtime", window.current_label.toolTip())
                self.assertIn("本次已保存 1 个试次", window.sequence_summary_label.text())
                self.assertTrue(any("开始后台写盘" in item for item in logs))
        finally:
            release_save.set()
            if window.save_thread is not None:
                deadline = time.time() + 1.0
                while window.save_thread is not None and time.time() < deadline:
                    self._pump(0.05)
            window.waiting_for_save = False
            window.close()

    def test_marker_failure_requests_emergency_raw_save(self) -> None:
        class FakeWorker:
            @staticmethod
            def supports_impedance_mode() -> bool:
                return False

        window = self._build_window_with_fake_session()
        stop_requests: list[str] = []
        errors: list[str] = []
        window.current_settings = self._build_settings()
        window.worker = FakeWorker()
        window.worker_stop_requested.connect(lambda: stop_requests.append("stop"))
        window.show_error = errors.append
        stop_voice_calls: list[str] = []
        window._stop_voice_prompt = lambda: stop_voice_calls.append("stop") or 0
        try:
            window._abort_session_due_to_marker_failure("marker write failed")

            self.assertTrue(window.marker_failure_active)
            self.assertTrue(window.capture_on_stop)
            self.assertTrue(window.waiting_for_save)
            self.assertTrue(window.emergency_save_only)
            self.assertIn("marker_write_failed", window.emergency_save_reason)
            self.assertEqual(stop_requests, ["stop"])
            self.assertEqual(stop_voice_calls, ["stop"])
            self.assertIn("紧急原始数据", window.countdown_label.text())
            self.assertTrue(any("紧急原始数据" in message for message in errors))
        finally:
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_phase_tick_exception_requests_emergency_raw_save(self) -> None:
        window = self._build_window_with_fake_session()
        stop_requests: list[str] = []
        errors: list[str] = []
        window.current_settings = self._build_settings()
        window.worker = object()
        window.worker_stop_requested.connect(lambda: stop_requests.append("stop"))
        window.show_error = errors.append
        window._stop_voice_prompt = lambda: None

        def raise_phase_error() -> None:
            raise RuntimeError("phase crashed")

        window.on_phase_tick = raise_phase_error
        try:
            window._on_phase_tick_guarded()

            self.assertTrue(window.runtime_failure_active)
            self.assertTrue(window.capture_on_stop)
            self.assertTrue(window.waiting_for_save)
            self.assertTrue(window.emergency_save_only)
            self.assertIn("phase_tick_exception", window.emergency_save_reason)
            self.assertEqual(stop_requests, ["stop"])
            self.assertTrue(any("紧急原始数据" in message for message in errors))
        finally:
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_worker_error_during_session_requests_emergency_raw_save(self) -> None:
        window = self._build_window_with_fake_session()
        stop_requests: list[str] = []
        errors: list[str] = []
        window.current_settings = self._build_settings()
        window.worker = object()
        window.worker_stop_requested.connect(lambda: stop_requests.append("stop"))
        window.show_error = errors.append
        window._stop_voice_prompt = lambda: None
        try:
            window.on_worker_error("device stream crashed")

            self.assertTrue(window.runtime_failure_active)
            self.assertTrue(window.capture_on_stop)
            self.assertTrue(window.waiting_for_save)
            self.assertTrue(window.emergency_save_only)
            self.assertIn("worker_error", window.emergency_save_reason)
            self.assertEqual(stop_requests, ["stop"])
            self.assertTrue(any("紧急原始数据" in message for message in errors))
        finally:
            window.waiting_for_save = False
            window.worker_thread = None
            window.close()

    def test_session_data_ready_can_emergency_save_without_event_log(self) -> None:
        window = MIDataCollectorWindow()
        logs: list[str] = []
        errors: list[str] = []
        window.log = logs.append
        window.show_error = errors.append
        window.capture_on_stop = True
        window.waiting_for_save = True
        window.emergency_save_only = True
        window.emergency_save_reason = "marker_write_failed: simulated"
        window.current_settings = self._build_settings()
        window.event_log = []
        window.trial_records = []
        window.use_separate_participant_screen = False

        payload = {
            "brainflow_data": np.zeros((10, 32), dtype=np.float32),
            "sampling_rate": 250.0,
            "selected_rows": list(range(8)),
            "marker_row": 8,
            "timestamp_row": None,
            "package_num_row": None,
            "board_descr": {},
        }
        recovery = {
            "session_dir": str(PROJECT_ROOT / "runtime" / "emergency_async"),
            "emergency_board_data_path": str(PROJECT_ROOT / "runtime" / "emergency_async" / "board.npy"),
            "emergency_meta_path": str(PROJECT_ROOT / "runtime" / "emergency_async" / "emergency.json"),
            "save_index": 1,
            "run_stem": "sub-test_ses-20260403_120000_run-001_emergency_raw",
            "sample_count": 32,
            "duration_sec": 0.128,
        }

        try:
            with mock.patch("mi_data_collector.save_mi_session") as regular_save:
                with mock.patch("mi_data_collector.save_mi_emergency_raw_capture", return_value=recovery):
                    window.on_session_data_ready(payload)
                    deadline = time.time() + 1.5
                    while window.save_thread is not None and time.time() < deadline:
                        self._pump(0.05)

            regular_save.assert_not_called()
            self.assertFalse(window.waiting_for_save)
            self.assertFalse(window.emergency_save_only)
            self.assertEqual(errors, [])
            self.assertEqual(window.current_label.text(), "当前任务：紧急原始数据已保存")
            self.assertIn("紧急原始数据", window.summary_label.text())
            self.assertTrue(any("紧急原始矩阵" in item for item in logs))
        finally:
            if window.save_thread is not None:
                deadline = time.time() + 1.0
                while window.save_thread is not None and time.time() < deadline:
                    self._pump(0.05)
            window.waiting_for_save = False
            window.close()

    def test_session_save_worker_emergency_saves_raw_when_export_fails(self) -> None:
        payload = {
            "brainflow_data": np.zeros((10, 32), dtype=np.float32),
            "sampling_rate": 250.0,
            "selected_rows": list(range(8)),
            "marker_row": 8,
            "timestamp_row": None,
            "package_num_row": None,
            "board_descr": {},
        }
        worker = SessionSaveWorker(
            payload=payload,
            settings=self._build_settings(),
            event_log=[make_event("session_start")],
            trial_records=[],
        )
        failures: list[str] = []
        completions: list[dict] = []
        worker.save_failed.connect(lambda message: failures.append(str(message)))
        worker.save_completed.connect(lambda result: completions.append(dict(result)))

        recovery = {
            "session_dir": str(PROJECT_ROOT / "runtime" / "recovery"),
            "emergency_board_data_path": str(PROJECT_ROOT / "runtime" / "recovery" / "board.npy"),
            "emergency_meta_path": str(PROJECT_ROOT / "runtime" / "recovery" / "emergency.json"),
            "save_index": 1,
            "run_stem": "sub-test_ses-20260403_120000_run-001_emergency_raw",
            "sample_count": 32,
            "duration_sec": 0.128,
        }
        with mock.patch("mi_data_collector.save_mi_session", side_effect=RuntimeError("derivative export failed")):
            with mock.patch("mi_data_collector.save_mi_emergency_raw_capture", return_value=recovery) as emergency_save:
                worker.run()

        self.assertEqual(failures, [])
        self.assertEqual(len(completions), 1)
        self.assertTrue(completions[0]["emergency_raw_only"])
        self.assertIn("derivative export failed", completions[0]["full_save_error"])
        self.assertEqual(completions[0]["board_data_path"], recovery["emergency_board_data_path"])
        self.assertEqual(completions[0]["emergency_meta_path"], recovery["emergency_meta_path"])
        emergency_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()

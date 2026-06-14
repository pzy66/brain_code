from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import replace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication

from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.config import build_hiwonder_camera_stream_url
from robot_workbench.flow_ui import (
    BrainFlowEegStreamThread,
    BrainRobotWorkbenchWindow,
    MiCueWidget,
    RobotCameraDisplayWidget,
    RobotCommandBackend,
    SignalPreviewWidget,
    SsvepStimulusWidget,
    WorkbenchConfig,
    _vision_targets_from_packet,
)


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _complete_connection_gate(window: BrainRobotWorkbenchWindow) -> None:
    window.connection_page.connect_eeg_cap()
    window.backend.connect_robot()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
    assert window.connection_page.btn_next.isEnabled() is True
    window.connection_page.btn_next.click()
    if app is not None:
        app.processEvents()


def _complete_pretraining(window: BrainRobotWorkbenchWindow) -> None:
    window.training_page.finish_training_for_test()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def test_signal_preview_accepts_real_8_channel_chunks() -> None:
    app = _ensure_app()
    preview = SignalPreviewWidget(window_seconds=1.0)
    preview.configure_stream(sampling_rate=250.0, channel_names=[f"Ch {idx}" for idx in range(1, 9)])

    chunk = np.arange(8 * 5, dtype=np.float32).reshape(8, 5)
    preview.append_chunk(chunk)
    app.processEvents()

    assert preview.live_stream_active is True
    assert len(preview.buffers) == 8
    assert list(preview.buffers[0])[-5:] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert list(preview.buffers[7])[-5:] == [35.0, 36.0, 37.0, 38.0, 39.0]
    preview.close()


def test_signal_preview_downsamples_large_windows_for_clear_rendering() -> None:
    values = np.linspace(-100.0, 100.0, 1250, dtype=np.float32)

    plotted = SignalPreviewWidget._downsample_for_plot(values, 160)
    center, low, high, scale = SignalPreviewWidget._robust_display_bounds(values)

    assert plotted.size <= 160
    assert plotted.min() == -100.0
    assert plotted.max() == 100.0
    assert center == 0.0
    assert low < 0.0 < high
    assert scale > 80.0


def test_brainflow_thread_waits_for_delayed_buffer(monkeypatch) -> None:
    import types

    _ensure_app()
    fake_brainflow = types.ModuleType("brainflow")
    fake_board_shim = types.ModuleType("brainflow.board_shim")
    fake_compat = types.ModuleType("brainflow_compat")

    class FakeParams:
        serial_port = ""

    class FakeBoard:
        def __init__(self, board_id, params):
            self.board_id = int(board_id)
            self.params = params
            self.count_calls = 0
            self.started = False
            self.released = False

        def prepare_session(self):
            return None

        def start_stream(self, *args, **kwargs):
            self.started = True

        def get_board_data_count(self):
            self.count_calls += 1
            if self.count_calls == 1:
                raise RuntimeError("BOARD_NOT_CREATED_ERROR: obtain buffer size")
            if self.count_calls < 4:
                return 0
            return 40

        def get_board_data(self, num_samples=None):
            samples = int(num_samples or 8)
            return np.tile(np.arange(samples, dtype=np.float64), (16, 1))

        def stop_stream(self):
            self.started = False

        def release_session(self):
            self.released = True

    class FakeBoardShim:
        def __new__(cls, board_id, params):
            return FakeBoard(board_id, params)

        @staticmethod
        def get_eeg_channels(board_id):
            return list(range(8))

        @staticmethod
        def get_sampling_rate(board_id):
            return 250

    fake_board_shim.BoardShim = FakeBoardShim
    fake_board_shim.BrainFlowInputParams = FakeParams
    monkeypatch.setitem(sys.modules, "brainflow", fake_brainflow)
    monkeypatch.setitem(sys.modules, "brainflow.board_shim", fake_board_shim)
    monkeypatch.setitem(sys.modules, "brainflow_compat", fake_compat)

    thread = BrainFlowEegStreamThread(serial_port="COM_TEST", board_id=0, poll_interval_sec=0.02)
    errors: list[str] = []
    samples: list[np.ndarray] = []
    statuses: list[str] = []
    thread.error_occurred.connect(lambda message: errors.append(str(message)))
    thread.status_changed.connect(lambda message: statuses.append(str(message)))
    thread.samples_ready.connect(lambda chunk: (samples.append(np.asarray(chunk)), thread.stop()))

    thread.run()

    assert not errors
    assert samples
    assert samples[-1].shape[0] == 8
    assert any("缓冲区已就绪" in item for item in statuses)


def test_connection_gate_does_not_start_eeg_stream_when_auto_start_disabled() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(
        WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False, eeg_signal_auto_start=False)
    )

    window.connection_page.eeg_serial_edit.setText("COM_TEST")
    window.connection_page.eeg_board_edit.setText("0")
    window.connection_page.connect_eeg_cap()
    app.processEvents()

    assert window.connection_page.eeg_connected is True
    assert window.eeg_thread is None
    assert "自动启动已关闭" in window.lbl_preview_status.text()

    window.close()
    app.processEvents()


def test_flow_window_unlocks_stages_in_new_ui_order() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))

    assert window.stack.currentIndex() == 0
    assert window.nav_buttons[1].isEnabled() is False
    assert window.nav_buttons[2].isEnabled() is False
    assert window.nav_buttons[3].isEnabled() is False

    _complete_connection_gate(window)
    assert window.stack.currentIndex() == 1
    assert window.nav_buttons[1].isEnabled() is True
    assert window.nav_buttons[2].isEnabled() is False
    assert window.nav_buttons[3].isEnabled() is False

    window.ssvep_page.complete_stage()
    assert window.nav_buttons[2].isEnabled() is True
    assert window.stack.currentIndex() == 2

    window.mi_page.complete_stage()
    assert window.stack.currentIndex() == 3
    assert window.training_page.training_running is True
    assert window.nav_buttons[3].isEnabled() is False

    _complete_pretraining(window)
    assert window.training_page.training_complete is True
    assert window.nav_buttons[3].isEnabled() is True
    assert window.stack.currentIndex() == 4

    window.close()
    app.processEvents()


def test_demo_connected_mode_opens_ssvep_training_ui_without_real_devices() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(
        WorkbenchConfig(
            demo_connected=True,
            robot_mode="real",
            move_stage_ms=500,
            camera_auto_start=True,
            vision_enabled=True,
            vision_auto_start=True,
            eeg_signal_auto_start=True,
        )
    )
    app.processEvents()

    assert window.config.robot_mode == "fake"
    assert window.config.camera_auto_start is False
    assert window.config.vision_enabled is False
    assert window.config.eeg_signal_auto_start is False
    assert window.connection_page.robot_connected is True
    assert window.connection_page.eeg_connected is True
    assert window.connection_page.demo_connected is True
    assert window.stack.currentIndex() == 1
    assert window.nav_buttons[1].isEnabled() is True
    assert window.nav_buttons[2].isEnabled() is False
    assert window.nav_buttons[3].isEnabled() is False
    assert window.eeg_thread is None

    window.close()
    app.processEvents()


def test_training_pages_have_stage_specific_visual_flows() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))

    _complete_connection_gate(window)

    assert window.stack.currentIndex() == 1
    assert window.ssvep_page.stage_kind == "ssvep"
    assert window.ssvep_page.ssvep_stimulus is not None
    stimulus = window.ssvep_page.ssvep_stimulus
    assert stimulus.minimumHeight() >= 400

    window.ssvep_page.start_demo()
    app.processEvents()
    assert stimulus.running is True
    assert window.ssvep_page.timer.interval() == 100
    initial_phase = stimulus.phase
    window.ssvep_page._tick()
    app.processEvents()
    assert stimulus.phase != initial_phase
    assert stimulus.progress > 0

    window.ssvep_page.complete_stage()
    app.processEvents()
    assert stimulus.running is False
    assert window.stack.currentIndex() == 2
    assert window.mi_page.stage_kind == "mi"
    assert window.mi_page.mi_cue is not None

    mi_cue = window.mi_page.mi_cue
    window.mi_page.start_demo()
    app.processEvents()
    assert mi_cue.running is True
    window.mi_page.complete_stage()
    app.processEvents()
    assert mi_cue.running is False
    assert window.stack.currentIndex() == 3
    assert window.training_page.training_running is True
    assert window.nav_buttons[3].isEnabled() is False
    _complete_pretraining(window)
    assert window.stack.currentIndex() == 4

    window.close()
    app.processEvents()


def test_ssvep_stimulus_uses_frequency_based_half_period_refresh() -> None:
    app = _ensure_app()
    stimulus = SsvepStimulusWidget()

    assert stimulus.frequencies == (9.0, 11.0, 13.0, 15.0)
    assert stimulus.PREPARE_SEC == 1.0
    assert stimulus.ACTIVE_SEC == 5.0
    assert stimulus.REST_SEC == 4.0
    assert stimulus.TARGET_REPEATS == 10
    assert stimulus.TOTAL_TRIALS == 40
    assert stimulus.TOTAL_SEC == 400.0
    assert stimulus.render_timer.interval() == 16
    assert stimulus.minimumHeight() >= 620
    assert stimulus.maximumHeight() > 1000
    rects = stimulus._target_rects(1200, 700)
    assert len(rects) == 4
    assert all(rect.width() >= 240 for rect in rects)
    assert rects[0].center().y() < rects[1].center().y()
    assert rects[2].center().y() > rects[1].center().y()
    assert rects[2].top() - rects[0].bottom() >= 48
    assert rects[1].center().x() < rects[0].center().x()
    assert rects[3].center().x() > rects[0].center().x()
    assert stimulus._target_flash_on(9.0, 0.0) is True
    assert stimulus._target_flash_on(9.0, 0.056) is False
    assert stimulus._target_flash_on(9.0, 0.112) is True
    assert stimulus._target_flash_on(15.0, 0.034) is False
    assert stimulus._target_flash_on(15.0, 0.067) is True

    stimulus.set_running(True)
    stimulus.set_progress(0)
    assert stimulus._phase_text() == "准备注视"
    assert stimulus.flicker_enabled is False
    assert stimulus.capture_label() == 0
    stimulus.set_progress(100.0 * (stimulus.PREPARE_SEC + 0.1) / stimulus.TOTAL_SEC)
    assert stimulus._phase_text() == "闪烁采集"
    assert stimulus.flicker_enabled is True
    assert stimulus.capture_label() == 1
    stimulus.set_progress(100.0 * (stimulus.PREPARE_SEC + stimulus.ACTIVE_SEC + 0.1) / stimulus.TOTAL_SEC)
    assert stimulus._phase_text() == "休息恢复"
    assert stimulus.flicker_enabled is False
    assert stimulus.capture_label() == 0
    assert stimulus.render_timer.isActive() is True
    stimulus.set_running(False)
    assert stimulus.render_timer.isActive() is False

    stimulus.close()
    app.processEvents()


def test_ssvep_training_active_sample_phase_matches_standard_protocol() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    stimulus = window.ssvep_page.ssvep_stimulus
    assert stimulus is not None

    window.ssvep_page.start_demo()
    for _ in range(9):
        window.ssvep_page._tick()
    assert stimulus._phase_text() == "准备注视"
    assert stimulus.flicker_enabled is False

    window.ssvep_page._tick()
    assert stimulus._phase_text() == "闪烁采集"
    assert stimulus.flicker_enabled is True

    for _ in range(49):
        window.ssvep_page._tick()
    assert stimulus._phase_text() == "闪烁采集"
    assert stimulus.flicker_enabled is True

    window.ssvep_page._tick()
    assert stimulus._phase_text() == "休息恢复"
    assert stimulus.flicker_enabled is False

    window.close()
    app.processEvents()


def test_mi_cue_uses_motor_imagery_collection_prompts() -> None:
    app = _ensure_app()
    cue = MiCueWidget()

    assert cue.minimumHeight() >= 520
    assert cue.DIRECTIONS == ("LEFT", "RIGHT", "FEET", "TONGUE")
    assert cue.TITLE_MAP["LEFT"] == "左手握拳"
    assert cue.TITLE_MAP["RIGHT"] == "右手握拳"
    assert cue.TITLE_MAP["FEET"] == "双脚运动"
    assert cue.TITLE_MAP["TONGUE"] == "舌头伸缩"
    assert cue.READY_SEC == 2.0
    assert cue.CUE_SEC == 4.0
    assert cue.REST_SEC == 2.0
    assert cue.TRIALS_PER_CLASS == 10
    assert cue.TOTAL_TRIALS == 40
    assert cue.TOTAL_SEC == 320.0
    assert set(cue.cue_asset_paths) == {"LEFT", "RIGHT", "FEET", "TONGUE"}
    assert all(path.exists() for path in cue.cue_asset_paths.values())
    assert all(not pixmap.isNull() for pixmap in cue.cue_pixmaps.values())
    assert cue.title_label.font().family() == "Microsoft YaHei UI"
    assert cue.title_label.font().pointSize() >= 22

    cue.set_running(True)
    cue.set_progress(0)
    assert cue._phase_text() == "准备阶段"
    assert cue.capture_label() == 0
    assert cue.title_label.text() == "准备阶段"
    assert "注视中央十字" in cue.subtitle_label.text()
    assert "Trial 1 / 40" in cue.status_label.text()

    cue.set_progress(100.0 * (MiCueWidget.READY_SEC + 0.1) / MiCueWidget.TOTAL_SEC)
    assert cue._phase_text() == "运动想象"
    assert cue.current_task_title() == "想象：左手握拳"
    assert cue.capture_label() == 1
    assert cue.title_label.text() == "想象任务：左手握拳"
    assert "标签 1" in cue.status_label.text()

    cue.set_progress(100.0 * (MiCueWidget.READY_SEC + MiCueWidget.CUE_SEC + 0.1) / MiCueWidget.TOTAL_SEC)
    assert cue._phase_text() == "休息恢复"
    assert cue.capture_label() == 0
    assert cue.title_label.text() == "休息恢复"

    second_trial_cue_progress = 100.0 * (MiCueWidget.TRIAL_SEC + MiCueWidget.READY_SEC + 0.1) / MiCueWidget.TOTAL_SEC
    cue.set_progress(second_trial_cue_progress)
    assert cue._phase_text() == "运动想象"
    assert cue.current_task_title() == "想象：右手握拳"
    assert cue.capture_label() == 2
    assert cue.title_label.text() == "想象任务：右手握拳"

    cue.close()
    app.processEvents()


def test_mi_capture_labels_only_motor_imagery_phase(tmp_path, monkeypatch) -> None:
    app = _ensure_app()
    monkeypatch.chdir(tmp_path)
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.configure_capture_stream(250.0, [f"Ch {index}" for index in range(1, 9)])

    window.mi_page.start_demo()
    chunk = np.ones((8, 12), dtype=np.float32)
    window.mi_page.append_eeg_chunk(chunk)
    assert window.mi_page.captured_sample_count == 0

    assert window.mi_page.mi_cue is not None
    window.mi_page.mi_cue.set_progress(100.0 * (MiCueWidget.READY_SEC + 0.1) / MiCueWidget.TOTAL_SEC)
    window.mi_page.append_eeg_chunk(chunk)
    assert window.mi_page.captured_sample_count == 12

    window.mi_page.complete_stage()
    saved_path = window.mi_page.last_capture_path
    assert saved_path is not None
    with np.load(saved_path) as payload:
        assert payload["samples"].shape == (8, 12)
        assert set(payload["labels"].tolist()) == {1}

    window.close()
    app.processEvents()


def test_training_stage_saves_labeled_eeg_samples_for_training(tmp_path, monkeypatch) -> None:
    app = _ensure_app()
    monkeypatch.chdir(tmp_path)
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))

    _complete_connection_gate(window)
    window.ssvep_page.configure_capture_stream(250.0, [f"Ch {index}" for index in range(1, 9)])
    window.ssvep_page.start_demo()
    chunk = np.ones((8, 12), dtype=np.float32)
    assert window.ssvep_page.ssvep_stimulus is not None
    active_progress = 100.0 * (SsvepStimulusWidget.PREPARE_SEC + 0.1) / SsvepStimulusWidget.TOTAL_SEC
    window.ssvep_page.ssvep_stimulus.set_progress(active_progress)
    window.ssvep_page.append_eeg_chunk(chunk)
    app.processEvents()

    assert window.ssvep_page.captured_sample_count == 12
    window.ssvep_page.complete_stage()
    app.processEvents()

    saved_path = window.ssvep_page.last_capture_path
    assert saved_path is not None
    assert saved_path.exists()
    with np.load(saved_path) as payload:
        assert payload["samples"].shape == (8, 12)
        assert payload["labels"].shape == (12,)
        assert set(payload["labels"].tolist()) == {1}
        assert float(payload["sampling_rate"][0]) == 250.0

    window.close()
    app.processEvents()


def test_training_stage_displays_capture_seconds_not_raw_samples(tmp_path, monkeypatch) -> None:
    app = _ensure_app()
    monkeypatch.chdir(tmp_path)
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))

    _complete_connection_gate(window)
    window.ssvep_page.configure_capture_stream(250.0, [f"Ch {index}" for index in range(1, 9)])
    window.ssvep_page.start_demo()
    assert window.ssvep_page.ssvep_stimulus is not None
    active_progress = 100.0 * (SsvepStimulusWidget.PREPARE_SEC + 0.1) / SsvepStimulusWidget.TOTAL_SEC
    window.ssvep_page.ssvep_stimulus.set_progress(active_progress)
    window.ssvep_page.append_eeg_chunk(np.ones((8, 250), dtype=np.float32))
    app.processEvents()

    assert window.ssvep_page.lbl_capture_samples.text() == "1.0 s"
    assert window.ssvep_page.lbl_capture_rate.text() == "0.004 s/样本"

    window.close()
    app.processEvents()


def test_connection_gate_requires_robot_and_eeg_before_next_stage() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))

    assert window.stack.currentIndex() == 0
    assert window.connection_page.btn_next.isEnabled() is False

    window.connection_page.connect_eeg_cap()
    app.processEvents()
    assert window.connection_page.btn_next.isEnabled() is False

    window.backend.connect_robot()
    app.processEvents()
    assert window.connection_page.btn_next.isEnabled() is True

    window.connection_page.btn_next.click()
    app.processEvents()
    assert window.stack.currentIndex() == 1

    window.close()
    app.processEvents()


def test_connection_gate_shows_robot_connection_progress() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    progress: list[tuple[int, str, str]] = []
    window.backend.connect_progress_changed.connect(
        lambda percent, phase, detail: progress.append((int(percent), str(phase), str(detail)))
    )

    assert window.connection_page.robot_progress.value() == 0
    assert window.connection_page.btn_robot_disconnect.isEnabled() is False

    window.backend.connect_robot()
    app.processEvents()

    assert progress
    assert progress[-1][0] == 100
    assert window.connection_page.robot_progress.value() == 100
    assert "连接完成" in window.connection_page.lbl_robot_phase.text()
    assert window.connection_page.btn_robot_connect.isEnabled() is False
    assert window.connection_page.btn_robot_disconnect.isEnabled() is True

    window.close()
    app.processEvents()


def test_connection_gate_uses_fullscreen_space_with_large_controls() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    window.resize(1920, 1080)
    window.show()
    app.processEvents()

    page = window.connection_page
    assert page.robot_card.width() >= 860
    assert page.eeg_card.width() >= 860
    assert page.btn_robot_connect.height() >= 44
    assert page.btn_eeg_connect.height() >= 44
    robot_top = page.robot_card.mapTo(page, page.robot_card.rect().topLeft()).y()
    next_top = page.btn_next.mapTo(page, page.btn_next.rect().topLeft()).y()
    assert robot_top < next_top

    window.close()
    app.processEvents()


def test_connection_gate_blocks_robot_control_until_pretraining_finishes() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))

    window.connection_page.connect_eeg_cap()
    window.backend.connect_robot()
    app.processEvents()
    assert window.connection_page.btn_robot_control.isEnabled() is False
    assert "训练完成" in window.connection_page.btn_robot_control.text()

    window.connection_page.btn_robot_control.click()
    app.processEvents()
    assert window.stack.currentIndex() == 0

    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    app.processEvents()
    assert window.stack.currentIndex() == 3
    assert window.connection_page.btn_robot_control.isEnabled() is False

    _complete_pretraining(window)
    assert window.stack.currentIndex() == 4
    assert window.nav_buttons[3].isEnabled() is True
    assert window.connection_page.btn_robot_control.isEnabled() is True

    window.close()
    app.processEvents()


def test_real_ros_connect_starts_robot_runtime_when_state_is_missing(monkeypatch) -> None:
    script = r'''
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication
import robot_workbench.flow_ui as flow_ui
from robot_workbench.flow_ui import RobotCommandBackend, WorkbenchConfig

app = QApplication.instance() or QApplication([])
attempts = []
starts = []

class FakeRosbridgeClient:
    def __init__(self, *args, state_callback=None, status_callback=None, **kwargs):
        self.index = len(attempts)
        attempts.append(self)
        self._connected = False
        self.state_callback = state_callback
        self.status_callback = status_callback

    def connect(self):
        self._connected = True
        if self.status_callback is not None:
            self.status_callback("fake rosbridge connected")

    def is_connected(self):
        return self._connected

    def latest_state_snapshot(self):
        if self.index == 0:
            return None
        return {"robot_cyl": {"theta_deg": 1.0, "radius_mm": 151.0, "z_mm": 160.0}}

    def close(self):
        self._connected = False

flow_ui.RosbridgeClient = FakeRosbridgeClient

def fake_start(self):
    starts.append(True)
    return True

RobotCommandBackend._start_remote_ros_runtime = fake_start
backend = RobotCommandBackend(
    WorkbenchConfig(
        robot_mode="real",
        robot_transport="ros",
        robot_runtime_auto_start=True,
        rosbridge_connect_timeout_sec=0.05,
        ros_state_timeout_sec=0.05,
    )
)
connected = []
poses = []
backend.connection_changed.connect(lambda value: connected.append(bool(value)))
backend.pose_changed.connect(lambda theta, radius, z: poses.append((theta, radius, z)))
progress = []
backend.connect_progress_changed.connect(lambda percent, phase, detail: progress.append((int(percent), str(phase), str(detail))))
backend._connect_worker()
assert len(attempts) == 2, attempts
assert starts == [True], starts
assert connected[-1] is True, connected
assert poses[-1] == (1.0, 151.0, 160.0), poses
assert any(item[1] == "自动启动远端程序" for item in progress), progress
assert any(item[1] == "重新连接机械臂" for item in progress), progress
assert progress[-1][0] == 100, progress
print("ok")
'''
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_connection_gate_relocks_flow_when_device_disconnects() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))

    _complete_connection_gate(window)
    assert window.stack.currentIndex() == 1

    window.connection_page.disconnect_eeg_cap()
    app.processEvents()
    assert window.stack.currentIndex() == 0
    assert window.nav_buttons[1].isEnabled() is False
    assert window.nav_buttons[2].isEnabled() is False
    assert window.nav_buttons[3].isEnabled() is False

    window.close()
    app.processEvents()


def test_robot_page_keyboard_flow_reaches_grab_stage() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page

    robot_page._start_integrated_flow()
    assert robot_page.control_phase == "MI_MOVE_1"
    assert robot_page.phase_remaining_ms == 20_000

    robot_page.phase_remaining_ms = robot_page.flow_timer.interval()
    robot_page._flow_tick()
    assert robot_page.control_phase == "DECIDE_1"

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    assert robot_page.control_phase == "SSVEP_TARGET_SELECT"
    assert robot_page.cam.ssv_flicker_enabled is True

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_3, Qt.NoModifier, "3"))
    assert robot_page.selected_target == "3"
    assert robot_page.cam.ssv_flicker_enabled is False
    assert robot_page.cam.active_id == "3"
    assert robot_page.control_phase == "GRASP_CONFIRM"
    assert "抓取确认信号" in robot_page.lbl_run_status.text()
    assert robot_page.popup_dialog.isHidden() is False
    assert "确认抓取目标 3" in robot_page.btn_primary.text()

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    assert robot_page.control_phase == "PICKING"

    window.close()
    app.processEvents()


def test_robot_page_pick_success_continues_to_carry_and_place(monkeypatch) -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    monkeypatch.setattr(window.backend, "pick_target", lambda target_id: None)
    monkeypatch.setattr(window.backend, "place", lambda: None)

    robot_page._start_integrated_flow()
    robot_page._enter_decision_stage_1()
    robot_page._enter_ssvep_target_selection()
    robot_page._lock_ssvep_target_and_confirm("2")
    assert robot_page.control_phase == "GRASP_CONFIRM"
    robot_page._execute_physical_grab()
    assert robot_page.control_phase == "PICKING"

    robot_page._on_command_finished("pick", True, "ok")
    assert robot_page.control_phase == "MI_MOVE_2"
    assert robot_page.flow_timer.isActive() is True

    robot_page.phase_remaining_ms = robot_page.flow_timer.interval()
    robot_page._flow_tick()
    assert robot_page.control_phase == "DECIDE_2"

    robot_page._primary_decision()
    assert robot_page.control_phase == "PLACING"
    robot_page._on_command_finished("place", True, "ok")
    assert robot_page.control_phase == "TASK_DONE"

    window.close()
    app.processEvents()


def test_robot_page_full_showcase_flow_loops_carry_decision(monkeypatch) -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    monkeypatch.setattr(window.backend, "pick_target", lambda target_id: None)
    monkeypatch.setattr(window.backend, "place", lambda: None)

    robot_page._start_integrated_flow()
    assert robot_page.control_phase == "MI_MOVE_1"
    assert robot_page.phase_remaining_ms == 20_000
    assert robot_page.cam.ssv_flicker_enabled is False

    robot_page.phase_remaining_ms = robot_page.flow_timer.interval()
    robot_page._flow_tick()
    assert robot_page.control_phase == "DECIDE_1"
    assert robot_page.popup_dialog.isHidden() is False
    assert robot_page.flash_box_confirm.isHidden() is False
    assert robot_page.flash_box_continue.isHidden() is False
    assert robot_page.flash_timer.isActive()

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    assert robot_page.control_phase == "SSVEP_TARGET_SELECT"
    assert robot_page.cam.ssv_flicker_enabled is True
    assert len(set(robot_page.cam.ssv_frequencies.values())) == 4

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_1, Qt.NoModifier, "1"))
    assert robot_page.control_phase == "GRASP_CONFIRM"
    assert robot_page.selected_target == "1"
    assert robot_page.cam.ssv_flicker_enabled is False
    assert robot_page.popup_dialog.isHidden() is False
    assert "抓取目标 1" in robot_page.flash_box_confirm.text()
    assert "重新选择" in robot_page.flash_box_continue.text()

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    assert robot_page.control_phase == "PICKING"
    assert robot_page.popup_dialog.isHidden()

    robot_page._on_command_finished("pick", True, "ok")
    assert robot_page.control_phase == "MI_MOVE_2"
    assert robot_page.phase_remaining_ms == 20_000

    robot_page.phase_remaining_ms = robot_page.flow_timer.interval()
    robot_page._flow_tick()
    assert robot_page.control_phase == "DECIDE_2"
    assert robot_page.popup_dialog.isHidden() is False
    assert "确认放下" in robot_page.flash_box_confirm.text()
    assert "继续搬运" in robot_page.flash_box_continue.text()

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_C, Qt.NoModifier, "c"))
    assert robot_page.control_phase == "MI_MOVE_2"
    assert robot_page.phase_remaining_ms == 20_000

    robot_page.phase_remaining_ms = robot_page.flow_timer.interval()
    robot_page._flow_tick()
    assert robot_page.control_phase == "DECIDE_2"
    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    assert robot_page.control_phase == "PLACING"
    robot_page._on_command_finished("place", True, "ok")
    assert robot_page.control_phase == "TASK_DONE"

    window.close()
    app.processEvents()


def test_robot_page_ignores_stale_robot_command_completion_after_stop(monkeypatch) -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    monkeypatch.setattr(window.backend, "pick_target", lambda target_id: None)

    robot_page._start_integrated_flow()
    robot_page._enter_decision_stage_1()
    robot_page._enter_ssvep_target_selection()
    robot_page._lock_ssvep_target_and_confirm("2")
    assert robot_page.control_phase == "GRASP_CONFIRM"
    robot_page._execute_physical_grab()
    assert robot_page.control_phase == "PICKING"

    robot_page._stop_current_task_safely()
    assert robot_page.control_phase == "IDLE"
    robot_page._on_command_finished("pick", True, "late ok")
    assert robot_page.control_phase == "IDLE"
    assert robot_page.flow_timer.isActive() is False

    robot_page.control_phase = "DECIDE_2"
    robot_page._on_command_finished("place", True, "late ok")
    assert robot_page.control_phase == "DECIDE_2"

    window.close()
    app.processEvents()


def test_robot_page_mi_stage_uses_keyboard_to_move_robot() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    poses: list[tuple[float, float, float]] = []
    window.backend.pose_changed.connect(lambda theta, radius, z: poses.append((theta, radius, z)))

    robot_page._start_integrated_flow()
    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_W, Qt.NoModifier, "w"))
    robot_page._flow_tick()
    robot_page.handle_key_release(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_W, Qt.NoModifier, "w"))

    assert robot_page.control_phase == "MI_MOVE_1"
    assert poses
    assert poses[-1][1] > 150.0

    window.close()
    app.processEvents()


def test_robot_page_manual_drive_moves_without_task_countdown() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    poses: list[tuple[float, float, float]] = []
    window.backend.pose_changed.connect(lambda theta, radius, z: poses.append((theta, radius, z)))

    assert robot_page.control_phase == "IDLE"
    robot_page._toggle_manual_drive()
    assert robot_page.control_phase == "MANUAL_DRIVE"
    assert robot_page.manual_drive_enabled is True

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_W, Qt.NoModifier, "w"))
    robot_page.handle_key_release(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_W, Qt.NoModifier, "w"))

    assert poses
    assert poses[-1][1] > 150.0
    assert not robot_page.pressed_move_keys

    robot_page._toggle_manual_drive()
    assert robot_page.control_phase == "IDLE"
    assert robot_page.manual_drive_enabled is False

    window.close()
    app.processEvents()


def test_robot_page_wasd_works_when_button_has_focus() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    poses: list[tuple[float, float, float]] = []
    window.backend.pose_changed.connect(lambda theta, radius, z: poses.append((theta, radius, z)))

    robot_page.btn_primary.setFocus(Qt.OtherFocusReason)
    QApplication.sendEvent(robot_page.btn_primary, QKeyEvent(QKeyEvent.KeyPress, Qt.Key_W, Qt.NoModifier, "w"))
    QApplication.sendEvent(robot_page.btn_primary, QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_W, Qt.NoModifier, "w"))
    app.processEvents()

    assert robot_page.manual_drive_enabled is True
    assert robot_page.control_phase == "MANUAL_DRIVE"
    assert poses
    assert poses[-1][1] > 150.0
    assert not robot_page.pressed_move_keys

    window.close()
    app.processEvents()


def test_robot_page_wasd_resets_error_state_before_manual_drive(monkeypatch) -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    reset_calls: list[bool] = []
    monkeypatch.setattr(
        window.backend,
        "latest_state_snapshot",
        lambda: {"state": "ERROR", "last_error_code": "aborted", "last_error": "Abort requested by operator."},
    )
    monkeypatch.setattr(window.backend, "reset", lambda: reset_calls.append(True))

    handled = robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_W, Qt.NoModifier, "w"))

    assert handled is True
    assert reset_calls == [True]
    assert robot_page.control_phase == "IDLE"
    assert robot_page.manual_drive_enabled is False
    assert "复位" in robot_page.lbl_run_status.text()

    window.close()
    app.processEvents()


def test_robot_page_uses_official_jetmax_camera_stream_url() -> None:
    app = _ensure_app()
    config = WorkbenchConfig(robot_host="10.1.2.3", camera_auto_start=False)
    window = BrainRobotWorkbenchWindow(config)

    assert window.robot_page.camera_stream_url == build_hiwonder_camera_stream_url("10.1.2.3")

    window.close()
    app.processEvents()


def test_robot_page_camera_overlay_tracks_flow_state() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page

    assert robot_page.cam.phase_title == "Idle"

    robot_page._start_integrated_flow()
    assert robot_page.cam.phase_title == "Stage 1 Motion Adjustment"
    assert robot_page.cam.countdown_text.endswith("s")

    robot_page.phase_remaining_ms = robot_page.flow_timer.interval()
    robot_page._flow_tick()
    assert robot_page.cam.phase_title == "Stage 1 Decision"

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    assert robot_page.cam.phase_title == "Stage 2 Target Selection"
    assert robot_page.cam.ssv_flicker_enabled is True

    window.close()
    app.processEvents()


def test_robot_page_uses_actionable_vision_target_for_center_pick(monkeypatch) -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(
        WorkbenchConfig(
            robot_mode="fake",
            move_stage_ms=500,
            camera_auto_start=False,
            vision_enabled=True,
            vision_auto_start=False,
        )
    )
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    robot_page.selected_target = "2"
    robot_page._enter_ssvep_grab_stage()
    target = VisionTarget(
        id=2,
        slot_id=2,
        bbox=(100.0, 100.0, 160.0, 150.0),
        center_px=(130.0, 125.0),
        raw_center=(130.0, 125.0),
        confidence=0.92,
        command_mode="world",
        command_point=(12.0, -145.0),
        actionable=True,
        grasp_angle_deg=18.0,
        grasp_angle_quality=0.9,
    )
    robot_page.vision_targets_by_id = {"2": target}
    vision_calls: list[tuple[str, VisionTarget]] = []
    fixed_calls: list[str] = []
    monkeypatch.setattr(window.backend, "pick_vision_target", lambda target_id, vision_target: vision_calls.append((target_id, vision_target)))
    monkeypatch.setattr(window.backend, "pick_target", lambda target_id: fixed_calls.append(target_id))

    robot_page._lock_ssvep_target_and_confirm("2")

    assert vision_calls == []
    assert fixed_calls == []
    assert robot_page.control_phase == "GRASP_CONFIRM"
    assert "抓取确认信号" in robot_page.lbl_run_status.text()

    robot_page._execute_physical_grab()

    assert vision_calls == [("2", target)]
    assert fixed_calls == []
    assert robot_page.control_phase == "PICKING"
    assert "视觉中心" in robot_page.lbl_run_status.text()

    window.close()
    app.processEvents()


def test_robot_page_limits_to_four_targets_and_picks_target_4(monkeypatch) -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(
        WorkbenchConfig(
            robot_mode="fake",
            move_stage_ms=500,
            camera_auto_start=False,
            vision_enabled=True,
            vision_auto_start=False,
            target_count=9,
            vision_max_targets=9,
        )
    )
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page

    assert len(robot_page.target_buttons) == 4
    assert "4" in robot_page.target_buttons
    assert "9" not in robot_page.target_buttons
    robot_page._start_integrated_flow()
    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_4, Qt.NoModifier, "4"))
    assert robot_page.selected_target == "4"
    assert robot_page.cam.active_id == "4"

    robot_page._enter_ssvep_grab_stage()
    target = VisionTarget(
        id=4,
        slot_id=4,
        bbox=(300.0, 180.0, 360.0, 240.0),
        center_px=(330.0, 210.0),
        raw_center=(330.0, 210.0),
        confidence=0.91,
        command_mode="world",
        command_point=(18.0, -156.0),
        actionable=True,
        grasp_angle_deg=9.0,
        grasp_angle_quality=0.8,
    )
    robot_page.vision_targets_by_id = {"4": target}
    vision_calls: list[tuple[str, VisionTarget]] = []
    monkeypatch.setattr(
        window.backend,
        "pick_vision_target",
        lambda target_id, vision_target: vision_calls.append((target_id, vision_target)),
    )

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_4, Qt.NoModifier, "4"))

    assert vision_calls == []
    assert robot_page.control_phase == "GRASP_CONFIRM"

    robot_page.handle_key_press(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_G, Qt.NoModifier, "g"))

    assert vision_calls == [("4", target)]
    assert robot_page.control_phase == "PICKING"

    window.close()
    app.processEvents()


def test_vision_packet_ignores_slots_above_four() -> None:
    targets = _vision_targets_from_packet(
        {
            "mapping_mode": "delta_servo",
            "alignment_target_pixel": [320.0, 240.0],
            "slots": [
                {
                    "slot_id": 4,
                    "valid": True,
                    "bbox": [300.0, 180.0, 360.0, 240.0],
                    "pixel_center_f": [330.0, 210.0],
                    "grasp_pixel_f": [331.0, 211.0],
                    "confidence": 0.91,
                    "command_mode": "world",
                    "command_point": [18.0, -156.0],
                    "actionable": True,
                    "grasp_angle_deg": 9.0,
                    "grasp_angle_quality": 0.8,
                },
                {
                    "slot_id": 5,
                    "valid": True,
                    "bbox": [100.0, 100.0, 140.0, 140.0],
                    "pixel_center_f": [120.0, 120.0],
                    "confidence": 0.99,
                    "command_point": [1.0, 2.0],
                    "actionable": True,
                }
            ],
        }
    )

    assert len(targets) == 1
    assert targets[0].slot_id == 4
    assert targets[0].command_point == (18.0, -156.0)
    assert targets[0].actionable is True


def test_robot_backend_routes_actionable_vision_target_to_ros_pick_world() -> None:
    app = _ensure_app()
    backend = RobotCommandBackend(WorkbenchConfig(robot_mode="real", robot_transport="ros"))
    calls: list[tuple[float, float, float | None]] = []

    class FakeRos:
        def is_connected(self) -> bool:
            return True

        def send_pick_world(self, x_mm, y_mm, *, sucker_rotation_deg=None, callback=None):  # noqa: ANN001
            calls.append((float(x_mm), float(y_mm), sucker_rotation_deg))

    backend._ros = FakeRos()  # noqa: SLF001
    target = VisionTarget(
        id=2,
        slot_id=2,
        bbox=(100.0, 100.0, 160.0, 150.0),
        center_px=(130.0, 125.0),
        raw_center=(130.0, 125.0),
        confidence=0.92,
        command_mode="world",
        command_point=(12.0, -145.0),
        actionable=True,
        grasp_angle_deg=18.0,
        grasp_angle_quality=0.9,
    )

    backend.pick_vision_target("2", target)

    assert calls == [(12.0, -145.0, 18.0)]
    backend.close()
    app.processEvents()


def test_robot_backend_reports_ros_service_errors_without_raising() -> None:
    app = _ensure_app()
    backend = RobotCommandBackend(WorkbenchConfig(robot_mode="real", robot_transport="ros"))
    finished: list[tuple[str, bool, str]] = []
    backend.command_finished.connect(lambda action, ok, message: finished.append((str(action), bool(ok), str(message))))

    class BrokenRos:
        def is_connected(self) -> bool:
            return True

        def latest_state_snapshot(self):  # noqa: ANN201
            return {
                "robot_cyl": {"theta_deg": 0.0, "radius_mm": 160.0, "z_mm": 210.0},
                "limits_cyl": {"theta_deg": [-120.0, 120.0], "radius_mm": [50.0, 280.0]},
                "limits_cyl_auto": {"theta_deg": [-120.0, 120.0], "radius_mm": [80.0, 260.0]},
            }

        def send_move_cyl_auto(self, *args, **kwargs):  # noqa: ANN002,ANN003,ANN201
            raise RuntimeError("move service unavailable")

        def send_pick_cyl(self, *args, **kwargs):  # noqa: ANN002,ANN003,ANN201
            raise RuntimeError("pick service unavailable")

    backend._ros = BrokenRos()  # noqa: SLF001
    target = VisionTarget(
        id=1,
        slot_id=1,
        bbox=(330.0, 210.0, 390.0, 270.0),
        center_px=(360.0, 240.0),
        raw_center=(360.0, 240.0),
        display_center=(360.0, 240.0),
        grasp_pixel=(360.0, 240.0),
        confidence=0.94,
        actionable=False,
        invalid_reason="calibration_profile_unavailable",
    )

    assert backend.align_to_vision_target("1", target) is True
    backend.pick_camera_center_target("1", target)

    assert ("vision-align", False, "move service unavailable") in finished
    assert ("pick", False, "pick service unavailable") in finished
    backend.close()
    app.processEvents()


def test_robot_page_aligns_before_pick_when_vision_center_is_not_ready(monkeypatch) -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(
        WorkbenchConfig(
            robot_mode="fake",
            move_stage_ms=500,
            camera_auto_start=False,
            vision_enabled=True,
            vision_auto_start=False,
        )
    )
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    robot_page.selected_target = "3"
    robot_page._enter_ssvep_grab_stage()
    target = VisionTarget(
        id=3,
        slot_id=3,
        bbox=(200.0, 160.0, 260.0, 210.0),
        center_px=(230.0, 185.0),
        raw_center=(230.0, 185.0),
        confidence=0.88,
        command_mode="world",
        command_point=None,
        actionable=False,
        invalid_reason="vision_servo_required",
        servo_required=True,
        servo_command_mode="cyl",
        servo_command_point=(6.0, 178.0),
    )
    robot_page.vision_targets_by_id = {"3": target}
    align_calls: list[tuple[str, VisionTarget]] = []
    pick_calls: list[tuple[str, VisionTarget]] = []
    monkeypatch.setattr(window.backend, "align_to_vision_target", lambda target_id, vision_target: align_calls.append((target_id, vision_target)) or True)
    monkeypatch.setattr(window.backend, "pick_vision_target", lambda target_id, vision_target: pick_calls.append((target_id, vision_target)))

    robot_page._lock_ssvep_target_and_confirm("3")

    assert align_calls == []
    assert pick_calls == []
    assert robot_page.control_phase == "GRASP_CONFIRM"

    robot_page._execute_physical_grab()

    assert align_calls == [("3", target)]
    assert pick_calls == []
    assert robot_page.control_phase == "GRASP_CONFIRM"
    assert "视觉对中" in robot_page.lbl_run_status.text()

    robot_page._on_command_finished("vision-align", True, "aligned")
    assert pick_calls == []
    assert "自动" in robot_page.lbl_run_status.text()

    ready_target = replace(
        target,
        command_point=(14.0, -148.0),
        actionable=True,
        invalid_reason="",
        servo_required=False,
    )
    robot_page._update_vision_targets([ready_target])

    assert pick_calls == [("3", ready_target)]
    assert robot_page.control_phase == "PICKING"
    assert "自动" in robot_page.lbl_run_status.text() or "视觉中心" in robot_page.lbl_run_status.text()

    window.close()
    app.processEvents()


def test_robot_page_camera_center_fallback_picks_without_calibration(monkeypatch) -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(
        WorkbenchConfig(
            robot_mode="fake",
            move_stage_ms=500,
            camera_auto_start=False,
            vision_enabled=True,
            vision_auto_start=False,
            vision_center_stable_frames=2,
        )
    )
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    robot_page = window.robot_page
    robot_page.selected_target = "1"
    robot_page._enter_ssvep_grab_stage()
    target = VisionTarget(
        id=1,
        slot_id=1,
        bbox=(330.0, 210.0, 390.0, 270.0),
        center_px=(360.0, 240.0),
        raw_center=(360.0, 240.0),
        display_center=(360.0, 240.0),
        grasp_pixel=(360.0, 240.0),
        confidence=0.94,
        command_mode="world",
        command_point=None,
        actionable=False,
        invalid_reason="calibration_profile_unavailable",
        servo_required=False,
    )
    robot_page.vision_targets_by_id = {"1": target}
    align_calls: list[tuple[str, VisionTarget]] = []
    pick_calls: list[tuple[str, VisionTarget]] = []
    monkeypatch.setattr(window.backend, "align_to_vision_target", lambda target_id, vision_target: align_calls.append((target_id, vision_target)) or True)
    monkeypatch.setattr(window.backend, "pick_camera_center_target", lambda target_id, vision_target: pick_calls.append((target_id, vision_target)))

    robot_page._lock_ssvep_target_and_confirm("1")
    robot_page._execute_physical_grab()

    assert align_calls == [("1", target)]
    assert pick_calls == []
    assert robot_page.control_phase == "GRASP_CONFIRM"

    robot_page._on_command_finished("vision-align", True, "aligned")
    assert pick_calls == []
    assert "新画面" in robot_page.lbl_run_status.text()

    centered_target = replace(
        target,
        bbox=(290.0, 210.0, 350.0, 270.0),
        center_px=(320.0, 240.0),
        raw_center=(320.0, 240.0),
        display_center=(320.0, 240.0),
        grasp_pixel=(320.0, 240.0),
    )
    robot_page._update_vision_targets([centered_target])
    assert pick_calls == []
    robot_page._update_vision_targets([centered_target])

    assert pick_calls == [("1", centered_target)]
    assert robot_page.control_phase == "PICKING"
    assert "稳定对中" in robot_page.lbl_run_status.text()

    window.close()
    app.processEvents()


def test_camera_overlay_cache_tracks_target_alignment_pixel() -> None:
    app = _ensure_app()
    widget = RobotCameraDisplayWidget()
    base = dict(
        id=2,
        slot_id=2,
        bbox=(100.0, 100.0, 160.0, 150.0),
        center_px=(130.0, 125.0),
        raw_center=(130.0, 125.0),
        confidence=0.92,
        command_mode="world",
        command_point=(12.0, -145.0),
        actionable=True,
    )
    first = VisionTarget(**base, alignment_target_pixel=(320.0, 240.0))
    second = VisionTarget(**base, alignment_target_pixel=(330.0, 240.0))

    widget.set_vision_payload(targets=[first], packet=None, status_text="视觉识别: 1 个目标")
    first_key = widget._vision_payload_key
    widget.set_vision_payload(targets=[second], packet=None, status_text="视觉识别: 1 个目标")

    assert first_key != widget._vision_payload_key
    widget.close()
    app.processEvents()


def test_robot_page_prioritizes_large_camera_region() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    window.resize(1440, 900)
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    window.show()
    app.processEvents()

    assert window.stack.currentIndex() == 4
    assert window.robot_page.video_container.width() >= 1100
    assert window.robot_page.video_container.height() >= 650

    window.close()
    app.processEvents()


def test_robot_page_expands_camera_region_for_full_hd_window() -> None:
    app = _ensure_app()
    window = BrainRobotWorkbenchWindow(WorkbenchConfig(robot_mode="fake", move_stage_ms=500, camera_auto_start=False))
    window.resize(1920, 1080)
    _complete_connection_gate(window)
    window.ssvep_page.complete_stage()
    window.mi_page.complete_stage()
    _complete_pretraining(window)
    window.show()
    app.processEvents()

    assert window.stack.currentIndex() == 4
    assert window.robot_page.left_panel.width() >= 280
    assert window.robot_page.video_container.width() >= 1500
    assert window.robot_page.video_container.height() >= 820
    assert window.robot_page.cam.size() == window.robot_page.video_container.size()

    window.close()
    app.processEvents()

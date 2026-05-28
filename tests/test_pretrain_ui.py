from __future__ import annotations

from brain_workspace.bootstrap import configure_qt_offscreen
from brain_workspace.paths import ensure_runtime_import_paths

configure_qt_offscreen()
ensure_runtime_import_paths()

from PyQt5.QtWidgets import QApplication  # noqa: E402

from pretrain_ui import PretrainWindow  # noqa: E402


def test_standalone_pretrain_window_starts_locked_until_device_ready() -> None:
    _app = QApplication.instance() or QApplication([])
    window = PretrainWindow()

    assert window.windowTitle() == "脑机接口预训练"
    assert not window.btn_start.isEnabled()

    window.apply_demo_device_ready()
    assert window.btn_start.isEnabled()
    assert window.status_badge.text() == "可以开始"
    window.close()


def test_standalone_pretrain_flow_can_advance_and_reset() -> None:
    _app = QApplication.instance() or QApplication([])
    window = PretrainWindow()
    window.apply_demo_device_ready()

    window.start_pretrain()
    assert window.pretrain_timer.isActive()
    window._advance()
    assert window.overall_progress.value() > 0
    assert window.btn_pause.isEnabled()

    window.reset_pretrain()
    assert not window.pretrain_timer.isActive()
    assert window.overall_progress.value() == 0
    assert window.btn_start.isEnabled()
    window.close()


def test_standalone_pretrain_finish_enters_robot_control_screen() -> None:
    _app = QApplication.instance() or QApplication([])
    window = PretrainWindow()
    window.apply_demo_device_ready()

    window._finish()

    assert window.control_screen_ready
    assert window.centralWidget().objectName() == "controlRoot"
    assert window.control_timer.isActive()
    assert window.camera_widget is not None
    assert window.control_status_label is not None
    assert window.control_state_nodes
    assert window.camera_thread is None
    assert not window.robot_connected
    window._advance_control_simulation()
    assert window.control_tick == 1
    assert "WASD" in window.control_status_label.text()

    window.select_block(2)
    assert window.selected_block_id == 2
    assert window.control_status_label.text() == "数字选块流程：小木块 2 已锁定"

    radius_before = float(window.robot_pose_cyl[1])
    window._handle_control_key_press("w")
    assert float(window.robot_pose_cyl[1]) > radius_before
    window._handle_control_key_release("w")
    window.close()


def test_pretrain_entrypoint_is_importable() -> None:
    from pretrain_ui.app import main

    assert callable(main)

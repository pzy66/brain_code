from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QApplication

from hybrid_controller.config import AppConfig
from hybrid_controller.coordinators import RobotCoordinator, SSVEPCoordinator, UiCoordinator, VisionCoordinator
from hybrid_controller.ui.main_window import MainWindow


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ui_coordinator_builds_typed_snapshot() -> None:
    config = AppConfig()
    robot = RobotCoordinator(config)
    vision = VisionCoordinator(config)
    ssvep = SSVEPCoordinator(config)

    robot.update(
        connected=True,
        health="ok",
        last_ack="ACK MOVE",
        preflight_ok=True,
        preflight_message="ready",
        robot_cyl={"theta_deg": 10.0, "radius_mm": 150.0, "z_mm": 205.0},
        auto_z_current=205.0,
        control_kernel="cylindrical_ros_teleop",
    )
    robot.set_scene_snapshot({"robot_cyl": {"theta_deg": 10.0, "radius_mm": 150.0, "z_mm": 205.0}})
    vision.update(health="camera_fps=30.0 infer_ms=12.0 slots=2", packet={"slots": []}, frame=None, flash_enabled=True)
    ssvep.update(
        running=True,
        stim_enabled=True,
        connected=True,
        mode="target_selection",
        runtime_status="running",
        profile_path=str(Path("datasets/profiles/hybrid_controller/ssvep_profiles/current_fbcca_profile.json")),
        profile_source="current",
    )

    snapshot = UiCoordinator().build_snapshot(
        controller_snapshot={
            "state": "s2_target_select",
            "context": {
                "motion_deadline_ts": None,
                "selected_target_id": 1,
                "selected_target_raw_center": (100.0, 100.0),
                "frozen_targets": [{"id": 1}],
                "carrying": False,
                "last_robot_status": "MOVE_DONE",
                "last_error": None,
            },
        },
        input_profile="bci_experimental",
        move_source="sim",
        decision_source="ssvep",
        robot_mode="real",
        vision_mode="robot_camera_detection",
        target_frequency_map=[("8Hz", 1)],
        last_ssvep_raw="state=selected",
        robot_state=robot.get_state(),
        vision_state=vision.get_state(),
        ssvep_state=ssvep.get_state(),
    )

    assert snapshot.task_state == "s2_target_select"
    assert snapshot.robot.connected is True
    assert snapshot.robot.control_kernel == "cylindrical_ros_teleop"
    assert snapshot.vision.health.startswith("camera_fps")
    assert snapshot.vision.flash_enabled is True
    assert snapshot.ssvep.stim_enabled is True
    assert snapshot.ssvep.profile_source == "current"
    assert snapshot.target_frequency_map == (("8Hz", 1),)


def test_main_window_accepts_app_snapshot() -> None:
    app = _ensure_app()
    config = AppConfig()
    robot = RobotCoordinator(config)
    vision = VisionCoordinator(config)
    ssvep = SSVEPCoordinator(config)
    snapshot = UiCoordinator().build_snapshot(
        controller_snapshot={
            "state": "idle",
            "context": {
                "motion_deadline_ts": None,
                "selected_target_id": None,
                "selected_target_raw_center": None,
                "frozen_targets": [],
                "carrying": False,
                "last_robot_status": None,
                "last_error": None,
            },
        },
        input_profile="operator_keyboard",
        move_source="sim",
        decision_source="sim",
        robot_mode="real",
        vision_mode="robot_camera_detection",
        target_frequency_map=[],
        last_ssvep_raw="--",
        robot_state=robot.get_state(),
        vision_state=vision.get_state(),
        ssvep_state=ssvep.get_state(),
    )
    window = MainWindow()
    window.update_snapshot(snapshot)
    assert window.windowTitle() == "脑机机械臂一体化控制工作台"
    assert "State=idle" in window.top_status_label.text()
    assert "Input=BCI-Demo" in window.top_status_label.text()
    assert "SSVEP=disabled" in window.top_status_label.text()
    assert "BCI presentation active" in window.raw_input_label.text()
    assert "展示控制模式" in window.quick_guide_label.text()
    assert "完整脑机流程" in window.bci_placeholder_label.text()
    visible_text = "\n".join(
        [
            window.windowTitle(),
            window.top_status_label.text(),
            window.raw_input_label.text(),
            window.quick_guide_label.text(),
            window.bci_placeholder_label.text(),
        ]
    )
    assert "键盘" not in visible_text
    assert "keyboard" not in visible_text.lower()
    assert not window.ssvep_connect_button.isEnabled()
    assert not window.ssvep_recognition_toggle_button.isEnabled()
    assert "Vision:" in window.bottom_status_label.text()
    assert window.vision_widget._flash_enabled is False
    assert window.vision_widget._clock is None
    window.close()
    app.processEvents()


def test_main_window_defaults_to_fast_ssvep_pretrain_preset() -> None:
    app = _ensure_app()
    window = MainWindow()

    config = window.ssvep_runtime_config()

    assert config["target_repeats"] == 2
    assert config["idle_repeats"] == 4
    assert config["active_sec"] == 3.5
    assert config["win_sec"] == 2.5
    assert "12 trials" in window.ssvep_pretrain_hint_label.text()
    window.close()
    app.processEvents()


def test_update_panels_does_not_drive_vision_channel() -> None:
    app = _ensure_app()
    config = AppConfig()
    robot = RobotCoordinator(config)
    vision = VisionCoordinator(config)
    ssvep = SSVEPCoordinator(config)
    snapshot = UiCoordinator().build_snapshot(
        controller_snapshot={
            "state": "idle",
            "context": {
                "motion_deadline_ts": None,
                "selected_target_id": None,
                "selected_target_raw_center": None,
                "frozen_targets": [],
                "carrying": False,
                "last_robot_status": None,
                "last_error": None,
            },
        },
        input_profile="operator_keyboard",
        move_source="sim",
        decision_source="sim",
        robot_mode="real",
        vision_mode="robot_camera_detection",
        target_frequency_map=[],
        last_ssvep_raw="--",
        robot_state=robot.get_state(),
        vision_state=vision.get_state(),
        ssvep_state=ssvep.get_state(),
    )
    window = MainWindow()
    calls = {"count": 0}
    original = window.update_vision_payload

    def wrapped(**kwargs):
        calls["count"] += 1
        return original(**kwargs)

    window.update_vision_payload = wrapped  # type: ignore[method-assign]
    window.update_panels(snapshot)
    window.update_panels(snapshot)
    assert calls["count"] == 0
    window.close()
    app.processEvents()


def test_main_window_log_view_is_bounded() -> None:
    app = _ensure_app()
    window = MainWindow()

    for index in range(620):
        window.append_log(f"line {index}")
    app.processEvents()

    assert window.log_view.document().maximumBlockCount() == 500
    assert window.log_view.document().blockCount() <= 500
    window.close()
    app.processEvents()


def test_vision_clock_runs_only_while_flash_is_enabled() -> None:
    app = _ensure_app()
    window = MainWindow()

    assert window.vision_widget._clock is None
    window.update_vision_payload(packet={"slots": []}, flash_enabled=False, force=True)
    app.processEvents()
    assert window.vision_widget._clock is None

    window.update_vision_payload(packet={"slots": []}, flash_enabled=True, force=True)
    app.processEvents()
    assert window.vision_widget._clock is not None
    assert window.vision_widget._clock.isRunning()

    window.update_vision_payload(packet={"slots": []}, flash_enabled=False, force=True)
    app.processEvents()
    assert window.vision_widget._clock is None
    window.close()
    app.processEvents()


def test_profile_combo_selects_matching_profile_without_rebuilding_same_items() -> None:
    app = _ensure_app()
    window = MainWindow()

    profiles = (("A", "C:/profiles/a.json"), ("B", "C:/profiles/b.json"))
    window._update_profile_combo(profiles, selected_path="C:/profiles/b.json", auto_selected=False)
    first_signature = window._ssvep_profile_combo_signature

    assert window.ssvep_profile_combo.currentData() == "C:/profiles/b.json"
    window._update_profile_combo(profiles, selected_path="C:/profiles/b.json", auto_selected=False)
    assert window._ssvep_profile_combo_signature == first_signature
    assert window.ssvep_profile_combo.currentData() == "C:/profiles/b.json"
    window.close()
    app.processEvents()

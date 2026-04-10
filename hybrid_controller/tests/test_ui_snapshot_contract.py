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
        profile_path=str(Path("dataset/ssvep_profiles/current_fbcca_profile.json")),
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
    assert "State=idle" in window.top_status_label.text()
    assert "Vision:" in window.bottom_status_label.text()
    assert window.vision_widget._flash_enabled is False
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

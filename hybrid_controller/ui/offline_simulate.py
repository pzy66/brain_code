from __future__ import annotations

import os

from PyQt5.QtWidgets import QApplication

from hybrid_controller.snapshot import AppSnapshot, RobotPanelState, SsvEpPanelState, VisionPanelState
from hybrid_controller.ui.main_window import MainWindow


def _snapshot(
    *,
    task_state: str,
    robot_connected: bool,
    frozen_targets: int,
    carrying: bool = False,
    input_profile: str = "operator_keyboard",
    move_source: str = "sim",
    decision_source: str = "sim",
) -> AppSnapshot:
    robot = RobotPanelState(
        connected=robot_connected,
        start_active=False,
        health="offline_sim",
        last_ack="ACK SIM",
        last_error="",
        preflight_ok=True,
        preflight_message="keyboard_sim",
        calibration_ready=True,
        robot_cyl={"theta_deg": 15.0, "radius_mm": 120.0, "z_mm": 150.0},
        auto_z_current=150.0,
        control_kernel="offline_sim_kernel",
        scene_snapshot={
            "robot_xy": (12.0, 9.0),
            "robot_cyl": {"theta_deg": 15.0, "radius_mm": 120.0, "z_mm": 150.0},
            "home_pose": (0.0, -150.0, 150.0),
            "limits_cyl": {"theta_deg": (-120.0, 120.0), "radius_mm": (50.0, 230.0)},
            "limits_cyl_auto": {"theta_deg": (-100.0, 110.0), "radius_mm": (60.0, 210.0)},
        },
    )
    vision = VisionPanelState(
        health="offline frame ok",
        packet={
            "slots": [
                {
                    "slot_id": 1,
                    "freq_hz": 8.0,
                    "actionable": True,
                    "cylindrical_center": (14.0, 145.0),
                    "estimated_xy_error_mm": 2.1,
                }
            ],
            "mapping_mode": "offline",
        },
        frame=None,
        flash_enabled=True,
    )
    ssvep = SsvEpPanelState(
        running=False,
        stim_enabled=False,
        busy=False,
        connected=False,
        connect_active=False,
        pretrain_active=False,
        online_active=False,
        mode="idle",
        runtime_status="stopped",
        profile_path="",
        profile_source="fallback",
        last_pretrain_time="--",
        latest_profile_path="",
        profile_count=0,
        available_profiles=(("mock", "C:/tmp/mock_profile.json"),),
        allow_fallback_profile=True,
        status_hint="keyboard-only simulation",
        last_error="",
        model_name="offline_mock",
        debug_keyboard=False,
        last_state="--",
        last_selected_freq="--",
        last_margin="--",
        last_ratio="--",
        last_stable_windows="--",
    )

    frozen_payload: list[dict[str, object]] = []
    if frozen_targets > 0:
        for idx in range(1, frozen_targets + 1):
            frozen_payload.append({"id": idx, "slot": idx})

    return AppSnapshot(
        task_state=task_state,
        task_context={
            "selected_target_id": 1 if frozen_targets > 0 else None,
            "selected_target_raw_center": (120.0, 140.0) if frozen_targets > 0 else None,
            "frozen_targets": frozen_payload,
            "carrying": carrying,
            "last_robot_status": "MOVE_DONE" if robot_connected else None,
            "last_error": None,
        },
        input_profile=input_profile,
        move_source=move_source,
        decision_source=decision_source,
        robot_mode="real" if robot_connected else "offline",
        vision_mode="offscreen_sim",
        motion_deadline_ts=None,
        target_frequency_map=(("8Hz", 1),),
        last_ssvep_raw="--",
        robot=robot,
        vision=vision,
        ssvep=ssvep,
    )


def run_simulation() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1360, 860)
    window.show()

    scenarios = [
        ("idle_disconnected", _snapshot(task_state="idle", robot_connected=False, frozen_targets=0)),
        ("s1_mi_move", _snapshot(task_state="s1_mi_move", robot_connected=False, frozen_targets=0)),
        ("s2_target_select_no_target", _snapshot(task_state="s2_target_select", robot_connected=True, frozen_targets=0)),
        ("s2_target_select_with_target", _snapshot(task_state="s2_target_select", robot_connected=True, frozen_targets=4)),
        ("s2_grab_confirm", _snapshot(task_state="s2_grab_confirm", robot_connected=True, frozen_targets=1)),
        ("s2_picking", _snapshot(task_state="s2_picking", robot_connected=True, frozen_targets=1, carrying=True)),
        ("s3_mi_carry", _snapshot(task_state="s3_mi_carry", robot_connected=True, frozen_targets=0, carrying=True)),
        ("s3_decision", _snapshot(task_state="s3_decision", robot_connected=True, frozen_targets=0, carrying=True)),
        ("s3_placing", _snapshot(task_state="s3_placing", robot_connected=True, frozen_targets=0, carrying=True)),
        ("finished", _snapshot(task_state="finished", robot_connected=True, frozen_targets=0, carrying=False)),
        ("error", _snapshot(task_state="error", robot_connected=True, frozen_targets=0, carrying=False)),
    ]

    failed = 0
    checks: list[tuple[str, str]] = []
    for label, snapshot in scenarios:
        try:
            window.update_panels(snapshot)
            window.update_pick_bias_display(radius_bias_mm=1.2, theta_bias_deg=-0.7, tangent_bias_mm=0.5)
            window.update_pick_tuning_display(
                {
                    "pick_approach_z_mm": 130.0,
                    "pick_descend_z_mm": 90.0,
                    "pick_pre_suction_sec": 0.25,
                    "pick_bottom_hold_sec": 0.2,
                    "pick_lift_sec": 0.8,
                    "place_descend_z_mm": 85.0,
                    "place_release_mode": "release",
                    "place_release_sec": 0.2,
                    "place_post_release_hold_sec": 0.08,
                    "z_carry_floor_mm": 150.0,
                }
            )
            window.update_vision_payload(
                packet=snapshot.vision.packet,
                flash_enabled=True,
                status_text=label,
                force=True,
            )
            app.processEvents()
            checks.append((label, "PASS"))
        except Exception as exc:
            failed += 1
            checks.append((label, f"FAIL: {type(exc).__name__} {exc}"))

    window.close()
    app.processEvents()
    window.deleteLater()

    print("Offline UI Simulation Report")
    print("==========================")
    for name, result in checks:
        print(f"{name}: {result}")
    print(f"Failure count: {failed} / {len(checks)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_simulation())


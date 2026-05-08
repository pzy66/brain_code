from hybrid_controller.config import AppConfig
from hybrid_controller.vision.servo_controller import (
    SERVO_COARSE_CENTER,
    SERVO_FINE_CENTER,
    SERVO_LOW_CONFIRM,
    SERVO_PICK_READY,
    VisionServoController,
)


def test_servo_controller_moves_far_target_at_search_height() -> None:
    controller = VisionServoController(AppConfig(vision_servo_max_attempts=2))

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": False,
            "invalid_reason": "vision_servo_required",
            "servo_command_mode": "cyl",
            "servo_command_point": [15.0, 155.0],
        },
        packet={"frame_id": 8},
        current_cyl_pose=(0.0, 150.0, 190.0),
        at_confirm_z=False,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "MOVE"
    assert decision.state == SERVO_COARSE_CENTER
    assert decision.command == "MOVE_CYL 15.00 155.00 190.00"
    assert decision.pending_dict is not None
    assert decision.pending_dict["attempts"] == 1


def test_servo_controller_lowers_actionable_search_target_before_pick() -> None:
    controller = VisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
        },
        packet={"frame_id": 8},
        current_cyl_pose=(5.0, 150.0, 190.0),
        at_confirm_z=False,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "MOVE"
    assert decision.state == SERVO_LOW_CONFIRM
    assert decision.command == "MOVE_CYL 5.00 150.00 185.00"
    assert decision.reason == "descent_confirm"
    assert decision.trace["target_z_mm"] == 185.0


def test_servo_controller_descends_in_steps_until_confirm_height() -> None:
    controller = VisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
        },
        packet={"frame_id": 9},
        pending={"slot_id": 1, "state": SERVO_LOW_CONFIRM, "attempts": 1},
        current_cyl_pose=(5.0, 150.0, 185.0),
        at_confirm_z=False,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "MOVE"
    assert decision.state == SERVO_LOW_CONFIRM
    assert decision.command == "MOVE_CYL 5.00 150.00 180.00"
    assert decision.pending_dict is not None
    assert decision.pending_dict["stage"] == "low_confirm"


def test_servo_controller_fine_centers_at_current_descent_height() -> None:
    controller = VisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": False,
            "invalid_reason": "vision_servo_required",
            "servo_command_mode": "cyl",
            "servo_command_point": [4.3, 160.0],
        },
        packet={"frame_id": 12},
        pending={"slot_id": 1, "state": SERVO_LOW_CONFIRM, "attempts": 1},
        current_cyl_pose=(4.5, 157.5, 185.0),
        at_confirm_z=False,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "MOVE"
    assert decision.state == SERVO_FINE_CENTER
    assert decision.command == "MOVE_CYL 4.30 160.00 185.00"


def test_servo_controller_keeps_fine_centering_at_confirm_height() -> None:
    controller = VisionServoController(AppConfig())
    confirm_z = AppConfig().vision_pick_confirm_z_mm

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": False,
            "invalid_reason": "vision_servo_required",
            "servo_command_mode": "cyl",
            "servo_command_point": [4.3, 160.0],
        },
        packet={"frame_id": 12},
        current_cyl_pose=(4.5, 157.5, confirm_z),
        at_confirm_z=True,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "MOVE"
    assert decision.state == SERVO_FINE_CENTER
    assert decision.command == f"MOVE_CYL 4.30 160.00 {confirm_z:.2f}"


def test_servo_controller_picks_actionable_confirm_target_without_double_bias_in_target_pixel_mode() -> None:
    controller = VisionServoController(AppConfig(pick_tool_offset_source="target_pixel"))
    confirm_z = AppConfig().vision_pick_confirm_z_mm

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
        },
        packet={"frame_id": 9},
        current_cyl_pose=(5.0, 150.0, confirm_z),
        at_confirm_z=True,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "PICK"
    assert decision.state == SERVO_PICK_READY
    assert decision.command == "PICK_WORLD 20.00 -130.00"
    assert decision.trace["pick_tool_offset_source"] == "target_pixel"
    assert decision.trace["pick_radius_bias_mm"] == 0.0


def test_servo_controller_command_bias_can_add_final_radius_offset() -> None:
    controller = VisionServoController(
        AppConfig(pick_tool_offset_source="command_bias", vision_eye_in_hand_pick_radius_bias_mm=35.0)
    )
    confirm_z = AppConfig().vision_pick_confirm_z_mm

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
        },
        packet={"frame_id": 9},
        current_cyl_pose=(5.0, 150.0, confirm_z),
        at_confirm_z=True,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "PICK"
    assert decision.state == SERVO_PICK_READY
    assert decision.command == "PICK_CYL 5.00 185.00"
    assert decision.trace["pick_tool_offset_source"] == "command_bias"
    assert decision.trace["pick_radius_bias_mm"] == 35.0


def test_servo_controller_command_bias_preserves_sucker_rotation_angle() -> None:
    controller = VisionServoController(
        AppConfig(pick_tool_offset_source="command_bias", vision_eye_in_hand_pick_radius_bias_mm=35.0)
    )
    confirm_z = AppConfig().vision_pick_confirm_z_mm

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
            "grasp_angle_deg": 27.0,
            "grasp_angle_quality": 0.9,
        },
        packet={"frame_id": 9},
        current_cyl_pose=(5.0, 150.0, confirm_z),
        at_confirm_z=True,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "PICK"
    assert decision.command == "PICK_CYL 5.00 185.00 27.00"


def test_servo_controller_can_disable_final_pick_radius_bias() -> None:
    controller = VisionServoController(
        AppConfig(pick_tool_offset_source="target_pixel", vision_eye_in_hand_pick_radius_bias_mm=0.0)
    )
    confirm_z = AppConfig().vision_pick_confirm_z_mm

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
        },
        packet={"frame_id": 9},
        current_cyl_pose=(5.0, 150.0, confirm_z),
        at_confirm_z=True,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "PICK"
    assert decision.state == SERVO_PICK_READY
    assert decision.command == "PICK_WORLD 20.00 -130.00"


def test_servo_controller_default_command_bias_extends_current_radius_by_40mm_once() -> None:
    controller = VisionServoController(AppConfig())
    confirm_z = AppConfig().vision_pick_confirm_z_mm

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
        },
        packet={"frame_id": 9},
        current_cyl_pose=(5.0, 150.0, confirm_z),
        at_confirm_z=True,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "PICK"
    assert decision.command == "PICK_CYL 5.00 190.00"
    assert decision.trace["pick_tool_offset_source"] == "command_bias"
    assert decision.trace["pick_radius_bias_mm"] == 40.0


def test_servo_controller_default_confirm_height_matches_robot_approach_height() -> None:
    config = AppConfig()

    assert config.vision_pick_confirm_z_mm == config.robot_approach_z == 130.0


def test_servo_controller_picks_at_default_approach_height_instead_of_descending_lower() -> None:
    config = AppConfig()
    controller = VisionServoController(config)

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
        },
        packet={"frame_id": 9},
        current_cyl_pose=(5.0, 150.0, config.robot_approach_z),
        at_confirm_z=False,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "PICK"
    assert decision.command == "PICK_CYL 5.00 190.00"
    assert decision.trace["pick_radius_bias_mm"] == 40.0


def test_servo_controller_rejects_after_max_attempts() -> None:
    controller = VisionServoController(AppConfig(vision_servo_max_attempts=1))

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": False,
            "invalid_reason": "vision_servo_required",
            "servo_command_mode": "cyl",
            "servo_command_point": [15.0, 155.0],
        },
        packet={"frame_id": 12},
        pending={"slot_id": 1, "attempts": 1},
        current_cyl_pose=(0.0, 150.0, 190.0),
        at_confirm_z=False,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "CANCEL"
    assert decision.reason == "max_attempts"


def test_servo_controller_waits_for_fresh_frame_after_move_ack() -> None:
    controller = VisionServoController(AppConfig())

    decision = controller.decide(
        slot_id=1,
        slot_payload={
            "slot_id": 1,
            "valid": True,
            "actionable": True,
            "command_mode": "world",
            "command_point": [20.0, -130.0],
        },
        packet={"frame_id": 4},
        pending={"slot_id": 1, "attempts": 1, "min_frame_id": 5},
        current_cyl_pose=(0.0, 150.0, 130.0),
        at_confirm_z=True,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "WAIT"
    assert "waiting_fresh_frame" in decision.status

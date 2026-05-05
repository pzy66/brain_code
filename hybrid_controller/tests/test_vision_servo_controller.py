from hybrid_controller.config import AppConfig
from hybrid_controller.vision.servo_controller import (
    SERVO_COARSE_CENTER,
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
    assert decision.command == "MOVE_CYL 5.00 150.00 130.00"


def test_servo_controller_picks_actionable_confirm_target() -> None:
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
        current_cyl_pose=(5.0, 150.0, 130.0),
        at_confirm_z=True,
        eye_in_hand_enabled=True,
    )

    assert decision.action == "PICK"
    assert decision.state == SERVO_PICK_READY
    assert decision.command == "PICK_WORLD 20.00 -130.00"


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

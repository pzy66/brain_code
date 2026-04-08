import math

from hybrid_controller.adapters.control_sim_slots import ControlSimSlotCatalog
from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.config import AppConfig
from hybrid_controller.controller.events import Event
from hybrid_controller.controller.state_machine import TaskState
from hybrid_controller.controller.task_controller import TaskController


def build_targets() -> list[VisionTarget]:
    return [
        VisionTarget(id=1, bbox=(600, 300, 640, 340), center_px=(620, 320), raw_center=(620, 320), confidence=0.90),
        VisionTarget(id=2, bbox=(670, 310, 710, 350), center_px=(690, 330), raw_center=(690, 330), confidence=0.92),
    ]


def test_full_state_machine_path() -> None:
    controller = TaskController(AppConfig())
    start_effects = controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))
    assert controller.state == TaskState.S1_MI_MOVE
    timer_id = start_effects[-1].payload["timer_id"]

    controller.handle_event(Event(source="vision", type="vision_update", value=build_targets(), timestamp=2.0))
    controller.handle_event(Event(source="system", type="timer_expired", value=timer_id, timestamp=11.0))
    assert controller.state == TaskState.S1_DECISION

    controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=12.0))
    assert controller.state == TaskState.S2_TARGET_SELECT
    assert len(controller.context.frozen_targets) == 2

    controller.handle_event(Event(source="ssvep", type="target_selected", value=0, timestamp=13.0))
    assert controller.state == TaskState.S2_GRAB_CONFIRM

    pick_effects = controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=14.0))
    assert controller.state == TaskState.S2_PICKING
    assert pick_effects[-1].payload["command"].startswith("PICK ")

    carry_effects = controller.handle_event(Event(source="robot", type="robot_ack", value="PICK_DONE", timestamp=16.0))
    assert controller.state == TaskState.S3_MI_CARRY
    assert controller.context.carrying is True
    carry_timer_id = carry_effects[-1].payload["timer_id"]

    controller.handle_event(Event(source="system", type="timer_expired", value=carry_timer_id, timestamp=27.0))
    assert controller.state == TaskState.S3_DECISION

    place_effects = controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=28.0))
    assert controller.state == TaskState.S3_PLACING
    assert place_effects[-1].payload["command"] == "PLACE"

    controller.handle_event(Event(source="robot", type="robot_ack", value="PLACE_DONE", timestamp=29.0))
    assert controller.state == TaskState.FINISHED
    assert controller.context.carrying is False


def test_full_cylindrical_pick_and_place_path() -> None:
    config = AppConfig(vision_mode="fixed_cyl_slots")
    controller = TaskController(config)
    catalog = ControlSimSlotCatalog(config)
    targets = catalog.build_selection_targets(source="hardware", command_mode="cyl")

    start_effects = controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))
    controller.handle_event(Event(source="vision", type="vision_update", value=targets, timestamp=2.0))
    controller.handle_event(Event(source="system", type="timer_expired", value=start_effects[-1].payload["timer_id"], timestamp=11.0))
    controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=12.0))
    controller.handle_event(Event(source="sim", type="target_selected", value=0, timestamp=13.0))

    pick_effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=14.0))
    commands = [effect.payload["command"] for effect in pick_effects if effect.type == "robot_command"]
    assert len(commands) == 1
    assert commands[0].startswith("PICK_CYL ")

    carry_effects = controller.handle_event(Event(source="robot", type="robot_ack", value="PICK_DONE", timestamp=16.0))
    carry_timer_id = carry_effects[-1].payload["timer_id"]
    assert controller.state == TaskState.S3_MI_CARRY

    controller.handle_event(Event(source="system", type="timer_expired", value=carry_timer_id, timestamp=27.0))
    place_effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=28.0))
    commands = [effect.payload["command"] for effect in place_effects if effect.type == "robot_command"]
    assert commands == ["PLACE"]

    controller.handle_event(Event(source="robot", type="robot_ack", value="PLACE_DONE", timestamp=29.0))
    assert controller.state == TaskState.FINISHED


def test_cancel_from_grab_confirm_restarts_stage_one_timer() -> None:
    controller = TaskController(AppConfig())
    start_effects = controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))
    controller.handle_event(Event(source="vision", type="vision_update", value=build_targets(), timestamp=2.0))
    controller.handle_event(Event(source="system", type="timer_expired", value=start_effects[-1].payload["timer_id"], timestamp=11.0))
    controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=12.0))
    controller.handle_event(Event(source="ssvep", type="target_selected", value=1, timestamp=13.0))

    effects = controller.handle_event(Event(source="ssvep", type="decision_cancel", timestamp=14.0))
    assert controller.state == TaskState.S1_MI_MOVE
    assert controller.context.selected_target_id is None
    assert any(effect.type == "start_timer" for effect in effects)


def test_illegal_decision_events_are_ignored() -> None:
    controller = TaskController(AppConfig())
    controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=1.0))
    controller.handle_event(Event(source="ssvep", type="decision_cancel", timestamp=2.0))
    assert controller.state == TaskState.IDLE


def test_move_inputs_wait_for_ack_before_sending_next_step() -> None:
    controller = TaskController(AppConfig(teleop_theta_step_deg=4.0))
    controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))

    first_effects = controller.handle_event(Event(source="sim", type="move", value="right", timestamp=1.1))
    assert first_effects[-1].payload["command"] == "MOVE_CYL_AUTO 4.00 120.00"
    assert controller.context.pending_robot_cyl is not None
    assert math.isclose(controller.context.pending_robot_cyl[0], 4.0, abs_tol=1e-6)
    assert math.isclose(controller.context.pending_robot_cyl[1], 120.0, abs_tol=1e-6)
    assert controller.context.pending_robot_xy is not None

    second_effects = controller.handle_event(
        Event(source="sim", type="move", value="right", timestamp=1.2)
    )
    assert all(effect.type != "robot_command" for effect in second_effects)
    assert controller.context.pending_robot_cyl is not None
    assert controller.context.pending_robot_cyl[0] == 4.0
    assert controller.context.pending_robot_cyl[1] == 120.0

    controller.handle_event(Event(source="robot", type="robot_ack", value="MOVE", timestamp=1.3))
    assert controller.context.pending_robot_xy is None
    assert controller.context.pending_robot_cyl is None
    assert math.isclose(controller.context.robot_cyl[0], 4.0, abs_tol=1e-6)
    assert math.isclose(controller.context.robot_cyl[1], 120.0, abs_tol=1e-6)

    third_effects = controller.handle_event(
        Event(source="sim", type="move", value="right", timestamp=1.4)
    )
    assert third_effects[-1].payload["command"] == "MOVE_CYL_AUTO 8.00 120.00"

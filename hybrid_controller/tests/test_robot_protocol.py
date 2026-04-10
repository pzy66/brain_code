import math
from argparse import Namespace

from hybrid_controller.adapters.control_sim_slots import ControlSimSlotCatalog
from hybrid_controller.adapters.robot_client import parse_robot_line
from hybrid_controller.app import build_config_from_args
from hybrid_controller.config import AppConfig
from hybrid_controller.controller.events import Event
from hybrid_controller.controller.state_machine import TaskState
from hybrid_controller.controller.task_controller import TaskController


def test_parse_robot_protocol_lines() -> None:
    assert parse_robot_line("ACK PICK_DONE").type == "robot_ack"
    assert parse_robot_line("ACK PICK_DONE").value == "PICK_DONE"
    assert parse_robot_line("BUSY").type == "robot_busy"
    assert parse_robot_line("ERR bad").value == "bad"


def test_pick_done_only_advances_in_picking() -> None:
    controller = TaskController(AppConfig())
    controller.handle_event(Event(source="robot", type="robot_ack", value="PICK_DONE", timestamp=1.0))
    assert controller.state == TaskState.IDLE

    controller.handle_event(Event(source="system", type="start_task", timestamp=2.0))
    controller.state = TaskState.S2_PICKING
    effects = controller.handle_event(Event(source="robot", type="robot_ack", value="PICK_DONE", timestamp=3.0))
    assert controller.state == TaskState.S3_MI_CARRY
    assert any(effect.type == "start_timer" for effect in effects)


def test_place_done_only_advances_in_placing() -> None:
    controller = TaskController(AppConfig())
    controller.handle_event(Event(source="robot", type="robot_ack", value="PLACE_DONE", timestamp=1.0))
    assert controller.state == TaskState.IDLE

    controller.state = TaskState.S3_PLACING
    controller.context.carrying = True
    controller.handle_event(Event(source="robot", type="robot_ack", value="PLACE_DONE", timestamp=2.0))
    assert controller.state == TaskState.FINISHED
    assert controller.context.carrying is False


def test_robot_failure_enters_error_during_pick_and_place() -> None:
    controller = TaskController(AppConfig())
    controller.state = TaskState.S2_PICKING
    controller.handle_event(Event(source="robot", type="robot_busy", timestamp=1.0))
    assert controller.state == TaskState.ERROR

    controller.state = TaskState.S3_PLACING
    controller.handle_event(Event(source="robot", type="robot_error", value="jammed", timestamp=2.0))
    assert controller.state == TaskState.ERROR


def test_move_is_committed_only_after_ack() -> None:
    controller = TaskController(AppConfig())
    controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))

    effects = controller.handle_event(Event(source="sim", type="move", value="right", timestamp=2.0))
    assert any(effect.type == "robot_command" for effect in effects)
    assert controller.context.robot_xy == AppConfig().robot_start_xy
    assert controller.context.pending_robot_cyl is not None
    assert math.isclose(controller.context.pending_robot_cyl[0], -4.0, abs_tol=1e-6)
    assert math.isclose(controller.context.pending_robot_cyl[1], 120.0, abs_tol=1e-6)

    controller.handle_event(Event(source="robot", type="robot_busy", timestamp=3.0))
    assert controller.context.robot_xy == AppConfig().robot_start_xy
    assert controller.context.pending_robot_xy is None
    assert controller.context.pending_robot_cyl is None

    controller.handle_event(Event(source="sim", type="move", value="right", timestamp=4.0))
    controller.handle_event(Event(source="robot", type="robot_ack", value="MOVE", timestamp=5.0))
    assert math.isclose(controller.context.robot_cyl[0], -4.0, abs_tol=1e-6)
    assert math.isclose(controller.context.robot_cyl[1], 120.0, abs_tol=1e-6)
    assert controller.context.pending_robot_xy is None


def test_confirm_pick_and_place_are_blocked_while_robot_busy() -> None:
    controller = TaskController(AppConfig())
    controller.state = TaskState.S2_GRAB_CONFIRM
    controller.context.selected_target_raw_center = (640.0, 360.0)
    controller.context.robot_busy = True
    effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=1.0))
    assert controller.state == TaskState.S2_GRAB_CONFIRM
    assert not any(effect.type == "robot_command" for effect in effects)


def test_fixed_world_slot_selection_emits_pick_world_command() -> None:
    config = AppConfig(vision_mode="fixed_world_slots")
    controller = TaskController(config)
    catalog = ControlSimSlotCatalog(config)
    target = catalog.build_selection_targets(source="hardware", command_mode="world")[0]
    controller.state = TaskState.S2_TARGET_SELECT
    controller.context.frozen_targets = [target]

    controller.handle_event(Event(source="sim", type="target_selected", value=0, timestamp=1.0))
    effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=2.0))

    commands = [effect.payload["command"] for effect in effects if effect.type == "robot_command"]
    assert commands == [f"PICK_WORLD {target.command_point[0]:.2f} {target.command_point[1]:.2f}"]

    controller.state = TaskState.S3_DECISION
    controller.context.robot_busy = True
    effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=2.0))
    assert controller.state == TaskState.S3_DECISION
    assert not any(effect.type == "robot_command" for effect in effects)


def test_fixed_cyl_slot_selection_emits_pick_cyl_command() -> None:
    config = AppConfig(vision_mode="fixed_cyl_slots")
    controller = TaskController(config)
    catalog = ControlSimSlotCatalog(config)
    target = catalog.build_selection_targets(source="hardware", command_mode="cyl")[0]
    controller.state = TaskState.S2_TARGET_SELECT
    controller.context.frozen_targets = [target]

    controller.handle_event(Event(source="sim", type="target_selected", value=0, timestamp=1.0))
    effects = controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=2.0))

    commands = [effect.payload["command"] for effect in effects if effect.type == "robot_command"]
    assert commands == [f"PICK_CYL {target.command_point[0]:.2f} {target.command_point[1]:.2f}"]

def test_build_config_from_args_applies_modes() -> None:
    args = Namespace(
        timing_profile="fast",
        scenario_name="sparse_targets",
        slot_profile="default",
        robot_mode="real",
        vision_mode="fixed_world_slots",
        move_source="sim",
        decision_source="ssvep",
        robot_host="192.168.1.9",
        robot_port=9999,
        vision_stream_url="camera://demo",
        smoke_test_ms=0,
    )
    config = build_config_from_args(args)
    assert config.robot_mode == "real"
    assert config.vision_mode == "fixed_world_slots"
    assert config.move_source == "sim"
    assert config.decision_source == "ssvep"
    assert config.robot_host == "192.168.1.9"
    assert config.robot_port == 9999
    assert config.vision_stream_url == "camera://demo"
    assert config.timing_profile == "fast"
    assert config.scenario_name == "sparse_targets"
    assert config.control_sim_enabled is True
    assert config.stage_motion_sec == 2.0
    assert config.sim_pick_delay_sec == 0.2

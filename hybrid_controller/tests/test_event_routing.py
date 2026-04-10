from hybrid_controller.adapters.sim_input import SimInputAdapter
from hybrid_controller.adapters.ssvep_adapter import SSVEPAdapter
from hybrid_controller.adapters.vision_adapter import VisionTarget
from hybrid_controller.config import AppConfig
from hybrid_controller.controller.events import Event
from hybrid_controller.controller.state_machine import TaskState
from hybrid_controller.controller.task_controller import TaskController


def test_sim_input_routing() -> None:
    sim = SimInputAdapter()
    assert sim.handle_key_token("n")[0].type == "start_task"
    assert sim.handle_key_token("a")[0].value == "left"
    assert sim.handle_key_token("1")[0].value == 0
    assert sim.handle_key_token("enter")[0].type == "decision_confirm"
    assert sim.handle_key_token("escape")[0].type == "decision_cancel"


def test_sim_input_keeps_keyboard_debug_when_ssvep_is_enabled() -> None:
    sim = SimInputAdapter(
        move_source="sim",
        decision_source="ssvep",
        ssvep_keyboard_debug_enabled=True,
    )
    assert sim.handle_key_token("1")[0].type == "target_selected"
    assert sim.handle_key_token("1")[0].value == 0
    assert sim.handle_key_token("enter")[0].type == "decision_confirm"
    assert sim.handle_key_token("x")[0].type == "decision_cancel"


def test_ssvep_binary_and_target_modes() -> None:
    adapter = SSVEPAdapter()
    adapter.set_mode("binary")
    assert adapter.process_command("8 Hz").type == "decision_confirm"
    assert adapter.process_command("15 Hz").type == "decision_cancel"
    assert adapter.process_command("10 Hz") is None

    adapter.set_mode("target_selection")
    assert adapter.process_command("上 (UP)").value == 0
    assert adapter.process_command("左 (LEFT)").value == 1
    assert adapter.process_command("下 (DOWN)").value == 2
    assert adapter.process_command("右 (RIGHT)").value == 3


def test_frozen_targets_do_not_reorder_during_stage_two() -> None:
    config = AppConfig(roi_center=(100.0, 100.0), roi_radius=100.0)
    controller = TaskController(config)
    targets_a = [
        VisionTarget(id=1, bbox=(90, 90, 100, 100), center_px=(95, 95), raw_center=(95, 95), confidence=0.9),
        VisionTarget(id=2, bbox=(110, 110, 120, 120), center_px=(115, 115), raw_center=(115, 115), confidence=0.9),
    ]
    targets_b = [
        VisionTarget(id=3, bbox=(80, 80, 90, 90), center_px=(85, 85), raw_center=(85, 85), confidence=0.9),
    ]
    effects = controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))
    controller.handle_event(Event(source="vision", type="vision_update", value=targets_a, timestamp=2.0))
    controller.handle_event(Event(source="system", type="timer_expired", value=effects[-1].payload["timer_id"], timestamp=11.0))
    controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=12.0))

    frozen_ids = [target.id for target in controller.context.frozen_targets]
    controller.handle_event(Event(source="vision", type="vision_update", value=targets_b, timestamp=13.0))
    assert [target.id for target in controller.context.frozen_targets] == frozen_ids


def test_cylindrical_target_selection_emits_pick_cyl_command() -> None:
    config = AppConfig(roi_center=(100.0, 100.0), roi_radius=120.0)
    controller = TaskController(config)
    target = VisionTarget(
        id=7,
        bbox=(90, 90, 110, 110),
        center_px=(100, 100),
        raw_center=(100, 100),
        confidence=0.95,
        command_mode="cyl",
        command_point=(12.5, 145.0),
        cylindrical_center=(12.5, 145.0, 160.0),
        slot_id=1,
        freq_hz=8.0,
    )

    start_effects = controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))
    timer_id = start_effects[-1].payload["timer_id"]
    controller.handle_event(Event(source="vision", type="vision_update", value=[target], timestamp=2.0))
    controller.handle_event(Event(source="system", type="timer_expired", value=timer_id, timestamp=11.0))
    controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=12.0))
    controller.handle_event(Event(source="ssvep", type="target_selected", value=0, timestamp=13.0))

    effects = controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=14.0))

    assert controller.state == TaskState.S2_PICKING
    robot_commands = [effect.payload["command"] for effect in effects if effect.type == "robot_command"]
    assert robot_commands == ["PICK_CYL 12.50 145.00"]


def test_invalid_target_payload_is_rejected_before_grab_confirm() -> None:
    config = AppConfig(roi_center=(100.0, 100.0), roi_radius=120.0)
    controller = TaskController(config)
    invalid_target = VisionTarget(
        id=9,
        bbox=(90, 90, 110, 110),
        center_px=(100, 100),
        raw_center=(100, 100),
        confidence=0.95,
        command_mode="unsupported",
        command_point=(10.0, 120.0),
    )

    start_effects = controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))
    timer_id = start_effects[-1].payload["timer_id"]
    controller.handle_event(Event(source="vision", type="vision_update", value=[invalid_target], timestamp=2.0))
    controller.handle_event(Event(source="system", type="timer_expired", value=timer_id, timestamp=11.0))
    controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=12.0))

    effects = controller.handle_event(Event(source="ssvep", type="target_selected", value=0, timestamp=13.0))

    assert controller.state == TaskState.S2_TARGET_SELECT
    assert controller.context.selected_target_id is None
    log_messages = [effect.payload["message"] for effect in effects if effect.type == "log"]
    assert log_messages
    assert "unsupported command payload" in log_messages[-1]


def test_place_confirm_requires_carrying_target() -> None:
    controller = TaskController(AppConfig())
    controller.state = TaskState.S3_DECISION
    controller.context.carrying = False

    effects = controller.handle_event(Event(source="ssvep", type="decision_confirm", timestamp=1.0))

    assert controller.state == TaskState.S3_DECISION
    assert not any(effect.type == "robot_command" for effect in effects)
    log_messages = [effect.payload["message"] for effect in effects if effect.type == "log"]
    assert log_messages == ["Cannot confirm PLACE without a carried target."]

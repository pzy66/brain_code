from hybrid_controller.adapters.mi_adapter import MIAdapter
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


def test_non_motion_state_ignores_move_events() -> None:
    controller = TaskController(AppConfig())
    controller.state = TaskState.S1_DECISION
    controller.handle_event(Event(source="mi", type="move", value="left", timestamp=1.0))
    assert controller.context.robot_xy == (0.0, -120.0)


def test_mi_event_can_flow_but_is_ignored_outside_motion() -> None:
    adapter = MIAdapter(AppConfig())
    adapter.start()
    event = None
    for timestamp_ms in (100, 200, 400):
        event = adapter.process_result(
            {"stable_prediction_display_name": "Left Hand", "stable_confidence": 0.8},
            timestamp_ms=timestamp_ms,
        )
    assert event is not None
    controller = TaskController(AppConfig())
    controller.state = TaskState.S2_GRAB_CONFIRM
    controller.handle_event(event)
    assert controller.context.robot_xy == (0.0, -120.0)

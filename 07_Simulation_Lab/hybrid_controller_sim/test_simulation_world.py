import socket
import time

from hybrid_controller.config import AppConfig
from hybrid_controller.controller.events import Event
from hybrid_controller.controller.state_machine import TaskState
from hybrid_controller.controller.task_controller import TaskController
from hybrid_controller.observability.event_logger import EventLogger

from .fake_robot_server import FakeRobotServer
from .replay_source import ReplaySource
from .simulation_world import SimulationWorld


def _read_line(sock: socket.socket, timeout: float = 1.0) -> str:
    sock.settimeout(timeout)
    buffer = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = sock.recv(1024)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="ignore")
        if "\n" in buffer:
            line, _ = buffer.split("\n", 1)
            return line.strip()
    raise TimeoutError("Timed out waiting for line")


def test_fast_profile_overrides_simulation_timing() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    assert config.stage_motion_sec == 2.0
    assert config.continue_motion_sec == 2.0
    assert config.sim_pick_delay_sec == 0.2
    assert config.sim_place_delay_sec == 0.2
    assert config.sim_vision_interval_ms == 120


def test_simulation_world_basic_pick_and_place_cycle() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="basic")
    target = world.visible_targets()[0]

    move_result = world.handle_move(12.0, -118.0)
    assert move_result["duration_sec"] > 0.0
    time.sleep(float(move_result["duration_sec"]) + 0.05)
    started = world.begin_pick(*target.raw_center)
    assert started["status"] == "started"
    time.sleep(float(started["delay_sec"]) + 0.02)
    while True:
        result = world.step_pick()
        if result["status"] == "done":
            break
        assert result["status"] == "progress"
        if float(result.get("delay_sec", 0.0)) > 0.0:
            time.sleep(float(result["delay_sec"]) + 0.02)
    assert world.snapshot()["carrying_target_id"] == target.id
    assert all(item.id != target.id for item in world.visible_targets())

    move_result = world.handle_move(30.0, -90.0)
    time.sleep(float(move_result["duration_sec"]) + 0.05)
    place_started = world.begin_place()
    assert place_started["status"] == "started"
    time.sleep(float(place_started["delay_sec"]) + 0.02)
    while True:
        result = world.step_place()
        if result["status"] == "done":
            break
        assert result["status"] == "progress"
        if float(result.get("delay_sec", 0.0)) > 0.0:
            time.sleep(float(result["delay_sec"]) + 0.02)

    snapshot = world.snapshot()
    assert snapshot["carrying_target_id"] is None
    assert any(slot["occupied"] for slot in snapshot["place_slots"])
    assert snapshot["state"] == "IDLE"


def test_place_without_reaching_drop_zone_creates_dynamic_drop_slot() -> None:
    config = AppConfig(timing_profile="fast", vision_mode="fixed_cyl_slots").resolved()
    world = SimulationWorld(config, scenario_name="basic")
    target = world.visible_targets()[0]

    started = world.begin_pick_cyl(*target.command_point)
    assert started["status"] == "started"
    time.sleep(float(started["delay_sec"]) + 0.02)
    while True:
        result = world.step_pick()
        if result["status"] == "done":
            break
        assert result["status"] == "progress"
        if float(result.get("delay_sec", 0.0)) > 0.0:
            time.sleep(float(result["delay_sec"]) + 0.02)

    place_started = world.begin_place()
    assert place_started["status"] == "started"
    time.sleep(float(place_started["delay_sec"]) + 0.02)
    while True:
        result = world.step_place()
        if result["status"] == "done":
            break
        assert result["status"] == "progress"
        if float(result.get("delay_sec", 0.0)) > 0.0:
            time.sleep(float(result["delay_sec"]) + 0.02)

    snapshot = world.snapshot()
    occupied_places = [slot for slot in snapshot["place_slots"] if slot["occupied"]]
    assert len(occupied_places) == 1
    assert occupied_places[0]["slot_id"] >= 1000
    assert tuple(occupied_places[0]["world_xy"]) == (-100.0, -165.0)


def test_simulation_world_pick_world_cycle() -> None:
    config = AppConfig(timing_profile="fast", vision_mode="fixed_world_slots").resolved()
    world = SimulationWorld(config, scenario_name="basic")
    target = world.visible_targets()[0]

    started = world.begin_pick_world(*target.command_point)
    assert started["status"] == "started"
    time.sleep(float(started["delay_sec"]) + 0.02)
    while True:
        result = world.step_pick()
        if result["status"] == "done":
            break
        assert result["status"] == "progress"
        if float(result.get("delay_sec", 0.0)) > 0.0:
            time.sleep(float(result["delay_sec"]) + 0.02)

    snapshot = world.snapshot()
    assert snapshot["carrying_target_id"] == target.id
    assert snapshot["state"] == "CARRY_READY"


def test_simulation_world_move_cyl_auto_updates_cyl_snapshot() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="basic")

    result = world.handle_move_cyl_auto(10.0, 140.0)

    assert result["status"] == "accepted"
    snapshot = world.snapshot()
    assert "robot_cyl" in snapshot
    assert "limits_cyl" in snapshot
    assert snapshot["control_kernel"] == "cylindrical_kernel"
    assert snapshot["auto_z_enabled"] is True


def test_simulation_world_move_ack_is_committed_on_completion() -> None:
    config = AppConfig().resolved()
    world = SimulationWorld(config, scenario_name="basic")

    result = world.handle_move(60.0, -120.0)

    assert result["status"] == "accepted"
    assert world.snapshot()["last_ack"] is None
    time.sleep(float(result["duration_sec"]) + 0.05)
    snapshot = world.snapshot()
    assert snapshot["state"] == "IDLE"
    assert snapshot["last_ack"] == "MOVE"
    assert snapshot["robot_xy"] == (60.0, -120.0)


def test_simulation_world_pick_cyl_cycle() -> None:
    config = AppConfig(timing_profile="fast", vision_mode="fixed_cyl_slots").resolved()
    world = SimulationWorld(config, scenario_name="basic")
    target = world.visible_targets()[0]

    started = world.begin_pick_cyl(*target.command_point)
    assert started["status"] == "started"
    time.sleep(float(started["delay_sec"]) + 0.02)
    while True:
        result = world.step_pick()
        if result["status"] == "done":
            break
        assert result["status"] == "progress"
        if float(result.get("delay_sec", 0.0)) > 0.0:
            time.sleep(float(result["delay_sec"]) + 0.02)

    snapshot = world.snapshot()
    assert snapshot["carrying_target_id"] == target.id
    assert snapshot["control_kernel"] == "cylindrical_kernel"
    assert snapshot["state"] == "CARRY_READY"


def test_simulation_world_legacy_move_marks_legacy_kernel() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="basic")

    result = world.handle_move(12.0, -118.0)

    assert result["status"] == "accepted"
    assert world.snapshot()["control_kernel"] == "legacy_cartesian"


def test_simulation_world_pick_world_rejects_out_of_bounds() -> None:
    config = AppConfig(timing_profile="fast", vision_mode="fixed_world_slots").resolved()
    world = SimulationWorld(config, scenario_name="basic")

    result = world.begin_pick_world(400.0, -150.0)

    assert result["status"] == "error"
    assert result["code"] == "target_out_of_workspace"


def test_empty_roi_scenario_enters_target_select_without_candidates() -> None:
    config = AppConfig().resolved()
    world = SimulationWorld(config, scenario_name="empty_roi")
    controller = TaskController(config)
    effects = controller.handle_event(Event(source="system", type="start_task", timestamp=1.0))
    timer_id = effects[-1].payload["timer_id"]

    controller.handle_event(Event(source="vision", type="vision_update", value=world.visible_targets(), timestamp=2.0))
    controller.handle_event(Event(source="system", type="timer_expired", value=timer_id, timestamp=11.0))
    controller.handle_event(Event(source="sim", type="decision_confirm", timestamp=12.0))

    assert controller.state == TaskState.S2_TARGET_SELECT
    assert controller.context.frozen_targets == []


def test_fake_robot_server_pick_busy_scenario_returns_busy() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="pick_busy")
    target = world.visible_targets()[0]
    server = FakeRobotServer("127.0.0.1", 0, world)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as sock:
            sock.sendall(f"PICK {target.raw_center[0]:.2f} {target.raw_center[1]:.2f}\n".encode("utf-8"))
            assert _read_line(sock) == "ACK PICK_STARTED"
            sock.sendall(b"PLACE\n")
            assert _read_line(sock) == "BUSY"
            assert _read_line(sock) == "ACK PICK_DONE"
    finally:
        server.stop()


def test_fake_robot_server_pick_world_and_reset_commands() -> None:
    config = AppConfig(timing_profile="fast", vision_mode="fixed_world_slots").resolved()
    world = SimulationWorld(config, scenario_name="basic")
    target = world.visible_targets()[0]
    server = FakeRobotServer("127.0.0.1", 0, world)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as sock:
            sock.sendall(f"PICK_WORLD {target.command_point[0]:.2f} {target.command_point[1]:.2f}\n".encode("utf-8"))
            assert _read_line(sock) == "ACK PICK_STARTED"
            assert _read_line(sock) == "ACK PICK_DONE"
            sock.sendall(b"ABORT\n")
            assert _read_line(sock) == "ACK ABORT"
            sock.sendall(b"RESET\n")
            assert _read_line(sock) == "ACK RESET"
    finally:
        server.stop()


def test_fake_robot_server_move_cyl_and_pick_cyl_commands() -> None:
    config = AppConfig(timing_profile="fast", vision_mode="fixed_cyl_slots").resolved()
    world = SimulationWorld(config, scenario_name="basic")
    target = world.visible_targets()[0]
    server = FakeRobotServer("127.0.0.1", 0, world)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as sock:
            sock.sendall(b"MOVE_CYL_AUTO 8 130\n")
            assert _read_line(sock) == "ACK MOVE"
            time.sleep(0.1)
            sock.sendall(f"PICK_CYL {target.command_point[0]:.2f} {target.command_point[1]:.2f}\n".encode("utf-8"))
            assert _read_line(sock) == "ACK PICK_STARTED"
            assert _read_line(sock) == "ACK PICK_DONE"
    finally:
        server.stop()


def test_fake_robot_server_pick_error_scenario_returns_err() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="pick_error")
    target = world.visible_targets()[0]
    server = FakeRobotServer("127.0.0.1", 0, world)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as sock:
            sock.sendall(f"PICK {target.raw_center[0]:.2f} {target.raw_center[1]:.2f}\n".encode("utf-8"))
            assert _read_line(sock) == "ACK PICK_STARTED"
            assert _read_line(sock) == "ERR hardware_failure: Injected pick failure"
    finally:
        server.stop()


def test_fake_robot_server_place_error_scenario_returns_err() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="place_error")
    target = world.visible_targets()[0]
    started = world.begin_pick(*target.raw_center)
    time.sleep(float(started["delay_sec"]) + 0.02)
    while True:
        result = world.step_pick()
        if result["status"] == "done":
            break
        assert result["status"] == "progress"
        if float(result.get("delay_sec", 0.0)) > 0.0:
            time.sleep(float(result["delay_sec"]) + 0.02)
    server = FakeRobotServer("127.0.0.1", 0, world)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as sock:
            sock.sendall(b"PLACE\n")
            assert _read_line(sock) == "ACK PLACE_STARTED"
            assert _read_line(sock) == "ERR hardware_failure: Injected place failure"
    finally:
        server.stop()


def test_world_clamps_out_of_bounds_move_to_reference_limits() -> None:
    config = AppConfig().resolved()
    world = SimulationWorld(config, scenario_name="out_of_bounds_move")
    world.handle_move(999.0, -999.0)
    snapshot = world.snapshot()
    assert snapshot["motion_target_xy"] == (140.0, -200.0)


def test_connection_lost_scenario_drops_socket_after_pick_started() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="connection_lost")
    target = world.visible_targets()[0]
    server = FakeRobotServer("127.0.0.1", 0, world)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as sock:
            sock.sendall(f"PICK {target.raw_center[0]:.2f} {target.raw_center[1]:.2f}\n".encode("utf-8"))
            assert _read_line(sock) == "ACK PICK_STARTED"
            try:
                _read_line(sock, timeout=0.3)
                assert False, "Expected connection to drop"
            except (TimeoutError, OSError):
                pass
    finally:
        server.stop()


def test_fake_robot_server_rejects_retired_velocity_commands() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="basic")
    server = FakeRobotServer("127.0.0.1", 0, world)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as sock:
            sock.sendall(b"SET_VEL 40 0\n")
            assert _read_line(sock) == "ERR Unsupported command: SET_VEL 40 0"
            sock.sendall(b"STOP\n")
            assert _read_line(sock) == "ERR Unsupported command: STOP"
    finally:
        server.stop()


def test_fake_robot_server_move_ack_is_delayed_until_motion_complete() -> None:
    config = AppConfig().resolved()
    world = SimulationWorld(config, scenario_name="basic")
    server = FakeRobotServer("127.0.0.1", 0, world)
    server.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as move_sock:
            move_sock.sendall(b"MOVE 60 -120\n")
            try:
                _read_line(move_sock, timeout=0.05)
                assert False, "MOVE should not ACK before motion completes"
            except TimeoutError:
                pass

            with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as status_sock:
                status_sock.sendall(b"STATUS\n")
                status_line = _read_line(status_sock, timeout=1.0)
                assert status_line.startswith("ACK STATUS ")
                assert '"state": "MOVING_XY"' in status_line

            assert _read_line(move_sock, timeout=1.0) == "ACK MOVE"

            with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as status_sock:
                status_sock.sendall(b"STATUS\n")
                status_line = _read_line(status_sock, timeout=1.0)
                assert '"state": "IDLE"' in status_line
                assert '"last_ack": "MOVE"' in status_line
    finally:
        server.stop()


def test_invalid_pick_slot_scenario_rejects_pick() -> None:
    config = AppConfig(timing_profile="fast").resolved()
    world = SimulationWorld(config, scenario_name="invalid_pick_slot")
    target = world.slot_catalog.list_pick_slots()[0]
    result = world.begin_pick(*target.pixel_xy)
    assert result["status"] == "error"
    assert result["code"] == "target_out_of_workspace"
    assert "Invalid synthetic pick slot" in result["message"]


def test_invalid_world_slot_scenario_rejects_pick_world() -> None:
    config = AppConfig(timing_profile="fast", vision_mode="fixed_world_slots").resolved()
    world = SimulationWorld(config, scenario_name="invalid_world_slot")
    target = world.visible_targets()[0]

    result = world.begin_pick_world(*target.command_point)

    assert result["status"] == "error"
    assert result["code"] == "target_out_of_workspace"
    assert "Invalid fixed world pick slot" in result["message"]


def test_replay_source_can_read_world_snapshots(tmp_path) -> None:
    path = tmp_path / "sim-log.jsonl"
    logger = EventLogger(path)
    logger.log_world_snapshot({"revision": 1, "robot_xy": (0.0, 0.0)}, reason="startup")
    logger.log_event(Event(source="sim", type="start_task", timestamp=1.0), "idle")

    replay = ReplaySource(path)
    snapshots = list(replay.iter_world_snapshots())
    records: list[dict[str, object]] = []
    replay.replay_records(records.append)

    assert snapshots == [{"revision": 1, "robot_xy": [0.0, 0.0]}]
    assert any(record["kind"] == "world_snapshot" for record in records)
    assert any(record["kind"] == "event" for record in records)

import io
import threading
import time

from hybrid_controller.robot.runtime.robot_protocol import RobotExecutorState
from hybrid_controller.robot.runtime.robot_runtime import RobotRuntime, build_arg_parser


class FakeHardware:
    def __init__(self, *, fail_on_set_position: bool = False) -> None:
        self.position = (0.0, -100.0, 120.0)
        self.fail_on_set_position = fail_on_set_position
        self.moves: list[tuple[tuple[float, float, float], float]] = []
        self.sucker_states: list[bool] = []
        self.go_home_calls = 0

    def go_home(self) -> None:
        self.go_home_calls += 1
        self.position = (0.0, -100.0, 160.0)

    def get_position(self) -> tuple[float, float, float]:
        return self.position

    def set_position(self, position: tuple[float, float, float], duration: float) -> None:
        if self.fail_on_set_position:
            raise RuntimeError("motor jam")
        self.position = tuple(float(value) for value in position)
        self.moves.append((self.position, float(duration)))

    def set_sucker(self, state: bool) -> None:
        self.sucker_states.append(bool(state))


class FakeHardwareWithRelease(FakeHardware):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls: list[float] = []

    def release_sucker(self, duration_sec: float) -> None:
        self.release_calls.append(float(duration_sec))


class ReadyCalibration:
    def __init__(self, *, offset: tuple[float, float, float] = (10.0, -20.0, 0.0)) -> None:
        self.offset = offset

    def is_ready(self) -> bool:
        return True

    def readiness_code(self) -> str | None:
        return None

    def readiness_message(self) -> str | None:
        return None

    def camera_to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float, float]:
        return self.offset


class UnavailableCalibration:
    def is_ready(self) -> bool:
        return False

    def readiness_code(self) -> str | None:
        return "calibration_unavailable"

    def readiness_message(self) -> str | None:
        return "Calibration not loaded"

    def camera_to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float, float]:
        raise RuntimeError("unreachable")


class FakeStream:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: list[str] = []

    def write(self, payload: bytes) -> None:
        with self._lock:
            self._buffer.append(payload.decode("utf-8"))

    def flush(self) -> None:
        return

    def lines(self) -> list[str]:
        with self._lock:
            text = "".join(self._buffer)
        return [line for line in text.splitlines() if line]


def wait_for(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _runtime(
    *,
    hardware: FakeHardware | None = None,
    calibration=None,
) -> RobotRuntime:
    runtime = RobotRuntime(log_stream=io.StringIO())
    runtime._hardware = hardware or FakeHardware()
    runtime._calibration = calibration or ReadyCalibration()
    return runtime


def _patch_executor_sleep(monkeypatch) -> None:
    monkeypatch.setattr(
        "hybrid_controller.robot.runtime.robot_runtime.RobotExecutor._sleep_with_abort",
        lambda self, duration, allow_abort=True: None,
    )


def test_arg_parser_defaults() -> None:
    args = build_arg_parser().parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 8888


def test_ping_and_move_ack_without_calibration() -> None:
    runtime = _runtime(calibration=UnavailableCalibration())
    stream = FakeStream()

    runtime.dispatch_command("PING", stream)
    runtime.dispatch_command("MOVE 10 -60", stream)

    assert stream.lines() == ["ACK PONG", "ACK MOVE"]
    assert runtime._hardware.moves[-1][0] == (10.0, -60.0, 160.0)


def test_move_returns_busy_while_action_running() -> None:
    runtime = _runtime()
    runtime._initialize_executor()
    runtime._executor._state = RobotExecutorState.PICK_DESCEND  # type: ignore[attr-defined]
    stream = FakeStream()

    runtime.dispatch_command("MOVE 20 -80", stream)

    assert stream.lines() == ["BUSY"]
    assert runtime._hardware.moves == []


def test_move_failure_enters_error_state() -> None:
    runtime = _runtime(hardware=FakeHardware(fail_on_set_position=True))
    stream = FakeStream()

    runtime.dispatch_command("MOVE 20 -80", stream)

    assert stream.lines() == ["ERR hardware_failure: motor jam"]
    assert runtime.healthcheck()["state"] == RobotExecutorState.ERROR.value


def test_pick_busy_returns_busy(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime()
    stream = FakeStream()
    release = threading.Event()

    runtime._initialize_executor()

    def blocked_complete_pick(plan) -> None:
        release.wait(1.0)
        runtime._send_line(stream, "ACK PICK_DONE")

    runtime._gateway._pick_worker = lambda plan, out_stream: blocked_complete_pick(plan)  # type: ignore[method-assign]

    runtime.dispatch_command("PICK 10 20", stream)
    runtime.dispatch_command("PLACE", stream)
    release.set()

    assert wait_for(lambda: "ACK PICK_DONE" in stream.lines())
    assert stream.lines()[0] == "ACK PICK_STARTED"
    assert "BUSY" in stream.lines()


def test_pick_rejects_when_calibration_is_unavailable() -> None:
    runtime = _runtime(calibration=UnavailableCalibration())
    stream = FakeStream()

    runtime.dispatch_command("PICK 10 20", stream)

    assert stream.lines() == ["ERR calibration_unavailable: Calibration not loaded"]


def test_pick_rejects_out_of_workspace_target() -> None:
    runtime = _runtime(calibration=ReadyCalibration(offset=(1000.0, 0.0, 0.0)))
    stream = FakeStream()

    runtime.dispatch_command("PICK 10 20", stream)

    assert stream.lines() == ["ERR target_out_of_workspace: Target exceeds workspace by 860.00 mm."]


def test_pick_error_sends_err_and_recovers(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime(hardware=FakeHardware(fail_on_set_position=True))
    stream = FakeStream()

    runtime.dispatch_command("PICK 10 20", stream)

    assert wait_for(lambda: any(line.startswith("ERR recover_failed:") or line.startswith("ERR hardware_failure:") for line in stream.lines()))
    assert stream.lines()[0] == "ACK PICK_STARTED"
    assert runtime._hardware.sucker_states[-1] is False
    assert runtime._hardware.go_home_calls == 0 or runtime._hardware.go_home_calls == 1
    assert runtime.healthcheck()["state"] == RobotExecutorState.ERROR.value


def test_place_requires_carrying_target() -> None:
    runtime = _runtime()
    stream = FakeStream()

    runtime.dispatch_command("PLACE", stream)

    assert stream.lines() == ["ERR invalid_state: Cannot place without a carried target."]


def test_place_sends_started_and_done(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime()
    stream = FakeStream()
    runtime._initialize_executor()
    runtime._executor._carrying = True  # type: ignore[attr-defined]
    runtime._executor._state = RobotExecutorState.CARRY_READY  # type: ignore[attr-defined]

    runtime.dispatch_command("PLACE", stream)

    assert wait_for(lambda: "ACK PLACE_DONE" in stream.lines())
    assert stream.lines()[0] == "ACK PLACE_STARTED"
    assert stream.lines()[-1] == "ACK PLACE_DONE"
    assert runtime._hardware.sucker_states[-1] is False
    assert runtime.healthcheck()["carrying"] is False


def test_healthcheck_includes_state_and_calibration() -> None:
    runtime = _runtime(calibration=UnavailableCalibration())
    status = runtime.healthcheck()
    assert status["state"] == RobotExecutorState.IDLE.value
    assert status["calibration_ready"] is False
    assert status["carrying"] is False
    assert status["control_kernel"] == "cylindrical_kernel"
    assert "robot_cyl" in status
    assert "limits_cyl" in status
    assert status["auto_z_enabled"] is True


def test_pick_world_succeeds_without_calibration(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime(calibration=UnavailableCalibration())
    stream = FakeStream()

    runtime.dispatch_command("PICK_WORLD -30 -150", stream)

    assert wait_for(lambda: "ACK PICK_DONE" in stream.lines())
    assert stream.lines()[0] == "ACK PICK_STARTED"
    assert runtime.healthcheck()["carrying"] is True


def test_pick_world_rejects_out_of_workspace() -> None:
    runtime = _runtime(calibration=UnavailableCalibration())
    stream = FakeStream()

    runtime.dispatch_command("PICK_WORLD 300 -150", stream)

    assert stream.lines() == ["ERR target_out_of_workspace: Radius 335.41 mm is outside limits."]


def test_move_cyl_and_move_cyl_auto_ack(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime()
    stream = FakeStream()

    runtime.dispatch_command("MOVE_CYL 10 130 160", stream)
    runtime.dispatch_command("MOVE_CYL_AUTO 5 120", stream)

    assert stream.lines() == ["ACK MOVE", "ACK MOVE"]
    status = runtime.healthcheck()
    assert "robot_cyl" in status
    assert status["state"] == RobotExecutorState.IDLE.value
    assert status["control_kernel"] == "cylindrical_kernel"
    assert len(runtime._hardware.moves) == 2


def test_pick_world_uses_settle_pose_for_post_pick_state(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime()
    stream = FakeStream()

    runtime.dispatch_command("PICK_WORLD 0 -170", stream)

    assert wait_for(lambda: "ACK PICK_DONE" in stream.lines())
    status = runtime.healthcheck()
    assert float(status["post_pick_settle_z"]) >= float(status["pick_tuning"]["z_carry_floor_mm"])
    assert status["state"] == RobotExecutorState.CARRY_READY.value


def test_place_release_prefers_release_api_when_available(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    hardware = FakeHardwareWithRelease()
    runtime = _runtime(hardware=hardware)
    stream = FakeStream()
    runtime._initialize_executor()
    runtime._executor._carrying = True  # type: ignore[attr-defined]
    runtime._executor._state = RobotExecutorState.CARRY_READY  # type: ignore[attr-defined]
    runtime._executor.set_pick_tuning({"place_release_mode": "release", "place_release_sec": 0.35})  # type: ignore[attr-defined]

    runtime.dispatch_command("PLACE", stream)

    assert wait_for(lambda: "ACK PLACE_DONE" in stream.lines())
    assert hardware.release_calls and abs(hardware.release_calls[-1] - 0.35) < 1e-6
    assert runtime.healthcheck()["release_mode_effective"] == "release"


def test_place_release_falls_back_to_set_state_when_release_unavailable(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime()
    stream = FakeStream()
    runtime._initialize_executor()
    runtime._executor._carrying = True  # type: ignore[attr-defined]
    runtime._executor._state = RobotExecutorState.CARRY_READY  # type: ignore[attr-defined]
    runtime._executor.set_pick_tuning({"place_release_mode": "release", "place_release_sec": 0.2})  # type: ignore[attr-defined]

    runtime.dispatch_command("PLACE", stream)

    assert wait_for(lambda: "ACK PLACE_DONE" in stream.lines())
    assert runtime._hardware.sucker_states[-1] is False
    assert runtime.healthcheck()["release_mode_effective"] == "off_fallback"


def test_legacy_move_marks_legacy_cartesian_kernel() -> None:
    runtime = _runtime()
    stream = FakeStream()

    runtime.dispatch_command("MOVE 10 -60", stream)

    assert stream.lines() == ["ACK MOVE"]
    assert runtime.healthcheck()["control_kernel"] == "legacy_cartesian"


def test_move_cyl_auto_rate_limits_z_change_for_small_radius_delta(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    hardware = FakeHardware()
    hardware.position = (0.0, -162.94, 212.8)
    runtime = _runtime(hardware=hardware)
    stream = FakeStream()

    runtime.dispatch_command("MOVE_CYL_AUTO 0 170", stream)

    assert stream.lines() == ["ACK MOVE"]
    final_pose, _ = runtime._hardware.moves[-1]
    assert final_pose[2] > 200.0


def test_pick_cyl_sends_started_and_done(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime()
    stream = FakeStream()

    runtime.dispatch_command("PICK_CYL 10 130", stream)

    assert wait_for(lambda: "ACK PICK_DONE" in stream.lines())
    assert stream.lines()[0] == "ACK PICK_STARTED"
    assert runtime.healthcheck()["control_kernel"] == "cylindrical_kernel"
    assert runtime.healthcheck()["carrying"] is True


def test_abort_and_reset_commands_round_trip(monkeypatch) -> None:
    _patch_executor_sleep(monkeypatch)
    runtime = _runtime()
    stream = FakeStream()

    runtime.dispatch_command("ABORT", stream)
    runtime.dispatch_command("RESET", stream)

    assert stream.lines() == ["ACK ABORT", "ACK RESET"]
    assert runtime.healthcheck()["state"] == RobotExecutorState.IDLE.value

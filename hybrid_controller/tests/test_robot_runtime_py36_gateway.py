from __future__ import annotations

import types

from hybrid_controller.robot.runtime.robot_runtime_py36 import RobotGateway


class _DummyExecutor:
    def __init__(self, *, state: str) -> None:
        self._state = str(state)
        self.legacy_kernel = types.SimpleNamespace(
            start_move=lambda sender, x, y: "ACK MOVE",
            start_pick_pixel=lambda sender, u, v: "ACK PICK_STARTED",
            start_pick_world=lambda sender, x, y: "ACK PICK_STARTED",
        )
        self.cylindrical_kernel = types.SimpleNamespace(
            start_move=lambda sender, theta, radius, z: "ACK MOVE",
            start_move_auto=lambda sender, theta, radius: "ACK MOVE",
            start_pick=lambda sender, theta, radius: "ACK PICK_STARTED",
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self._state,
            "busy": False,
            "robot_xy": [0.0, -120.0],
            "robot_z": 160.0,
            "last_error_code": "aborted" if self._state == "ERROR" else "",
            "last_error": "Abort requested by operator." if self._state == "ERROR" else "",
        }

    def reset(self) -> str:
        self._state = "IDLE"
        return "ACK RESET"

    def abort(self) -> str:
        self._state = "ERROR"
        return "ACK ABORT"

    def start_place(self, sender) -> str:  # noqa: ANN001
        return "ACK PLACE_STARTED"


def test_gateway_blocks_motion_commands_in_error_state() -> None:
    gateway = RobotGateway(_DummyExecutor(state="ERROR"))

    response = gateway.handle("PICK_WORLD -30 -150", sender=None)

    assert isinstance(response, str)
    assert response.startswith("ERR invalid_state:")
    assert "ERROR state" in response


def test_gateway_allows_status_and_reset_in_error_state() -> None:
    executor = _DummyExecutor(state="ERROR")
    gateway = RobotGateway(executor)

    status_response = gateway.handle("STATUS", sender=None)
    reset_response = gateway.handle("RESET", sender=None)
    post_reset_status = gateway.handle("STATUS", sender=None)

    assert isinstance(status_response, str) and status_response.startswith("ACK STATUS ")
    assert reset_response == "ACK RESET"
    assert isinstance(post_reset_status, str) and '"state":"IDLE"' in post_reset_status

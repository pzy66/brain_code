from __future__ import annotations

from hybrid_controller.adapters.rosbridge_client import RosbridgeClient


class _FakeTopic:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def publish(self, message) -> None:
        self.messages.append(dict(message))


def test_rosbridge_client_emits_ack_when_last_ack_changes() -> None:
    events: list[tuple[str, str]] = []
    client = RosbridgeClient(
        "127.0.0.1",
        9091,
        event_callback=lambda event: events.append((str(event.type), str(event.value))),
    )

    client._handle_state(
        {
            "state": "IDLE",
            "state_seq": 1,
            "robot_ts": 1.0,
            "last_ack": "",
        }
    )
    client._handle_state(
        {
            "state": "IDLE",
            "state_seq": 2,
            "robot_ts": 2.0,
            "last_ack": "MOVE",
        }
    )

    assert ("robot_ack", "MOVE") in events


def test_rosbridge_client_drops_out_of_order_state_seq() -> None:
    states: list[int] = []
    client = RosbridgeClient(
        "127.0.0.1",
        9091,
        state_callback=lambda snapshot: states.append(int(snapshot.get("state_seq", 0))),
    )

    client._handle_state({"state": "IDLE", "state_seq": 3, "robot_ts": 3.0})
    client._handle_state({"state": "IDLE", "state_seq": 2, "robot_ts": 2.0})

    assert states == [3]
    latest = client.latest_state_snapshot()
    assert isinstance(latest, dict)
    assert int(latest.get("state_seq", 0)) == 3


def test_rosbridge_client_accepts_equal_seq_with_newer_robot_ts() -> None:
    robot_ts_values: list[float] = []
    client = RosbridgeClient(
        "127.0.0.1",
        9091,
        state_callback=lambda snapshot: robot_ts_values.append(float(snapshot.get("robot_ts", 0.0) or 0.0)),
    )

    client._handle_state({"state": "IDLE", "state_seq": 7, "robot_ts": 10.0})
    client._handle_state({"state": "IDLE", "state_seq": 7, "robot_ts": 10.1})
    client._handle_state({"state": "IDLE", "state_seq": 7, "robot_ts": 10.05})

    # Equal seq + newer timestamp is valid heartbeat, older timestamp is dropped.
    assert robot_ts_values == [10.0, 10.1]
    latest = client.latest_state_snapshot()
    assert isinstance(latest, dict)
    assert float(latest.get("robot_ts", 0.0) or 0.0) == 10.1


def test_rosbridge_stop_teleop_preserves_sequence_and_explicit_z_mode() -> None:
    client = RosbridgeClient("127.0.0.1", 9091)
    topic = _FakeTopic()
    client._teleop_topic = topic

    client.stop_teleop(use_auto_z=False, cmd_seq=42, client_ts=1234.5)

    assert topic.messages == [
        {
            "theta_rate_deg_s": 0.0,
            "radius_rate_mm_s": 0.0,
            "z_rate_mm_s": 0.0,
            "use_auto_z": False,
            "enabled": False,
            "cmd_seq": 42,
            "client_ts": 1234.5,
        }
    ]

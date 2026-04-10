from __future__ import annotations

import threading
import time

from hybrid_controller.adapters.remote_snapshot_poller import RemoteSnapshotPoller
from hybrid_controller.runtime_state import RobotSnapshotEnvelope


def test_remote_snapshot_poller_emits_requested_snapshot() -> None:
    received: list[RobotSnapshotEnvelope] = []
    done = threading.Event()

    def fetch_snapshot() -> RobotSnapshotEnvelope:
        return RobotSnapshotEnvelope(
            payload={"state": "IDLE"},
            ts=time.time(),
            transport="tcp",
            ok=True,
            error="",
        )

    def on_snapshot(envelope: RobotSnapshotEnvelope) -> None:
        received.append(envelope)
        done.set()

    poller = RemoteSnapshotPoller(
        interval_ms=200,
        fetch_snapshot=fetch_snapshot,
        on_snapshot=on_snapshot,
    )
    poller.start()
    try:
        poller.request_now()
        assert done.wait(1.0), "poller should emit one requested snapshot"
        assert received
        assert received[0].ok is True
        assert received[0].payload == {"state": "IDLE"}
    finally:
        poller.stop()


def test_remote_snapshot_poller_converts_fetch_errors_to_error_envelope() -> None:
    received: list[RobotSnapshotEnvelope] = []
    done = threading.Event()

    def fetch_snapshot() -> RobotSnapshotEnvelope:
        raise RuntimeError("fetch_failed")

    def on_snapshot(envelope: RobotSnapshotEnvelope) -> None:
        received.append(envelope)
        done.set()

    poller = RemoteSnapshotPoller(
        interval_ms=200,
        fetch_snapshot=fetch_snapshot,
        on_snapshot=on_snapshot,
    )
    poller.start()
    try:
        poller.request_now()
        assert done.wait(1.0), "poller should emit an error envelope when fetch fails"
        assert received
        first = received[0]
        assert first.ok is False
        assert "fetch_failed" in first.error
    finally:
        poller.stop()


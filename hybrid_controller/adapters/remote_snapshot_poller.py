from __future__ import annotations

import threading
import time
from typing import Callable

from hybrid_controller.runtime_state import RobotSnapshotEnvelope


SnapshotFetch = Callable[[], RobotSnapshotEnvelope]
SnapshotCallback = Callable[[RobotSnapshotEnvelope], None]


class RemoteSnapshotPoller:
    """Background poller that periodically fetches robot snapshots.

    Polling is serialized in one thread to avoid overlapping requests.
    """

    def __init__(
        self,
        *,
        interval_ms: int,
        fetch_snapshot: SnapshotFetch,
        on_snapshot: SnapshotCallback,
    ) -> None:
        self._interval_sec = max(0.02, float(interval_ms) / 1000.0)
        self._fetch_snapshot = fetch_snapshot
        self._on_snapshot = on_snapshot
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._running_lock = threading.Lock()

    def start(self) -> None:
        with self._running_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._wake_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="remote-snapshot-poller",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._running_lock:
            thread = self._thread
            self._thread = None
            if thread is None:
                return
            self._stop_event.set()
            self._wake_event.set()
        thread.join(timeout=2.0)
        self._stop_event.clear()
        self._wake_event.clear()

    def request_now(self) -> None:
        self._wake_event.set()

    def _run_loop(self) -> None:
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            wait_sec = max(0.0, next_deadline - now)
            self._wake_event.wait(timeout=wait_sec)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break
            envelope = self._safe_fetch_snapshot()
            try:
                self._on_snapshot(envelope)
            except Exception:
                # Keep poller alive even if callback handling fails.
                pass
            next_deadline = time.monotonic() + self._interval_sec

    def _safe_fetch_snapshot(self) -> RobotSnapshotEnvelope:
        try:
            envelope = self._fetch_snapshot()
            if isinstance(envelope, RobotSnapshotEnvelope):
                return envelope
            return RobotSnapshotEnvelope(
                payload=None,
                ts=time.time(),
                transport="unknown",
                ok=False,
                error="invalid_snapshot_envelope",
            )
        except Exception as error:  # pragma: no cover - defensive fallback
            return RobotSnapshotEnvelope(
                payload=None,
                ts=time.time(),
                transport="poller",
                ok=False,
                error=str(error),
            )


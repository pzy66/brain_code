from __future__ import annotations

from dataclasses import dataclass

from hybrid_controller.adapters.input_provider import InputFrame, InputHealth, InputProvider
from hybrid_controller.mi.ports import BrainFlowPortResolver


@dataclass(slots=True)
class MiInputProvider(InputProvider):
    backend: str = "brainflow"
    poll_interval_ms: int = 50

    def __post_init__(self) -> None:
        self._mode = "move"
        self._connected = False
        self._running = False
        self._last_error = "not_implemented"
        self._resolver = BrainFlowPortResolver()

    def connect(self) -> bool:
        # Placeholder phase: no runtime BrainFlow dependency is introduced.
        self._connected = False
        self._last_error = "not_implemented"
        return False

    def start(self) -> bool:
        self._running = False
        self._last_error = "not_implemented"
        return False

    def stop(self) -> None:
        self._running = False

    def read_frame(self) -> InputFrame | None:
        return None

    def set_mode(self, mode: str) -> None:
        self._mode = str(mode or "move").strip().lower() or "move"

    def healthcheck(self) -> InputHealth:
        ports = self._resolver.list_candidate_ports()
        status = "not_implemented"
        if not ports:
            status = "not_implemented:no_serial_port"
        return InputHealth(
            source="mi",
            connected=self._connected,
            running=self._running,
            ready=False,
            status=status,
            last_error=self._last_error,
        )


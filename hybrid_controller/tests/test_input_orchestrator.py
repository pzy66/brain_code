from __future__ import annotations

from dataclasses import dataclass

from hybrid_controller.adapters.input_orchestrator import InputOrchestrator
from hybrid_controller.adapters.input_provider import InputFrame, InputHealth
from hybrid_controller.adapters.sim_input import SimInputAdapter
from hybrid_controller.adapters.ssvep_adapter import SSVEPAdapter


@dataclass
class _StubMiProvider:
    ready: bool = True
    direction: str | None = "left"

    def __post_init__(self) -> None:
        self.connected = False
        self.running = False
        self.mode = "move"
        self._read_count = 0

    def connect(self) -> bool:
        self.connected = self.ready
        return self.connected

    def start(self) -> bool:
        self.running = self.ready
        return self.running

    def stop(self) -> None:
        self.running = False

    def set_mode(self, mode: str) -> None:
        self.mode = str(mode)

    def healthcheck(self) -> InputHealth:
        return InputHealth(
            source="mi",
            connected=self.connected,
            running=self.running,
            ready=self.ready,
            status="ready" if self.ready else "not_ready",
            last_error="--" if self.ready else "stub_not_ready",
        )

    def read_frame(self) -> InputFrame | None:
        self._read_count += 1
        if not self.ready or self.direction is None:
            return None
        return InputFrame(
            source="mi",
            direction=self.direction,  # type: ignore[arg-type]
            confidence=0.9,
            timestamp=float(self._read_count),
        )


def test_orchestrator_sim_and_ssvep_routes() -> None:
    sim_input = SimInputAdapter(move_source="sim", decision_source="ssvep", ssvep_keyboard_debug_enabled=True)
    ssvep_adapter = SSVEPAdapter((8.0, 10.0, 12.0, 15.0))
    ssvep_adapter.set_mode("target_selection")
    orchestrator = InputOrchestrator(
        sim_input=sim_input,
        ssvep_adapter=ssvep_adapter,
        move_source="sim",
        decision_source="ssvep",
        ssvep_keyboard_debug_enabled=True,
        mi_provider=None,
        mi_enabled=False,
    )

    events = orchestrator.handle_key_token("1")
    assert len(events) == 1
    assert events[0].type == "target_selected"
    assert events[0].source == "sim"

    ssvep_events = orchestrator.handle_ssvep_command("8 Hz")
    assert len(ssvep_events) == 1
    assert ssvep_events[0].type == "target_selected"
    assert ssvep_events[0].source == "ssvep"


def test_orchestrator_mi_missing_provider_reports_health() -> None:
    orchestrator = InputOrchestrator(
        sim_input=SimInputAdapter(),
        ssvep_adapter=SSVEPAdapter(),
        move_source="mi",
        decision_source="sim",
        ssvep_keyboard_debug_enabled=True,
        mi_provider=None,
        mi_enabled=True,
    )
    health = orchestrator.initialize()
    assert health is not None
    assert health.status == "provider_missing"
    result = orchestrator.poll()
    assert result.mi_health is not None
    assert result.mi_health.status == "provider_missing"
    assert result.events == ()


def test_orchestrator_mi_cooldown_blocks_duplicate_direction() -> None:
    provider = _StubMiProvider(ready=True, direction="forward")
    orchestrator = InputOrchestrator(
        sim_input=SimInputAdapter(),
        ssvep_adapter=SSVEPAdapter(),
        move_source="mi",
        decision_source="sim",
        ssvep_keyboard_debug_enabled=True,
        mi_provider=provider,
        mi_enabled=True,
        mi_command_cooldown_ms=500,
    )
    health = orchestrator.initialize()
    assert health is not None
    assert health.ready is True

    first = orchestrator.poll()
    second = orchestrator.poll()
    assert len(first.events) == 1
    assert first.events[0].value == "forward"
    assert second.events == ()


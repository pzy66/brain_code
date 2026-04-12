from __future__ import annotations

import time
from dataclasses import dataclass

from hybrid_controller.adapters.input_provider import InputHealth, InputProvider
from hybrid_controller.adapters.sim_input import SimInputAdapter
from hybrid_controller.adapters.ssvep_adapter import SSVEPAdapter
from hybrid_controller.controller.events import Event


@dataclass(frozen=True, slots=True)
class InputPollResult:
    events: tuple[Event, ...] = ()
    mi_health: InputHealth | None = None


class InputOrchestrator:
    def __init__(
        self,
        *,
        sim_input: SimInputAdapter,
        ssvep_adapter: SSVEPAdapter,
        move_source: str,
        decision_source: str,
        ssvep_keyboard_debug_enabled: bool,
        mi_provider: InputProvider | None = None,
        mi_enabled: bool = False,
        mi_command_cooldown_ms: int = 120,
    ) -> None:
        self.sim_input = sim_input
        self.ssvep_adapter = ssvep_adapter
        self.mi_provider = mi_provider
        self.move_source = str(move_source or "sim").strip().lower()
        self.decision_source = str(decision_source or "sim").strip().lower()
        self.ssvep_keyboard_debug_enabled = bool(ssvep_keyboard_debug_enabled)
        self.mi_enabled = bool(mi_enabled)
        self.mi_command_cooldown_sec = max(float(mi_command_cooldown_ms) / 1000.0, 0.0)
        self._mi_ready = False
        self._last_mi_emit_ts = 0.0
        self._last_mi_direction: str | None = None

    def set_sources(
        self,
        *,
        move_source: str,
        decision_source: str,
        ssvep_keyboard_debug_enabled: bool,
        mi_enabled: bool,
    ) -> None:
        self.move_source = str(move_source or "sim").strip().lower()
        self.decision_source = str(decision_source or "sim").strip().lower()
        self.ssvep_keyboard_debug_enabled = bool(ssvep_keyboard_debug_enabled)
        self.mi_enabled = bool(mi_enabled)
        self.sim_input.set_sources(
            move_source=self.move_source,
            decision_source=self.decision_source,
            ssvep_keyboard_debug_enabled=self.ssvep_keyboard_debug_enabled,
        )

    def initialize(self) -> InputHealth | None:
        if self.move_source != "mi":
            return None
        if self.mi_provider is None:
            self._mi_ready = False
            return InputHealth(
                source="mi",
                connected=False,
                running=False,
                ready=False,
                status="provider_missing",
                last_error="provider_missing",
            )
        if not self.mi_enabled:
            self._mi_ready = False
            return InputHealth(
                source="mi",
                connected=False,
                running=False,
                ready=False,
                status="disabled",
                last_error="mi_disabled",
            )
        self.mi_provider.set_mode("move")
        connected = bool(self.mi_provider.connect())
        running = bool(self.mi_provider.start()) if connected else False
        health = self.mi_provider.healthcheck()
        self._mi_ready = bool(connected and running and health.ready)
        return InputHealth(
            source=health.source,
            connected=connected and health.connected,
            running=running and health.running,
            ready=self._mi_ready and health.ready,
            status=health.status,
            last_error=health.last_error,
        )

    def shutdown(self) -> None:
        if self.mi_provider is not None:
            try:
                self.mi_provider.stop()
            except Exception:
                pass
        self._mi_ready = False
        self._last_mi_emit_ts = 0.0
        self._last_mi_direction = None

    def handle_key_token(self, token: str) -> list[Event]:
        return self.sim_input.handle_key_token(token)

    def handle_ssvep_command(self, command: object) -> list[Event]:
        event = self.ssvep_adapter.process_command(command)
        return [event] if event is not None else []

    def poll(self) -> InputPollResult:
        if self.move_source != "mi":
            return InputPollResult()
        if self.mi_provider is None:
            return InputPollResult(
                events=(),
                mi_health=InputHealth(
                    source="mi",
                    connected=False,
                    running=False,
                    ready=False,
                    status="provider_missing",
                    last_error="provider_missing",
                ),
            )
        if not self.mi_enabled:
            return InputPollResult(
                events=(),
                mi_health=InputHealth(
                    source="mi",
                    connected=False,
                    running=False,
                    ready=False,
                    status="disabled",
                    last_error="mi_disabled",
                ),
            )

        health = self.mi_provider.healthcheck()
        frame = self.mi_provider.read_frame()
        events: list[Event] = []
        if (
            frame is not None
            and frame.direction in {"left", "right", "forward", "backward"}
            and bool(health.ready)
            and self._should_emit_move(str(frame.direction))
        ):
            events.append(
                Event(
                    source="mi",
                    type="move",
                    value=str(frame.direction),
                    confidence=frame.confidence,
                    timestamp=time.time(),
                )
            )
        return InputPollResult(events=tuple(events), mi_health=health)

    def _should_emit_move(self, direction: str) -> bool:
        now = time.monotonic()
        if self.mi_command_cooldown_sec <= 0.0:
            self._last_mi_emit_ts = now
            self._last_mi_direction = direction
            return True
        if (
            self._last_mi_direction == direction
            and (now - self._last_mi_emit_ts) < self.mi_command_cooldown_sec
        ):
            return False
        self._last_mi_emit_ts = now
        self._last_mi_direction = direction
        return True


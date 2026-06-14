from __future__ import annotations

from typing import Optional

from hybrid_controller.controller.events import Event


class SSVEPAdapter:
    def __init__(self, freqs: tuple[float, float, float, float] = (8.0, 10.0, 12.0, 15.0)) -> None:
        self.mode = "idle"
        self.freqs = tuple(float(freq) for freq in freqs)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def process_command(self, command: object) -> Optional[Event]:
        token = self._normalize_command_token(command)
        if token is None:
            return None
        if self.mode == "binary":
            confirm_token = self._format_freq_token(self.freqs[0])
            cancel_token = self._format_freq_token(self.freqs[3])
            if token == confirm_token:
                return Event(source="ssvep", type="decision_confirm", value=f"{confirm_token}hz")
            if token == cancel_token:
                return Event(source="ssvep", type="decision_cancel", value=f"{cancel_token}hz")
            return None
        if self.mode == "target_selection":
            mapping = {self._format_freq_token(freq): index for index, freq in enumerate(self.freqs)}
            if token in mapping:
                return Event(source="ssvep", type="target_selected", value=mapping[token])
        return None

    @staticmethod
    def _format_freq_token(freq: float) -> str:
        return str(int(freq)) if float(freq).is_integer() else f"{freq:g}"

    def _normalize_command_token(self, command: object) -> Optional[str]:
        text = str(command or "").strip().lower()
        if not text:
            return None
        aliases = {
            self._format_freq_token(self.freqs[0]): ("up",),
            self._format_freq_token(self.freqs[1]): ("left",),
            self._format_freq_token(self.freqs[2]): ("down",),
            self._format_freq_token(self.freqs[3]): ("right",),
        }
        for token, extra_aliases in aliases.items():
            if token in text:
                return token
            if any(alias in text for alias in extra_aliases):
                return token
        return None

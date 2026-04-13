from __future__ import annotations

from typing import Any, Optional

from ._legacy_adapter import LegacyDecoderAdapter


class TDCADecoder(LegacyDecoderAdapter):
    def __init__(self, *, win_sec: float, step_sec: float, model_params: Optional[dict[str, Any]] = None) -> None:
        params = dict(model_params or {})
        params.setdefault("delay_steps", 3)
        super().__init__(
            model_name="tdca",
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=params,
        )


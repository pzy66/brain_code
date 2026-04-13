from __future__ import annotations

from typing import Any, Optional

from ._legacy_adapter import LegacyDecoderAdapter


class FBCCADecoder(LegacyDecoderAdapter):
    def __init__(self, *, win_sec: float, step_sec: float, model_params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            model_name="fbcca",
            win_sec=win_sec,
            step_sec=step_sec,
            model_params=model_params,
        )


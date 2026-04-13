from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from async_fbcca_idle_standalone import create_decoder, normalize_model_name

from .base import BaseDecoder, DecoderOutput, decoder_output_from_payload


class LegacyDecoderAdapter(BaseDecoder):
    """
    Adapter around standalone decoders to keep algorithm behavior unchanged
    while exposing a unified BaseDecoder API.
    """

    def __init__(
        self,
        *,
        model_name: str,
        win_sec: float,
        step_sec: float,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=normalize_model_name(model_name), win_sec=win_sec, step_sec=step_sec)
        self.model_params = dict(model_params or {})
        self._decoder = None

    def fit(
        self,
        train_trials: Sequence[tuple[Any, np.ndarray]],
        fs: int,
        freqs: Sequence[float],
        channels: Sequence[int],
    ) -> None:
        self.freqs = tuple(float(item) for item in freqs)  # type: ignore[assignment]
        if len(self.freqs) != 4:
            raise ValueError("decoder expects exactly 4 frequencies")
        self.channels = tuple(int(item) for item in channels)
        self.configure_runtime(int(fs))
        self._decoder = create_decoder(
            self.name,
            sampling_rate=int(fs),
            freqs=self.freqs,
            win_sec=self.win_sec,
            step_sec=self.step_sec,
            model_params=dict(self.model_params),
        )
        if getattr(self._decoder, "requires_fit", False):
            self._decoder.fit(train_trials)

    @property
    def decoder(self) -> Any:
        if self._decoder is None:
            raise RuntimeError(f"decoder '{self.name}' is not fitted")
        return self._decoder

    def score_window(self, window: np.ndarray) -> np.ndarray:
        matrix = np.ascontiguousarray(np.asarray(window, dtype=np.float64))
        return np.asarray(self.decoder.score_window(matrix), dtype=np.float64)

    def predict_window(self, window: np.ndarray) -> DecoderOutput:
        matrix = np.ascontiguousarray(np.asarray(window, dtype=np.float64))
        payload = dict(self.decoder.analyze_window(matrix))
        return decoder_output_from_payload(payload)


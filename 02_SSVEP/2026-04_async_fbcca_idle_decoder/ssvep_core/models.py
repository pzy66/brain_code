from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from async_fbcca_idle_standalone import (
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_NH,
    TrialSpec,
    create_decoder,
    normalize_model_name,
)

SUPPORTED_MODEL_NAMES = tuple(str(name) for name in DEFAULT_BENCHMARK_MODELS)


@dataclass
class DecoderModelAdapter:
    model_name: str
    sampling_rate: int
    freqs: tuple[float, float, float, float]
    win_sec: float
    step_sec: float
    model_params: dict[str, Any]

    def __post_init__(self) -> None:
        params = dict(self.model_params or {})
        params.setdefault("Nh", DEFAULT_NH)
        self.model_params = params
        self.model_name = normalize_model_name(self.model_name)
        self.decoder = create_decoder(
            self.model_name,
            sampling_rate=int(self.sampling_rate),
            freqs=self.freqs,
            win_sec=float(self.win_sec),
            step_sec=float(self.step_sec),
            model_params=self.model_params,
        )

    def fit(
        self,
        calib_trials: Sequence[tuple[TrialSpec, np.ndarray]],
        *,
        fs: Optional[int] = None,
        channels: Optional[Sequence[int]] = None,
    ) -> dict[str, Any]:
        if fs is not None and int(fs) != int(self.sampling_rate):
            self.sampling_rate = int(fs)
            self.decoder.configure_runtime(self.sampling_rate)
        if self.decoder.requires_fit:
            self.decoder.fit(calib_trials)
        return self.get_state()

    def score(self, window_eeg: np.ndarray) -> np.ndarray:
        scores = self.decoder.score_window(np.asarray(window_eeg, dtype=np.float64))
        return np.asarray(scores, dtype=np.float64)

    def analyze(self, window_eeg: np.ndarray) -> dict[str, Any]:
        return dict(self.decoder.analyze_window(np.asarray(window_eeg, dtype=np.float64)))

    def reset(self) -> None:
        self.decoder.reset()

    def get_state(self) -> dict[str, Any]:
        return {
            "model_name": str(self.model_name),
            "state": self.decoder.get_state(),
            "model_params": dict(self.model_params),
        }


def create_model_adapter(
    model_name: str,
    *,
    sampling_rate: int,
    freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    model_params: Optional[dict[str, Any]] = None,
) -> DecoderModelAdapter:
    return DecoderModelAdapter(
        model_name=normalize_model_name(model_name),
        sampling_rate=int(sampling_rate),
        freqs=tuple(float(freq) for freq in freqs),  # type: ignore[arg-type]
        win_sec=float(win_sec),
        step_sec=float(step_sec),
        model_params=dict(model_params or {}),
    )


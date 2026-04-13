from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RollingFeatureHistory:
    window_size: int = 4

    def __post_init__(self) -> None:
        self.window_size = max(1, int(self.window_size))
        self._pred_freq_history: deque[Optional[float]] = deque(maxlen=self.window_size)
        self._margin_history: deque[float] = deque(maxlen=self.window_size)
        self._ratio_history: deque[float] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._pred_freq_history.clear()
        self._margin_history.clear()
        self._ratio_history.clear()

    def update(self, *, pred_freq: Optional[float], margin: float, ratio: float) -> dict[str, float]:
        freq_value = None if pred_freq is None else float(pred_freq)
        self._pred_freq_history.append(freq_value)
        self._margin_history.append(float(margin))
        self._ratio_history.append(float(ratio))

        if freq_value is None or len(self._pred_freq_history) == 0:
            consistency = 0.0
        else:
            matches = sum(1 for item in self._pred_freq_history if item is not None and abs(float(item) - freq_value) <= 1e-8)
            consistency = float(matches / max(len(self._pred_freq_history), 1))

        margin_mean = float(np.mean(np.asarray(self._margin_history, dtype=float))) if self._margin_history else 0.0
        ratio_mean = float(np.mean(np.asarray(self._ratio_history, dtype=float))) if self._ratio_history else 0.0

        return {
            "consistency": float(consistency),
            "margin_mean_k": float(margin_mean),
            "ratio_mean_k": float(ratio_mean),
        }


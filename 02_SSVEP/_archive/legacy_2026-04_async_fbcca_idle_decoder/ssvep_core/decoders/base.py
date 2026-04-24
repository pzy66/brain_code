from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DecoderOutput:
    pred_freq: float
    scores: np.ndarray
    top1_score: float
    top2_score: float
    margin: float
    ratio: float
    entropy: float
    normalized_top1: float

    def as_feature_row(self) -> dict[str, float]:
        return {
            "pred_freq": float(self.pred_freq),
            "top1_score": float(self.top1_score),
            "top2_score": float(self.top2_score),
            "margin": float(self.margin),
            "ratio": float(self.ratio),
            "entropy": float(self.entropy),
            "score_entropy": float(self.entropy),
            "normalized_top1": float(self.normalized_top1),
        }


def decoder_output_from_payload(payload: dict[str, Any]) -> DecoderOutput:
    scores = np.asarray(payload.get("scores", []), dtype=np.float64)
    return DecoderOutput(
        pred_freq=float(payload.get("pred_freq", 0.0)),
        scores=scores,
        top1_score=float(payload.get("top1_score", 0.0)),
        top2_score=float(payload.get("top2_score", 0.0)),
        margin=float(payload.get("margin", 0.0)),
        ratio=float(payload.get("ratio", 0.0)),
        entropy=float(payload.get("entropy", payload.get("score_entropy", 0.0))),
        normalized_top1=float(payload.get("normalized_top1", 0.0)),
    )


class BaseDecoder(ABC):
    name: str

    def __init__(self, *, name: str, win_sec: float, step_sec: float) -> None:
        self.name = str(name)
        self.win_sec = float(win_sec)
        self.step_sec = float(step_sec)
        self.fs = 0
        self.freqs: tuple[float, float, float, float] = (8.0, 10.0, 12.0, 15.0)
        self.channels: tuple[int, ...] = ()
        self.win_samples = 0
        self.step_samples = 0

    def configure_runtime(self, fs: int) -> None:
        self.fs = int(fs)
        self.win_samples = max(1, int(round(self.win_sec * float(self.fs))))
        self.step_samples = max(1, int(round(self.step_sec * float(self.fs))))

    @abstractmethod
    def fit(
        self,
        train_trials: Sequence[tuple[Any, np.ndarray]],
        fs: int,
        freqs: Sequence[float],
        channels: Sequence[int],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def score_window(self, window: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_window(self, window: np.ndarray) -> DecoderOutput:
        raise NotImplementedError

    def iter_window_features(
        self,
        segment: np.ndarray,
        *,
        expected_freq: Optional[float],
        label: str,
        trial_id: int = -1,
        block_index: int = -1,
    ) -> list[dict[str, Any]]:
        segment_matrix = np.asarray(segment, dtype=np.float64)
        if segment_matrix.ndim != 2:
            raise ValueError("segment must have shape (samples, channels)")
        if self.win_samples <= 0 or self.step_samples <= 0:
            raise RuntimeError("decoder runtime is not configured")
        if segment_matrix.shape[0] < self.win_samples:
            raise ValueError("segment is shorter than decoder window")
        rows: list[dict[str, Any]] = []
        for window_index, start in enumerate(range(0, segment_matrix.shape[0] - self.win_samples + 1, self.step_samples)):
            window = np.ascontiguousarray(segment_matrix[start : start + self.win_samples], dtype=np.float64)
            output = self.predict_window(window)
            correct = expected_freq is not None and abs(float(output.pred_freq) - float(expected_freq)) <= 1e-8
            row: dict[str, Any] = {
                "label": str(label),
                "expected_freq": None if expected_freq is None else float(expected_freq),
                **output.as_feature_row(),
                "correct": bool(correct),
                "trial_id": int(trial_id),
                "block_index": int(block_index),
                "window_index": int(window_index),
            }
            rows.append(row)
        return rows


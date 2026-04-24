from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from async_fbcca_idle_standalone import DEFAULT_BENCHMARK_MODELS, normalize_model_name


@dataclass(frozen=True)
class ModelSpec:
    name: str
    decoder_name: str
    legacy: bool = False
    realtime_enabled: bool = False


class ModelRegistry:
    """Unified registry used by training/eval UI and offline pipeline."""

    _SPECS: tuple[ModelSpec, ...] = tuple(
        [ModelSpec(name=str(item), decoder_name=str(item), legacy=True) for item in DEFAULT_BENCHMARK_MODELS]
        + [
            ModelSpec(name="etrca_r", decoder_name="trca_r", legacy=True),
            ModelSpec(name="tdca_v2", decoder_name="tdca", realtime_enabled=True),
            ModelSpec(name="fbcca_v1", decoder_name="fbcca", legacy=True),
        ]
    )
    _ALIASES: dict[str, str] = {
        "trca-r": "trca_r",
        "etrca": "etrca_r",
        "etrca-r": "etrca_r",
        "tdca-v2": "tdca_v2",
        "fbcca-v1": "fbcca_v1",
    }

    @classmethod
    def normalize(cls, model_name: str) -> str:
        key = str(model_name).strip().lower()
        key = cls._ALIASES.get(key, key)
        if key in {spec.name for spec in cls._SPECS}:
            return key
        return normalize_model_name(key)

    @classmethod
    def decoder_name(cls, model_name: str) -> str:
        normalized = cls.normalize(model_name)
        for spec in cls._SPECS:
            if spec.name == normalized:
                return spec.decoder_name
        return normalize_model_name(normalized)

    @classmethod
    def list_models(cls, *, task: str = "benchmark") -> list[str]:
        task_name = str(task).strip().lower()
        if task_name == "realtime":
            values = [spec.name for spec in cls._SPECS if spec.realtime_enabled]
            return values or ["tdca_v2"]
        if task_name == "legacy":
            return [spec.name for spec in cls._SPECS if spec.legacy]
        if task_name == "benchmark":
            return [
                "cca",
                "itcca",
                "ecca",
                "msetcca",
                "fbcca",
                "trca",
                "trca_r",
                "sscor",
                "tdca",
                "etrca_r",
            ]
        return [spec.name for spec in cls._SPECS]

    @classmethod
    def resolve_many(cls, model_names: Iterable[str], *, task: str = "benchmark") -> tuple[str, ...]:
        values = list(model_names)
        if not values:
            values = cls.list_models(task=task)
        ordered: list[str] = []
        for item in values:
            name = cls.normalize(item)
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered)


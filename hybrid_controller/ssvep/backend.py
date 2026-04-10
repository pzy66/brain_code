from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


class AsyncFbccaBackend:
    def _core(self):
        return importlib.import_module("hybrid_controller.ssvep.experimental.async_fbcca_idle")

    def _workers(self):
        return importlib.import_module("hybrid_controller.ssvep.experimental.single_model")

    def create_single_model_config(self, **kwargs: Any) -> Any:
        return self._workers().SingleModelConfig(**kwargs)

    def device_check_worker(self, **kwargs: Any) -> Any:
        return self._workers().DeviceCheckWorker(**kwargs)

    def pretrain_worker(self, config: Any) -> Any:
        return self._workers().PretrainWorker(config)

    def online_worker(self, config: Any) -> Any:
        return self._workers().OnlineWorker(config)

    def load_profile(self, path: Path | str, *, fallback_freqs: tuple[float, ...], require_exists: bool) -> Any:
        return self._core().load_profile(Path(path), fallback_freqs=fallback_freqs, require_exists=require_exists)

    def default_profile(self, freqs: tuple[float, ...]) -> Any:
        return self._core().default_profile(freqs)

    def save_profile(self, profile: Any, path: Path | str) -> None:
        self._core().save_profile(profile, Path(path))

    def normalize_serial_port(self, serial_port: str | None) -> str:
        return str(self._core().normalize_serial_port(serial_port))

    def serial_port_is_auto(self, serial_port: str | None) -> bool:
        return bool(self._core().serial_port_is_auto(serial_port))

    def list_serial_port_candidates(self) -> list[str]:
        candidates = self._core().list_serial_port_candidates()
        return [str(item) for item in candidates]

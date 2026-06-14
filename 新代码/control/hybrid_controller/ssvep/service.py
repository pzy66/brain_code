from __future__ import annotations

from pathlib import Path

from hybrid_controller.config import AppConfig

from .backend import AsyncFbccaBackend
from .profiles import ProfileStore


class AsyncFbccaService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.backend = AsyncFbccaBackend()
        self.profile_store = ProfileStore(config.ssvep_profile_dir, config.ssvep_current_profile_path)

    def create_single_model_config(
        self,
        *,
        profile_path: Path,
    ):
        return self.backend.create_single_model_config(
            serial_port=str(self.config.ssvep_serial_port),
            board_id=int(self.config.ssvep_board_id),
            freqs=tuple(float(freq) for freq in self.config.ssvep_freqs),
            model_name=str(self.config.ssvep_model_name),
            profile_path=Path(profile_path),
            prepare_sec=float(self.config.ssvep_pretrain_prepare_sec),
            active_sec=float(self.config.ssvep_pretrain_active_sec),
            rest_sec=float(self.config.ssvep_pretrain_rest_sec),
            target_repeats=int(self.config.ssvep_pretrain_target_repeats),
            idle_repeats=int(self.config.ssvep_pretrain_idle_repeats),
            win_sec=float(self.config.ssvep_win_sec),
            step_sec=float(self.config.ssvep_step_sec),
        )

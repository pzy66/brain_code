from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt5.QtCore import QThread, Qt

from hybrid_controller.config import AppConfig

from .reference_loader import load_module


class SSVEPRuntime:
    def __init__(
        self,
        config: AppConfig,
        command_callback: Callable[[object], None],
        status_callback: Callable[[str], None],
    ) -> None:
        self.config = config
        self.command_callback = command_callback
        self.status_callback = status_callback
        self.mode = "idle"
        self._module = None
        self._runtime_config: dict[str, Any] = {
            "serial_port": config.ssvep_serial_port,
            "board_id": config.ssvep_board_id,
            "sampling_rate": config.ssvep_sampling_rate,
            "freqs": tuple(config.ssvep_freqs),
            "win_sec": config.ssvep_win_sec,
            "step_sec": config.ssvep_step_sec,
            "score_threshold": config.ssvep_score_threshold,
            "ratio_threshold": config.ssvep_ratio_threshold,
            "history_len": config.ssvep_history_len,
        }
        self.eeg_thread: Optional[QThread] = None
        self.analysis_thread: Optional[QThread] = None
        self.worker = None
        self.analysis_worker = None
        self.analyzer = None

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.status_callback(f"SSVEP mode -> {mode}")

    def set_runtime_config(self, **kwargs: Any) -> None:
        self._runtime_config.update(kwargs)

    def start(self) -> None:
        if self.worker is not None:
            return
        module = self._load_module()
        self.analyzer = module.RealTimeFBCCA(
            sampling_rate=self._runtime_config["sampling_rate"],
            freqs=self._runtime_config["freqs"],
            win_sec=self._runtime_config["win_sec"],
            step_sec=self._runtime_config["step_sec"],
            score_threshold=self._runtime_config["score_threshold"],
            conf_ratio_th=self._runtime_config["ratio_threshold"],
            history_len=self._runtime_config["history_len"],
        )
        self.eeg_thread = QThread()
        self.analysis_thread = QThread()
        self.worker = module.EEGWorker(
            self._runtime_config["board_id"],
            self._runtime_config["serial_port"],
            self.analyzer,
        )
        self.analysis_worker = module.AnalysisWorker(self.analyzer)

        self.worker.moveToThread(self.eeg_thread)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.eeg_thread.started.connect(self.worker.start_collection)
        self.worker.finished.connect(self.eeg_thread.quit)
        self.worker.finished.connect(self.analysis_thread.quit)
        self.worker.data_ready.connect(self.analysis_worker.process_data, type=Qt.QueuedConnection)
        self.worker.status_changed.connect(self.status_callback)
        self.worker.error_occurred.connect(self.status_callback)
        self.analysis_worker.result_ready.connect(lambda text: self.status_callback(f"SSVEP raw: {text.splitlines()[0]}"))
        self.analysis_worker.status_ready.connect(self.status_callback)
        self.analysis_worker.command_detected.connect(self.command_callback)
        self.eeg_thread.finished.connect(self.worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)

        self.analysis_thread.start()
        self.eeg_thread.start()
        self.status_callback("SSVEP runtime started.")

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        if self.eeg_thread is not None:
            self.eeg_thread.quit()
            self.eeg_thread.wait(2000)
            self.eeg_thread = None
        if self.analysis_thread is not None:
            self.analysis_thread.quit()
            self.analysis_thread.wait(2000)
            self.analysis_thread = None
        self.worker = None
        self.analysis_worker = None
        self.analyzer = None
        self.status_callback("SSVEP runtime stopped.")

    def healthcheck(self) -> dict[str, object]:
        return {
            "running": self.worker is not None,
            "mode": self.mode,
            "serial_port": self._runtime_config["serial_port"],
        }

    def _load_module(self):
        if self._module is None:
            self._module = load_module("hybrid_reference_ssvep_demo", self.config.ssvep_reference_path)
        return self._module

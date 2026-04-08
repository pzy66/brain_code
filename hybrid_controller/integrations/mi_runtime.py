from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PyQt5.QtCore import QThread, Qt

from hybrid_controller.config import AppConfig

from .reference_loader import load_module


class MIRuntime:
    def __init__(
        self,
        config: AppConfig,
        result_callback: Callable[[dict[str, object]], None],
        status_callback: Callable[[str], None],
    ) -> None:
        self.config = config
        self.result_callback = result_callback
        self.status_callback = status_callback
        self.enabled = True
        self._module = None
        self._runtime_config: dict[str, Any] = {
            "realtime_mode": config.mi_realtime_mode,
            "serial_port": config.mi_serial_port,
            "board_id": config.mi_board_id,
            "model_path": Path(config.mi_model_path),
        }
        self.predictor = None
        self.worker = None
        self.analysis_worker = None
        self.eeg_thread: Optional[QThread] = None
        self.analysis_thread: Optional[QThread] = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.status_callback(f"MI runtime enabled={self.enabled}")

    def set_runtime_config(self, **kwargs: Any) -> None:
        self._runtime_config.update(kwargs)

    def start(self) -> None:
        if self.worker is not None:
            return
        module = self._load_module()
        runtime_config, predictor = self._build_predictor(module)
        self.predictor = predictor
        self.worker = module.EEGWorker(runtime_config, predictor)
        self.analysis_worker = module.AnalysisWorker(predictor, runtime_config["realtime_mode"])
        self.eeg_thread = QThread()
        self.analysis_thread = QThread()

        self.worker.moveToThread(self.eeg_thread)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.eeg_thread.started.connect(self.worker.start_collection)
        self.worker.finished.connect(self.eeg_thread.quit)
        self.worker.finished.connect(self.analysis_thread.quit)
        self.worker.status_changed.connect(self.status_callback)
        self.worker.error_occurred.connect(self.status_callback)
        self.worker.sampling_rate_ready.connect(self.analysis_worker.set_live_sampling_rate)
        self.worker.data_ready.connect(self.analysis_worker.process_data, type=Qt.QueuedConnection)
        self.analysis_worker.result_ready.connect(self._handle_result)
        self.analysis_worker.status_ready.connect(self.status_callback)
        self.eeg_thread.finished.connect(self.worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)

        self.analysis_thread.start()
        self.eeg_thread.start()
        self.status_callback("MI runtime started.")

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
        self.predictor = None
        self.status_callback("MI runtime stopped.")

    def healthcheck(self) -> dict[str, object]:
        return {
            "running": self.worker is not None,
            "enabled": self.enabled,
            "model_path": str(self._runtime_config["model_path"]),
        }

    def _handle_result(self, result: dict[str, object]) -> None:
        if not self.enabled:
            return
        self.result_callback(result)

    def _load_module(self):
        if self._module is None:
            self._module = load_module("hybrid_reference_mi_runtime", self.config.mi_reference_path)
        return self._module

    def _build_predictor(self, module):
        runtime_config = dict(module.USER_CONFIG)
        runtime_config.update(self._runtime_config)
        runtime_config["realtime_mode"] = str(runtime_config.get("realtime_mode", "continuous")).lower()

        model_path = Path(runtime_config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"MI model file not found: {model_path}")

        artifact = module.load_realtime_model(model_path)

        recommended_runtime = {}
        if bool(runtime_config.get("use_artifact_recommended_thresholds", True)):
            recommended_runtime = dict(artifact.get("recommended_runtime") or {})
        recommended_gate_runtime = {}
        control_gate_artifact = artifact.get("control_gate") if isinstance(artifact, dict) else None
        if bool(runtime_config.get("use_artifact_recommended_gate_thresholds", True)) and isinstance(control_gate_artifact, dict):
            recommended_gate_runtime = dict(control_gate_artifact.get("recommended_runtime") or {})

        confidence_threshold = float(recommended_runtime.get("confidence_threshold", runtime_config["confidence_threshold"]))
        margin_threshold = float(recommended_runtime.get("margin_threshold", runtime_config["margin_threshold"]))
        gate_confidence_threshold = float(
            recommended_gate_runtime.get("confidence_threshold", runtime_config["gate_confidence_threshold"])
        )
        gate_margin_threshold = float(
            recommended_gate_runtime.get("margin_threshold", runtime_config["gate_margin_threshold"])
        )

        predictor = module.RealtimeMIPredictor(
            artifact=artifact,
            history_len=int(runtime_config["history_len"]),
            confidence_threshold=confidence_threshold,
            gate_confidence_threshold=gate_confidence_threshold,
            probability_smoothing=float(runtime_config["probability_smoothing"]),
            margin_threshold=margin_threshold,
            gate_margin_threshold=gate_margin_threshold,
            switch_delta=float(runtime_config["switch_delta"]),
            hold_confidence_drop=float(runtime_config["hold_confidence_drop"]),
            hold_margin_drop=float(runtime_config["hold_margin_drop"]),
            release_windows=int(runtime_config["release_windows"]),
            gate_release_windows=int(runtime_config["gate_release_windows"]),
            min_stable_windows=int(runtime_config["min_stable_windows"]),
            flatline_std_threshold=float(runtime_config["flatline_std_threshold"]),
            dominant_channel_ratio_threshold=float(runtime_config["dominant_channel_ratio_threshold"]),
            max_bad_channels=int(runtime_config["max_bad_channels"]),
            artifact_freeze_windows=int(runtime_config["artifact_freeze_windows"]),
        )
        runtime_config["confidence_threshold"] = confidence_threshold
        runtime_config["margin_threshold"] = margin_threshold
        runtime_config["gate_confidence_threshold"] = gate_confidence_threshold
        runtime_config["gate_margin_threshold"] = gate_margin_threshold
        return runtime_config, predictor

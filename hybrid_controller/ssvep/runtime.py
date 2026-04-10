from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PyQt5.QtCore import QThread, Qt

from hybrid_controller.config import AppConfig
from hybrid_controller.ssvep.profiles import (
    ProfileSummary,
    ProfileStore,
    build_timestamped_profile_path,
    infer_profile_timestamp,
)
from hybrid_controller.ssvep.service import AsyncFbccaService


class SSVEPRuntime:
    def __init__(
        self,
        config: AppConfig,
        command_callback: Callable[[object], None],
        status_callback: Callable[[str], None],
        state_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.config = config
        self.command_callback = command_callback
        self.status_callback = status_callback
        self.state_callback = state_callback
        self.mode = "idle"
        self._service = AsyncFbccaService(config)
        self._backend = self._service.backend
        self._profile_store: ProfileStore = self._service.profile_store
        self._runtime_config: dict[str, Any] = {
            "serial_port": config.ssvep_serial_port,
            "board_id": config.ssvep_board_id,
            "freqs": tuple(config.ssvep_freqs),
            "model_name": config.ssvep_model_name,
            "profile_dir": Path(config.ssvep_profile_dir),
            "current_profile_path": Path(config.ssvep_current_profile_path),
            "default_profile_path": Path(config.ssvep_default_profile_path),
            "allow_fallback_profile": bool(config.ssvep_allow_fallback_profile),
            "auto_use_latest_profile": bool(config.ssvep_auto_use_latest_profile),
            "prefer_default_profile": bool(config.ssvep_prefer_default_profile),
            "recent_profile_limit": int(config.ssvep_recent_profile_limit),
            "prepare_sec": float(config.ssvep_pretrain_prepare_sec),
            "active_sec": float(config.ssvep_pretrain_active_sec),
            "rest_sec": float(config.ssvep_pretrain_rest_sec),
            "target_repeats": int(config.ssvep_pretrain_target_repeats),
            "idle_repeats": int(config.ssvep_pretrain_idle_repeats),
            "win_sec": float(config.ssvep_win_sec),
            "step_sec": float(config.ssvep_step_sec),
        }
        self.connect_thread: Optional[QThread] = None
        self.pretrain_thread: Optional[QThread] = None
        self.online_thread: Optional[QThread] = None
        self.connect_worker = None
        self.pretrain_worker = None
        self.online_worker = None
        self.connected = False
        self.device_info: dict[str, object] = {}
        self.current_profile_path: Path | None = None
        self.current_profile_source = "uninitialized"
        self._session_profile_path: Path | None = None
        self.last_pretrain_time: str | None = None
        self.last_result: dict[str, object] | None = None
        self.last_error: str | None = None
        self._last_command_signature: tuple[str, float] | None = None
        self._ensure_profile_dir()
        self._bootstrap_current_profile_state()

    def set_mode(self, mode: str) -> None:
        next_mode = str(mode)
        if next_mode == self.mode:
            return
        self.mode = next_mode
        self.status_callback(f"SSVEP mode -> {self.mode}")
        self._emit_state()

    def set_runtime_config(self, **kwargs: Any) -> None:
        self._runtime_config.update(kwargs)
        if "profile_dir" in kwargs or "current_profile_path" in kwargs or "default_profile_path" in kwargs:
            self._ensure_profile_dir()
        self._emit_state()

    def connect_device(self) -> None:
        if self.connect_worker is not None:
            self.status_callback("SSVEP device connect already running.")
            return
        try:
            requested_serial = self._backend.normalize_serial_port(self._runtime_config["serial_port"])
            if self._backend.serial_port_is_auto(requested_serial):
                candidates = self._backend.list_serial_port_candidates()
                preview = ", ".join(candidates[:8]) if candidates else "(none)"
                self.status_callback(f"SSVEP device connect started (auto scan): {preview}")
            else:
                self.status_callback(f"SSVEP device connect started (serial={requested_serial}).")
            self.connect_thread = QThread()
            self.connect_worker = self._backend.device_check_worker(
                serial_port=requested_serial,
                board_id=self._runtime_config["board_id"],
            )
            self.connect_worker.moveToThread(self.connect_thread)
            self.connect_thread.started.connect(self.connect_worker.run)
            self.connect_worker.connected.connect(self._on_device_connected)
            self.connect_worker.error.connect(self._on_runtime_error)
            self.connect_worker.finished.connect(self.connect_thread.quit)
            self.connect_worker.finished.connect(self.connect_worker.deleteLater)
            self.connect_thread.finished.connect(self._cleanup_connect_worker)
            self.connect_thread.start()
            self._emit_state()
        except Exception as error:
            self.connect_thread = None
            self.connect_worker = None
            self.last_error = str(error)
            self.status_callback(f"SSVEP device connect unavailable: {error}")
            self._emit_state()

    def start_pretrain(self) -> None:
        if self.pretrain_worker is not None:
            self.status_callback("SSVEP pretrain already running.")
            return
        try:
            self._stop_online_worker(wait_ms=2000)
            timestamped_path = build_timestamped_profile_path(self._profile_dir())
            config = self._backend.create_single_model_config(
                serial_port=str(self._runtime_config["serial_port"]),
                board_id=int(self._runtime_config["board_id"]),
                freqs=tuple(float(freq) for freq in self._runtime_config["freqs"]),
                model_name=str(self._runtime_config["model_name"]),
                profile_path=Path(timestamped_path),
                allow_default_profile=False,
                prepare_sec=float(self._runtime_config["prepare_sec"]),
                active_sec=float(self._runtime_config["active_sec"]),
                rest_sec=float(self._runtime_config["rest_sec"]),
                target_repeats=int(self._runtime_config["target_repeats"]),
                idle_repeats=int(self._runtime_config["idle_repeats"]),
                win_sec=float(self._runtime_config["win_sec"]),
                step_sec=float(self._runtime_config["step_sec"]),
            )
            self.pretrain_thread = QThread()
            self.pretrain_worker = self._backend.pretrain_worker(config)
            self.pretrain_worker.moveToThread(self.pretrain_thread)
            self.pretrain_thread.started.connect(self.pretrain_worker.run)
            self.pretrain_worker.log.connect(lambda text: self.status_callback(f"SSVEP pretrain: {text}"))
            self.pretrain_worker.phase_changed.connect(self._on_pretrain_phase_changed)
            self.pretrain_worker.profile_ready.connect(self._on_profile_ready)
            self.pretrain_worker.error.connect(self._on_runtime_error)
            self.pretrain_worker.finished.connect(self.pretrain_thread.quit)
            self.pretrain_worker.finished.connect(self.pretrain_worker.deleteLater)
            self.pretrain_thread.finished.connect(self._cleanup_pretrain_worker)
            self.pretrain_thread.start()
            self.status_callback(f"SSVEP pretrain started -> {timestamped_path.name}")
            self._emit_state()
        except Exception as error:
            self.pretrain_thread = None
            self.pretrain_worker = None
            self.last_error = str(error)
            self.status_callback(f"SSVEP pretrain unavailable: {error}")
            self._emit_state()

    def load_profile_from_path(self, profile_path: Path | str) -> Path:
        path = Path(profile_path).resolve()
        profile = self._backend.load_profile(path, fallback_freqs=self._runtime_config["freqs"], require_exists=True)
        self._ensure_profile_compatible(profile=profile, profile_path=path)
        self._session_profile_path = path
        self.current_profile_path = path
        self.current_profile_source = "session"
        self.last_pretrain_time = infer_profile_timestamp(path)
        self.last_error = None
        self.status_callback(f"SSVEP profile selected for current session: {path}")
        if self.state_callback is not None:
            self.state_callback(
                {
                    "type": "profile_loaded",
                    "profile_path": str(path),
                    "profile_source": self.current_profile_source,
                    "model_name": str(profile.model_name),
                }
            )
        self._emit_state()
        return path

    def start(self) -> None:
        self.start_online()

    def start_online(self) -> None:
        if self.online_worker is not None:
            self.status_callback("SSVEP runtime already running.")
            return
        try:
            active_profile_path, profile_source = self._resolve_active_profile_for_online()
            config = self._backend.create_single_model_config(
                serial_port=str(self._runtime_config["serial_port"]),
                board_id=int(self._runtime_config["board_id"]),
                freqs=tuple(float(freq) for freq in self._runtime_config["freqs"]),
                model_name=str(self._runtime_config["model_name"]),
                profile_path=Path(active_profile_path),
                allow_default_profile=bool(self._runtime_config["allow_fallback_profile"]),
                prepare_sec=float(self._runtime_config["prepare_sec"]),
                active_sec=float(self._runtime_config["active_sec"]),
                rest_sec=float(self._runtime_config["rest_sec"]),
                target_repeats=int(self._runtime_config["target_repeats"]),
                idle_repeats=int(self._runtime_config["idle_repeats"]),
                win_sec=float(self._runtime_config["win_sec"]),
                step_sec=float(self._runtime_config["step_sec"]),
            )
            self.current_profile_path = Path(active_profile_path)
            self.current_profile_source = str(profile_source)
            self.last_error = None
            self._last_command_signature = None
            self.online_thread = QThread()
            self.online_worker = self._backend.online_worker(config)
            self.online_worker.moveToThread(self.online_thread)
            self.online_thread.started.connect(self.online_worker.run)
            self.online_worker.log.connect(lambda text: self.status_callback(f"SSVEP online: {text}"))
            self.online_worker.phase_changed.connect(self._on_online_phase_changed)
            self.online_worker.result.connect(self._on_online_result, type=Qt.QueuedConnection)
            self.online_worker.error.connect(self._on_runtime_error)
            self.online_worker.finished.connect(self.online_thread.quit)
            self.online_worker.finished.connect(self.online_worker.deleteLater)
            self.online_thread.finished.connect(self._cleanup_online_worker)
            self.online_thread.start()
            self.status_callback(
                f"SSVEP online started ({self.current_profile_source}) -> {Path(active_profile_path).name}"
            )
            self._emit_state()
        except Exception as error:
            self.online_thread = None
            self.online_worker = None
            self.last_error = str(error)
            self.status_callback(f"SSVEP online unavailable: {error}")
            self._emit_state()

    def stop(self) -> None:
        self.stop_online()
        self._stop_pretrain_worker(wait_ms=2000)
        self._stop_connect_worker(wait_ms=1000)
        self.status_callback("SSVEP runtime stopped.")
        self._emit_state()

    def stop_online(self) -> None:
        self._stop_online_worker(wait_ms=2000)
        self.mode = "idle"
        self._last_command_signature = None
        self.status_callback("SSVEP online stopped.")
        self._emit_state()

    def clear_session_profile(self) -> None:
        self._session_profile_path = None
        current_path = self._current_profile_path()
        if current_path.exists():
            self.current_profile_path = current_path
            self.current_profile_source = "current"
        else:
            default_path = self._resolve_default_profile()
            if default_path is not None:
                self.current_profile_path = default_path
                self.current_profile_source = "default"
            else:
                self.current_profile_path = None
                self.current_profile_source = "latest" if self._runtime_config["auto_use_latest_profile"] else "fallback"
        self.status_callback("SSVEP profile selection reset to automatic mode.")
        self._emit_state()

    def healthcheck(self) -> dict[str, object]:
        return {
            "running": self.online_worker is not None,
            "mode": self.mode,
            "connected": self.connected,
            "profile_path": None if self.current_profile_path is None else str(self.current_profile_path),
            "profile_source": self.current_profile_source,
        }

    def _bootstrap_current_profile_state(self) -> None:
        current_path = self._current_profile_path()
        if current_path.exists():
            self.current_profile_path = current_path
            self.current_profile_source = "current"
            self.last_pretrain_time = infer_profile_timestamp(current_path)
            return
        default_path = self._resolve_default_profile()
        if default_path is not None:
            self.current_profile_path = default_path
            self.current_profile_source = "default"
            self.last_pretrain_time = infer_profile_timestamp(default_path)
        else:
            self.current_profile_path = None
            self.current_profile_source = "fallback"

    def _resolve_active_profile_for_online(self) -> tuple[Path, str]:
        if self._session_profile_path is not None and self._session_profile_path.exists():
            profile = self._backend.load_profile(
                self._session_profile_path,
                fallback_freqs=self._runtime_config["freqs"],
                require_exists=True,
            )
            self._ensure_profile_compatible(profile=profile, profile_path=self._session_profile_path)
            self.last_pretrain_time = infer_profile_timestamp(self._session_profile_path)
            return self._session_profile_path, "session"

        if bool(self._runtime_config["auto_use_latest_profile"]):
            latest_profile = self._resolve_latest_compatible_profile()
            if latest_profile is not None:
                self.current_profile_path = latest_profile.path
                self.current_profile_source = "latest"
                self.last_pretrain_time = latest_profile.timestamp or infer_profile_timestamp(latest_profile.path)
                self.status_callback(f"SSVEP auto profile -> {latest_profile.name}")
                return latest_profile.path, "latest"

        current_path = self._current_profile_path()
        if current_path.exists():
            profile = self._backend.load_profile(current_path, fallback_freqs=self._runtime_config["freqs"], require_exists=True)
            self._ensure_profile_compatible(profile=profile, profile_path=current_path)
            self.last_pretrain_time = infer_profile_timestamp(current_path)
            return current_path, "current"

        if bool(self._runtime_config["prefer_default_profile"]):
            default_path = self._resolve_default_profile()
            if default_path is not None:
                self.last_pretrain_time = infer_profile_timestamp(default_path)
                self.status_callback(f"SSVEP default profile -> {default_path.name}")
                return default_path, "default"

        if not bool(self._runtime_config["allow_fallback_profile"]):
            raise FileNotFoundError(
                "SSVEP profile unavailable: no session/latest/current/default profile, and fallback is disabled."
            )
        fallback_path = self._profile_dir() / "fallback_fbcca_profile.json"
        fallback_profile = self._backend.default_profile(self._runtime_config["freqs"])
        self._backend.save_profile(fallback_profile, fallback_path)
        self.last_pretrain_time = infer_profile_timestamp(fallback_path)
        return fallback_path, "fallback"

    def _emit_state(self) -> None:
        if self.state_callback is None:
            return
        profile_summaries = self.list_profiles(limit=int(self._runtime_config["recent_profile_limit"]))
        latest_profile = self._resolve_latest_compatible_profile()
        payload: dict[str, object] = {
            "type": "runtime_state",
            "running": self.online_worker is not None,
            "busy": any(
                worker is not None
                for worker in (
                    self.connect_worker,
                    self.pretrain_worker,
                    self.online_worker,
                )
            ),
            "connect_active": self.connect_worker is not None,
            "pretrain_active": self.pretrain_worker is not None,
            "online_active": self.online_worker is not None,
            "mode": self.mode,
            "connected": self.connected,
            "profile_path": None if self.current_profile_path is None else str(self.current_profile_path),
            "profile_source": self.current_profile_source,
            "last_pretrain_time": self.last_pretrain_time,
            "model_name": str(self._runtime_config["model_name"]),
            "last_result": dict(self.last_result) if isinstance(self.last_result, dict) else None,
            "last_error": self.last_error,
            "allow_fallback_profile": bool(self._runtime_config["allow_fallback_profile"]),
            "default_profile_path": str(self._default_profile_path()),
            "default_profile_available": self._default_profile_path().exists(),
            "latest_profile_path": str(latest_profile.path) if latest_profile is not None else "--",
            "profile_count": len(profile_summaries),
            "session_profile_path": str(self._session_profile_path) if self._session_profile_path else None,
            "available_profiles": [
                {
                    "name": summary.name,
                    "path": str(summary.path),
                    "timestamp": summary.timestamp,
                    "kind": summary.kind,
                    "display_name": summary.display_name,
                }
                for summary in profile_summaries
            ],
            "status_hint": self._build_status_hint(profile_summaries),
        }
        self.state_callback(payload)

    def _ensure_profile_dir(self) -> None:
        self._profile_dir().mkdir(parents=True, exist_ok=True)
        self._current_profile_path().parent.mkdir(parents=True, exist_ok=True)
        self._default_profile_path().parent.mkdir(parents=True, exist_ok=True)

    def _profile_dir(self) -> Path:
        return Path(self._runtime_config["profile_dir"]).resolve()

    def _current_profile_path(self) -> Path:
        return Path(self._runtime_config["current_profile_path"]).resolve()

    def _default_profile_path(self) -> Path:
        return Path(self._runtime_config["default_profile_path"]).resolve()

    def _copy_profile_to_current(self, source_path: Path | str) -> None:
        self._profile_store.copy_to_current(source_path)

    def list_profiles(self, *, limit: int | None = None) -> list[ProfileSummary]:
        return self._profile_store.list_profiles(limit=limit)

    def latest_profile(self) -> ProfileSummary | None:
        return self._resolve_latest_compatible_profile()

    def status_hint(self, summaries: list[ProfileSummary] | None = None) -> str:
        if summaries is None:
            summaries = self.list_profiles(limit=int(self._runtime_config["recent_profile_limit"]))
        return self._build_status_hint(summaries)

    def _resolve_latest_compatible_profile(self) -> ProfileSummary | None:
        for summary in self._profile_store.list_profiles(include_current_alias=False, include_fallback=False):
            if summary.kind != "history":
                continue
            try:
                profile = self._backend.load_profile(
                    summary.path,
                    fallback_freqs=self._runtime_config["freqs"],
                    require_exists=True,
                )
                self._ensure_profile_compatible(profile=profile, profile_path=summary.path)
                return summary
            except Exception:
                continue
        return None

    def _resolve_default_profile(self) -> Path | None:
        default_path = self._default_profile_path()
        if not default_path.exists():
            return None
        try:
            profile = self._backend.load_profile(
                default_path,
                fallback_freqs=self._runtime_config["freqs"],
                require_exists=True,
            )
            self._ensure_profile_compatible(profile=profile, profile_path=default_path)
        except Exception:
            return None
        return default_path

    def _ensure_profile_compatible(self, *, profile: Any, profile_path: Path) -> None:
        expected_model = str(self._runtime_config["model_name"]).strip().lower()
        actual_model = str(getattr(profile, "model_name", "")).strip().lower()
        if actual_model != expected_model:
            raise ValueError(
                f"SSVEP profile model mismatch: expected '{expected_model}', got '{actual_model}' ({profile_path})."
            )
        expected_freqs = tuple(float(freq) for freq in self._runtime_config["freqs"])
        actual_freqs = tuple(float(freq) for freq in getattr(profile, "freqs", ()))
        if len(actual_freqs) != len(expected_freqs) or any(
            abs(left - right) > 1e-6 for left, right in zip(actual_freqs, expected_freqs)
        ):
            raise ValueError(
                f"SSVEP profile freqs mismatch: expected {expected_freqs}, got {actual_freqs} ({profile_path})."
            )

    def _build_status_hint(self, summaries: list[ProfileSummary]) -> str:
        if self.last_error and not self.connected and self.connect_worker is None and self.online_worker is None:
            return f"设备连接失败：{self.last_error}"
        if self.online_worker is not None:
            if self.current_profile_source == "session":
                return "当前使用手动选择的 profile（仅本次会话生效）。"
            if self.current_profile_source == "default":
                return "当前使用默认通用 profile，建议后续做一次个人预训练。"
            if self.current_profile_source == "fallback":
                return "当前使用默认 fallback profile，可先预训练提升稳定性。"
            return "在线识别运行中。"
        if self.pretrain_worker is not None:
            return "正在采集并训练 FBCCA profile。"
        if self.connect_worker is not None:
            return "正在连接脑电设备。"
        if self.current_profile_source == "default":
            return "当前使用默认通用 profile。你也可以手动选择历史 profile 或先预训练。"
        if self.current_profile_source == "fallback":
            if summaries:
                return "未设置当前 profile。可选择“自动（最新训练）”，或直接用 fallback 启动。"
            return "当前没有已训练 profile。可先预训练，或直接用 fallback 启动。"
        if self.current_profile_source == "latest":
            return "当前使用自动模式：已选最新可兼容 profile。"
        if self.current_profile_source == "session":
            return "当前已手动选择 profile（仅本次会话生效）。"
        return "设备已就绪，可直接开始 SSVEP 识别。"

    def _on_device_connected(self, payload: object) -> None:
        if isinstance(payload, dict):
            self.connected = True
            self.device_info = dict(payload)
        else:
            self.connected = True
            self.device_info = {}
        self.last_error = None
        self.status_callback(
            "SSVEP device connected: serial={} fs={}Hz".format(
                self.device_info.get("resolved_serial_port", self._runtime_config["serial_port"]),
                self.device_info.get("sampling_rate", "--"),
            )
        )
        if self.state_callback is not None:
            self.state_callback({"type": "device_connected", "device_info": dict(self.device_info)})
        self._emit_state()

    def _on_pretrain_phase_changed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        title = str(payload.get("title", "SSVEP pretrain"))
        detail = str(payload.get("detail", ""))
        remaining = payload.get("remaining_sec")
        suffix = "" if remaining in {None, ""} else f" ({remaining}s)"
        self.status_callback(f"{title}{suffix}: {detail}")
        if self.state_callback is not None:
            self.state_callback({"type": "pretrain_phase", "payload": dict(payload)})
        self._emit_state()

    def _on_online_phase_changed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        title = str(payload.get("title", "SSVEP online"))
        detail = str(payload.get("detail", ""))
        self.status_callback(f"{title}: {detail}")
        if self.state_callback is not None:
            self.state_callback({"type": "online_phase", "payload": dict(payload)})
        self._emit_state()

    def _on_profile_ready(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        profile_path = Path(str(payload.get("profile_path", ""))).resolve()
        if profile_path.exists():
            self._copy_profile_to_current(profile_path)
            self.current_profile_path = self._current_profile_path()
            self.current_profile_source = "trained"
            self.last_pretrain_time = infer_profile_timestamp(profile_path)
        summary_text = str(payload.get("summary_text", "")).strip()
        self.last_error = None
        self.status_callback(
            f"SSVEP pretrain completed -> {profile_path.name}"
            + (f" | {summary_text}" if summary_text else "")
        )
        if self.state_callback is not None:
            self.state_callback(
                {
                    "type": "profile_ready",
                    "profile_path": str(profile_path),
                    "current_profile_path": str(self.current_profile_path) if self.current_profile_path else None,
                    "profile_source": self.current_profile_source,
                    "summary": payload.get("summary"),
                    "summary_text": summary_text,
                }
            )
        self._emit_state()

    def _on_online_result(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self.last_result = dict(payload)
        raw_summary = "state={state} selected={selected_freq} pred={pred_freq} margin={margin:.4f} ratio={ratio:.4f} stable={stable_windows}".format(
            state=str(payload.get("state", "--")),
            selected_freq="None" if payload.get("selected_freq") is None else f"{float(payload['selected_freq']):g}Hz",
            pred_freq="None" if payload.get("pred_freq") is None else f"{float(payload['pred_freq']):g}Hz",
            margin=float(payload.get("margin", 0.0)),
            ratio=float(payload.get("ratio", 0.0)),
            stable_windows=int(payload.get("stable_windows", 0)),
        )
        self.status_callback(f"SSVEP raw: {raw_summary}")
        self._maybe_emit_selected_command(payload)
        if self.state_callback is not None:
            self.state_callback({"type": "online_result", "payload": dict(payload)})
        self._emit_state()

    def _maybe_emit_selected_command(self, payload: dict[str, object]) -> None:
        state = str(payload.get("state", "")).strip().lower()
        selected_freq = payload.get("selected_freq")
        if state != "selected" or selected_freq is None:
            self._last_command_signature = None
            return
        signature = (state, float(selected_freq))
        if signature == self._last_command_signature:
            return
        self._last_command_signature = signature
        self.command_callback(f"{float(selected_freq):g} Hz")

    def _on_runtime_error(self, message: object) -> None:
        text = str(message)
        self.last_error = text
        self.status_callback(text)
        if self.state_callback is not None:
            self.state_callback({"type": "runtime_error", "message": text})
        self._emit_state()

    def _cleanup_connect_worker(self) -> None:
        if self.connect_thread is not None:
            self.connect_thread.deleteLater()
        self.connect_thread = None
        self.connect_worker = None
        self._emit_state()

    def _cleanup_pretrain_worker(self) -> None:
        if self.pretrain_thread is not None:
            self.pretrain_thread.deleteLater()
        self.pretrain_thread = None
        self.pretrain_worker = None
        self._emit_state()

    def _cleanup_online_worker(self) -> None:
        if self.online_thread is not None:
            self.online_thread.deleteLater()
        self.online_thread = None
        self.online_worker = None
        self._emit_state()

    def _stop_connect_worker(self, *, wait_ms: int) -> None:
        if self.connect_thread is not None:
            self.connect_thread.quit()
            self.connect_thread.wait(wait_ms)

    def _stop_pretrain_worker(self, *, wait_ms: int) -> None:
        if self.pretrain_worker is not None:
            self.pretrain_worker.request_stop()
        if self.pretrain_thread is not None:
            self.pretrain_thread.quit()
            self.pretrain_thread.wait(wait_ms)

    def _stop_online_worker(self, *, wait_ms: int) -> None:
        if self.online_worker is not None:
            self.online_worker.request_stop()
        if self.online_thread is not None:
            self.online_thread.quit()
            self.online_thread.wait(wait_ms)

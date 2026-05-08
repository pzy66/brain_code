from __future__ import annotations

import argparse
import math
import re
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
try:
    from PyQt5.QtTextToSpeech import QTextToSpeech
except Exception:
    QTextToSpeech = None

PROJECT_DIR = Path(__file__).resolve().parents[1]
BRAIN_CODE_ROOT = PROJECT_DIR.parent
if str(BRAIN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRAIN_CODE_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from brain_workspace.paths import DATASETS_ROOT, SSVEP_DATASET_DIR, resolve_data_path
from ssvep_core.async_fbcca_idle_standalone import (
    BoardShim,
    DEFAULT_BOARD_ID,
    DEFAULT_STREAM_WARMUP_SEC,
    describe_runtime_error,
    ensure_stream_ready,
    normalize_serial_port,
    parse_freqs,
    prepare_board_session,
    read_recent_eeg_segment,
)
from apps.async_fbcca_validation_ui import (
    FourArrowStimWidget,
    PHASE_CAL_ACTIVE,
    PHASE_CAL_PREPARE,
    PHASE_CAL_REST,
    PHASE_ERROR,
    PHASE_STOPPED,
    STIMULUS_MODE_ELAPSED_TIME_SINE,
    STIMULUS_MODE_FRAME_LOCKED_SINE,
    STIMULUS_MODES,
    validate_stimulus_mode,
)
from ssvep_core.dataset import (
    ENHANCED_45M_PROTOCOL,
    CollectionProtocol,
    build_collection_trials,
    save_collection_dataset_bundle,
    sanitize_collection_token,
)
from ssvep_core.stimulus_profiles import (
    DEFAULT_STIMULUS_PROFILE_ID,
    find_matching_stimulus_profile_id,
    STIMULUS_PROFILES,
    STIMULUS_PROFILE_COMFORT_FBCCA_V1,
    get_stimulus_profile,
    profile_matches_freqs,
    select_stimulus_mode_for_profile,
    stimulus_profile_metadata,
    validate_stimulus_profile_id,
)

try:
    import winsound
except Exception:
    winsound = None


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SSVEP_DATASET_DIR
MIN_TRIAL_QUALITY_RATIO = 0.90
MAX_TRIAL_RETRIES = 3
MIN_ACTIVE_SEC_FOR_TRAINING = 1.5
MIN_PREPARE_SEC_FOR_VOICE = 1.0
MIN_REST_SEC_BETWEEN_TRIALS = 2.0

ACTIVE_START_TONE_HZ = 1200
ACTIVE_START_TONE_MS = 120
PREPARE_TONE_HZ = 660
PREPARE_TONE_MS = 90
PREPARE_TONE_GAP_MS = 80
ACTIVE_END_TONE_HZ = 760
ACTIVE_END_TONE_MS = 90
ACTIVE_END_TONE_GAP_MS = 50
ACTIVE_END_CONFIRM_TONE_HZ = 540
ACTIVE_END_CONFIRM_TONE_MS = 80
TONE_EVENT_PREPARE_START = "prepare_start"
TONE_EVENT_ACTIVE_START = "active_start"
TONE_EVENT_ACTIVE_END = "active_end"
ACTIVE_START_CUE_SEC = float(ACTIVE_START_TONE_MS) / 1000.0
VOICE_PROMPT_GUARD_SEC = 0.8
VOICE_PROMPT_FINISH_TIMEOUT_SEC = 5.0
VOICE_PROMPT_RATE = 0.0
VOICE_PROMPT_DIRECTIONS_CN = ("看上方", "看左方", "看下方", "看右方")
_DEFAULT_STIMULUS_PROFILE = get_stimulus_profile(DEFAULT_STIMULUS_PROFILE_ID)
STIM_REFRESH_RATE_HZ = float(_DEFAULT_STIMULUS_PROFILE.refresh_rate_hz)
STIM_MEAN = float(_DEFAULT_STIMULUS_PROFILE.mean)
STIM_AMP = float(_DEFAULT_STIMULUS_PROFILE.amp)
STIM_PHI = float(_DEFAULT_STIMULUS_PROFILE.phi)
STIM_RAMP_SEC = float(_DEFAULT_STIMULUS_PROFILE.ramp_sec)
DEFAULT_COLLECTION_STIMULUS_PROFILE_ID = DEFAULT_STIMULUS_PROFILE_ID
DEFAULT_COLLECTION_STIMULUS_MODE = _DEFAULT_STIMULUS_PROFILE.fallback_mode
DEFAULT_COLLECTION_FREQS = tuple(float(freq) for freq in _DEFAULT_STIMULUS_PROFILE.freqs)
DEFAULT_COLLECTION_FREQS_CSV = ",".join(f"{freq:g}" for freq in DEFAULT_COLLECTION_FREQS)
STIM_FRAME_FORMULA = "luminance(frame)=mean+amp*sin(2*pi*freq*frame/refresh_rate_hz+phi)"
ACTIVE_STIMULUS_ARM_SEC = 1.0 / STIM_REFRESH_RATE_HZ
STIMULUS_PHASE_APPLY_TIMEOUT_SEC = 1.0
STIMULUS_BACKEND_PYQT_FULLSCREEN = "pyqt_fullscreen"
STIMULUS_BACKEND_HEADLESS_NO_VISUAL = "headless_no_visual"
STIMULUS_BACKENDS = (STIMULUS_BACKEND_PYQT_FULLSCREEN, STIMULUS_BACKEND_HEADLESS_NO_VISUAL)


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def resolve_dataset_dir(value: str | Path | None = None) -> Path:
    return resolve_data_path(
        value,
        base=DATASETS_ROOT,
        default=DEFAULT_DATASET_DIR,
        purpose="SSVEP dataset dir",
    )


def wallclock_iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

DEFAULT_STABLE_PREPARE_SEC = 1.0
DEFAULT_STABLE_ACTIVE_SEC = 5.0
DEFAULT_STABLE_REST_SEC = 4.0
DEFAULT_STABLE_TARGET_REPEATS = 10
DEFAULT_STABLE_IDLE_REPEATS = 20
DEFAULT_STABLE_SWITCH_TRIALS = 14
DEFAULT_STABLE_LONG_IDLE_SEC = 0.0
DEFAULT_PRESET_NAME = "stable_12m"


@dataclass(frozen=True)
class ProtocolPreset:
    key: str
    display: str
    prepare_sec: float
    active_sec: float
    rest_sec: float
    target_repeats: int
    idle_repeats: int
    switch_trials: int
    long_idle_sec: float


STABLE_12M_PRESET = ProtocolPreset(
    key="stable_12m",
    display="稳态12分钟 (1+5+4, 目标10 空闲20 切换14)",
    prepare_sec=DEFAULT_STABLE_PREPARE_SEC,
    active_sec=DEFAULT_STABLE_ACTIVE_SEC,
    rest_sec=DEFAULT_STABLE_REST_SEC,
    target_repeats=DEFAULT_STABLE_TARGET_REPEATS,
    idle_repeats=DEFAULT_STABLE_IDLE_REPEATS,
    switch_trials=DEFAULT_STABLE_SWITCH_TRIALS,
    long_idle_sec=DEFAULT_STABLE_LONG_IDLE_SEC,
)
ENHANCED_45M_PRESET = ProtocolPreset(
    key="enhanced_45m",
    display="增强长程 (1+4+2, 目标24 空闲48 切换32)",
    prepare_sec=float(ENHANCED_45M_PROTOCOL.prepare_sec),
    active_sec=float(ENHANCED_45M_PROTOCOL.active_sec),
    rest_sec=max(float(MIN_REST_SEC_BETWEEN_TRIALS), float(ENHANCED_45M_PROTOCOL.rest_sec)),
    target_repeats=int(ENHANCED_45M_PROTOCOL.target_repeats),
    idle_repeats=int(ENHANCED_45M_PROTOCOL.idle_repeats),
    switch_trials=int(ENHANCED_45M_PROTOCOL.switch_trials),
    long_idle_sec=float(ENHANCED_45M_PROTOCOL.long_idle_sec),
)
CUSTOM_PRESET = ProtocolPreset(
    key="custom",
    display="自定义 (手动设置)",
    prepare_sec=DEFAULT_STABLE_PREPARE_SEC,
    active_sec=DEFAULT_STABLE_ACTIVE_SEC,
    rest_sec=DEFAULT_STABLE_REST_SEC,
    target_repeats=DEFAULT_STABLE_TARGET_REPEATS,
    idle_repeats=DEFAULT_STABLE_IDLE_REPEATS,
    switch_trials=DEFAULT_STABLE_SWITCH_TRIALS,
    long_idle_sec=DEFAULT_STABLE_LONG_IDLE_SEC,
)
COLLECTION_PRESETS: dict[str, ProtocolPreset] = {
    STABLE_12M_PRESET.key: STABLE_12M_PRESET,
    ENHANCED_45M_PRESET.key: ENHANCED_45M_PRESET,
    CUSTOM_PRESET.key: CUSTOM_PRESET,
}


def normalize_preset_name(raw: Optional[str]) -> str:
    value = str(raw or "").strip().lower()
    if value in COLLECTION_PRESETS:
        return value
    return CUSTOM_PRESET.key


def trial_count_for_protocol(
    target_repeats: int,
    idle_repeats: int,
    switch_trials: int,
    *,
    long_idle_sec: float = 0.0,
) -> int:
    long_idle_trials = 1 if float(long_idle_sec) > 0.0 else 0
    return int(
        max(0, int(target_repeats)) * 4
        + max(0, int(idle_repeats))
        + max(0, int(switch_trials))
        + long_idle_trials
    )


def estimate_round_seconds(
    *,
    prepare_sec: float,
    active_sec: float,
    rest_sec: float,
    target_repeats: int,
    idle_repeats: int,
    switch_trials: int,
    long_idle_sec: float = 0.0,
    refresh_rate_hz: float = STIM_REFRESH_RATE_HZ,
) -> float:
    base_trial_count = trial_count_for_protocol(
        target_repeats,
        idle_repeats,
        switch_trials,
        long_idle_sec=0.0,
    )
    trial_cue_sec = float(ACTIVE_START_CUE_SEC + estimate_active_stimulus_arm_sec(refresh_rate_hz))
    total_sec = float(base_trial_count) * float(
        max(0.0, prepare_sec) + trial_cue_sec + max(0.0, active_sec) + max(0.0, rest_sec)
    )
    if float(long_idle_sec) > 0.0:
        total_sec += float(max(0.0, prepare_sec) + trial_cue_sec + max(0.0, long_idle_sec) + max(0.0, rest_sec))
    return total_sec


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    mins, secs = divmod(total, 60)
    return f"{mins}m {secs:02d}s"


def tone_sequence_for_event(event: str) -> tuple[tuple[int, int], ...]:
    value = str(event or "").strip().lower()
    if value == TONE_EVENT_PREPARE_START:
        return (
            (PREPARE_TONE_HZ, PREPARE_TONE_MS),
            (0, PREPARE_TONE_GAP_MS),
            (PREPARE_TONE_HZ, PREPARE_TONE_MS),
        )
    if value == TONE_EVENT_ACTIVE_START:
        return ((ACTIVE_START_TONE_HZ, ACTIVE_START_TONE_MS),)
    if value == TONE_EVENT_ACTIVE_END:
        return (
            (ACTIVE_END_TONE_HZ, ACTIVE_END_TONE_MS),
            (0, ACTIVE_END_TONE_GAP_MS),
            (ACTIVE_END_CONFIRM_TONE_HZ, ACTIVE_END_CONFIRM_TONE_MS),
        )
    return ()


def tone_sequence_duration_sec(event: str) -> float:
    return float(sum(duration_ms for _frequency, duration_ms in tone_sequence_for_event(event)) / 1000.0)


def prompt_text_for_freq(freqs: Sequence[float], target_freq: Optional[float]) -> str:
    if target_freq is None:
        return "看中间"
    for index, freq in enumerate(tuple(float(value) for value in freqs)):
        if abs(float(freq) - float(target_freq)) < 1e-8:
            if 0 <= index < len(VOICE_PROMPT_DIRECTIONS_CN):
                return VOICE_PROMPT_DIRECTIONS_CN[index]
            break
    return "看目标"


def prompt_text_for_trial(freqs: Sequence[float], trial: Any) -> str:
    return prompt_text_for_freq(freqs, getattr(trial, "expected_freq", None))


def _play_tone_sequence_async(sequence: Sequence[tuple[int, int]]) -> None:
    items = tuple((int(freq), int(duration_ms)) for freq, duration_ms in sequence if int(duration_ms) > 0)
    if not items:
        return
    if winsound is not None:
        def _beep_sequence() -> None:
            for frequency, duration_ms in items:
                try:
                    if int(frequency) <= 0:
                        time.sleep(max(0.0, float(duration_ms) / 1000.0))
                    else:
                        winsound.Beep(int(frequency), int(duration_ms))
                except Exception:
                    pass

        threading.Thread(target=_beep_sequence, daemon=True).start()
        return
    app = QApplication.instance()
    if app is not None:
        app.beep()


def _play_tone_sequence_sync(sequence: Sequence[tuple[int, int]]) -> None:
    items = tuple((int(freq), int(duration_ms)) for freq, duration_ms in sequence if int(duration_ms) > 0)
    if not items:
        return
    if winsound is not None:
        for frequency, duration_ms in items:
            try:
                if int(frequency) <= 0:
                    time.sleep(max(0.0, float(duration_ms) / 1000.0))
                else:
                    winsound.Beep(int(frequency), int(duration_ms))
            except Exception:
                time.sleep(max(0.0, float(duration_ms) / 1000.0))
        return
    app = QApplication.instance()
    if app is not None:
        app.beep()
    total_ms = sum(int(duration_ms) for _frequency, duration_ms in items)
    time.sleep(max(0.0, float(total_ms) / 1000.0))


def play_collection_tone_event(payload: dict[str, Any]) -> None:
    _play_tone_sequence_async(tone_sequence_for_event(str(payload.get("event", ""))))


def play_collection_tone_event_sync(payload: dict[str, Any]) -> None:
    _play_tone_sequence_sync(tone_sequence_for_event(str(payload.get("event", ""))))


def resolve_collection_stim_refresh_rate_hz(screen: Optional[Any] = None) -> float:
    target_screen = screen
    if target_screen is None:
        app = QApplication.instance()
        if app is not None:
            target_screen = app.primaryScreen()
    if target_screen is None:
        return float(STIM_REFRESH_RATE_HZ)
    try:
        hz = float(target_screen.refreshRate())
    except Exception:
        return float(STIM_REFRESH_RATE_HZ)
    if not np.isfinite(hz) or hz <= 1.0:
        return float(STIM_REFRESH_RATE_HZ)
    return float(hz)


def validate_stimulus_frequency_set(freqs: Sequence[float], *, refresh_rate_hz: float) -> None:
    hz = float(refresh_rate_hz)
    if not np.isfinite(hz) or hz <= 1.0:
        raise ValueError("stim_refresh_rate_hz must be > 1")
    values = tuple(float(freq) for freq in freqs)
    if len(values) != 4:
        raise ValueError("stimulus frequency list must contain exactly 4 values")
    if any((not np.isfinite(freq)) or freq <= 0.0 for freq in values):
        raise ValueError("stimulus frequencies must be positive finite values")
    nyquist_hz = hz / 2.0
    too_high = [freq for freq in values if freq >= nyquist_hz]
    if too_high:
        joined = ", ".join(f"{freq:g}" for freq in too_high)
        raise ValueError(
            f"stimulus frequencies must be below half the display refresh rate "
            f"({nyquist_hz:g}Hz for {hz:g}Hz display); invalid: {joined}"
        )


def resolve_collection_stimulus_mode(
    *,
    stimulus_profile_id: str,
    refresh_rate_hz: float,
    requested_mode: Optional[str] = None,
) -> tuple[str, str]:
    requested = str(requested_mode or "").strip().lower()
    manual_mode = "" if requested in ("", "auto") else validate_stimulus_mode(requested)
    mode, reason = select_stimulus_mode_for_profile(
        stimulus_profile_id,
        refresh_rate_hz=float(refresh_rate_hz),
        requested_mode=manual_mode or "auto",
    )
    return validate_stimulus_mode(mode), str(reason)


def validate_stimulus_backend(value: str) -> str:
    backend = str(value or "").strip().lower()
    if backend not in STIMULUS_BACKENDS:
        joined = "|".join(STIMULUS_BACKENDS)
        raise ValueError(f"stimulus_backend must be one of: {joined}")
    return backend


def stimulus_backend_metadata(backend: str) -> dict[str, Any]:
    resolved_backend = validate_stimulus_backend(backend)
    rendered = resolved_backend == STIMULUS_BACKEND_PYQT_FULLSCREEN
    return {
        "stimulus_backend": resolved_backend,
        "stimulus_rendered_by_this_process": bool(rendered),
        "stimulus_mode_applies_to_rendered_stimulus": bool(rendered),
    }


def estimate_active_stimulus_arm_sec(refresh_rate_hz: float) -> float:
    hz = float(refresh_rate_hz)
    if not np.isfinite(hz) or hz <= 1.0:
        hz = float(STIM_REFRESH_RATE_HZ)
    return 1.0 / hz


def estimate_stimulus_sample_window_frame_offset(refresh_rate_hz: float) -> int:
    hz = float(refresh_rate_hz)
    if not np.isfinite(hz) or hz <= 1.0:
        hz = float(STIM_REFRESH_RATE_HZ)
    offset_sec = estimate_active_stimulus_arm_sec(hz) + float(ACTIVE_START_CUE_SEC)
    return max(0, int(round(offset_sec * hz)))


def _stimulus_freq_key(freq: float) -> str:
    return f"{float(freq):g}Hz"


def stimulus_sample_window_alignment_metadata(
    freqs: Sequence[float],
    *,
    refresh_rate_hz: float,
    backend: str,
    base_phi: float = STIM_PHI,
) -> dict[str, Any]:
    resolved_backend = validate_stimulus_backend(backend)
    if resolved_backend != STIMULUS_BACKEND_PYQT_FULLSCREEN:
        return {
            "stimulus_sequence_t0_reference": "not_rendered_by_this_process",
            "stim_phi_reference": "not_applicable",
            "stimulus_sample_window_alignment": "no_visual_backend_in_this_process",
            "stimulus_sample_window_offset_source": "not_applicable",
            "stimulus_sample_window_frame_offset_estimate": None,
            "stimulus_sample_window_offset_sec_estimate": None,
            "stimulus_sample_window_display_frame_offset_sec_estimate": None,
            "stimulus_sample_window_phase_rad_by_freq": {},
            "stimulus_sample_window_phase_cycles_by_freq": {},
        }

    hz = float(refresh_rate_hz)
    if not np.isfinite(hz) or hz <= 1.0:
        hz = float(STIM_REFRESH_RATE_HZ)
    offset_frames = estimate_stimulus_sample_window_frame_offset(hz)
    nominal_offset_sec = estimate_active_stimulus_arm_sec(hz) + float(ACTIVE_START_CUE_SEC)
    frame_offset_sec = float(offset_frames) / hz
    phase_rad_by_freq: dict[str, float] = {}
    phase_cycles_by_freq: dict[str, float] = {}
    for freq in tuple(float(item) for item in freqs):
        cycles = (float(base_phi) / (2.0 * math.pi)) + (freq * float(offset_frames) / hz)
        wrapped_cycles = cycles % 1.0
        key = _stimulus_freq_key(freq)
        phase_cycles_by_freq[key] = float(wrapped_cycles)
        phase_rad_by_freq[key] = float(wrapped_cycles * 2.0 * math.pi)
    return {
        "stimulus_sequence_t0_reference": "active_phase_ui_apply",
        "stim_phi_reference": "stimulus_sequence_t0",
        "stimulus_sample_window_alignment": "prearmed_before_eeg_buffer_clear",
        "stimulus_sample_window_offset_source": "estimated_one_ui_ack_frame_plus_start_tone_rounded_to_display_frame",
        "stimulus_sample_window_frame_offset_estimate": int(offset_frames),
        "stimulus_sample_window_offset_sec_estimate": float(nominal_offset_sec),
        "stimulus_sample_window_display_frame_offset_sec_estimate": float(frame_offset_sec),
        "stimulus_sample_window_phase_rad_by_freq": phase_rad_by_freq,
        "stimulus_sample_window_phase_cycles_by_freq": phase_cycles_by_freq,
    }


def make_collection_stim_widget(
    freqs: Sequence[float],
    *,
    refresh_rate_hz: Optional[float] = None,
    stimulus_mode: str = DEFAULT_COLLECTION_STIMULUS_MODE,
    stimulus_profile_id: str = DEFAULT_COLLECTION_STIMULUS_PROFILE_ID,
    parent: Optional[QWidget] = None,
) -> FourArrowStimWidget:
    profile = get_stimulus_profile(stimulus_profile_id)
    resolved_refresh_rate_hz = (
        resolve_collection_stim_refresh_rate_hz() if refresh_rate_hz is None else float(refresh_rate_hz)
    )
    resolved_stimulus_mode = validate_stimulus_mode(stimulus_mode)
    validate_stimulus_frequency_set(freqs, refresh_rate_hz=resolved_refresh_rate_hz)
    return FourArrowStimWidget(
        freqs=tuple(float(freq) for freq in freqs),
        refresh_rate_hz=resolved_refresh_rate_hz,
        mean=float(profile.mean),
        amp=float(profile.amp),
        phi=float(profile.phi),
        stimulus_mode=resolved_stimulus_mode,
        stimulus_profile_id=str(profile.profile_id),
        ramp_sec=float(profile.ramp_sec),
        parent=parent,
    )


class SpeechPromptPlayer(QObject):
    playback_finished = pyqtSignal(int)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._engine = QTextToSpeech(parent) if QTextToSpeech is not None else None
        self._pending_request_id = 0
        self._awaiting_completion = False
        if self._engine is not None:
            try:
                self._engine.setRate(float(VOICE_PROMPT_RATE))
                self._engine.setVolume(1.0)
                self._engine.stateChanged.connect(self._on_state_changed)
            except Exception:
                pass

    @staticmethod
    def _fallback_duration_ms(text: str) -> int:
        compact_len = len(str(text or "").strip())
        return int(min(1800, max(650, 350 + compact_len * 140)))

    def say(self, text: str, *, request_id: int = 0) -> None:
        rid = int(request_id)
        message = str(text or "").strip()
        if rid <= 0:
            return
        if not message or self._engine is None:
            QTimer.singleShot(0, lambda rid=rid: self.playback_finished.emit(rid))
            return
        try:
            self._pending_request_id = rid
            self._awaiting_completion = False
            self._engine.stop()
            self._awaiting_completion = True
            self._engine.say(message)
            QTimer.singleShot(self._fallback_duration_ms(message), lambda rid=rid: self._finish_if_pending(rid))
        except Exception:
            self._awaiting_completion = False
            self._pending_request_id = 0
            QTimer.singleShot(0, lambda rid=rid: self.playback_finished.emit(rid))

    def stop(self) -> None:
        self._awaiting_completion = False
        self._pending_request_id = 0
        if self._engine is None:
            return
        try:
            self._engine.stop()
        except Exception:
            pass

    def _finish_if_pending(self, request_id: int) -> None:
        if not self._awaiting_completion:
            return
        if int(request_id) != int(self._pending_request_id):
            return
        self._awaiting_completion = False
        self._pending_request_id = 0
        self.playback_finished.emit(int(request_id))

    def _on_state_changed(self, state) -> None:
        if not self._awaiting_completion:
            return
        ready_state = getattr(QTextToSpeech, "Ready", None)
        if ready_state is None:
            return
        try:
            is_ready = int(state) == int(ready_state)
        except Exception:
            is_ready = state == ready_state
        if not is_ready:
            return
        request_id = int(self._pending_request_id)
        if request_id > 0:
            self._finish_if_pending(request_id)


class CollectionFullscreenStimWindow(QMainWindow):
    escape_requested = pyqtSignal()
    active_phase_frame_presented = pyqtSignal(object)

    def __init__(
        self,
        *,
        freqs: Sequence[float],
        refresh_rate_hz: float,
        stimulus_mode: str,
        stimulus_profile_id: str = DEFAULT_COLLECTION_STIMULUS_PROFILE_ID,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._allow_close = False
        self.setWindowTitle("SSVEP 全屏刺激")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setStyleSheet("background-color: black;")
        self.stim = make_collection_stim_widget(
            freqs,
            refresh_rate_hz=refresh_rate_hz,
            stimulus_mode=stimulus_mode,
            stimulus_profile_id=stimulus_profile_id,
        )
        self.stim.active_phase_frame_presented.connect(self.active_phase_frame_presented.emit)
        self.setCentralWidget(self.stim)

    def apply_phase(self, phase: dict[str, Any]) -> None:
        self.stim.apply_phase(phase)

    def close_from_owner(self) -> None:
        self._allow_close = True
        try:
            self.stim.stop_clock()
        except Exception:
            pass
        self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.escape_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._allow_close:
            try:
                self.stim.stop_clock()
            except Exception:
                pass
            event.accept()
            return
        self.escape_requested.emit()
        event.ignore()


def _validate_collection_protocol_legacy_unused(*, active_sec: float) -> None:
    if float(active_sec) < float(MIN_ACTIVE_SEC_FOR_TRAINING):
        raise ValueError(
            f"active_sec 必须 >= {MIN_ACTIVE_SEC_FOR_TRAINING:.1f}s，"
            "否则不满足训练质量门槛"
        )


def _validate_collection_protocol(
    *,
    active_sec: float,
    long_idle_sec: float = 0.0,
    prepare_sec: float = DEFAULT_STABLE_PREPARE_SEC,
    rest_sec: float = DEFAULT_STABLE_REST_SEC,
) -> None:
    if float(prepare_sec) < float(MIN_PREPARE_SEC_FOR_VOICE):
        raise ValueError(f"prepare_sec must be >= {MIN_PREPARE_SEC_FOR_VOICE:.1f}s")
    if float(active_sec) < float(MIN_ACTIVE_SEC_FOR_TRAINING):
        raise ValueError(f"active_sec must be >= {MIN_ACTIVE_SEC_FOR_TRAINING:.1f}s")
    if float(rest_sec) < float(MIN_REST_SEC_BETWEEN_TRIALS):
        raise ValueError(f"rest_sec must be >= {MIN_REST_SEC_BETWEEN_TRIALS:.1f}s")
    if float(long_idle_sec) < 0.0:
        raise ValueError("long_idle_sec must be >= 0")
    if 0.0 < float(long_idle_sec) < float(MIN_ACTIVE_SEC_FOR_TRAINING):
        raise ValueError(f"long_idle_sec must be 0 or >= {MIN_ACTIVE_SEC_FOR_TRAINING:.1f}s")


def _auto_session_base_id(subject_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_subject = sanitize_collection_token(subject_id or "subject", default="subject")
    return f"{clean_subject}_collection_{stamp}"


def _strip_round_suffix(session_base: str) -> str:
    return re.sub(r"_r\d+$", "", str(session_base).strip(), flags=re.IGNORECASE)


def _build_round_session_id(session_base: str, round_index: int) -> str:
    base = sanitize_collection_token(_strip_round_suffix(session_base), default="session")
    return f"{base}_r{int(round_index):02d}"


def build_collection_output_session_id(
    session_id: str,
    *,
    collection_aborted: bool,
    dataset_dir: Optional[Path] = None,
    stamp: Optional[str] = None,
) -> str:
    base = sanitize_collection_token(session_id or "session", default="session")
    suffix = str(stamp or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()
    suffix = re.sub(r"[^0-9A-Za-z_-]+", "_", suffix) or datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{base}_aborted_{suffix}" if bool(collection_aborted) else base
    if dataset_dir is None:
        return candidate

    root = Path(dataset_dir).expanduser().resolve()
    session_dir = root / candidate
    if not (session_dir / "session_manifest.json").exists() and not (session_dir / "raw_trials.npz").exists():
        return candidate

    rerun_base = f"{candidate}_rerun_{suffix}"
    rerun_candidate = rerun_base
    counter = 2
    while True:
        rerun_dir = root / rerun_candidate
        if not (rerun_dir / "session_manifest.json").exists() and not (rerun_dir / "raw_trials.npz").exists():
            return rerun_candidate
        rerun_candidate = f"{rerun_base}_{counter}"
        counter += 1


@dataclass(frozen=True)
class CollectionConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    subject_id: str
    session_id: str
    session_index: int
    dataset_dir: Path
    protocol_name: str = DEFAULT_PRESET_NAME
    prepare_sec: float = DEFAULT_STABLE_PREPARE_SEC
    active_sec: float = DEFAULT_STABLE_ACTIVE_SEC
    rest_sec: float = DEFAULT_STABLE_REST_SEC
    target_repeats: int = DEFAULT_STABLE_TARGET_REPEATS
    idle_repeats: int = DEFAULT_STABLE_IDLE_REPEATS
    switch_trials: int = DEFAULT_STABLE_SWITCH_TRIALS
    long_idle_sec: float = DEFAULT_STABLE_LONG_IDLE_SEC
    seed: int = 20260410
    rounds_planned: int = 1
    round_index: int = 1
    estimated_round_sec: float = 0.0
    stimulus_profile_id: str = DEFAULT_COLLECTION_STIMULUS_PROFILE_ID
    stim_refresh_rate_hz: float = STIM_REFRESH_RATE_HZ
    stimulus_mode: str = DEFAULT_COLLECTION_STIMULUS_MODE
    stimulus_mode_selection_reason: str = ""
    comfort_rating: Optional[int] = None
    screen_brightness_note: str = "not_recorded"
    stimulus_backend: str = STIMULUS_BACKEND_HEADLESS_NO_VISUAL
    sync_stimulus_phase: bool = False
    sync_voice_prompt: bool = False
    simulation_only: bool = False


class DeviceCheckWorker(QObject):
    connected = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, *, serial_port: str, board_id: int) -> None:
        super().__init__()
        self.serial_port = normalize_serial_port(serial_port)
        self.board_id = int(board_id)

    @pyqtSlot()
    def run(self) -> None:
        board = None
        try:
            board, resolved_port, attempted_ports = prepare_board_session(self.board_id, self.serial_port)
            fs = int(BoardShim.get_sampling_rate(self.board_id))
            board.start_stream(450000)
            ready = int(ensure_stream_ready(board, fs))
            self.connected.emit(
                {
                    "requested_serial_port": self.serial_port,
                    "resolved_serial_port": resolved_port,
                    "attempted_ports": attempted_ports,
                    "sampling_rate": fs,
                    "ready_samples": ready,
                }
            )
        except Exception as exc:
            self.error.emit(f"连接失败：{describe_runtime_error(exc, serial_port=self.serial_port)}")
        finally:
            if board is not None:
                try:
                    board.stop_stream()
                except Exception:
                    pass
                try:
                    board.release_session()
                except Exception:
                    pass
            self.finished.emit()


class CollectionWorker(QObject):
    phase_changed = pyqtSignal(object)
    trial_tone_event = pyqtSignal(object)
    voice_prompt_event = pyqtSignal(object)
    log = pyqtSignal(str)
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: CollectionConfig) -> None:
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()
        self._stimulus_phase_applied_event = threading.Event()
        self._stimulus_phase_payload_lock = threading.Lock()
        self._last_stimulus_phase_payload: dict[str, Any] = {}
        self._voice_prompt_finished_event = threading.Event()
        self._voice_prompt_request_id = 0
        self._current_trial_active_sec = float(config.active_sec)

    def request_stop(self) -> None:
        self._stop_event.set()
        self._stimulus_phase_applied_event.set()
        self._voice_prompt_finished_event.set()

    def _emit_phase(
        self,
        mode: str,
        title: str,
        detail: str,
        *,
        flicker: bool,
        cue_freq: Optional[float],
        active_sec: Optional[float] = None,
    ) -> None:
        phase_active_sec = float(self.config.active_sec if active_sec is None else active_sec)
        if active_sec is None and mode == PHASE_CAL_ACTIVE and bool(flicker):
            phase_active_sec = float(self._current_trial_active_sec)
        self.phase_changed.emit(
            {
                "mode": mode,
                "title": title,
                "detail": detail,
                "flicker": flicker,
                "cue_freq": cue_freq,
                "active_sec": float(phase_active_sec),
                "stimulus_profile_id": str(self.config.stimulus_profile_id),
            }
        )

    def _clear_stimulus_phase_ack(self) -> None:
        self._stimulus_phase_applied_event.clear()
        with self._stimulus_phase_payload_lock:
            self._last_stimulus_phase_payload = {}

    def notify_stimulus_phase_applied(self, payload: Optional[dict[str, Any]] = None) -> None:
        if payload is None or str(payload.get("mode", "")) == PHASE_CAL_ACTIVE:
            ack_payload = dict(payload or {})
            ack_payload["ack_wall_time"] = wallclock_iso_timestamp()
            ack_payload["ack_perf_counter_sec"] = float(time.perf_counter())
            with self._stimulus_phase_payload_lock:
                self._last_stimulus_phase_payload = ack_payload
            self._stimulus_phase_applied_event.set()

    def notify_voice_prompt_finished(self, request_id: int) -> None:
        if int(request_id) <= 0:
            return
        if int(request_id) == int(self._voice_prompt_request_id):
            self._voice_prompt_finished_event.set()

    def _wait_for_stimulus_phase_applied(self) -> tuple[bool, dict[str, Any]]:
        if not bool(self.config.sync_stimulus_phase):
            return False, {}
        if self._stop_event.is_set():
            return False, {}
        ready = self._stimulus_phase_applied_event.wait(float(STIMULUS_PHASE_APPLY_TIMEOUT_SEC))
        if not ready and not self._stop_event.is_set():
            self.log.emit(
                "警告：未在超时时间内收到 UI active 刺激确认；"
                "将丢弃本次尝试并重采，避免保存视觉起点不可信的数据。"
            )
            return False, {}
        with self._stimulus_phase_payload_lock:
            payload = dict(self._last_stimulus_phase_payload)
        return bool(ready), payload

    def _sleep_interruptible(self, seconds: float) -> bool:
        return bool(self._stop_event.wait(max(0.0, float(seconds))))

    def _wait_for_voice_prompt_finished(
        self,
        request_id: int,
        *,
        trial_index: int = 0,
        total_trials: int = 0,
        retry_index: int = 0,
        timeout_sec: Optional[float] = None,
        log_timeout: bool = True,
    ) -> float:
        if not bool(self.config.sync_voice_prompt):
            return 0.0
        if int(request_id) <= 0 or self._stop_event.is_set():
            return 0.0
        timeout = (
            float(VOICE_PROMPT_FINISH_TIMEOUT_SEC)
            if timeout_sec is None
            else max(0.0, float(timeout_sec))
        )
        start = time.perf_counter()
        ready = self._voice_prompt_finished_event.wait(timeout)
        elapsed = float(max(time.perf_counter() - start, 0.0))
        if not ready and not self._stop_event.is_set():
            if bool(log_timeout):
                self.log.emit(
                    "警告：未在超时时间内收到语音提示完成确认；"
                    "将先强制停止当前语音，避免与后续提示音重叠。"
                )
            self._emit_voice_prompt(
                stop=True,
                trial_index=int(trial_index),
                total_trials=int(total_trials),
                retry_index=int(retry_index),
            )
            self._voice_prompt_finished_event.set()
        return elapsed

    def _run_prepare_window(
        self,
        *,
        request_id: int,
        trial_index: int,
        total_trials: int,
        retry_index: int,
    ) -> bool:
        prepare_sec = max(0.0, float(self.config.prepare_sec))
        started = time.perf_counter()
        voice_wait_sec = 0.0
        if int(request_id) > 0 and prepare_sec > 0.0:
            voice_wait_sec = self._wait_for_voice_prompt_finished(
                int(request_id),
                trial_index=int(trial_index),
                total_trials=int(total_trials),
                retry_index=int(retry_index),
                timeout_sec=prepare_sec,
                log_timeout=False,
            )
        elif int(request_id) > 0:
            self._emit_voice_prompt(
                stop=True,
                trial_index=int(trial_index),
                total_trials=int(total_trials),
                retry_index=int(retry_index),
            )
            self._voice_prompt_finished_event.set()
        elapsed = max(float(voice_wait_sec), max(0.0, time.perf_counter() - started))
        remaining_sec = max(0.0, prepare_sec - elapsed)
        prepare_tone_sec = tone_sequence_duration_sec(TONE_EVENT_PREPARE_START)
        if prepare_sec > 0.0 and remaining_sec >= prepare_tone_sec:
            tone_payload = self._emit_tone(
                event=TONE_EVENT_PREPARE_START,
                trial_index=int(trial_index),
                total_trials=int(total_trials),
                retry_index=int(retry_index),
            )
            play_collection_tone_event(tone_payload)
            elapsed = max(float(voice_wait_sec), max(0.0, time.perf_counter() - started))
        if self._sleep_interruptible(max(0.0, prepare_sec - elapsed)):
            return True
        self._emit_voice_prompt(
            stop=True,
            trial_index=int(trial_index),
            total_trials=int(total_trials),
            retry_index=int(retry_index),
        )
        return False

    def _emit_tone(self, *, event: str, trial_index: int, total_trials: int, retry_index: int) -> dict[str, Any]:
        payload = {
            "event": str(event),
            "round_index": int(self.config.round_index),
            "trial_index": int(trial_index),
            "total_trials": int(total_trials),
            "retry_index": int(retry_index),
        }
        self.trial_tone_event.emit(payload)
        return payload

    def _emit_voice_prompt(
        self,
        *,
        text: str = "",
        stop: bool = False,
        trial_index: int,
        total_trials: int,
        retry_index: int,
    ) -> int:
        request_id = 0
        if not bool(stop) and str(text or "").strip():
            self._voice_prompt_request_id += 1
            request_id = int(self._voice_prompt_request_id)
            self._voice_prompt_finished_event.clear()
        self.voice_prompt_event.emit(
            {
                "text": str(text),
                "stop": bool(stop),
                "request_id": int(request_id),
                "round_index": int(self.config.round_index),
                "trial_index": int(trial_index),
                "total_trials": int(total_trials),
                "retry_index": int(retry_index),
            }
        )
        return int(request_id)

    def _build_done_payload(
        self,
        *,
        collection_aborted: bool,
        collected_trials: int,
        total_trials: int,
        metadata: Optional[dict[str, Any]] = None,
        failure_reason: str = "",
    ) -> dict[str, Any]:
        payload = {
            "collection_aborted": bool(collection_aborted),
            "collected_trials": int(collected_trials),
            "total_trials": int(total_trials),
            "round_index": int(self.config.round_index),
            "rounds_planned": int(self.config.rounds_planned),
            "dataset_manifest": "",
            "dataset_npz": "",
        }
        if metadata:
            payload.update(dict(metadata))
        if str(failure_reason or "").strip():
            payload["failure_reason"] = str(failure_reason)
        return payload

    def _build_protocol_config(
        self,
        *,
        collection_aborted: bool,
        output_session_id: str,
        total_trials: int,
        saved_trial_count: int,
        failure_reason: str = "",
        continuous_board_error: str = "",
    ) -> dict[str, Any]:
        profile = get_stimulus_profile(self.config.stimulus_profile_id)
        protocol_config = {
            "collection_aborted": bool(collection_aborted),
            "requested_session_id": str(self.config.session_id),
            "saved_session_id": str(output_session_id),
            "planned_total_trials": int(total_trials),
            "saved_trial_count": int(saved_trial_count),
            "protocol_name": str(self.config.protocol_name),
            "prepare_sec": float(self.config.prepare_sec),
            "active_sec": float(self.config.active_sec),
            "rest_sec": float(self.config.rest_sec),
            "long_idle_sec": float(self.config.long_idle_sec),
            "target_repeats": int(self.config.target_repeats),
            "idle_repeats": int(self.config.idle_repeats),
            "switch_trials": int(self.config.switch_trials),
            "session_index": int(self.config.session_index),
            "seed": int(self.config.seed),
            "round_index": int(self.config.round_index),
            "rounds_planned": int(self.config.rounds_planned),
            "preset_name": str(self.config.protocol_name),
            "estimated_round_sec": float(self.config.estimated_round_sec),
            "voice_prompt_guard_sec": float(VOICE_PROMPT_GUARD_SEC),
            "voice_prompt_timing_policy": "voice_then_prepare_tone_inside_prepare_window_stop_before_active",
            "voice_prompt_extra_wait_sec": 0.0,
            "voice_prompt_finish_timeout_sec": float(VOICE_PROMPT_FINISH_TIMEOUT_SEC),
            "active_start_cue_sec": float(ACTIVE_START_CUE_SEC),
            "active_start_buffer_clear_timing": "before_start_cue",
            "active_saved_window": "last_active_sec_after_start_cue",
            "active_end_cue_timing": "after_segment_capture",
            "active_stimulus_arm_sec_estimate": float(
                estimate_active_stimulus_arm_sec(self.config.stim_refresh_rate_hz)
            ),
            "stim_frame_formula": str(STIM_FRAME_FORMULA),
            "sync_stimulus_phase": bool(self.config.sync_stimulus_phase),
        }
        protocol_config.update(
            stimulus_profile_metadata(
                self.config.stimulus_profile_id,
                stimulus_mode=str(self.config.stimulus_mode),
                refresh_rate_hz=float(self.config.stim_refresh_rate_hz),
                freqs=self.config.freqs,
                mode_selection_reason=str(self.config.stimulus_mode_selection_reason),
                comfort_rating=self.config.comfort_rating,
                screen_brightness_note=str(self.config.screen_brightness_note),
            )
        )
        protocol_config.update(stimulus_backend_metadata(self.config.stimulus_backend))
        protocol_config.update(
            stimulus_sample_window_alignment_metadata(
                self.config.freqs,
                refresh_rate_hz=self.config.stim_refresh_rate_hz,
                backend=self.config.stimulus_backend,
                base_phi=float(profile.phi),
            )
        )
        if bool(collection_aborted):
            protocol_config["aborted_reason"] = "runtime_failure" if str(failure_reason).strip() else "user_stop"
        if str(failure_reason or "").strip():
            protocol_config["failure_reason"] = str(failure_reason)
        if str(continuous_board_error or "").strip():
            protocol_config["continuous_board_error"] = str(continuous_board_error)
        return protocol_config

    def _save_collected_dataset(
        self,
        *,
        active_serial: str,
        sampling_rate: int,
        eeg_channels: Sequence[int],
        total_trials: int,
        trial_segments: Sequence[tuple[Any, np.ndarray]],
        quality_rows: Sequence[dict[str, Any]],
        collection_aborted: bool,
        failure_reason: str = "",
        continuous_board_data: Optional[np.ndarray] = None,
        continuous_board_error: str = "",
    ) -> dict[str, Any]:
        output_session_id = build_collection_output_session_id(
            self.config.session_id,
            collection_aborted=collection_aborted,
            dataset_dir=self.config.dataset_dir,
        )
        protocol_config = self._build_protocol_config(
            collection_aborted=collection_aborted,
            output_session_id=output_session_id,
            total_trials=total_trials,
            saved_trial_count=len(trial_segments),
            failure_reason=failure_reason,
            continuous_board_error=continuous_board_error,
        )
        return save_collection_dataset_bundle(
            dataset_root=self.config.dataset_dir,
            session_id=output_session_id,
            subject_id=self.config.subject_id,
            serial_port=active_serial,
            board_id=self.config.board_id,
            sampling_rate=int(sampling_rate),
            freqs=self.config.freqs,
            board_eeg_channels=tuple(int(ch) for ch in eeg_channels),
            protocol_config=protocol_config,
            trial_segments=trial_segments,
            quality_rows=quality_rows,
            continuous_board_data=continuous_board_data,
            continuous_board_info=self._continuous_board_info(continuous_board_data),
        )

    def _save_raw_board_fallback(
        self,
        *,
        active_serial: str,
        sampling_rate: int,
        eeg_channels: Sequence[int],
        total_trials: int,
        raw_board_chunks: Sequence[np.ndarray],
        failure_reason: str,
    ) -> Optional[dict[str, Any]]:
        continuous_board_data, continuous_board_error = self._try_concat_board_data_chunks(raw_board_chunks)
        if continuous_board_data is None:
            return None
        return self._save_collected_dataset(
            active_serial=active_serial,
            sampling_rate=int(sampling_rate),
            eeg_channels=eeg_channels,
            total_trials=int(total_trials),
            trial_segments=[],
            quality_rows=[],
            collection_aborted=True,
            failure_reason=failure_reason,
            continuous_board_data=continuous_board_data,
            continuous_board_error=continuous_board_error,
        )

    def _build_trial_plan(self) -> list[Any]:
        protocol = CollectionProtocol(
            name=str(self.config.protocol_name),
            prepare_sec=float(self.config.prepare_sec),
            active_sec=float(self.config.active_sec),
            rest_sec=float(self.config.rest_sec),
            target_repeats=int(self.config.target_repeats),
            idle_repeats=int(self.config.idle_repeats),
            switch_trials=int(self.config.switch_trials),
            long_idle_sec=float(self.config.long_idle_sec),
        )
        return build_collection_trials(
            self.config.freqs,
            protocol=protocol,
            seed=self.config.seed,
            session_index=self.config.session_index,
        )

    @staticmethod
    def _append_board_data_chunk(chunks: list[np.ndarray], data: Any) -> int:
        try:
            matrix = np.asarray(data, dtype=np.float64)
        except Exception:
            return 0
        if matrix.ndim != 2 or int(matrix.shape[1]) <= 0:
            return 0
        if chunks and int(chunks[0].shape[0]) != int(matrix.shape[0]):
            return 0
        chunks.append(np.ascontiguousarray(matrix, dtype=np.float64))
        return int(matrix.shape[1])

    def _drain_board_data(self, board: Any, chunks: list[np.ndarray]) -> int:
        data = board.get_board_data()
        return self._append_board_data_chunk(chunks, data)

    @staticmethod
    def _concat_board_data_chunks(chunks: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        valid = [np.asarray(chunk, dtype=np.float64) for chunk in chunks if np.asarray(chunk).ndim == 2]
        if not valid:
            return None
        if any(int(chunk.shape[1]) <= 0 for chunk in valid):
            valid = [chunk for chunk in valid if int(chunk.shape[1]) > 0]
        if not valid:
            return None
        row_count = int(valid[0].shape[0])
        compatible = [chunk for chunk in valid if int(chunk.shape[0]) == row_count]
        if not compatible:
            return None
        return np.ascontiguousarray(np.concatenate(compatible, axis=1), dtype=np.float64)

    @classmethod
    def _try_concat_board_data_chunks(cls, chunks: Sequence[np.ndarray]) -> tuple[Optional[np.ndarray], str]:
        try:
            return cls._concat_board_data_chunks(chunks), ""
        except Exception as error:
            return None, str(error)

    def _continuous_board_info(self, continuous_board_data: Optional[np.ndarray]) -> dict[str, Any]:
        if continuous_board_data is None:
            return {}
        info: dict[str, Any] = {
            "source": "brainflow_get_board_data_drain_chunks",
            "shape": [int(continuous_board_data.shape[0]), int(continuous_board_data.shape[1])],
        }
        for key, method_name in (
            ("marker_channel", "get_marker_channel"),
            ("timestamp_channel", "get_timestamp_channel"),
            ("package_num_channel", "get_package_num_channel"),
        ):
            try:
                info[key] = int(getattr(BoardShim, method_name)(self.config.board_id))
            except Exception:
                info[key] = None
        return info

    def _run_simulation_only(self) -> None:
        trials = self._build_trial_plan()
        total = len(trials)
        executed_trials = 0
        self.log.emit(
            f"流程测试开始：session={self.config.session_id}，"
            "不连接板卡、不保存数据，将完整运行语音/提示音/闪烁刺激/休息流程。"
        )
        for index, trial in enumerate(trials, start=1):
            if self._stop_event.is_set():
                break
            cue_freq = None if trial.expected_freq is None else float(trial.expected_freq)
            trial_label_lower = str(trial.label).strip().lower()
            is_long_idle = "long_idle" in trial_label_lower or "long idle" in trial_label_lower
            trial_active_sec = (
                float(self.config.long_idle_sec)
                if is_long_idle and float(self.config.long_idle_sec) > 0.0
                else float(self.config.active_sec)
            )
            self._current_trial_active_sec = float(trial_active_sec)
            prompt_base = (
                f"第 {self.config.round_index} 轮 Trial {index}/{total} 空闲（看中心）"
                if cue_freq is None
                else f"第 {self.config.round_index} 轮 Trial {index}/{total} 注视 {trial.label}"
            )
            if is_long_idle:
                prompt_base = (
                    f"第 {self.config.round_index} 轮 Trial {index}/{total} Long Idle "
                    "（保持看中心，不看任何目标）"
                )
            self._emit_phase(PHASE_CAL_PREPARE, "准备", prompt_base, flicker=False, cue_freq=cue_freq)
            self.log.emit(f"{prompt_base} [流程测试]")
            request_id = self._emit_voice_prompt(
                text=prompt_text_for_trial(self.config.freqs, trial),
                trial_index=index,
                total_trials=total,
                retry_index=0,
            )
            if self._run_prepare_window(
                request_id=request_id,
                trial_index=index,
                total_trials=total,
                retry_index=0,
            ):
                break
            self._clear_stimulus_phase_ack()
            self._emit_phase(PHASE_CAL_ACTIVE, "采样即将开始", prompt_base, flicker=True, cue_freq=cue_freq)
            self._wait_for_stimulus_phase_applied()
            if self._stop_event.is_set():
                break

            tone_payload = self._emit_tone(
                event=TONE_EVENT_ACTIVE_START,
                trial_index=index,
                total_trials=total,
                retry_index=0,
            )
            play_collection_tone_event_sync(tone_payload)
            if self._stop_event.is_set():
                break
            self._emit_phase(PHASE_CAL_ACTIVE, "流程测试中", prompt_base, flicker=True, cue_freq=cue_freq)
            if self._sleep_interruptible(trial_active_sec):
                break
            self._emit_phase(
                PHASE_CAL_REST,
                "流程测试采样结束",
                "测试模式：未连接设备，未读取 EEG 数据。",
                flicker=False,
                cue_freq=None,
            )
            tone_payload = self._emit_tone(
                event=TONE_EVENT_ACTIVE_END,
                trial_index=index,
                total_trials=total,
                retry_index=0,
            )
            play_collection_tone_event(tone_payload)
            executed_trials += 1
            self._emit_phase(PHASE_CAL_REST, "休息", "请放松并正常眨眼。", flicker=False, cue_freq=None)
            if self._sleep_interruptible(self.config.rest_sec):
                break

        collection_aborted = bool(self._stop_event.is_set() and executed_trials < total)
        if collection_aborted:
            self._emit_phase(
                PHASE_STOPPED,
                "流程测试已停止",
                "已停止当前流程测试；未连接设备，未保存数据。",
                flicker=False,
                cue_freq=None,
            )
        else:
            self._emit_phase(
                PHASE_STOPPED,
                "流程测试完成",
                "已完整走完闪烁刺激和采集流程；未连接设备，未保存数据。",
                flicker=False,
                cue_freq=None,
            )
        self.done.emit(
            self._build_done_payload(
                collection_aborted=collection_aborted,
                collected_trials=0,
                total_trials=total,
                metadata={
                    "simulation_only": True,
                    "executed_trials": int(executed_trials),
                    "session_id": str(self.config.session_id),
                },
            )
        )

    @pyqtSlot()
    def run(self) -> None:
        board = None
        active_serial = self.config.serial_port
        fs = 0
        eeg_channels: tuple[int, ...] = ()
        trials: list[Any] = []
        collected: list[tuple[Any, np.ndarray]] = []
        quality_rows: list[dict[str, Any]] = []
        raw_board_chunks: list[np.ndarray] = []
        total = 0
        try:
            if bool(self.config.simulation_only):
                self._run_simulation_only()
                return
            board, resolved_port, attempted_ports = prepare_board_session(self.config.board_id, self.config.serial_port)
            active_serial = resolved_port
            fs = int(BoardShim.get_sampling_rate(self.config.board_id))
            eeg_channels = tuple(int(ch) for ch in BoardShim.get_eeg_channels(self.config.board_id))
            board.start_stream(450000)
            ready = ensure_stream_ready(board, fs)
            self.log.emit(
                f"采集开始：请求串口={self.config.serial_port} -> 实际={resolved_port}，"
                f"尝试={attempted_ports}，fs={fs}Hz，通道={list(eeg_channels)}，缓存就绪={ready}，"
                f"轮次={self.config.round_index}/{self.config.rounds_planned}"
            )
            if self._sleep_interruptible(max(2.0, DEFAULT_STREAM_WARMUP_SEC)):
                metadata = None
                try:
                    self._drain_board_data(board, raw_board_chunks)
                    metadata = self._save_raw_board_fallback(
                        active_serial=active_serial,
                        sampling_rate=fs,
                        eeg_channels=eeg_channels,
                        total_trials=0,
                        raw_board_chunks=raw_board_chunks,
                        failure_reason="user_stop_during_warmup",
                    )
                except Exception as save_exc:
                    self.error.emit(
                        "预热阶段停止后原始板卡数据保存失败："
                        f"{describe_runtime_error(save_exc, serial_port=active_serial)}"
                    )
                stopped_detail = (
                    "尚未进入 Trial，但已保存原始板卡数据。"
                    if metadata
                    else "尚未进入 Trial，未保存数据。"
                )
                self._emit_phase(PHASE_STOPPED, "采集已停止", stopped_detail, flicker=False, cue_freq=None)
                self.done.emit(
                    self._build_done_payload(
                        collection_aborted=True,
                        collected_trials=0,
                        total_trials=0,
                        metadata=metadata,
                        failure_reason="user_stop_during_warmup",
                    )
                )
                return
            self._drain_board_data(board, raw_board_chunks)
            trials = self._build_trial_plan()
            minimum_samples = max(1, int(round(1.5 * fs)))
            total = len(trials)
            for index, trial in enumerate(trials, start=1):
                if self._stop_event.is_set():
                    break
                cue_freq = None if trial.expected_freq is None else float(trial.expected_freq)
                trial_label_lower = str(trial.label).strip().lower()
                is_long_idle = "long_idle" in trial_label_lower or "long idle" in trial_label_lower
                trial_active_sec = (
                    float(self.config.long_idle_sec)
                    if is_long_idle and float(self.config.long_idle_sec) > 0.0
                    else float(self.config.active_sec)
                )
                self._current_trial_active_sec = float(trial_active_sec)
                active_samples = int(round(trial_active_sec * fs))
                prompt_base = (
                    f"第{self.config.round_index}轮 Trial {index}/{total} 空闲（看中心）"
                    if cue_freq is None
                    else f"第{self.config.round_index}轮 Trial {index}/{total} 注视 {trial.label}"
                )
                if is_long_idle:
                    prompt_base = (
                        f"Round {self.config.round_index} Trial {index}/{total} long-idle "
                        "(keep looking at center, avoid all targets)"
                    )
                voice_text = prompt_text_for_trial(self.config.freqs, trial)
                accepted_segment: Optional[np.ndarray] = None
                accepted_used_samples = 0
                accepted_shortfall_ratio = 1.0
                retry_count = 0
                available_samples = 0
                active_start_tone_started_at = ""
                active_window_started_at = ""
                active_window_ended_at = ""
                segment_captured_at = ""
                active_end_tone_started_at = ""
                stimulus_phase_apply_requested_at = ""
                stimulus_first_frame_presented_at = ""
                stimulus_first_frame_presented_t_sec: Optional[float] = None
                stimulus_first_frame_frame_index: Optional[int] = None
                stimulus_first_frame_cue_freq: Optional[float] = None
                stimulus_first_frame_mode = ""
                stimulus_first_frame_ack_latency_sec: Optional[float] = None
                stimulus_first_frame_ack_timed_out = False
                stimulus_frame_interval_stats: dict[str, Any] = {}
                board_buffer_cleared_at = ""
                board_buffer_clear_samples = 0
                while retry_count <= MAX_TRIAL_RETRIES:
                    prompt = (
                        prompt_base
                        if retry_count == 0
                        else f"{prompt_base} | 重采 {retry_count}/{MAX_TRIAL_RETRIES}"
                    )
                    self._emit_phase(PHASE_CAL_PREPARE, "准备", prompt, flicker=False, cue_freq=cue_freq)
                    self.log.emit(prompt)
                    request_id = self._emit_voice_prompt(
                        text=voice_text,
                        trial_index=index,
                        total_trials=total,
                        retry_index=retry_count,
                    )
                    if self._run_prepare_window(
                        request_id=request_id,
                        trial_index=index,
                        total_trials=total,
                        retry_index=retry_count,
                    ):
                        break
                    self._clear_stimulus_phase_ack()
                    stimulus_phase_apply_requested_at = wallclock_iso_timestamp()
                    stimulus_phase_apply_perf = time.perf_counter()
                    self._emit_phase(
                        PHASE_CAL_ACTIVE,
                        "采样即将开始",
                        prompt,
                        flicker=True,
                        cue_freq=cue_freq,
                        active_sec=trial_active_sec,
                    )
                    stimulus_ack_ready, stimulus_ack_payload = self._wait_for_stimulus_phase_applied()
                    if self._stop_event.is_set():
                        break
                    if bool(self.config.sync_stimulus_phase) and not stimulus_ack_ready:
                        stimulus_first_frame_ack_timed_out = True
                        retry_count += 1
                        self.log.emit(
                            f"Trial {index} 未收到首帧呈现确认，已丢弃本次尝试 "
                            f"({retry_count}/{MAX_TRIAL_RETRIES})。"
                        )
                        if retry_count > MAX_TRIAL_RETRIES:
                            raise RuntimeError(f"Trial {index} 连续未收到 UI 首帧确认，流程中止")
                        self._emit_phase(
                            PHASE_CAL_REST,
                            "重采中",
                            "未收到刺激首帧确认，正在重采该 Trial。",
                            flicker=False,
                            cue_freq=None,
                        )
                        if self._sleep_interruptible(max(0.2, self.config.rest_sec * 0.5)):
                            break
                        continue
                    if stimulus_ack_ready:
                        stimulus_first_frame_ack_timed_out = False
                        stimulus_first_frame_presented_at = str(stimulus_ack_payload.get("ack_wall_time", ""))
                        stimulus_first_frame_mode = str(stimulus_ack_payload.get("mode", ""))
                        try:
                            stimulus_first_frame_presented_t_sec = float(
                                stimulus_ack_payload.get("presented_t_sec")
                            )
                        except Exception:
                            stimulus_first_frame_presented_t_sec = None
                        try:
                            stimulus_first_frame_frame_index = int(stimulus_ack_payload.get("frame_index"))
                        except Exception:
                            stimulus_first_frame_frame_index = None
                        try:
                            raw_cue_freq = stimulus_ack_payload.get("cue_freq")
                            stimulus_first_frame_cue_freq = None if raw_cue_freq is None else float(raw_cue_freq)
                        except Exception:
                            stimulus_first_frame_cue_freq = None
                        try:
                            ack_perf = float(stimulus_ack_payload.get("ack_perf_counter_sec"))
                            stimulus_first_frame_ack_latency_sec = max(0.0, ack_perf - float(stimulus_phase_apply_perf))
                        except Exception:
                            stimulus_first_frame_ack_latency_sec = None
                        frame_stats = stimulus_ack_payload.get("frame_interval_stats", {})
                        stimulus_frame_interval_stats = dict(frame_stats) if isinstance(frame_stats, dict) else {}

                    board_buffer_cleared_at = wallclock_iso_timestamp()
                    board_buffer_clear_samples = self._drain_board_data(board, raw_board_chunks)
                    # The EEG segment is saved as the last active_sec samples after this cue,
                    # so the synchronous "beep" duration is discarded from the saved window.
                    active_start_tone_started_at = wallclock_iso_timestamp()
                    tone_payload = self._emit_tone(
                        event=TONE_EVENT_ACTIVE_START,
                        trial_index=index,
                        total_trials=total,
                        retry_index=retry_count,
                    )
                    play_collection_tone_event_sync(tone_payload)
                    active_window_started_at = wallclock_iso_timestamp()
                    if self._stop_event.is_set():
                        break
                    self._emit_phase(
                        PHASE_CAL_ACTIVE,
                        "采集中",
                        prompt,
                        flicker=True,
                        cue_freq=cue_freq,
                        active_sec=trial_active_sec,
                    )
                    if self._sleep_interruptible(trial_active_sec):
                        break
                    active_window_ended_at = wallclock_iso_timestamp()
                    segment, used_samples, available_samples = read_recent_eeg_segment(
                        board,
                        eeg_channels,
                        target_samples=active_samples,
                        minimum_samples=minimum_samples,
                    )
                    segment_captured_at = wallclock_iso_timestamp()
                    self._emit_phase(PHASE_CAL_REST, "采样结束", "正在读取数据。", flicker=False, cue_freq=None)
                    # The end cue follows the segment capture, so it is not included in the saved active window.
                    active_end_tone_started_at = wallclock_iso_timestamp()
                    tone_payload = self._emit_tone(
                        event=TONE_EVENT_ACTIVE_END,
                        trial_index=index,
                        total_trials=total,
                        retry_index=retry_count,
                    )
                    play_collection_tone_event(tone_payload)
                    shortfall_ratio = float(max(active_samples - int(used_samples), 0) / max(active_samples, 1))
                    sample_ratio = float(int(used_samples) / max(active_samples, 1))
                    if sample_ratio >= float(MIN_TRIAL_QUALITY_RATIO):
                        accepted_segment = np.ascontiguousarray(segment, dtype=np.float64)
                        accepted_used_samples = int(used_samples)
                        accepted_shortfall_ratio = float(shortfall_ratio)
                        break

                    retry_count += 1
                    self.log.emit(
                        f"Trial {index} 样本不足：{used_samples}/{active_samples} "
                        f"(比例={sample_ratio:.3f}, 缓冲区={available_samples})。"
                    )
                    if retry_count > MAX_TRIAL_RETRIES:
                        raise RuntimeError(
                            f"Trial {index} 连续 {MAX_TRIAL_RETRIES} 次仍未通过质量门槛 "
                            f"(used={used_samples}, target={active_samples})"
                        )
                    self._emit_phase(
                        PHASE_CAL_REST,
                        "重采中",
                        "样本不足，正在重采该 Trial。",
                        flicker=False,
                        cue_freq=None,
                    )
                    if self._sleep_interruptible(max(0.2, self.config.rest_sec * 0.5)):
                        break

                if self._stop_event.is_set():
                    break
                if accepted_segment is None:
                    raise RuntimeError(f"Trial {index} 未采到有效片段，流程中止")

                stim_profile = get_stimulus_profile(self.config.stimulus_profile_id)
                collected.append((trial, accepted_segment))
                quality_rows.append(
                    {
                        "order_index": int(index - 1),
                        "target_samples": int(active_samples),
                        "used_samples": int(accepted_used_samples),
                        "active_sec": float(trial_active_sec),
                        "sample_ratio": float(accepted_used_samples / max(active_samples, 1)),
                        "shortfall_ratio": float(accepted_shortfall_ratio),
                        "retry_count": int(retry_count),
                        "available_samples": int(available_samples),
                        "active_start_tone_started_at": str(active_start_tone_started_at),
                        "active_window_started_at": str(active_window_started_at),
                        "active_window_ended_at": str(active_window_ended_at),
                        "segment_captured_at": str(segment_captured_at),
                        "active_end_tone_started_at": str(active_end_tone_started_at),
                        "stimulus_phase_apply_requested_at": str(stimulus_phase_apply_requested_at),
                        "stimulus_first_frame_presented_at": str(stimulus_first_frame_presented_at),
                        "stimulus_first_frame_presented_t_sec": stimulus_first_frame_presented_t_sec,
                        "stimulus_first_frame_frame_index": stimulus_first_frame_frame_index,
                        "stimulus_first_frame_cue_freq": stimulus_first_frame_cue_freq,
                        "stimulus_first_frame_mode": str(stimulus_first_frame_mode),
                        "stimulus_first_frame_ack_latency_sec": stimulus_first_frame_ack_latency_sec,
                        "stimulus_first_frame_ack_timed_out": bool(stimulus_first_frame_ack_timed_out),
                        "stimulus_frame_interval_stats": dict(stimulus_frame_interval_stats),
                        "stimulus_profile_id": str(self.config.stimulus_profile_id),
                        "stim_mean": float(stim_profile.mean),
                        "stim_amp": float(stim_profile.amp),
                        "ramp_sec": float(stim_profile.ramp_sec),
                        "board_buffer_cleared_at": str(board_buffer_cleared_at),
                        "board_buffer_clear_samples": int(board_buffer_clear_samples),
                    }
                )
                self._emit_phase(PHASE_CAL_REST, "休息", "请放松并正常眨眼。", flicker=False, cue_freq=None)
                if self._sleep_interruptible(self.config.rest_sec):
                    break

            collection_aborted = bool(self._stop_event.is_set() and len(collected) < len(trials))
            if not collected:
                if board is not None:
                    try:
                        self._drain_board_data(board, raw_board_chunks)
                    except Exception:
                        pass
                metadata = None
                if int(fs) > 0 and eeg_channels:
                    fallback_reason = "user_stop_before_valid_trial" if collection_aborted else "no_valid_trial"
                    try:
                        metadata = self._save_raw_board_fallback(
                            active_serial=active_serial,
                            sampling_rate=fs,
                            eeg_channels=eeg_channels,
                            total_trials=total,
                            raw_board_chunks=raw_board_chunks,
                            failure_reason=fallback_reason,
                        )
                    except Exception as save_exc:
                        self.error.emit(
                            "未采到有效 Trial，且原始板卡数据保存失败："
                            f"{describe_runtime_error(save_exc, serial_port=active_serial)}"
                        )
                if collection_aborted:
                    saved_message = (
                        "未采到有效 Trial，但已保存原始板卡数据。"
                        if metadata
                        else "未采到有效 Trial，未保存数据。"
                    )
                    self._emit_phase(PHASE_STOPPED, "采集已停止", saved_message, flicker=False, cue_freq=None)
                    self.done.emit(
                        self._build_done_payload(
                            collection_aborted=True,
                            collected_trials=0,
                            total_trials=len(trials),
                            metadata=metadata,
                        )
                    )
                    return
                if metadata:
                    self._emit_phase(
                        PHASE_STOPPED,
                        "采集已停止",
                        "未采到有效 Trial，但已保存原始板卡数据。",
                        flicker=False,
                        cue_freq=None,
                    )
                    self.done.emit(
                        self._build_done_payload(
                            collection_aborted=True,
                            collected_trials=0,
                            total_trials=len(trials),
                            metadata=metadata,
                            failure_reason="no_valid_trial",
                        )
                    )
                    self.error.emit(
                        "没有采集到任何有效 Trial，但已保存原始板卡数据："
                        f"manifest={metadata.get('dataset_manifest', '')}"
                    )
                    return
                raise RuntimeError("没有采集到任何 Trial")
            if board is not None:
                try:
                    self._drain_board_data(board, raw_board_chunks)
                except Exception:
                    pass
            continuous_board_data, continuous_board_error = self._try_concat_board_data_chunks(raw_board_chunks)
            metadata = self._save_collected_dataset(
                active_serial=active_serial,
                sampling_rate=fs,
                eeg_channels=eeg_channels,
                total_trials=total,
                trial_segments=collected,
                quality_rows=quality_rows,
                collection_aborted=collection_aborted,
                continuous_board_data=continuous_board_data,
                continuous_board_error=continuous_board_error,
            )
            self._emit_phase(
                PHASE_STOPPED,
                "采集已停止" if collection_aborted else "采集完成",
                "已保存部分数据，本轮未计为完成。" if collection_aborted else "数据已保存。",
                flicker=False,
                cue_freq=None,
            )
            self.done.emit(
                self._build_done_payload(
                    collection_aborted=collection_aborted,
                    collected_trials=len(collected),
                    total_trials=len(trials),
                    metadata=metadata,
                )
            )
        except Exception as exc:
            failure_reason = str(exc)
            if int(fs) > 0 and eeg_channels:
                try:
                    if board is not None:
                        try:
                            self._drain_board_data(board, raw_board_chunks)
                        except Exception:
                            pass
                    if collected:
                        continuous_board_data, continuous_board_error = self._try_concat_board_data_chunks(raw_board_chunks)
                        metadata = self._save_collected_dataset(
                            active_serial=active_serial,
                            sampling_rate=fs,
                            eeg_channels=eeg_channels,
                            total_trials=max(int(total), len(trials)),
                            trial_segments=collected,
                            quality_rows=quality_rows,
                            collection_aborted=True,
                            failure_reason=failure_reason,
                            continuous_board_data=continuous_board_data,
                            continuous_board_error=continuous_board_error,
                        )
                    else:
                        metadata = self._save_raw_board_fallback(
                            active_serial=active_serial,
                            sampling_rate=fs,
                            eeg_channels=eeg_channels,
                            total_trials=max(int(total), len(trials)),
                            raw_board_chunks=raw_board_chunks,
                            failure_reason=failure_reason,
                        )
                        if metadata is None:
                            raise RuntimeError("没有可保存的原始板卡数据")
                except Exception as save_exc:
                    self.error.emit(
                        "采集失败，且兜底数据保存失败："
                        f"{describe_runtime_error(exc, serial_port=active_serial)}；"
                        f"save_error={describe_runtime_error(save_exc, serial_port=active_serial)}"
                    )
                    self._emit_phase(
                        PHASE_ERROR,
                        "采集错误",
                        f"{exc}；兜底数据保存失败：{save_exc}",
                        flicker=False,
                        cue_freq=None,
                    )
                    return
                saved_detail = "部分数据" if collected else "原始板卡数据"
                self._emit_phase(
                    PHASE_STOPPED,
                    "采集已停止",
                    f"采集中途失败，但已保存{saved_detail}，本轮未计为完成。",
                    flicker=False,
                    cue_freq=None,
                )
                self.done.emit(
                    self._build_done_payload(
                        collection_aborted=True,
                        collected_trials=len(collected),
                        total_trials=max(int(total), len(trials)),
                        metadata=metadata,
                        failure_reason=failure_reason,
                    )
                )
                self.error.emit(
                    f"采集失败，但已保存{saved_detail}："
                    f"{describe_runtime_error(exc, serial_port=active_serial)}；"
                    f"manifest={metadata.get('dataset_manifest', '')}；"
                    f"npz={metadata.get('dataset_npz', '')}"
                )
                return
            self.error.emit(f"采集失败：{describe_runtime_error(exc, serial_port=active_serial)}")
            self._emit_phase(PHASE_ERROR, "采集错误", str(exc), flicker=False, cue_freq=None)
        finally:
            if board is not None:
                try:
                    board.stop_stream()
                except Exception:
                    pass
                try:
                    board.release_session()
                except Exception:
                    pass
            self.finished.emit()


def run_collection_cli(config: CollectionConfig) -> dict[str, Any]:
    profile = get_stimulus_profile(config.stimulus_profile_id)
    config = replace(
        config,
        stimulus_backend=STIMULUS_BACKEND_HEADLESS_NO_VISUAL,
        stimulus_mode_selection_reason=str(config.stimulus_mode_selection_reason or "headless_no_visual"),
        screen_brightness_note=str(config.screen_brightness_note or profile.screen_brightness_note),
        sync_stimulus_phase=False,
        sync_voice_prompt=False,
    )
    worker = CollectionWorker(config)
    state: dict[str, Any] = {}

    def _log(text: str) -> None:
        print(text, flush=True)

    worker.log.connect(_log)  # type: ignore[arg-type]
    worker.done.connect(lambda payload: state.update(payload))  # type: ignore[arg-type]
    worker.error.connect(lambda text: (_log(text), state.setdefault("error", text)))  # type: ignore[arg-type]
    worker.run()
    if "error" in state:
        raise RuntimeError(str(state["error"]))
    return state


class DatasetCollectionWindow(QMainWindow):
    def __init__(
        self,
        *,
        serial_port: str,
        board_id: int,
        freqs: Sequence[float],
        simulation_only_default: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP 数据采集")
        self.resize(1280, 880)

        self.default_serial = normalize_serial_port(serial_port)
        self.default_board_id = int(board_id)
        self.default_freqs = tuple(float(freq) for freq in freqs)

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[CollectionWorker] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceCheckWorker] = None
        self.fullscreen_window: Optional[CollectionFullscreenStimWindow] = None

        self.rounds_completed = 0
        self._session_base_auto_cache: Optional[str] = None
        self._updating_preset = False
        self.speech_prompt_player = SpeechPromptPlayer(parent=self)
        self.speech_prompt_player.playback_finished.connect(self._on_voice_prompt_finished)

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QWidget(root)
        left_layout = QVBoxLayout(left)
        form = QFormLayout()

        self.serial_edit = QLineEdit(self.default_serial)
        self.board_edit = QLineEdit(str(self.default_board_id))
        self.freqs_edit = QLineEdit(",".join(f"{freq:g}" for freq in self.default_freqs))
        self.subject_edit = QLineEdit("subject001")
        self.session_index_spin = QSpinBox()
        self.session_index_spin.setRange(1, 999)
        self.session_index_spin.setValue(1)
        self.rounds_planned_spin = QSpinBox()
        self.rounds_planned_spin.setRange(1, 99)
        self.rounds_planned_spin.setValue(1)
        self.current_round_value = QLabel("1")
        self.completed_rounds_value = QLabel("0")
        self.session_base_edit = QLineEdit("")
        self.dataset_dir_edit = QLineEdit(str(DEFAULT_DATASET_DIR))
        self._set_dataset_dir_text(DEFAULT_DATASET_DIR)
        self.preset_combo = QComboBox()
        for preset in (STABLE_12M_PRESET, ENHANCED_45M_PRESET, CUSTOM_PRESET):
            self.preset_combo.addItem(preset.display, preset.key)
        self.preset_combo.setCurrentText(STABLE_12M_PRESET.display)
        self.simulation_only_check = QCheckBox("流程测试模式（不连接板卡，不保存数据）")
        self.simulation_only_check.setChecked(bool(simulation_only_default))
        self.stimulus_profile_combo = QComboBox()
        for profile_id, profile in STIMULUS_PROFILES.items():
            self.stimulus_profile_combo.addItem(f"{profile_id} ({profile.mean:.2f}/{profile.amp:.2f})", profile_id)
        profile_index = self.stimulus_profile_combo.findData(DEFAULT_COLLECTION_STIMULUS_PROFILE_ID)
        self.stimulus_profile_combo.setCurrentIndex(max(0, profile_index))
        self.stimulus_mode_combo = QComboBox()
        self.stimulus_mode_combo.addItem("Auto (profile decides)", "auto")
        self.stimulus_mode_combo.addItem("Elapsed time sine（采集默认，按时间计算）", STIMULUS_MODE_ELAPSED_TIME_SINE)
        self.stimulus_mode_combo.addItem("Frame locked sine（可选，按帧采样）", STIMULUS_MODE_FRAME_LOCKED_SINE)
        self.stimulus_mode_combo.setCurrentIndex(0)
        self.stim_refresh_rate_spin = QDoubleSpinBox()
        self.stim_refresh_rate_spin.setRange(0.0, 1000.0)
        self.stim_refresh_rate_spin.setDecimals(2)
        self.stim_refresh_rate_spin.setSingleStep(1.0)
        self.stim_refresh_rate_spin.setSpecialValueText("Auto")
        self.stim_refresh_rate_spin.setValue(0.0)
        self.prepare_spin = QDoubleSpinBox()
        self.prepare_spin.setRange(MIN_PREPARE_SEC_FOR_VOICE, 20.0)
        self.prepare_spin.setDecimals(1)
        self.prepare_spin.setSingleStep(0.5)
        self.active_spin = QDoubleSpinBox()
        self.active_spin.setRange(MIN_ACTIVE_SEC_FOR_TRAINING, 20.0)
        self.active_spin.setDecimals(1)
        self.active_spin.setSingleStep(0.5)
        self.rest_spin = QDoubleSpinBox()
        self.rest_spin.setRange(MIN_REST_SEC_BETWEEN_TRIALS, 20.0)
        self.rest_spin.setDecimals(1)
        self.rest_spin.setSingleStep(0.5)
        self.long_idle_spin = QDoubleSpinBox()
        self.long_idle_spin.setRange(0.0, 300.0)
        self.long_idle_spin.setDecimals(1)
        self.long_idle_spin.setSingleStep(5.0)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 60)
        self.idle_spin = QSpinBox()
        self.idle_spin.setRange(1, 120)
        self.switch_spin = QSpinBox()
        self.switch_spin.setRange(0, 120)
        self.estimate_label = QLabel("预计时长：--")

        form.addRow("串口", self.serial_edit)
        form.addRow("板卡 ID", self.board_edit)
        form.addRow("刺激频率", self.freqs_edit)
        form.addRow("被试 ID", self.subject_edit)
        form.addRow("起始轮次", self.session_index_spin)
        form.addRow("计划轮数", self.rounds_planned_spin)
        form.addRow("当前轮次", self.current_round_value)
        form.addRow("已完成轮次", self.completed_rounds_value)
        form.addRow("会话基础 ID（可选）", self.session_base_edit)
        form.addRow("数据集目录", self.dataset_dir_edit)
        form.addRow("预设协议", self.preset_combo)
        form.addRow("运行方式", self.simulation_only_check)
        form.addRow("Stimulus profile", self.stimulus_profile_combo)
        form.addRow("刺激生成方式", self.stimulus_mode_combo)
        form.addRow("刺激刷新率Hz（0=自动）", self.stim_refresh_rate_spin)
        form.addRow("准备时长（秒）", self.prepare_spin)
        form.addRow("采集时长（秒）", self.active_spin)
        form.addRow("休息时长（秒）", self.rest_spin)
        form.addRow("目标重复次数", self.target_spin)
        form.addRow("空闲重复次数", self.idle_spin)
        form.addRow("切换 Trial 数", self.switch_spin)
        form.addRow("单轮预计时长", self.estimate_label)
        form.addRow("Long Idle (sec, 0=off)", self.long_idle_spin)
        left_layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_pick_dir = QPushButton("选择目录")
        self.btn_connect = QPushButton("连接设备")
        self.btn_start = QPushButton("开始本轮采集")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_pick_dir)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        left_layout.addLayout(row)

        self.phase_label = QLabel("空闲")
        self.phase_label.setStyleSheet("font-size:16px; font-weight:600;")
        left_layout.addWidget(self.phase_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text, 1)

        right = QWidget(root)
        right_layout = QVBoxLayout(right)
        initial_refresh_rate = float(self.stim_refresh_rate_spin.value())
        self.stim = make_collection_stim_widget(
            self.default_freqs,
            refresh_rate_hz=None if initial_refresh_rate <= 1.0 else initial_refresh_rate,
            stimulus_mode=self._current_stimulus_mode(),
            stimulus_profile_id=self._current_stimulus_profile_id(),
        )
        self.stim.active_phase_frame_presented.connect(self._on_active_phase_frame_presented)
        right_layout.addWidget(self.stim, 1)

        layout.addWidget(left, 0)
        layout.addWidget(right, 1)

        self.config_widgets = [
            self.serial_edit,
            self.board_edit,
            self.freqs_edit,
            self.subject_edit,
            self.session_index_spin,
            self.rounds_planned_spin,
            self.session_base_edit,
            self.dataset_dir_edit,
            self.preset_combo,
            self.simulation_only_check,
            self.stimulus_profile_combo,
            self.stimulus_mode_combo,
            self.stim_refresh_rate_spin,
            self.prepare_spin,
            self.active_spin,
            self.rest_spin,
            self.long_idle_spin,
            self.target_spin,
            self.idle_spin,
            self.switch_spin,
        ]

        self.btn_pick_dir.clicked.connect(self._pick_dataset_dir)
        self.btn_connect.clicked.connect(self._connect_device)
        self.btn_start.clicked.connect(self._start_collection)
        self.btn_stop.clicked.connect(self._stop_collection)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.stimulus_profile_combo.currentIndexChanged.connect(self._on_stimulus_profile_changed)
        self.stimulus_mode_combo.currentIndexChanged.connect(self._on_stimulus_mode_changed)
        self.stim_refresh_rate_spin.valueChanged.connect(self._on_stim_refresh_rate_changed)
        self.prepare_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.active_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.rest_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.long_idle_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.target_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.idle_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.switch_spin.valueChanged.connect(self._on_protocol_value_changed)
        self.rounds_planned_spin.valueChanged.connect(self._on_round_control_changed)
        self.session_index_spin.valueChanged.connect(self._on_round_control_changed)
        self.subject_edit.textChanged.connect(self._on_session_base_source_changed)
        self.session_base_edit.textChanged.connect(self._on_session_base_source_changed)

        self._apply_preset(DEFAULT_PRESET_NAME)
        self._refresh_estimate_label()
        self._refresh_round_status()

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {text}")

    def _set_running(self, running: bool) -> None:
        self.btn_connect.setEnabled(not running)
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        for widget in self.config_widgets:
            widget.setEnabled(not running)

    def _set_dataset_dir_text(self, value: str | Path) -> None:
        text = str(value)
        self.dataset_dir_edit.setText(text)
        self.dataset_dir_edit.setToolTip(text)
        self.dataset_dir_edit.setCursorPosition(0)

    def _pick_dataset_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录", self.dataset_dir_edit.text().strip())
        if path:
            self._set_dataset_dir_text(path)

    def _on_session_base_source_changed(self, _value: str) -> None:
        if self.worker_thread is not None:
            return
        self._session_base_auto_cache = None

    def _on_round_control_changed(self, _value: int) -> None:
        if self.worker_thread is not None:
            return
        self.rounds_completed = 0
        self._session_base_auto_cache = None
        self._refresh_round_status()

    def _current_preset_name(self) -> str:
        value = self.preset_combo.currentData()
        return normalize_preset_name(str(value) if value is not None else self.preset_combo.currentText())

    def _current_stimulus_mode(self) -> str:
        value = self.stimulus_mode_combo.currentData()
        raw = str(value) if value is not None else self.stimulus_mode_combo.currentText()
        if not hasattr(self, "stim_refresh_rate_spin"):
            return DEFAULT_COLLECTION_STIMULUS_MODE
        if str(raw).strip().lower() == "auto":
            mode, _reason = resolve_collection_stimulus_mode(
                stimulus_profile_id=self._current_stimulus_profile_id(),
                refresh_rate_hz=self._resolve_stim_refresh_rate_hz(),
                requested_mode="auto",
            )
            return mode
        return validate_stimulus_mode(raw)

    def _current_stimulus_mode_selection_reason(self) -> str:
        value = self.stimulus_mode_combo.currentData()
        raw = str(value) if value is not None else self.stimulus_mode_combo.currentText()
        if not hasattr(self, "stim_refresh_rate_spin"):
            return "initializing"
        _mode, reason = resolve_collection_stimulus_mode(
            stimulus_profile_id=self._current_stimulus_profile_id(),
            refresh_rate_hz=self._resolve_stim_refresh_rate_hz(),
            requested_mode=raw,
        )
        return str(reason)

    def _current_stimulus_profile_id(self) -> str:
        if not hasattr(self, "stimulus_profile_combo"):
            return DEFAULT_COLLECTION_STIMULUS_PROFILE_ID
        value = self.stimulus_profile_combo.currentData()
        return validate_stimulus_profile_id(str(value) if value is not None else self.stimulus_profile_combo.currentText())

    def _set_stimulus_profile_selection(self, profile_id: str) -> None:
        resolved = validate_stimulus_profile_id(profile_id)
        index = self.stimulus_profile_combo.findData(resolved)
        if index >= 0 and self.stimulus_profile_combo.currentIndex() != index:
            self.stimulus_profile_combo.setCurrentIndex(index)

    def _apply_preset(self, preset_name: str) -> None:
        key = normalize_preset_name(preset_name)
        preset = COLLECTION_PRESETS.get(key, CUSTOM_PRESET)
        self._updating_preset = True
        try:
            self.prepare_spin.setValue(float(preset.prepare_sec))
            self.active_spin.setValue(float(preset.active_sec))
            self.rest_spin.setValue(float(preset.rest_sec))
            self.long_idle_spin.setValue(float(preset.long_idle_sec))
            self.target_spin.setValue(int(preset.target_repeats))
            self.idle_spin.setValue(int(preset.idle_repeats))
            self.switch_spin.setValue(int(preset.switch_trials))
            self.preset_combo.setCurrentText(preset.display)
        finally:
            self._updating_preset = False

    def _on_preset_changed(self, _display: str) -> None:
        key = self._current_preset_name()
        if key in (STABLE_12M_PRESET.key, ENHANCED_45M_PRESET.key):
            self._apply_preset(key)
        self._refresh_estimate_label()

    def _on_protocol_value_changed(self, _value: float) -> None:
        if not self._updating_preset and self._current_preset_name() != CUSTOM_PRESET.key:
            self.preset_combo.setCurrentText(CUSTOM_PRESET.display)
        self._refresh_estimate_label()

    def _on_stimulus_profile_changed(self, _index: int) -> None:
        if self.worker_thread is not None:
            return
        try:
            profile_id = self._current_stimulus_profile_id()
            profile = get_stimulus_profile(profile_id)
            desired_freqs = tuple(float(freq) for freq in profile.freqs)
            self.freqs_edit.setText(",".join(f"{freq:g}" for freq in desired_freqs))
            self.stim.mean = float(profile.mean)
            self.stim.amp = float(profile.amp)
            self.stim.phi = float(profile.phi)
            self.stim.ramp_sec = float(profile.ramp_sec)
            self.stim.stimulus_profile_id = str(profile.profile_id)
            self._sync_stim_freqs(
                desired_freqs,
                refresh_rate_hz=self._resolve_stim_refresh_rate_hz(),
                stimulus_mode=self._current_stimulus_mode(),
                stimulus_profile_id=profile_id,
            )
            self._refresh_estimate_label()
        except Exception as exc:
            self._log(f"stimulus profile error: {exc}")

    def _on_stimulus_mode_changed(self, _index: int) -> None:
        if self.worker_thread is not None:
            return
        try:
            self.stim.stimulus_mode = self._current_stimulus_mode()
            self.stim.stimulus_profile_id = self._current_stimulus_profile_id()
            self.stim.update()
        except Exception as exc:
            self._log(f"刺激模式错误：{exc}")

    def _on_stim_refresh_rate_changed(self, _value: float) -> None:
        if self.worker_thread is not None:
            return
        try:
            self._sync_stim_freqs(
                parse_freqs(self.freqs_edit.text().strip()),
                refresh_rate_hz=self._resolve_stim_refresh_rate_hz(),
                stimulus_mode=self._current_stimulus_mode(),
                stimulus_profile_id=self._current_stimulus_profile_id(),
            )
        except Exception as exc:
            self._log(f"刺激刷新率错误：{exc}")

    def _sync_profile_selection_from_freqs(self, freqs: Sequence[float]) -> None:
        matched_profile_id = find_matching_stimulus_profile_id(freqs)
        if matched_profile_id is None:
            return
        self._set_stimulus_profile_selection(matched_profile_id)

    def _round_index_for_next_run(self) -> int:
        return int(self.session_index_spin.value()) + int(self.rounds_completed)

    def _refresh_round_status(self) -> None:
        planned = int(self.rounds_planned_spin.value())
        current_round = self._round_index_for_next_run()
        remaining = max(0, planned - int(self.rounds_completed))
        self.current_round_value.setText(str(current_round))
        self.completed_rounds_value.setText(f"{self.rounds_completed}（剩余 {remaining}）")

    def _refresh_estimate_label(self) -> None:
        trial_count = trial_count_for_protocol(
            target_repeats=int(self.target_spin.value()),
            idle_repeats=int(self.idle_spin.value()),
            switch_trials=int(self.switch_spin.value()),
            long_idle_sec=float(self.long_idle_spin.value()),
        )
        round_sec = estimate_round_seconds(
            prepare_sec=float(self.prepare_spin.value()),
            active_sec=float(self.active_spin.value()),
            rest_sec=float(self.rest_spin.value()),
            target_repeats=int(self.target_spin.value()),
            idle_repeats=int(self.idle_spin.value()),
            switch_trials=int(self.switch_spin.value()),
            long_idle_sec=float(self.long_idle_spin.value()),
            refresh_rate_hz=self._resolve_stim_refresh_rate_hz(),
        )
        planned = int(self.rounds_planned_spin.value())
        total_sec = round_sec * float(planned)
        self.estimate_label.setText(
            f"每轮 {trial_count} 个 Trial，单轮约 {format_duration(round_sec)}，"
            f"总计约 {format_duration(total_sec)}"
        )

    def _stim_target_screen(self):
        return self.screen() or QApplication.primaryScreen()

    def _resolve_stim_refresh_rate_hz(self) -> float:
        if hasattr(self, "stim_refresh_rate_spin"):
            manual_hz = float(self.stim_refresh_rate_spin.value())
            if np.isfinite(manual_hz) and manual_hz > 1.0:
                return float(manual_hz)
        return resolve_collection_stim_refresh_rate_hz(self._stim_target_screen())

    def _sync_stim_freqs(
        self,
        freqs: Sequence[float],
        *,
        refresh_rate_hz: Optional[float] = None,
        stimulus_mode: Optional[str] = None,
        stimulus_profile_id: Optional[str] = None,
    ) -> None:
        values = tuple(float(freq) for freq in freqs)
        resolved_refresh_rate_hz = (
            self._resolve_stim_refresh_rate_hz() if refresh_rate_hz is None else float(refresh_rate_hz)
        )
        resolved_profile_id = (
            self._current_stimulus_profile_id()
            if stimulus_profile_id is None
            else validate_stimulus_profile_id(stimulus_profile_id)
        )
        profile = get_stimulus_profile(resolved_profile_id)
        resolved_stimulus_mode = (
            self._current_stimulus_mode() if stimulus_mode is None else validate_stimulus_mode(stimulus_mode)
        )
        validate_stimulus_frequency_set(values, refresh_rate_hz=resolved_refresh_rate_hz)
        self.stim.freqs = values
        self.stim.refresh_rate_hz = resolved_refresh_rate_hz
        self.stim.stimulus_mode = resolved_stimulus_mode
        self.stim.stimulus_profile_id = str(profile.profile_id)
        self.stim.mean = float(profile.mean)
        self.stim.amp = float(profile.amp)
        self.stim.phi = float(profile.phi)
        self.stim.ramp_sec = float(profile.ramp_sec)
        self.stim.update()

    def _show_fullscreen_stimulus(
        self,
        freqs: Sequence[float],
        *,
        refresh_rate_hz: float,
        stimulus_mode: str,
        stimulus_profile_id: str,
    ) -> None:
        self._close_fullscreen_stimulus()
        screen = self._stim_target_screen()
        window = CollectionFullscreenStimWindow(
            freqs=freqs,
            refresh_rate_hz=refresh_rate_hz,
            stimulus_mode=stimulus_mode,
            stimulus_profile_id=stimulus_profile_id,
        )
        window.escape_requested.connect(self._stop_collection)
        window.active_phase_frame_presented.connect(self._on_active_phase_frame_presented)
        if screen is not None:
            window.setGeometry(screen.geometry())
        self.fullscreen_window = window
        window.showFullScreen()
        window.raise_()
        window.activateWindow()

    def _close_fullscreen_stimulus(self) -> None:
        window = self.fullscreen_window
        self.fullscreen_window = None
        if window is None:
            return
        try:
            window.escape_requested.disconnect(self._stop_collection)
        except Exception:
            pass
        try:
            window.active_phase_frame_presented.disconnect(self._on_active_phase_frame_presented)
        except Exception:
            pass
        window.close_from_owner()

    def _phase_for_embedded_preview(self, phase: dict[str, Any]) -> dict[str, Any]:
        payload = dict(phase)
        if self.fullscreen_window is not None:
            payload["flicker"] = False
        return payload

    def _resolve_session_base(self, subject_id: str) -> str:
        raw = _strip_round_suffix(self.session_base_edit.text().strip())
        if raw:
            return raw
        if self._session_base_auto_cache is None:
            self._session_base_auto_cache = _auto_session_base_id(subject_id)
        return self._session_base_auto_cache

    def _read_config(self, *, round_index_override: Optional[int] = None) -> CollectionConfig:
        serial_port = normalize_serial_port(self.serial_edit.text().strip())
        board_id = int(self.board_edit.text().strip())
        freqs = parse_freqs(self.freqs_edit.text().strip())
        self._sync_profile_selection_from_freqs(freqs)
        subject_id = sanitize_collection_token(self.subject_edit.text().strip() or "subject001", default="subject001")
        simulation_only = bool(self.simulation_only_check.isChecked())
        rounds_planned = int(self.rounds_planned_spin.value())
        round_index = int(round_index_override) if round_index_override is not None else int(self.session_index_spin.value())
        session_base = self._resolve_session_base(subject_id)
        session_id = _build_round_session_id(session_base, round_index)
        dataset_dir = resolve_dataset_dir(self.dataset_dir_edit.text().strip())
        protocol_name = self._current_preset_name()
        stimulus_profile_id = self._current_stimulus_profile_id()
        stimulus_mode = self._current_stimulus_mode()
        stimulus_mode_selection_reason = self._current_stimulus_mode_selection_reason()
        prepare_sec = float(self.prepare_spin.value())
        active_sec = float(self.active_spin.value())
        long_idle_sec = float(self.long_idle_spin.value())
        rest_sec = float(self.rest_spin.value())
        _validate_collection_protocol(
            prepare_sec=prepare_sec,
            active_sec=active_sec,
            rest_sec=rest_sec,
            long_idle_sec=long_idle_sec,
        )
        stim_refresh_rate_hz = self._resolve_stim_refresh_rate_hz()
        validate_stimulus_frequency_set(freqs, refresh_rate_hz=stim_refresh_rate_hz)
        if not profile_matches_freqs(stimulus_profile_id, freqs):
            self._log("warning: selected stimulus profile freqs differ from the UI frequency list")
        target_repeats = int(self.target_spin.value())
        idle_repeats = int(self.idle_spin.value())
        switch_trials = int(self.switch_spin.value())
        estimated_round_sec = estimate_round_seconds(
            prepare_sec=prepare_sec,
            active_sec=active_sec,
            rest_sec=rest_sec,
            target_repeats=target_repeats,
            idle_repeats=idle_repeats,
            switch_trials=switch_trials,
            long_idle_sec=long_idle_sec,
            refresh_rate_hz=stim_refresh_rate_hz,
        )
        return CollectionConfig(
            serial_port=serial_port,
            board_id=board_id,
            freqs=freqs,
            subject_id=subject_id,
            session_id=session_id,
            session_index=round_index,
            dataset_dir=dataset_dir,
            protocol_name=protocol_name,
            prepare_sec=prepare_sec,
            active_sec=active_sec,
            rest_sec=rest_sec,
            target_repeats=target_repeats,
            idle_repeats=idle_repeats,
            switch_trials=switch_trials,
            long_idle_sec=long_idle_sec,
            rounds_planned=rounds_planned,
            round_index=round_index,
            estimated_round_sec=estimated_round_sec,
            stimulus_profile_id=stimulus_profile_id,
            stim_refresh_rate_hz=stim_refresh_rate_hz,
            stimulus_mode=stimulus_mode,
            stimulus_mode_selection_reason=stimulus_mode_selection_reason,
            simulation_only=simulation_only,
        )

    def _connect_device(self) -> None:
        if self.worker_thread is not None:
            self._log("采集中，请先停止再重新连接设备。")
            return
        if self.connect_thread is not None:
            self._log("正在连接设备，请稍候。")
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"配置错误：{exc}")
            return
        if bool(cfg.simulation_only):
            self.phase_label.setText("流程测试模式")
            self._log("当前为流程测试模式：不会连接设备；直接点击“开始本轮采集”即可演练语音、提示音、闪烁刺激和 trial 流程。")
            return
        worker = DeviceCheckWorker(serial_port=cfg.serial_port, board_id=cfg.board_id)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.connected.connect(self._on_connected)
        worker.error.connect(self._on_connect_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_connect_finished)
        thread.started.connect(worker.run)
        self.connect_worker = worker
        self.connect_thread = thread
        self.phase_label.setText("连接中...")
        thread.start()

    def _on_connected(self, payload: dict[str, Any]) -> None:
        self.phase_label.setText("设备已连接")
        resolved_port = normalize_serial_port(str(payload.get("resolved_serial_port", "")))
        attempted_ports = payload.get("attempted_ports", [])
        log_payload = dict(payload)
        log_payload["attempted_ports"] = (
            list(attempted_ports) if isinstance(attempted_ports, (list, tuple)) else attempted_ports
        )
        if resolved_port:
            self.serial_edit.setText(resolved_port)
        self._log(
            "连接成功：请求串口={requested_serial_port}，实际串口={resolved_serial_port}，"
            "尝试端口={attempted_ports}，采样率={sampling_rate}Hz，缓存就绪={ready_samples}。"
            "已回写实际串口到输入框，下次采集将沿用该串口。".format(**log_payload)
        )

    def _on_connect_error(self, text: str) -> None:
        self.phase_label.setText("连接失败")
        self._log(text)

    def _on_connect_finished(self) -> None:
        self.connect_worker = None
        self.connect_thread = None

    def _start_collection(self) -> None:
        if self.worker_thread is not None:
            return
        if self.connect_thread is not None:
            self._log("设备正在连接，请等待完成。")
            return
        planned = int(self.rounds_planned_spin.value())
        if self.rounds_completed >= planned:
            self.phase_label.setText("计划轮次已完成")
            self._log("计划轮次已全部完成。请调整轮次设置后继续。")
            return
        round_index = self._round_index_for_next_run()
        try:
            cfg = self._read_config(round_index_override=round_index)
        except Exception as exc:
            self._log(f"配置错误：{exc}")
            return
        self._log(
            f"开始第 {cfg.round_index}/{cfg.rounds_planned} 轮：session={cfg.session_id}，"
            f"预计时长={format_duration(cfg.estimated_round_sec)}"
        )
        stim_refresh_rate_hz = self._resolve_stim_refresh_rate_hz()
        validate_stimulus_frequency_set(cfg.freqs, refresh_rate_hz=stim_refresh_rate_hz)
        stimulus_profile_id = self._current_stimulus_profile_id()
        stimulus_mode = self._current_stimulus_mode()
        cfg = replace(
            cfg,
            stim_refresh_rate_hz=stim_refresh_rate_hz,
            stimulus_profile_id=stimulus_profile_id,
            stimulus_mode=stimulus_mode,
            stimulus_mode_selection_reason=self._current_stimulus_mode_selection_reason(),
            stimulus_backend=STIMULUS_BACKEND_PYQT_FULLSCREEN,
            sync_stimulus_phase=True,
            sync_voice_prompt=True,
        )
        if bool(cfg.simulation_only):
            self._log("当前为流程测试模式：不连接板卡，不保存数据，但会正常运行闪烁刺激与采集流程。")
        self._log(
            f"刺激刷新率={stim_refresh_rate_hz:g}Hz，目标频率={','.join(f'{freq:g}' for freq in cfg.freqs)}Hz，"
            f"刺激模式={stimulus_mode}"
        )
        self._sync_stim_freqs(
            cfg.freqs,
            refresh_rate_hz=stim_refresh_rate_hz,
            stimulus_mode=stimulus_mode,
            stimulus_profile_id=stimulus_profile_id,
        )
        self._show_fullscreen_stimulus(
            cfg.freqs,
            refresh_rate_hz=stim_refresh_rate_hz,
            stimulus_mode=stimulus_mode,
            stimulus_profile_id=stimulus_profile_id,
        )
        worker = CollectionWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.error.connect(self._on_error)
        worker.done.connect(self._on_done)
        worker.phase_changed.connect(self._on_phase_changed)
        worker.voice_prompt_event.connect(self._on_voice_prompt_event)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.started.connect(worker.run)
        self.worker = worker
        self.worker_thread = thread
        self._set_running(True)
        self.phase_label.setText("正在启动本轮采集...")
        thread.start()

    def _stop_collection(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.btn_stop.setEnabled(False)
            self.phase_label.setText("正在停止...")
        self.speech_prompt_player.stop()
        try:
            self.stim.apply_phase(
                {
                    "mode": PHASE_STOPPED,
                    "title": "采集已停止",
                    "detail": "正在结束当前采集流程。",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        except Exception:
            pass
        self._close_fullscreen_stimulus()
        if self.worker is None:
            self._set_running(False)

    def _on_phase_changed(self, phase: dict[str, Any]) -> None:
        self.phase_label.setText(str(phase.get("title", "采集中")))
        self.stim.apply_phase(self._phase_for_embedded_preview(phase))
        if self.fullscreen_window is not None:
            self.fullscreen_window.apply_phase(phase)

    def _notify_active_stimulus_applied(self, phase: dict[str, Any]) -> None:
        if self.worker is not None:
            self.worker.notify_stimulus_phase_applied(phase)

    def _on_active_phase_frame_presented(self, payload: dict[str, Any]) -> None:
        self._notify_active_stimulus_applied(dict(payload))

    def _on_voice_prompt_event(self, payload: dict[str, Any]) -> None:
        if bool(payload.get("stop", False)):
            self.speech_prompt_player.stop()
            return
        self.speech_prompt_player.say(
            str(payload.get("text", "")),
            request_id=int(payload.get("request_id", 0) or 0),
        )

    def _on_voice_prompt_finished(self, request_id: int) -> None:
        if self.worker is not None:
            self.worker.notify_voice_prompt_finished(int(request_id))

    def _on_error(self, text: str) -> None:
        self._log(text)
        self.speech_prompt_player.stop()
        self._close_fullscreen_stimulus()

    def _on_done(self, payload: dict[str, Any]) -> None:
        self.speech_prompt_player.stop()
        self._close_fullscreen_stimulus()
        if bool(payload.get("simulation_only", False)):
            executed = int(payload.get("executed_trials", 0))
            total = int(payload.get("total_trials", 0))
            aborted = bool(payload.get("collection_aborted", False))
            self._refresh_round_status()
            if aborted:
                self._log(
                    f"流程测试已停止：已执行 {executed}/{total} 个 Trial；"
                    "未连接设备，未保存数据，本轮未计入完成轮次。"
                )
                self.phase_label.setText("流程测试已停止")
            else:
                self._log(
                    f"流程测试完成：已执行 {executed}/{total} 个 Trial；"
                    "未连接设备，未保存数据，本轮未计入完成轮次。"
                )
                self.phase_label.setText("流程测试完成")
            return
        aborted = bool(payload.get("collection_aborted", False))
        if aborted:
            self._refresh_round_status()
            collected = int(payload.get("collected_trials", 0))
            total = int(payload.get("total_trials", 0))
            manifest = str(payload.get("dataset_manifest", "") or "")
            npz_path = str(payload.get("dataset_npz", "") or "")
            if manifest:
                saved_label = "原始板卡数据" if collected <= 0 else "部分数据"
                self._log(
                    f"第 {payload.get('round_index', self._round_index_for_next_run())}/"
                    f"{payload.get('rounds_planned', self.rounds_planned_spin.value())} 轮已停止："
                    f"已保存{saved_label} {collected}/{total}，manifest={manifest}，npz={npz_path}；"
                    "本轮未计入完成轮次。"
                )
            else:
                self._log("本轮已停止：未采到有效 Trial，未保存数据；本轮未计入完成轮次。")
            self.phase_label.setText("本轮已停止，未计入完成轮次")
            return
        self.rounds_completed += 1
        self._refresh_round_status()
        self._log(
            "第 {round_index}/{rounds_planned} 轮完成：采集={collected_trials}/{total_trials}，"
            "manifest={dataset_manifest}，npz={dataset_npz}".format(**payload)
        )
        planned = int(self.rounds_planned_spin.value())
        if self.rounds_completed < planned:
            self.phase_label.setText("本轮完成，请手动开始下一轮")
        else:
            self.phase_label.setText("计划轮次已全部完成")

    def _on_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.speech_prompt_player.stop()
        self._close_fullscreen_stimulus()
        self._set_running(False)

    def closeEvent(self, event) -> None:
        if self.worker_thread is not None:
            self._stop_collection()
            self._log("正在停止采集并保存数据，请等待完成后再关闭窗口。")
            event.ignore()
            return
        if self.connect_thread is not None:
            self._log("正在连接设备，请等待连接结束后再关闭窗口。")
            event.ignore()
            return
        self.speech_prompt_player.stop()
        self._close_fullscreen_stimulus()
        try:
            self.stim.stop_clock()
        except Exception:
            pass
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP 数据采集 UI / CLI")
    parser.add_argument("--serial-port", type=str, default="auto")
    parser.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    parser.add_argument("--freqs", type=str, default=DEFAULT_COLLECTION_FREQS_CSV)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--subject-id", type=str, default="subject001")
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--session-index", type=int, default=1)
    parser.add_argument("--rounds-planned", type=int, default=1)
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET_NAME,
        help="stable_12m|enhanced_45m|custom",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="",
        help="已弃用参数，等价于 --preset",
    )
    parser.add_argument("--prepare-sec", type=float, default=DEFAULT_STABLE_PREPARE_SEC)
    parser.add_argument("--active-sec", type=float, default=DEFAULT_STABLE_ACTIVE_SEC)
    parser.add_argument("--rest-sec", type=float, default=DEFAULT_STABLE_REST_SEC)
    parser.add_argument("--long-idle-sec", type=float, default=DEFAULT_STABLE_LONG_IDLE_SEC)
    parser.add_argument("--target-repeats", type=int, default=DEFAULT_STABLE_TARGET_REPEATS)
    parser.add_argument("--idle-repeats", type=int, default=DEFAULT_STABLE_IDLE_REPEATS)
    parser.add_argument("--switch-trials", type=int, default=DEFAULT_STABLE_SWITCH_TRIALS)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument(
        "--stimulus-profile-id",
        type=str,
        default=DEFAULT_COLLECTION_STIMULUS_PROFILE_ID,
        choices=tuple(sorted(STIMULUS_PROFILES)),
        help="stimulus profile id, default comfort_fbcca_v1",
    )
    parser.add_argument(
        "--stimulus-mode",
        type=str,
        default="auto",
        choices=("auto",) + STIMULUS_MODES,
        help="auto|elapsed_time_sine|frame_locked_sine",
    )
    parser.add_argument(
        "--stim-refresh-rate-hz",
        type=float,
        default=STIM_REFRESH_RATE_HZ,
        help="Stimulus refresh rate for rendering; use 0 to auto-detect from Qt screen.",
    )
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="流程测试模式：不连接板卡、不保存数据，只运行提示与刺激流程",
    )
    parser.add_argument("--headless", action="store_true", help="仅命令行采集，不启动 UI")
    return parser


def _resolve_cli_protocol(
    *,
    preset_name: str,
    prepare_sec: float,
    active_sec: float,
    rest_sec: float,
    long_idle_sec: float,
    target_repeats: int,
    idle_repeats: int,
    switch_trials: int,
) -> tuple[str, float, float, float, float, int, int, int]:
    preset_key = normalize_preset_name(preset_name)
    if preset_key in (STABLE_12M_PRESET.key, ENHANCED_45M_PRESET.key):
        preset = COLLECTION_PRESETS[preset_key]
        return (
            preset_key,
            float(preset.prepare_sec),
            float(preset.active_sec),
            float(preset.rest_sec),
            float(preset.long_idle_sec),
            int(preset.target_repeats),
            int(preset.idle_repeats),
            int(preset.switch_trials),
        )
    return (
        CUSTOM_PRESET.key,
        float(prepare_sec),
        float(active_sec),
        float(rest_sec),
        float(long_idle_sec),
        int(target_repeats),
        int(idle_repeats),
        int(switch_trials),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    freqs = parse_freqs(args.freqs)
    stimulus_profile_id = validate_stimulus_profile_id(str(args.stimulus_profile_id))
    requested_refresh_rate_hz = float(args.stim_refresh_rate_hz)
    refresh_rate_is_manual = bool(np.isfinite(requested_refresh_rate_hz) and requested_refresh_rate_hz > 1.0)
    if refresh_rate_is_manual:
        stim_refresh_rate_hz = float(requested_refresh_rate_hz)
    else:
        stim_refresh_rate_hz = float(STIM_REFRESH_RATE_HZ)
    validate_stimulus_frequency_set(freqs, refresh_rate_hz=stim_refresh_rate_hz)
    stimulus_mode, stimulus_mode_selection_reason = resolve_collection_stimulus_mode(
        stimulus_profile_id=stimulus_profile_id,
        refresh_rate_hz=stim_refresh_rate_hz,
        requested_mode=str(args.stimulus_mode),
    )
    requested_preset = str(args.protocol).strip() or str(args.preset).strip()
    (
        protocol_name,
        prepare_sec,
        active_sec,
        rest_sec,
        long_idle_sec,
        target_repeats,
        idle_repeats,
        switch_trials,
    ) = _resolve_cli_protocol(
        preset_name=requested_preset,
        prepare_sec=float(args.prepare_sec),
        active_sec=float(args.active_sec),
        rest_sec=float(args.rest_sec),
        long_idle_sec=float(args.long_idle_sec),
        target_repeats=int(args.target_repeats),
        idle_repeats=int(args.idle_repeats),
        switch_trials=int(args.switch_trials),
    )
    subject_id = sanitize_collection_token(str(args.subject_id).strip() or "subject001", default="subject001")
    round_index = int(args.round_index) if int(args.round_index) > 0 else int(args.session_index)
    session_base = _strip_round_suffix(str(args.session_id).strip()) or _auto_session_base_id(subject_id)
    session_id = _build_round_session_id(session_base, round_index)
    estimated_round_sec = estimate_round_seconds(
        prepare_sec=prepare_sec,
        active_sec=active_sec,
        rest_sec=rest_sec,
        target_repeats=target_repeats,
        idle_repeats=idle_repeats,
        switch_trials=switch_trials,
        long_idle_sec=long_idle_sec,
        refresh_rate_hz=stim_refresh_rate_hz,
    )
    _validate_collection_protocol(
        prepare_sec=prepare_sec,
        active_sec=active_sec,
        rest_sec=rest_sec,
        long_idle_sec=long_idle_sec,
    )
    config = CollectionConfig(
        serial_port=normalize_serial_port(args.serial_port),
        board_id=int(args.board_id),
        freqs=freqs,
        subject_id=subject_id,
        session_id=session_id,
        session_index=round_index,
        dataset_dir=resolve_dataset_dir(args.dataset_dir),
        protocol_name=protocol_name,
        prepare_sec=prepare_sec,
        active_sec=active_sec,
        rest_sec=rest_sec,
        target_repeats=target_repeats,
        idle_repeats=idle_repeats,
        switch_trials=switch_trials,
        long_idle_sec=long_idle_sec,
        seed=int(args.seed),
        rounds_planned=max(1, int(args.rounds_planned)),
        round_index=round_index,
        estimated_round_sec=estimated_round_sec,
        stimulus_profile_id=stimulus_profile_id,
        stim_refresh_rate_hz=stim_refresh_rate_hz,
        stimulus_mode=stimulus_mode,
        stimulus_mode_selection_reason=stimulus_mode_selection_reason,
        simulation_only=bool(args.simulation_only),
    )
    if bool(args.headless):
        if bool(config.simulation_only):
            print(
                "Simulation-only + headless will not render visual stimulus, "
                "but will still execute the scripted timing flow without board/save.",
                flush=True,
            )
        else:
            print(
                "Headless mode does not render visual stimulus; "
                f"stimulus_backend={STIMULUS_BACKEND_HEADLESS_NO_VISUAL}.",
                flush=True,
            )
        payload = run_collection_cli(config)
        if bool(config.simulation_only):
            print("Simulation-only run finished: no dataset saved.", flush=True)
        else:
            print(f"Dataset manifest: {payload.get('dataset_manifest', '')}", flush=True)
        return 0

    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = DatasetCollectionWindow(
        serial_port=config.serial_port,
        board_id=config.board_id,
        freqs=config.freqs,
        simulation_only_default=config.simulation_only,
    )
    window.stim_refresh_rate_spin.setValue(float(config.stim_refresh_rate_hz) if refresh_rate_is_manual else 0.0)
    window._set_dataset_dir_text(config.dataset_dir)
    window.subject_edit.setText(config.subject_id)
    window.session_index_spin.setValue(config.session_index)
    window.rounds_planned_spin.setValue(config.rounds_planned)
    window.session_base_edit.setText(session_base)
    stimulus_profile_index = window.stimulus_profile_combo.findData(config.stimulus_profile_id)
    if stimulus_profile_index >= 0:
        window.stimulus_profile_combo.setCurrentIndex(stimulus_profile_index)
    raw_mode = str(args.stimulus_mode)
    stimulus_mode_index = window.stimulus_mode_combo.findData(raw_mode if raw_mode == "auto" else config.stimulus_mode)
    if stimulus_mode_index >= 0:
        window.stimulus_mode_combo.setCurrentIndex(stimulus_mode_index)
    if config.protocol_name in COLLECTION_PRESETS:
        window.preset_combo.setCurrentText(COLLECTION_PRESETS[config.protocol_name].display)
    else:
        window.preset_combo.setCurrentText(CUSTOM_PRESET.display)
    window.prepare_spin.setValue(config.prepare_sec)
    window.active_spin.setValue(config.active_sec)
    window.rest_spin.setValue(config.rest_sec)
    window.long_idle_spin.setValue(config.long_idle_sec)
    window.target_spin.setValue(config.target_repeats)
    window.idle_spin.setValue(config.idle_repeats)
    window.switch_spin.setValue(config.switch_trials)
    window._refresh_estimate_label()
    window._refresh_round_status()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

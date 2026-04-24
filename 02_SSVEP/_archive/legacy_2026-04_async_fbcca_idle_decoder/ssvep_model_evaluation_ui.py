from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
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

from async_fbcca_idle_standalone import (
    BoardShim,
    DEFAULT_BENCHMARK_CAL_IDLE_REPEATS,
    DEFAULT_BENCHMARK_CAL_TARGET_REPEATS,
    DEFAULT_BENCHMARK_DATASET_ROOT,
    DEFAULT_BENCHMARK_EVAL_IDLE_REPEATS,
    DEFAULT_BENCHMARK_EVAL_TARGET_REPEATS,
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BENCHMARK_SWITCH_TRIALS,
    DEFAULT_BOARD_ID,
    DEFAULT_CHANNEL_WEIGHT_MODE,
    DEFAULT_DYNAMIC_STOP_ALPHA,
    DEFAULT_DYNAMIC_STOP_ENABLED,
    DEFAULT_GATE_POLICY,
    DEFAULT_PROFILE_PATH,
    DEFAULT_WIN_SEC_CANDIDATES,
    benchmark_rank_key,
    describe_runtime_error,
    ensure_stream_ready,
    normalize_serial_port,
    parse_channel_mode_list,
    parse_freqs,
    parse_model_list,
    prepare_board_session,
    profile_meets_acceptance,
)
from async_fbcca_validation_ui import (
    FourArrowStimWidget,
    PHASE_CAL_ACTIVE,
    PHASE_CAL_REST,
    PHASE_ERROR,
    PHASE_IDLE,
    PHASE_STOPPED,
    PHASE_VALIDATION,
)


THIS_DIR = Path(__file__).resolve().parent
STANDALONE_SCRIPT = THIS_DIR / "async_fbcca_idle_standalone.py"
DEFAULT_REPORT_DIR = THIS_DIR / "profiles"
DEFAULT_DATASET_DIR = DEFAULT_BENCHMARK_DATASET_ROOT

TRIAL_LINE_PATTERN = re.compile(
    r"^(Calibration|Evaluation)\s+(\d+)/(\d+)\s+(focus\s+(.+?)|idle\b.*?)(?:\s+\((\d+)s\))?$",
    flags=re.IGNORECASE,
)
REST_LINE_PATTERN = re.compile(r"^Rest(?:\s+\((\d+)s\))?$", flags=re.IGNORECASE)
MODEL_LINE_PATTERN = re.compile(r"^Benchmark model:\s*(.+?)\s*$", flags=re.IGNORECASE)
FREQ_TOKEN_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*Hz", flags=re.IGNORECASE)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_non_conflicting_report_path(report_path: Path, *, now_stamp: Optional[str] = None) -> Path:
    candidate = Path(report_path).expanduser().resolve()
    if not candidate.exists():
        return candidate
    stamp = str(now_stamp or _now_stamp())
    return candidate.with_name(f"{candidate.stem}_{stamp}{candidate.suffix}")


def phase_from_benchmark_line(line: str) -> Optional[dict[str, Any]]:
    text = str(line).strip()
    if not text:
        return None
    rest_match = REST_LINE_PATTERN.match(text)
    if rest_match is not None:
        remaining = int(rest_match.group(1) or 0)
        return {
            "mode": PHASE_CAL_REST,
            "title": "Rest",
            "detail": "Rest and relax.",
            "remaining_sec": remaining,
            "flicker": False,
            "cue_freq": None,
        }
    trial_match = TRIAL_LINE_PATTERN.match(text)
    if trial_match is not None:
        stage = str(trial_match.group(1)).capitalize()
        index = int(trial_match.group(2))
        total = int(trial_match.group(3))
        focus_token_raw = str(trial_match.group(4)).strip()
        focus_token = focus_token_raw.lower()
        cue_freq: Optional[float] = None
        freq_match = FREQ_TOKEN_PATTERN.search(focus_token)
        if freq_match is not None:
            cue_freq = float(freq_match.group(1))
        remaining = int(trial_match.group(6) or 0)
        if "idle" in focus_token:
            detail = f"{stage} {index}/{total}: idle trial (look center, avoid targets)"
            cue_freq = None
        else:
            focus_display = focus_token_raw.replace("focus ", "", 1).strip()
            if cue_freq is not None:
                detail = f"{stage} {index}/{total}: focus {focus_display} ({cue_freq:g}Hz)"
            else:
                detail = f"{stage} {index}/{total}: focus {focus_display}"
        return {
            "mode": PHASE_CAL_ACTIVE,
            "title": f"{stage} Trial {index}/{total}",
            "detail": detail,
            "remaining_sec": remaining,
            "flicker": True,
            "cue_freq": cue_freq,
        }
    model_match = MODEL_LINE_PATTERN.match(text)
    if model_match is not None:
        model_name = str(model_match.group(1))
        return {
            "mode": PHASE_VALIDATION,
            "title": f"Evaluating model: {model_name}",
            "detail": "Running model scoring and gate search.",
            "flicker": False,
            "cue_freq": None,
        }
    return None


def rank_model_results(model_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    successful: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for item in model_results:
        cloned = dict(item)
        if "metrics" in cloned:
            successful.append(cloned)
        else:
            cloned["rank"] = None
            failed.append(cloned)
    successful.sort(key=lambda item: benchmark_rank_key(item["metrics"]))
    for index, item in enumerate(successful, start=1):
        item["rank"] = int(index)
        item["meets_acceptance"] = bool(profile_meets_acceptance(item["metrics"]))
    return [*successful, *failed]


def enrich_benchmark_report(report_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(report_payload)
    ranked = rank_model_results(payload.get("model_results", []))
    payload["model_results"] = ranked
    successful = [item for item in ranked if "metrics" in item]
    accepted = [item for item in successful if bool(item.get("meets_acceptance"))]
    chosen = accepted[0] if accepted else (successful[0] if successful else None)
    if chosen is not None:
        payload["chosen_model"] = str(chosen.get("model_name"))
        payload["chosen_metrics"] = dict(chosen.get("metrics", {}))
        payload["chosen_rank"] = int(chosen.get("rank", 0) or 0)
        payload["chosen_meets_acceptance"] = bool(chosen.get("meets_acceptance", False))
    else:
        payload["chosen_model"] = None
        payload["chosen_metrics"] = {}
        payload["chosen_rank"] = None
        payload["chosen_meets_acceptance"] = False
    return payload


def render_benchmark_markdown(report_payload: dict[str, Any]) -> str:
    payload = enrich_benchmark_report(report_payload)
    generated_at = str(payload.get("generated_at", ""))
    chosen_model = payload.get("chosen_model")
    chosen_rank = payload.get("chosen_rank")
    chosen_ok = bool(payload.get("chosen_meets_acceptance", False))
    rank_policy = payload.get("metric_definition", {}).get("ranking_policy", {})
    switch_metric_mode = (
        payload.get("metric_definition", {})
        .get("switch_latency_s", {})
        .get("mode", "")
    )
    robustness = payload.get("robustness", {})
    robust_recommendation = payload.get("robust_recommendation")
    lines = [
        "# SSVEP Benchmark Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Serial port: `{payload.get('serial_port', '')}`",
        f"- Board ID: `{payload.get('board_id', '')}`",
        f"- Chosen model: `{chosen_model}` (rank={chosen_rank}, meets_acceptance={chosen_ok})",
        f"- Dataset dir: `{payload.get('dataset_dir', '')}`",
        f"- Dataset manifest: `{payload.get('dataset_manifest', '')}`",
        f"- Dataset npz: `{payload.get('dataset_npz', '')}`",
        f"- Ranking min control_recall: `{rank_policy.get('min_control_recall_for_ranking', '')}`",
        f"- switch_latency_s mode: `{switch_metric_mode}`",
        f"- Robustness modes: `{','.join(str(item) for item in robustness.get('channel_modes', []))}`",
        f"- Robustness seeds: `{','.join(str(item) for item in robustness.get('seeds', []))}`",
        "",
        "## Ranked Models (Primary Run)",
        "",
        "| Rank | Model | Impl | idle_fp_per_min | control_recall | switch_detect_rate | switch_latency_s | release_latency_s | itr_bpm | inference_ms | Accept |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    ranked = payload.get("model_results", [])
    for item in ranked:
        if "metrics" not in item:
            continue
        metrics = item["metrics"]
        lines.append(
            "| {rank} | {model} | {impl} | {idle:.4f} | {recall:.4f} | {switch_detect:.4f} | {switch:.4f} | {release:.4f} | {itr:.4f} | {infer:.4f} | {ok} |".format(
                rank=int(item.get("rank", 0) or 0),
                model=str(item.get("model_name", "")),
                impl=str(item.get("implementation_level", "")),
                idle=float(metrics.get("idle_fp_per_min", float("inf"))),
                recall=float(metrics.get("control_recall", 0.0)),
                switch_detect=float(metrics.get("switch_detect_rate", 0.0)),
                switch=float(metrics.get("switch_latency_s", float("inf"))),
                release=float(metrics.get("release_latency_s", float("inf"))),
                itr=float(metrics.get("itr_bpm", 0.0)),
                infer=float(metrics.get("inference_ms", float("inf"))),
                ok="Y" if bool(item.get("meets_acceptance")) else "N",
            )
        )
    failed = [item for item in ranked if "metrics" not in item]
    if failed:
        lines.extend(["", "## Failed Models", ""])
        for item in failed:
            lines.append(f"- `{item.get('model_name', '')}`: {item.get('error', 'unknown error')}")

    if isinstance(robust_recommendation, dict):
        lines.extend(
            [
                "",
                "## Robust Recommendation",
                "",
                f"- channel_mode: `{robust_recommendation.get('channel_mode', '')}`",
                    f"- model: `{robust_recommendation.get('model_name', '')}`",
                    f"- rank: `{robust_recommendation.get('rank', '')}`",
                ]
            )

    chosen_fixed = payload.get("chosen_fixed_window_metrics", {})
    chosen_delta = payload.get("chosen_dynamic_delta", {})
    if isinstance(chosen_fixed, dict) and isinstance(chosen_delta, dict) and chosen_fixed:
        lines.extend(
            [
                "",
                "## Dynamic vs Fixed (Chosen Model)",
                "",
                f"- dynamic switch_latency_s: `{payload.get('chosen_metrics', {}).get('switch_latency_s', '')}`",
                f"- fixed switch_latency_s: `{chosen_fixed.get('switch_latency_s', '')}`",
                f"- delta switch_latency_s: `{chosen_delta.get('switch_latency_s', '')}`",
                f"- dynamic release_latency_s: `{payload.get('chosen_metrics', {}).get('release_latency_s', '')}`",
                f"- fixed release_latency_s: `{chosen_fixed.get('release_latency_s', '')}`",
                f"- delta release_latency_s: `{chosen_delta.get('release_latency_s', '')}`",
            ]
        )

    robustness_modes = robustness.get("by_mode", {}) if isinstance(robustness, dict) else {}
    if isinstance(robustness_modes, dict) and robustness_modes:
        lines.extend(["", "## Robustness Summary", ""])
        for mode_name in sorted(robustness_modes.keys()):
            mode_payload = robustness_modes.get(mode_name, {})
            ranked_models = list(mode_payload.get("ranked_models", [])) if isinstance(mode_payload, dict) else []
            if not ranked_models:
                continue
            lines.extend(
                [
                    f"### Mode `{mode_name}`",
                    "",
                    "| Rank | Model | Mean Rank | Std Rank | Mean idle_fp_per_min | Mean control_recall | Mean switch_detect_rate | Mean switch_latency_s | Mean release_latency_s | Runs |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for item in ranked_models:
                metrics_mean = dict(item.get("metrics_mean", {}))
                lines.append(
                    "| {rank} | {model} | {mean_rank:.3f} | {std_rank:.3f} | {idle:.3f} | {recall:.3f} | {switch_detect:.3f} | {switch_lat:.3f} | {release_lat:.3f} | {runs_success}/{runs_total} |".format(
                        rank=int(item.get("rank", 0) or 0),
                        model=str(item.get("model_name", "")),
                        mean_rank=float(item.get("mean_rank", float("inf"))),
                        std_rank=float(item.get("std_rank", float("inf"))),
                        idle=float(metrics_mean.get("idle_fp_per_min", float("inf"))),
                        recall=float(metrics_mean.get("control_recall", 0.0)),
                        switch_detect=float(metrics_mean.get("switch_detect_rate", 0.0)),
                        switch_lat=float(metrics_mean.get("switch_latency_s", float("inf"))),
                        release_lat=float(metrics_mean.get("release_latency_s", float("inf"))),
                        runs_success=int(item.get("runs_success", 0)),
                        runs_total=int(item.get("runs_total", 0)),
                    )
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_benchmark_report_bundle(report_payload: dict[str, Any], report_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = enrich_benchmark_report(report_payload)
    report_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = report_path.with_suffix(".md")
    markdown_path.write_text(render_benchmark_markdown(enriched), encoding="utf-8")
    return report_path, markdown_path, enriched


@dataclass(frozen=True)
class EvalConfig:
    serial_port: str
    board_id: int
    freqs: tuple[float, float, float, float]
    model_names: tuple[str, ...]
    channel_modes: tuple[str, ...]
    multi_seed_count: int
    output_profile_path: Path
    report_path: Path
    dataset_dir: Path


def build_benchmark_command(config: EvalConfig, *, python_executable: Optional[str] = None) -> list[str]:
    python_bin = str(python_executable or sys.executable)
    command = [
        python_bin,
        str(STANDALONE_SCRIPT),
        "benchmark",
        "--serial-port",
        str(config.serial_port),
        "--board-id",
        str(int(config.board_id)),
        "--freqs",
        ",".join(f"{freq:g}" for freq in config.freqs),
        "--output-profile",
        str(config.output_profile_path),
        "--report-path",
        str(config.report_path),
        "--dataset-dir",
        str(config.dataset_dir),
        "--models",
        ",".join(config.model_names),
        "--channel-modes",
        ",".join(config.channel_modes),
        "--multi-seed-count",
        str(int(config.multi_seed_count)),
        "--prepare-sec",
        "1.0",
        "--active-sec",
        "4.0",
        "--rest-sec",
        "1.0",
        "--calibration-target-repeats",
        str(int(DEFAULT_BENCHMARK_CAL_TARGET_REPEATS)),
        "--calibration-idle-repeats",
        str(int(DEFAULT_BENCHMARK_CAL_IDLE_REPEATS)),
        "--eval-target-repeats",
        str(int(DEFAULT_BENCHMARK_EVAL_TARGET_REPEATS)),
        "--eval-idle-repeats",
        str(int(DEFAULT_BENCHMARK_EVAL_IDLE_REPEATS)),
        "--eval-switch-trials",
        str(int(DEFAULT_BENCHMARK_SWITCH_TRIALS)),
        "--step-sec",
        "0.25",
        "--win-candidates",
        ",".join(f"{value:g}" for value in DEFAULT_WIN_SEC_CANDIDATES),
    ]
    command.extend(["--gate-policy", str(DEFAULT_GATE_POLICY)])
    command.extend(["--channel-weight-mode", str(DEFAULT_CHANNEL_WEIGHT_MODE)])
    command.extend(["--dynamic-stop-alpha", f"{float(DEFAULT_DYNAMIC_STOP_ALPHA):g}"])
    if not bool(DEFAULT_DYNAMIC_STOP_ENABLED):
        command.append("--disable-dynamic-stop")
    return command


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
            self.error.emit(f"Connect failed: {describe_runtime_error(exc, serial_port=self.serial_port)}")
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


class EvaluationWorker(QObject):
    log = pyqtSignal(str)
    phase_changed = pyqtSignal(object)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, config: EvalConfig) -> None:
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()
        self._process: Optional[subprocess.Popen] = None

    def request_stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    @pyqtSlot()
    def run(self) -> None:
        cmd = build_benchmark_command(self.config)
        self.log.emit(f"Start benchmark: {' '.join(cmd)}")
        self.phase_changed.emit(
            {
                "mode": PHASE_VALIDATION,
                "title": "Benchmark running",
                "detail": "Follow prompts in log; flicker starts only during trial collection.",
                "flicker": False,
                "cue_freq": None,
            }
        )
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(THIS_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert self._process.stdout is not None
            while True:
                if self._stop_event.is_set() and self._process.poll() is None:
                    self.log.emit("Stopping benchmark process...")
                    try:
                        self._process.terminate()
                    except Exception:
                        pass
                line = self._process.stdout.readline()
                if line:
                    clean = line.rstrip("\n")
                    self.log.emit(clean)
                    parsed = phase_from_benchmark_line(clean)
                    if parsed is not None:
                        self.phase_changed.emit(parsed)
                    continue
                code = self._process.poll()
                if code is not None:
                    if code != 0:
                        if self._stop_event.is_set():
                            self.log.emit(f"Benchmark terminated after stop request (code={code}).")
                            break
                        raise RuntimeError(f"benchmark subprocess exited with code={code}")
                    break
                time.sleep(0.05)

            if self._stop_event.is_set():
                self.phase_changed.emit(
                    {
                        "mode": PHASE_STOPPED,
                        "title": "Benchmark stopped",
                        "detail": "Stopped by user.",
                        "flicker": False,
                        "cue_freq": None,
                    }
                )
                self.finished.emit({"stopped": True})
                return

            if not self.config.report_path.exists():
                raise RuntimeError(f"benchmark report not found: {self.config.report_path}")
            raw_report = json.loads(self.config.report_path.read_text(encoding="utf-8"))
            report_path, markdown_path, enriched_report = save_benchmark_report_bundle(raw_report, self.config.report_path)
            self.finished.emit(
                {
                    "stopped": False,
                    "report_path": str(report_path),
                    "markdown_path": str(markdown_path),
                    "report": enriched_report,
                }
            )
            self.phase_changed.emit(
                {
                    "mode": PHASE_STOPPED,
                    "title": "Benchmark completed",
                    "detail": "Evaluation finished. Flicker stopped.",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
        except Exception as exc:
            if self._stop_event.is_set():
                self.phase_changed.emit(
                    {
                        "mode": PHASE_STOPPED,
                        "title": "Benchmark stopped",
                        "detail": "Stopped by user.",
                        "flicker": False,
                        "cue_freq": None,
                    }
                )
                self.finished.emit({"stopped": True})
            else:
                self.phase_changed.emit(
                    {
                        "mode": PHASE_ERROR,
                        "title": "Benchmark error",
                        "detail": str(exc),
                        "flicker": False,
                        "cue_freq": None,
                    }
                )
                self.error.emit(str(exc))
        finally:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            self._process = None


class ModelEvaluationWindow(QMainWindow):
    def __init__(self, *, serial_port: str, board_id: int, freqs: tuple[float, float, float, float]) -> None:
        super().__init__()
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[EvaluationWorker] = None
        self.connect_thread: Optional[QThread] = None
        self.connect_worker: Optional[DeviceCheckWorker] = None
        self.last_report_payload: Optional[dict[str, Any]] = None
        self.last_markdown_path: Optional[Path] = None

        self.initial_serial_port = normalize_serial_port(serial_port)
        self.initial_board_id = int(board_id)
        self.initial_freqs = tuple(float(freq) for freq in freqs)
        self.detected_refresh_hz = self._detect_refresh_rate()
        self._build_ui()

    @staticmethod
    def _detect_refresh_rate() -> float:
        app = QApplication.instance()
        if app is None:
            return 60.0
        screen = app.primaryScreen()
        if screen is None:
            return 60.0
        refresh = float(screen.refreshRate() or 0.0)
        if not np.isfinite(refresh) or refresh < 30.0 or refresh > 360.0:
            return 60.0
        return refresh

    def _build_ui(self) -> None:
        self.setWindowTitle("SSVEP Model Evaluation UI")
        self.resize(1460, 900)
        root = QWidget(self)
        self.setCentralWidget(root)
        root.setStyleSheet(
            """
            QWidget { background-color: black; color: #dce5ea; }
            QLabel { color: #dce5ea; }
            QLineEdit, QSpinBox, QPlainTextEdit {
                background-color: #121417;
                color: #dce5ea;
                border: 1px solid #2d3943;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: #1b242b;
                color: #eaf0f4;
                border: 1px solid #33414c;
                border-radius: 5px;
                padding: 6px 12px;
            }
            QPushButton:disabled {
                color: #7b8893;
                background-color: #11161a;
                border: 1px solid #202830;
            }
            """
        )

        layout = QHBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left.setMaximumWidth(480)

        title = QLabel("SSVEP 多模型评测（内置刺激）")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        left_layout.addWidget(title)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.serial_port_edit = QLineEdit(self.initial_serial_port)
        self.board_id_spin = QSpinBox()
        self.board_id_spin.setRange(-1, 9999)
        self.board_id_spin.setValue(self.initial_board_id)
        self.freqs_edit = QLineEdit(",".join(f"{freq:g}" for freq in self.initial_freqs))
        self.models_edit = QLineEdit(",".join(DEFAULT_BENCHMARK_MODELS))
        self.channel_modes_edit = QLineEdit("auto,all8")
        self.multi_seed_count_spin = QSpinBox()
        self.multi_seed_count_spin.setRange(1, 20)
        self.multi_seed_count_spin.setValue(5)
        self.profile_path_edit = QLineEdit(str(DEFAULT_PROFILE_PATH))
        self.report_path_edit = QLineEdit(str(DEFAULT_REPORT_DIR / f"benchmark_report_{_now_stamp()}.json"))
        self.dataset_dir_edit = QLineEdit(str(DEFAULT_DATASET_DIR))
        self._config_widgets = [
            self.serial_port_edit,
            self.board_id_spin,
            self.freqs_edit,
            self.models_edit,
            self.channel_modes_edit,
            self.multi_seed_count_spin,
            self.profile_path_edit,
            self.report_path_edit,
            self.dataset_dir_edit,
        ]

        form.addRow("Serial Port", self.serial_port_edit)
        form.addRow("Board ID", self.board_id_spin)
        form.addRow("Frequencies", self.freqs_edit)
        form.addRow("Models", self.models_edit)
        form.addRow("Channel Modes", self.channel_modes_edit)
        form.addRow("Multi-seed Count", self.multi_seed_count_spin)
        form.addRow("Output Profile", self.profile_path_edit)
        form.addRow("Report JSON", self.report_path_edit)
        form.addRow("Dataset Dir", self.dataset_dir_edit)
        left_layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("连接设备")
        self.btn_start = QPushButton("开始评测")
        self.btn_stop = QPushButton("停止")
        self.btn_export = QPushButton("导出报告")
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(False)
        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_export)
        left_layout.addLayout(btn_row)

        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(220)
        left_layout.addWidget(self.summary_text, 1)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(260)
        left_layout.addWidget(self.log_text, 1)
        layout.addWidget(left, 0)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)
        self.phase_label = QLabel("Ready")
        self.phase_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #89d7ff;")
        center_layout.addWidget(self.phase_label, 0)
        self.phase_detail_label = QLabel("Connect device, then start benchmark.")
        self.phase_detail_label.setWordWrap(True)
        self.phase_detail_label.setStyleSheet("font-size: 13px; color: #b8c7d2;")
        center_layout.addWidget(self.phase_detail_label, 0)
        self.stim_widget = FourArrowStimWidget(
            freqs=self.initial_freqs,
            refresh_rate_hz=self.detected_refresh_hz,
            mean=0.5,
            amp=0.5,
            phi=0.0,
        )
        center_layout.addWidget(self.stim_widget, 1)
        layout.addWidget(center, 1)

        self.btn_connect.clicked.connect(self.connect_device)
        self.btn_start.clicked.connect(self.start_evaluation)
        self.btn_stop.clicked.connect(self.stop_evaluation)
        self.btn_export.clicked.connect(self.export_report)

        self.on_phase_changed(
            {
                "mode": PHASE_IDLE,
                "title": "Ready",
                "detail": "先连接设备，再开始评测。",
                "flicker": False,
                "cue_freq": None,
            }
        )
        self._set_summary(
            f"Ready. 连接设备 -> 开始评测\nStim refresh rate: {self.detected_refresh_hz:.1f} Hz"
        )

    def _set_summary(self, text: str) -> None:
        self.summary_text.setPlainText(str(text))

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for widget in self._config_widgets:
            widget.setEnabled(bool(enabled))

    def _sync_stim_freqs(self, freqs: Sequence[float]) -> None:
        normalized = tuple(float(freq) for freq in freqs)
        if tuple(self.stim_widget.freqs) != normalized:
            self.stim_widget.freqs = normalized
            self.stim_widget.update()

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {message}")
        bar = self.log_text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _collect_config(self) -> EvalConfig:
        freqs = parse_freqs(self.freqs_edit.text())
        models = parse_model_list(self.models_edit.text())
        channel_modes = parse_channel_mode_list(self.channel_modes_edit.text())
        output_profile = Path(self.profile_path_edit.text().strip()).expanduser().resolve()
        report_path_input = Path(self.report_path_edit.text().strip()).expanduser().resolve()
        report_path = resolve_non_conflicting_report_path(report_path_input)
        if report_path != report_path_input:
            self.report_path_edit.setText(str(report_path))
            self._log(f"Report path exists, switched to new file: {report_path}")
        dataset_dir = Path(self.dataset_dir_edit.text().strip()).expanduser().resolve()
        return EvalConfig(
            serial_port=normalize_serial_port(self.serial_port_edit.text().strip()),
            board_id=int(self.board_id_spin.value()),
            freqs=freqs,
            model_names=models,
            channel_modes=channel_modes,
            multi_seed_count=int(self.multi_seed_count_spin.value()),
            output_profile_path=output_profile,
            report_path=report_path,
            dataset_dir=dataset_dir,
        )

    def connect_device(self) -> None:
        if self.worker_thread is not None:
            self._log("Benchmark is running; stop it first.")
            return
        if self.connect_thread is not None:
            self._log("Device connection is in progress.")
            return
        serial_port = normalize_serial_port(self.serial_port_edit.text().strip())
        board_id = int(self.board_id_spin.value())
        self.connect_thread = QThread(self)
        self.connect_worker = DeviceCheckWorker(serial_port=serial_port, board_id=board_id)
        self.connect_worker.moveToThread(self.connect_thread)
        self.connect_thread.started.connect(self.connect_worker.run)
        self.connect_worker.connected.connect(self.on_connected)
        self.connect_worker.error.connect(self.on_connect_error)
        self.connect_worker.finished.connect(self.connect_thread.quit)
        self.connect_thread.finished.connect(self.connect_thread.deleteLater)
        self.connect_thread.finished.connect(self._cleanup_connect_worker)
        self.btn_connect.setEnabled(False)
        self._set_inputs_enabled(False)
        self._log(f"Connecting device serial={serial_port}, board_id={board_id} ...")
        self.connect_thread.start()

    @pyqtSlot(object)
    def on_connected(self, payload: dict[str, Any]) -> None:
        resolved = str(payload.get("resolved_serial_port", ""))
        if resolved:
            self.serial_port_edit.setText(resolved)
        self._log(
            f"Connected: requested={payload.get('requested_serial_port')} -> {resolved}, "
            f"fs={payload.get('sampling_rate')}Hz, ready_samples={payload.get('ready_samples')}"
        )

    @pyqtSlot(str)
    def on_connect_error(self, message: str) -> None:
        self._log(message)

    @pyqtSlot()
    def _cleanup_connect_worker(self) -> None:
        self.connect_worker = None
        self.connect_thread = None
        self.btn_connect.setEnabled(self.worker_thread is None)
        self._set_inputs_enabled(self.worker_thread is None)

    def start_evaluation(self) -> None:
        if self.worker_thread is not None:
            self._log("Benchmark already running.")
            return
        if self.connect_thread is not None:
            self._log("Device connection in progress; wait.")
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            self._log(f"Invalid config: {exc}")
            return
        self._sync_stim_freqs(config.freqs)

        self.worker_thread = QThread(self)
        self.worker = EvaluationWorker(config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self._log)
        self.worker.phase_changed.connect(self.on_phase_changed)
        self.worker.error.connect(self.on_eval_error)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.on_eval_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._cleanup_eval_worker)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_connect.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._set_inputs_enabled(False)
        self._set_summary("Benchmark running...")
        self._log("Benchmark started.")
        self.worker_thread.start()

    def stop_evaluation(self) -> None:
        worker = self.worker
        if worker is None:
            self._log("No active benchmark worker.")
            return
        self._log("Stop requested.")
        worker.request_stop()

    @pyqtSlot(object)
    def on_phase_changed(self, payload: dict[str, Any]) -> None:
        self.phase_label.setText(str(payload.get("title", "")))
        detail = str(payload.get("detail", "")).strip()
        remaining = int(payload.get("remaining_sec", 0) or 0)
        if remaining > 0:
            detail = f"{detail}  ({remaining}s)"
        self.phase_detail_label.setText(detail)
        self.stim_widget.apply_phase(payload)

    @pyqtSlot(str)
    def on_eval_error(self, message: str) -> None:
        self._log(f"Benchmark error: {message}")
        self._set_summary(f"Benchmark failed.\n\n{message}")
        self.on_phase_changed(
            {
                "mode": PHASE_ERROR,
                "title": "Benchmark error",
                "detail": message,
                "flicker": False,
                "cue_freq": None,
            }
        )

    @pyqtSlot(object)
    def on_eval_finished(self, payload: dict[str, Any]) -> None:
        if bool(payload.get("stopped", False)):
            self._log("Benchmark stopped.")
            self._set_summary("Benchmark stopped by user.")
            self.on_phase_changed(
                {
                    "mode": PHASE_STOPPED,
                    "title": "Benchmark stopped",
                    "detail": "Stopped by user.",
                    "flicker": False,
                    "cue_freq": None,
                }
            )
            return
        report = payload.get("report") or {}
        report_path = str(payload.get("report_path", ""))
        markdown_path = str(payload.get("markdown_path", ""))
        dataset_dir = str(report.get("dataset_dir", ""))
        dataset_manifest = str(report.get("dataset_manifest", ""))
        dataset_npz = str(report.get("dataset_npz", ""))
        self.last_report_payload = dict(report)
        self.last_markdown_path = Path(markdown_path) if markdown_path else None
        chosen = report.get("chosen_model")
        chosen_rank = report.get("chosen_rank")
        chosen_metrics = report.get("chosen_metrics", {})
        robust = report.get("robustness", {})
        robust_rec = report.get("robust_recommendation", {})
        robust_modes = ",".join(str(item) for item in robust.get("channel_modes", [])) if isinstance(robust, dict) else ""
        robust_seeds = ",".join(str(item) for item in robust.get("seeds", [])) if isinstance(robust, dict) else ""
        robust_auto_top = None
        if isinstance(robust, dict):
            by_mode = robust.get("by_mode", {})
            if isinstance(by_mode, dict):
                auto_payload = by_mode.get("auto", {})
                ranked_models = list(auto_payload.get("ranked_models", [])) if isinstance(auto_payload, dict) else []
                if ranked_models:
                    robust_auto_top = dict(ranked_models[0])
        summary = [
            f"Report JSON: {report_path}",
            f"Report MD: {markdown_path}",
            f"Dataset Dir: {dataset_dir}",
            f"Dataset Manifest: {dataset_manifest}",
            f"Dataset NPZ: {dataset_npz}",
            "",
            f"Chosen model: {chosen} (rank={chosen_rank})",
            f"idle_fp_per_min={chosen_metrics.get('idle_fp_per_min')}",
            f"control_recall={chosen_metrics.get('control_recall')}",
            f"switch_detect_rate={chosen_metrics.get('switch_detect_rate')}",
            f"switch_latency_s={chosen_metrics.get('switch_latency_s')}",
            f"release_latency_s={chosen_metrics.get('release_latency_s')}",
            f"detection_latency_s={chosen_metrics.get('detection_latency_s')}",
            f"itr_bpm={chosen_metrics.get('itr_bpm')}",
            f"inference_ms={chosen_metrics.get('inference_ms')}",
            "",
            f"Robustness modes={robust_modes}",
            f"Robustness seeds={robust_seeds}",
        ]
        if isinstance(robust_rec, dict) and robust_rec:
            summary.extend(
                [
                    f"Robust recommendation={robust_rec.get('model_name')}@{robust_rec.get('channel_mode')} "
                    f"(rank={robust_rec.get('rank')})",
                ]
            )
        if isinstance(robust_auto_top, dict):
            metrics_mean = dict(robust_auto_top.get("metrics_mean", {}))
            summary.extend(
                [
                    "Auto-mode robust top:",
                    f"model={robust_auto_top.get('model_name')} rank={robust_auto_top.get('rank')} "
                    f"mean_rank={robust_auto_top.get('mean_rank')}",
                    f"idle_fp_mean={metrics_mean.get('idle_fp_per_min')} "
                    f"recall_mean={metrics_mean.get('control_recall')} "
                    f"switch_detect_mean={metrics_mean.get('switch_detect_rate')} "
                    f"switch_latency_mean={metrics_mean.get('switch_latency_s')} "
                    f"release_latency_mean={metrics_mean.get('release_latency_s')}",
                ]
            )
        self._set_summary("\n".join(summary))
        self._log("Benchmark completed.")
        self.btn_export.setEnabled(True)
        self.on_phase_changed(
            {
                "mode": PHASE_STOPPED,
                "title": "Benchmark completed",
                "detail": "评测结束，可导出报告。",
                "flicker": False,
                "cue_freq": None,
            }
        )

    @pyqtSlot()
    def _cleanup_eval_worker(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_connect.setEnabled(self.connect_thread is None)
        self._set_inputs_enabled(self.connect_thread is None)

    def export_report(self) -> None:
        if self.last_report_payload is None:
            self._log("No report to export.")
            return
        suggested = f"benchmark_report_{_now_stamp()}.md"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Benchmark Markdown",
            str((DEFAULT_REPORT_DIR / suggested).resolve()),
            "Markdown (*.md)",
        )
        if not output_path:
            return
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_benchmark_markdown(self.last_report_payload), encoding="utf-8")
        self._log(f"Markdown exported: {out}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        if self.connect_thread is not None:
            self.connect_thread.quit()
            self.connect_thread.wait(3000)
        self.stim_widget.stop_clock()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP multi-model benchmark evaluation UI")
    parser.add_argument("--serial-port", type=str, default="auto")
    parser.add_argument("--board-id", type=int, default=DEFAULT_BOARD_ID)
    parser.add_argument("--freqs", type=str, default="8,10,12,15")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = ModelEvaluationWindow(
        serial_port=args.serial_port,
        board_id=args.board_id,
        freqs=parse_freqs(args.freqs),
    )
    window.dataset_dir_edit.setText(str(Path(args.dataset_dir).expanduser().resolve()))
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

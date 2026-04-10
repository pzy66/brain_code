from __future__ import annotations

import argparse
import time
import threading
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from PyQt5.QtCore import QObject, QThread, QUrl, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDesktopServices, QFont
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
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BENCHMARK_MULTI_SEED_COUNT,
    DEFAULT_BENCHMARK_CHANNEL_MODES,
    DEFAULT_CHANNEL_WEIGHT_MODE,
    DEFAULT_DECISION_TIME_MODE,
    DEFAULT_DYNAMIC_STOP_ALPHA,
    DEFAULT_DYNAMIC_STOP_ENABLED,
    DEFAULT_EXPORT_FIGURES,
    DEFAULT_GATE_POLICY,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_PROFILE_PATH,
    DEFAULT_RANKING_POLICY,
    DEFAULT_WIN_SEC_CANDIDATES,
    parse_channel_mode_list,
    parse_decision_time_mode,
    parse_gate_policy,
    parse_metric_scope,
    parse_model_list,
    parse_ranking_policy,
)
from ssvep_core.train_eval import OfflineTrainEvalConfig, run_offline_train_eval


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = THIS_DIR / "profiles"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class TrainEvalUIConfig:
    session1_manifest: Path
    session2_manifest: Optional[Path]
    output_profile_path: Path
    report_path: Path
    model_names: tuple[str, ...]
    channel_modes: tuple[str, ...]
    multi_seed_count: int
    gate_policy: str
    channel_weight_mode: Optional[str]
    metric_scope: str
    decision_time_mode: str
    export_figures: bool
    ranking_policy: str
    dynamic_stop_enabled: bool
    dynamic_stop_alpha: float
    win_candidates: tuple[float, ...]
    seed: int


class TrainEvalWorker(QObject):
    log = pyqtSignal(str)
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: TrainEvalUIConfig) -> None:
        super().__init__()
        self.config = config

    @pyqtSlot()
    def run(self) -> None:
        try:
            cfg = OfflineTrainEvalConfig(
                dataset_manifest_session1=self.config.session1_manifest,
                dataset_manifest_session2=self.config.session2_manifest,
                output_profile_path=self.config.output_profile_path,
                report_path=self.config.report_path,
                model_names=self.config.model_names,
                channel_modes=self.config.channel_modes,
                multi_seed_count=self.config.multi_seed_count,
                win_candidates=self.config.win_candidates,
                gate_policy=self.config.gate_policy,
                channel_weight_mode=self.config.channel_weight_mode,
                metric_scope=self.config.metric_scope,
                decision_time_mode=self.config.decision_time_mode,
                export_figures=bool(self.config.export_figures),
                ranking_policy=self.config.ranking_policy,
                dynamic_stop_enabled=self.config.dynamic_stop_enabled,
                dynamic_stop_alpha=self.config.dynamic_stop_alpha,
                seed=self.config.seed,
            )
            payload = run_offline_train_eval(cfg, log_fn=self.log.emit)
            self.done.emit(payload)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class TrainingEvaluationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP Training & Evaluation UI")
        self.resize(1180, 780)

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TrainEvalWorker] = None
        self._last_report_path: Optional[Path] = None
        self._last_figures_dir: Optional[Path] = None

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        form = QFormLayout()
        self.session1_edit = QLineEdit("")
        self.session2_edit = QLineEdit("")
        self.output_profile_edit = QLineEdit(str(DEFAULT_PROFILE_PATH))
        self.report_edit = QLineEdit(str(DEFAULT_REPORT_DIR / f"offline_train_eval_{_now_stamp()}.json"))
        self.models_edit = QLineEdit(",".join(DEFAULT_BENCHMARK_MODELS))
        self.channel_modes_edit = QLineEdit(",".join(DEFAULT_BENCHMARK_CHANNEL_MODES))
        self.multi_seed_spin = QSpinBox()
        self.multi_seed_spin.setRange(1, 20)
        self.multi_seed_spin.setValue(DEFAULT_BENCHMARK_MULTI_SEED_COUNT)
        self.gate_policy_edit = QLineEdit(DEFAULT_GATE_POLICY)
        self.weight_mode_edit = QLineEdit(str(DEFAULT_CHANNEL_WEIGHT_MODE))
        self.metric_scope_edit = QLineEdit(DEFAULT_METRIC_SCOPE)
        self.decision_time_mode_edit = QLineEdit(DEFAULT_DECISION_TIME_MODE)
        self.export_figures_edit = QLineEdit("1" if DEFAULT_EXPORT_FIGURES else "0")
        self.ranking_policy_edit = QLineEdit(DEFAULT_RANKING_POLICY)
        self.dynamic_stop_edit = QLineEdit("1" if DEFAULT_DYNAMIC_STOP_ENABLED else "0")
        self.dynamic_alpha_edit = QLineEdit(f"{DEFAULT_DYNAMIC_STOP_ALPHA:g}")
        self.win_candidates_edit = QLineEdit(",".join(f"{item:g}" for item in DEFAULT_WIN_SEC_CANDIDATES))
        self.seed_edit = QLineEdit("20260410")

        form.addRow("Session1 Manifest", self.session1_edit)
        form.addRow("Session2 Manifest (optional)", self.session2_edit)
        form.addRow("Output Profile", self.output_profile_edit)
        form.addRow("Report JSON", self.report_edit)
        form.addRow("Models", self.models_edit)
        form.addRow("Channel Modes", self.channel_modes_edit)
        form.addRow("Multi Seed Count", self.multi_seed_spin)
        form.addRow("Gate Policy", self.gate_policy_edit)
        form.addRow("Channel Weight Mode", self.weight_mode_edit)
        form.addRow("Metric Scope", self.metric_scope_edit)
        form.addRow("Decision Time Mode", self.decision_time_mode_edit)
        form.addRow("Export Figures (1/0)", self.export_figures_edit)
        form.addRow("Ranking Policy", self.ranking_policy_edit)
        form.addRow("Dynamic Stop (1/0)", self.dynamic_stop_edit)
        form.addRow("Dynamic Alpha", self.dynamic_alpha_edit)
        form.addRow("Win Candidates", self.win_candidates_edit)
        form.addRow("Seed", self.seed_edit)
        layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_pick_s1 = QPushButton("选择Session1")
        self.btn_pick_s2 = QPushButton("选择Session2")
        self.btn_pick_profile = QPushButton("选择Profile输出")
        self.btn_pick_report = QPushButton("选择报告输出")
        self.btn_run = QPushButton("开始训练评测")
        self.btn_open_report_dir = QPushButton("打开报告目录")
        self.btn_open_figures_dir = QPushButton("打开图表目录")
        self.btn_open_report_dir.setEnabled(False)
        self.btn_open_figures_dir.setEnabled(False)
        row.addWidget(self.btn_pick_s1)
        row.addWidget(self.btn_pick_s2)
        row.addWidget(self.btn_pick_profile)
        row.addWidget(self.btn_pick_report)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_open_report_dir)
        row.addWidget(self.btn_open_figures_dir)
        layout.addLayout(row)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-size:16px; font-weight:600;")
        layout.addWidget(self.status_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, 1)

        self.btn_pick_s1.clicked.connect(self._pick_session1)
        self.btn_pick_s2.clicked.connect(self._pick_session2)
        self.btn_pick_profile.clicked.connect(self._pick_profile)
        self.btn_pick_report.clicked.connect(self._pick_report)
        self.btn_run.clicked.connect(self._start_run)
        self.btn_open_report_dir.clicked.connect(self._open_report_dir)
        self.btn_open_figures_dir.clicked.connect(self._open_figures_dir)

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {text}")

    def _pick_json(self, target: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, str(Path(target.text().strip()).parent), "JSON (*.json)")
        if path:
            target.setText(path)

    def _pick_session1(self) -> None:
        self._pick_json(self.session1_edit, "Select session1 manifest")

    def _pick_session2(self) -> None:
        self._pick_json(self.session2_edit, "Select session2 manifest")

    def _pick_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Output profile path", self.output_profile_edit.text().strip(), "JSON (*.json)")
        if path:
            self.output_profile_edit.setText(path)

    def _pick_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Report JSON path", self.report_edit.text().strip(), "JSON (*.json)")
        if path:
            self.report_edit.setText(path)

    def _open_path(self, path: Optional[Path]) -> None:
        if path is None:
            self._log("Path is not available yet.")
            return
        target = Path(path).expanduser().resolve()
        if not target.exists():
            self._log(f"Path not found: {target}")
            return
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _open_report_dir(self) -> None:
        if self._last_report_path is not None:
            self._open_path(self._last_report_path.parent)
            return
        self._open_path(Path(self.report_edit.text().strip()).expanduser().resolve().parent)

    def _open_figures_dir(self) -> None:
        self._open_path(self._last_figures_dir)

    def _read_config(self) -> TrainEvalUIConfig:
        s1 = Path(self.session1_edit.text().strip()).expanduser().resolve()
        if not s1.exists():
            raise FileNotFoundError(f"session1 manifest not found: {s1}")
        s2_raw = self.session2_edit.text().strip()
        s2 = Path(s2_raw).expanduser().resolve() if s2_raw else None
        if s2 is not None and not s2.exists():
            raise FileNotFoundError(f"session2 manifest not found: {s2}")
        output_profile = Path(self.output_profile_edit.text().strip()).expanduser().resolve()
        report_path = Path(self.report_edit.text().strip()).expanduser().resolve()
        model_names = tuple(parse_model_list(self.models_edit.text().strip()))
        channel_modes = tuple(parse_channel_mode_list(self.channel_modes_edit.text().strip()))
        gate_policy = parse_gate_policy(self.gate_policy_edit.text().strip())
        metric_scope = parse_metric_scope(self.metric_scope_edit.text().strip())
        decision_time_mode = parse_decision_time_mode(self.decision_time_mode_edit.text().strip())
        ranking_policy = parse_ranking_policy(self.ranking_policy_edit.text().strip())
        export_figures = bool(int(self.export_figures_edit.text().strip() or "1"))
        weight_mode_raw = self.weight_mode_edit.text().strip()
        channel_weight_mode = None if not weight_mode_raw else weight_mode_raw
        dynamic_stop_enabled = bool(int(self.dynamic_stop_edit.text().strip() or "1"))
        dynamic_stop_alpha = float(self.dynamic_alpha_edit.text().strip())
        win_candidates = tuple(
            float(item.strip())
            for item in self.win_candidates_edit.text().split(",")
            if item.strip()
        )
        seed = int(self.seed_edit.text().strip())
        return TrainEvalUIConfig(
            session1_manifest=s1,
            session2_manifest=s2,
            output_profile_path=output_profile,
            report_path=report_path,
            model_names=model_names,
            channel_modes=channel_modes,
            multi_seed_count=int(self.multi_seed_spin.value()),
            gate_policy=gate_policy,
            channel_weight_mode=channel_weight_mode,
            metric_scope=metric_scope,
            decision_time_mode=decision_time_mode,
            export_figures=export_figures,
            ranking_policy=ranking_policy,
            dynamic_stop_enabled=dynamic_stop_enabled,
            dynamic_stop_alpha=dynamic_stop_alpha,
            win_candidates=win_candidates,
            seed=seed,
        )

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        if running:
            self.btn_open_report_dir.setEnabled(False)
            self.btn_open_figures_dir.setEnabled(False)

    def _start_run(self) -> None:
        if self.worker_thread is not None:
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"Config error: {exc}")
            return
        worker = TrainEvalWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.done.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.started.connect(worker.run)
        self.worker = worker
        self.worker_thread = thread
        self._set_running(True)
        self.status_label.setText("Running training/evaluation...")
        thread.start()

    def _on_done(self, payload: dict[str, Any]) -> None:
        report_path = payload.get("report_path") or self.report_edit.text().strip()
        self._last_report_path = Path(str(report_path)).expanduser().resolve()
        figures_payload = dict(payload.get("figures", {}))
        figures_dir_raw = figures_payload.get("dir")
        self._last_figures_dir = (
            Path(str(figures_dir_raw)).expanduser().resolve()
            if figures_dir_raw
            else None
        )
        self.btn_open_report_dir.setEnabled(True)
        self.btn_open_figures_dir.setEnabled(self._last_figures_dir is not None)
        async_metrics = dict(payload.get("chosen_async_metrics", {}))
        metrics_4 = dict(payload.get("chosen_metrics_4class", {}))
        summary = (
            "Summary: "
            f"idle_fp={float(async_metrics.get('idle_fp_per_min', float('inf'))):.4f}, "
            f"recall={float(async_metrics.get('control_recall', 0.0)):.4f}, "
            f"switch={float(async_metrics.get('switch_latency_s', float('inf'))):.4f}s, "
            f"release={float(async_metrics.get('release_latency_s', float('inf'))):.4f}s, "
            f"acc4={float(metrics_4.get('acc', 0.0)):.4f}, "
            f"macroF1_4={float(metrics_4.get('macro_f1', 0.0)):.4f}, "
            f"itr4={float(metrics_4.get('itr_bpm', 0.0)):.4f}"
        )
        self._log(summary)
        if bool(payload.get("profile_saved", False)):
            self.status_label.setText("Training/evaluation done")
            self._log(
                f"Done. chosen_model={payload.get('chosen_model')} report={payload.get('mode')} "
                f"profile={payload.get('chosen_profile_path')} report_json={self._last_report_path}"
            )
            return
        self.status_label.setText("Done (no accepted model)")
        self._log(
            "Done without profile save: no model met acceptance. "
            f"recommended={payload.get('recommended_model')} report={payload.get('mode')}"
        )

    def _on_error(self, text: str) -> None:
        self.status_label.setText("Training/evaluation failed")
        self._log(text)

    def _on_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_running(False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP training/evaluation UI or CLI")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="session1 manifest path")
    parser.add_argument("--dataset-manifest-session2", type=Path, default=None, help="session2 manifest path")
    parser.add_argument("--output-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_DIR / f"offline_train_eval_{_now_stamp()}.json")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_BENCHMARK_MODELS))
    parser.add_argument("--channel-modes", type=str, default=",".join(DEFAULT_BENCHMARK_CHANNEL_MODES))
    parser.add_argument("--multi-seed-count", type=int, default=DEFAULT_BENCHMARK_MULTI_SEED_COUNT)
    parser.add_argument("--gate-policy", type=str, default=DEFAULT_GATE_POLICY)
    parser.add_argument("--channel-weight-mode", type=str, default=str(DEFAULT_CHANNEL_WEIGHT_MODE))
    parser.add_argument("--metric-scope", type=str, default=DEFAULT_METRIC_SCOPE)
    parser.add_argument("--decision-time-mode", type=str, default=DEFAULT_DECISION_TIME_MODE)
    parser.add_argument("--export-figures", type=int, default=1 if DEFAULT_EXPORT_FIGURES else 0)
    parser.add_argument("--ranking-policy", type=str, default=DEFAULT_RANKING_POLICY)
    parser.add_argument("--dynamic-stop-enabled", type=int, default=1)
    parser.add_argument("--dynamic-stop-alpha", type=float, default=DEFAULT_DYNAMIC_STOP_ALPHA)
    parser.add_argument("--win-candidates", type=str, default=",".join(f"{item:g}" for item in DEFAULT_WIN_SEC_CANDIDATES))
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.headless):
        if args.dataset_manifest is None:
            raise ValueError("--dataset-manifest is required in --headless mode")
        config = OfflineTrainEvalConfig(
            dataset_manifest_session1=Path(args.dataset_manifest).expanduser().resolve(),
            dataset_manifest_session2=(
                None
                if args.dataset_manifest_session2 is None
                else Path(args.dataset_manifest_session2).expanduser().resolve()
            ),
            output_profile_path=Path(args.output_profile).expanduser().resolve(),
            report_path=Path(args.report_path).expanduser().resolve(),
            model_names=tuple(parse_model_list(args.models)),
            channel_modes=tuple(parse_channel_mode_list(args.channel_modes)),
            multi_seed_count=int(args.multi_seed_count),
            gate_policy=parse_gate_policy(args.gate_policy),
            channel_weight_mode=(None if str(args.channel_weight_mode).strip() == "" else str(args.channel_weight_mode).strip()),
            metric_scope=parse_metric_scope(args.metric_scope),
            decision_time_mode=parse_decision_time_mode(args.decision_time_mode),
            export_figures=bool(int(args.export_figures)),
            ranking_policy=parse_ranking_policy(args.ranking_policy),
            dynamic_stop_enabled=bool(int(args.dynamic_stop_enabled)),
            dynamic_stop_alpha=float(args.dynamic_stop_alpha),
            win_candidates=tuple(float(item.strip()) for item in str(args.win_candidates).split(",") if item.strip()),
            seed=int(args.seed),
        )
        run_offline_train_eval(config, log_fn=lambda text: print(text, flush=True))
        return 0

    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = TrainingEvaluationWindow()
    if args.dataset_manifest is not None:
        window.session1_edit.setText(str(Path(args.dataset_manifest).expanduser().resolve()))
    if args.dataset_manifest_session2 is not None:
        window.session2_edit.setText(str(Path(args.dataset_manifest_session2).expanduser().resolve()))
    window.output_profile_edit.setText(str(Path(args.output_profile).expanduser().resolve()))
    window.report_edit.setText(str(Path(args.report_path).expanduser().resolve()))
    window.models_edit.setText(str(args.models))
    window.channel_modes_edit.setText(str(args.channel_modes))
    window.multi_seed_spin.setValue(int(args.multi_seed_count))
    window.gate_policy_edit.setText(str(args.gate_policy))
    window.weight_mode_edit.setText(str(args.channel_weight_mode))
    window.metric_scope_edit.setText(str(args.metric_scope))
    window.decision_time_mode_edit.setText(str(args.decision_time_mode))
    window.export_figures_edit.setText("1" if bool(int(args.export_figures)) else "0")
    window.ranking_policy_edit.setText(str(args.ranking_policy))
    window.dynamic_stop_edit.setText("1" if bool(int(args.dynamic_stop_enabled)) else "0")
    window.dynamic_alpha_edit.setText(f"{float(args.dynamic_stop_alpha):g}")
    window.win_candidates_edit.setText(str(args.win_candidates))
    window.seed_edit.setText(str(int(args.seed)))
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from PyQt5.QtCore import QObject, QThread, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from async_fbcca_idle_standalone import (
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_BENCHMARK_CHANNEL_MODES,
    DEFAULT_BENCHMARK_MULTI_SEED_COUNT,
    DEFAULT_CHANNEL_WEIGHT_MODE,
    DEFAULT_COMPUTE_BACKEND_NAME,
    DEFAULT_DATA_POLICY,
    DEFAULT_DYNAMIC_STOP_ALPHA,
    DEFAULT_DYNAMIC_STOP_ENABLED,
    DEFAULT_EXPORT_FIGURES,
    DEFAULT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_JOINT_WEIGHT_ITERS,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_PAPER_DECISION_TIME_MODE,
    DEFAULT_PROFILE_PATH,
    DEFAULT_RANKING_POLICY,
    DEFAULT_SPATIAL_FILTER_MODE,
    DEFAULT_SUBBAND_WEIGHT_MODE,
    DEFAULT_SPATIAL_RANK_CANDIDATES,
    DEFAULT_SPATIAL_SOURCE_MODEL,
    DEFAULT_WIN_SEC_CANDIDATES,
    parse_channel_mode_list,
    parse_compute_backend_name,
    parse_data_policy,
    parse_decision_time_mode,
    parse_gate_policy,
    parse_gpu_cache_policy,
    parse_gpu_precision,
    parse_metric_scope,
    parse_model_list,
    normalize_model_name,
    parse_ranking_policy,
    parse_spatial_filter_mode,
    parse_spatial_rank_candidates,
    parse_spatial_source_model,
    parse_subband_weight_mode,
)
from ssvep_core.dataset import discover_collection_manifests
from ssvep_core.train_eval import (
    DEFAULT_EVALUATION_MODE,
    DEFAULT_FORCE_INCLUDE_MODELS,
    DEFAULT_FBCCA_WEIGHT_CV_FOLDS,
    DEFAULT_PROGRESS_HEARTBEAT_SEC,
    DEFAULT_QUICK_SCREEN_TOP_K,
    DEFAULT_TRAIN_EVAL_TASK,
    OfflineTrainEvalConfig,
    run_offline_train_eval,
)
from ssvep_core.registry import ModelRegistry
from ssvep_server_train_client import (
    DEFAULT_REMOTE_COMPUTE_BACKEND,
    DEFAULT_REMOTE_GPU_CACHE_POLICY,
    DEFAULT_REMOTE_GPU_DEVICE,
    DEFAULT_REMOTE_GPU_PRECISION,
    DEFAULT_REMOTE_GPU_WARMUP,
    DEFAULT_REMOTE_MULTI_SEED_COUNT,
    DEFAULT_REMOTE_WIN_CANDIDATES,
    ServerConfig,
    SSHClient,
    _find_dataset_by_manifest,
    build_train_command,
    download_results,
    now_run_id,
    preflight_cuda_or_fail,
    read_remote_status,
    start_remote_task,
    upload_dataset,
)

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = THIS_DIR / "profiles"
DEFAULT_DATASET_ROOT = THIS_DIR / "profiles" / "datasets"
DEFAULT_REPORT_ROOT = THIS_DIR / "profiles" / "reports" / "train_eval"
TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND = "cuda"
TRAIN_EVAL_DEFAULT_GPU_PRECISION = "float32"
CORE_COMPARE_MODELS = ("tdca", "trca_r", "etrca_r", "fbcca")
SIMPLE_MODE_MODELS = CORE_COMPARE_MODELS
BASELINE_COMPARE_MODELS = tuple(ModelRegistry.list_models(task="benchmark"))
QUICK_MODE_MODELS = (
    "fbcca_fixed_all8",
    "fbcca_cw_all8",
    "fbcca_sw_all8",
    "fbcca_cw_sw_all8",
)
SIMPLE_MODE_CHANNEL_MODES = tuple(str(name) for name in DEFAULT_BENCHMARK_CHANNEL_MODES)
SIMPLE_MODE_MULTI_SEED_COUNT = int(DEFAULT_BENCHMARK_MULTI_SEED_COUNT)
QUICK_MODE_CHANNEL_MODES = ("all8",)
QUICK_MODE_MULTI_SEED_COUNT = 1
QUICK_MODE_WIN_CANDIDATES = (1.5,)
QUICK_MODE_JOINT_WEIGHT_ITERS = 1
QUICK_MODE_WEIGHT_CV_FOLDS = 2
QUICK_MODE_QUICK_SCREEN_TOP_K = 2
QUICK_MODE_FORCE_INCLUDE_MODELS = ("fbcca_fixed_all8", "fbcca_cw_sw_all8")
QUICK_MODE_CHANNEL_WEIGHT_MODE = "fbcca_diag"
QUICK_MODE_SUBBAND_WEIGHT_MODE = "chen_ab_subject"
QUICK_MODE_SPATIAL_FILTER_MODE = "none"
MODEL_COMPARE_MODELS = CORE_COMPARE_MODELS
MODEL_COMPARE_CHANNEL_MODES = ("all8",)
MODEL_COMPARE_MULTI_SEED_COUNT = 5
MODEL_COMPARE_WIN_CANDIDATES = (2.5, 3.0, 3.5, 4.0)
MODEL_COMPARE_JOINT_WEIGHT_ITERS = 1
MODEL_COMPARE_WEIGHT_CV_FOLDS = 2
MODEL_COMPARE_QUICK_SCREEN_TOP_K = len(MODEL_COMPARE_MODELS)
MODEL_COMPARE_FORCE_INCLUDE_MODELS = MODEL_COMPARE_MODELS
MODEL_COMPARE_CHANNEL_WEIGHT_MODE = "none"
MODEL_COMPARE_SUBBAND_WEIGHT_MODE = "chen_fixed"
MODEL_COMPARE_SPATIAL_FILTER_MODE = "none"
WEIGHTED_COMPARE_MODELS = tuple(
    dict.fromkeys(
        (
            "legacy_fbcca_202603",
            "fbcca_fixed_all8",
            "fbcca_cw_all8",
            "fbcca_sw_all8",
            "fbcca_cw_sw_all8",
        )
        + tuple(
            name
            for name in ModelRegistry.list_models(task="benchmark")
            if str(name) not in {"legacy_fbcca_202603", "fbcca"}
        )
    )
)
WEIGHTED_COMPARE_CHANNEL_MODES = ("all8",)
WEIGHTED_COMPARE_MULTI_SEED_COUNT = 5
WEIGHTED_COMPARE_WIN_CANDIDATES = (2.5, 3.0, 3.5, 4.0)
WEIGHTED_COMPARE_JOINT_WEIGHT_ITERS = 1
WEIGHTED_COMPARE_WEIGHT_CV_FOLDS = 2
WEIGHTED_COMPARE_QUICK_SCREEN_TOP_K = len(WEIGHTED_COMPARE_MODELS)
WEIGHTED_COMPARE_FORCE_INCLUDE_MODELS = WEIGHTED_COMPARE_MODELS
WEIGHTED_COMPARE_CHANNEL_WEIGHT_MODE = "none"
WEIGHTED_COMPARE_SUBBAND_WEIGHT_MODE = "chen_fixed"
WEIGHTED_COMPARE_SPATIAL_FILTER_MODE = "none"
DEFAULT_SERVER_HOST = "10.72.128.221"
DEFAULT_SERVER_PORT = 22
DEFAULT_SERVER_USERNAME = "zhangkexin"
DEFAULT_REMOTE_POLL_INTERVAL_MS = 5000


def _parse_task(raw: str) -> str:
    value = str(raw or DEFAULT_TRAIN_EVAL_TASK).strip().lower()
    aliases = {
        "fbcca": "fbcca-weights",
        "fbcca_weight": "fbcca-weights",
        "fbcca_weights": "fbcca-weights",
        "weights": "fbcca-weights",
        "compare": "model-compare",
        "model_compare": "model-compare",
        "models": "model-compare",
        "weighted_compare": "fbcca-weighted-compare",
        "fbcca_weighted_compare": "fbcca-weighted-compare",
        "weights_compare": "fbcca-weighted-compare",
    }
    value = aliases.get(value, value)
    if value not in {"fbcca-weights", "model-compare", "fbcca-weighted-compare"}:
        raise ValueError(f"unsupported train-eval task: {raw}")
    return value


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _bool_from_text(value: Any, *, default: bool = False) -> bool:
    raw = str(value if value is not None else "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _parse_manifest_csv(raw: str) -> tuple[Path, ...]:
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    return tuple(Path(item).expanduser().resolve() for item in items)


def _apply_quick_mode_args(args: argparse.Namespace) -> None:
    args.task = "fbcca-weights"
    args.models = ",".join(QUICK_MODE_MODELS)
    args.channel_modes = ",".join(QUICK_MODE_CHANNEL_MODES)
    args.multi_seed_count = int(QUICK_MODE_MULTI_SEED_COUNT)
    args.channel_weight_mode = str(QUICK_MODE_CHANNEL_WEIGHT_MODE)
    args.subband_weight_mode = str(QUICK_MODE_SUBBAND_WEIGHT_MODE)
    args.spatial_filter_mode = str(QUICK_MODE_SPATIAL_FILTER_MODE)
    args.joint_weight_iters = int(QUICK_MODE_JOINT_WEIGHT_ITERS)
    args.weight_cv_folds = int(QUICK_MODE_WEIGHT_CV_FOLDS)
    args.win_candidates = ",".join(f"{float(value):g}" for value in QUICK_MODE_WIN_CANDIDATES)
    args.evaluation_mode = str(DEFAULT_EVALUATION_MODE)
    args.quick_screen_top_k = int(QUICK_MODE_QUICK_SCREEN_TOP_K)
    args.force_include_models = ",".join(QUICK_MODE_FORCE_INCLUDE_MODELS)
    args.compute_backend = str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND)
    args.gpu_precision = str(TRAIN_EVAL_DEFAULT_GPU_PRECISION)


def _apply_model_compare_args(args: argparse.Namespace) -> None:
    args.task = "model-compare"
    args.models = ",".join(MODEL_COMPARE_MODELS)
    args.channel_modes = ",".join(MODEL_COMPARE_CHANNEL_MODES)
    args.multi_seed_count = int(MODEL_COMPARE_MULTI_SEED_COUNT)
    args.channel_weight_mode = str(MODEL_COMPARE_CHANNEL_WEIGHT_MODE)
    args.subband_weight_mode = str(MODEL_COMPARE_SUBBAND_WEIGHT_MODE)
    args.spatial_filter_mode = str(MODEL_COMPARE_SPATIAL_FILTER_MODE)
    args.joint_weight_iters = int(MODEL_COMPARE_JOINT_WEIGHT_ITERS)
    args.weight_cv_folds = int(MODEL_COMPARE_WEIGHT_CV_FOLDS)
    args.win_candidates = ",".join(f"{float(value):g}" for value in MODEL_COMPARE_WIN_CANDIDATES)
    args.evaluation_mode = str(DEFAULT_EVALUATION_MODE)
    args.quick_screen_top_k = int(MODEL_COMPARE_QUICK_SCREEN_TOP_K)
    args.force_include_models = ",".join(MODEL_COMPARE_FORCE_INCLUDE_MODELS)
    args.compute_backend = str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND)
    args.gpu_precision = str(TRAIN_EVAL_DEFAULT_GPU_PRECISION)


def _apply_weighted_compare_args(args: argparse.Namespace) -> None:
    args.task = "fbcca-weighted-compare"
    args.models = ",".join(WEIGHTED_COMPARE_MODELS)
    args.channel_modes = ",".join(WEIGHTED_COMPARE_CHANNEL_MODES)
    args.multi_seed_count = int(WEIGHTED_COMPARE_MULTI_SEED_COUNT)
    args.channel_weight_mode = str(WEIGHTED_COMPARE_CHANNEL_WEIGHT_MODE)
    args.subband_weight_mode = str(WEIGHTED_COMPARE_SUBBAND_WEIGHT_MODE)
    args.spatial_filter_mode = str(WEIGHTED_COMPARE_SPATIAL_FILTER_MODE)
    args.joint_weight_iters = int(WEIGHTED_COMPARE_JOINT_WEIGHT_ITERS)
    args.weight_cv_folds = int(WEIGHTED_COMPARE_WEIGHT_CV_FOLDS)
    args.win_candidates = ",".join(f"{float(value):g}" for value in WEIGHTED_COMPARE_WIN_CANDIDATES)
    args.evaluation_mode = str(DEFAULT_EVALUATION_MODE)
    args.quick_screen_top_k = int(WEIGHTED_COMPARE_QUICK_SCREEN_TOP_K)
    args.force_include_models = ",".join(WEIGHTED_COMPARE_FORCE_INCLUDE_MODELS)
    args.compute_backend = str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND)
    args.gpu_precision = str(TRAIN_EVAL_DEFAULT_GPU_PRECISION)


@dataclass(frozen=True)
class TrainEvalUIConfig:
    session1_manifest: Path
    session2_manifest: Optional[Path]
    dataset_manifests: tuple[Path, ...]
    dataset_root: Path
    dataset_selection_snapshot: dict[str, Any]
    quality_min_sample_ratio: float
    quality_max_retry_count: int
    strict_protocol_consistency: bool
    strict_subject_consistency: bool
    output_profile_path: Path
    report_path: Path
    report_root_dir: Path
    organize_report_dir: bool
    model_names: tuple[str, ...]
    channel_modes: tuple[str, ...]
    multi_seed_count: int
    gate_policy: str
    channel_weight_mode: Optional[str]
    subband_weight_mode: Optional[str]
    spatial_filter_mode: Optional[str]
    spatial_rank_candidates: tuple[int, ...]
    joint_weight_iters: int
    weight_cv_folds: int
    spatial_source_model: str
    metric_scope: str
    decision_time_mode: str
    async_decision_time_mode: str
    data_policy: str
    export_figures: bool
    ranking_policy: str
    dynamic_stop_enabled: bool
    dynamic_stop_alpha: float
    win_candidates: tuple[float, ...]
    seed: int
    evaluation_mode: str
    quick_screen_top_k: int
    force_include_models: tuple[str, ...]
    progress_heartbeat_sec: float
    compute_backend: str
    gpu_device: int
    gpu_precision: str
    gpu_warmup: bool
    gpu_cache_policy: str
    task: str


class TrainEvalWorker(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(object)
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
                dataset_manifests=self.config.dataset_manifests,
                output_profile_path=self.config.output_profile_path,
                report_path=self.config.report_path,
                report_root_dir=self.config.report_root_dir,
                organize_report_dir=bool(self.config.organize_report_dir),
                dataset_selection_snapshot=dict(self.config.dataset_selection_snapshot),
                quality_min_sample_ratio=float(self.config.quality_min_sample_ratio),
                quality_max_retry_count=int(self.config.quality_max_retry_count),
                strict_protocol_consistency=bool(self.config.strict_protocol_consistency),
                strict_subject_consistency=bool(self.config.strict_subject_consistency),
                model_names=self.config.model_names,
                channel_modes=self.config.channel_modes,
                multi_seed_count=self.config.multi_seed_count,
                win_candidates=self.config.win_candidates,
                gate_policy=self.config.gate_policy,
                channel_weight_mode=self.config.channel_weight_mode,
                subband_weight_mode=self.config.subband_weight_mode,
                spatial_filter_mode=self.config.spatial_filter_mode,
                spatial_rank_candidates=self.config.spatial_rank_candidates,
                joint_weight_iters=self.config.joint_weight_iters,
                weight_cv_folds=self.config.weight_cv_folds,
                spatial_source_model=self.config.spatial_source_model,
                metric_scope=self.config.metric_scope,
                decision_time_mode=self.config.decision_time_mode,
                async_decision_time_mode=self.config.async_decision_time_mode,
                data_policy=self.config.data_policy,
                export_figures=bool(self.config.export_figures),
                ranking_policy=self.config.ranking_policy,
                dynamic_stop_enabled=self.config.dynamic_stop_enabled,
                dynamic_stop_alpha=self.config.dynamic_stop_alpha,
                seed=self.config.seed,
                evaluation_mode=self.config.evaluation_mode,
                quick_screen_top_k=self.config.quick_screen_top_k,
                force_include_models=self.config.force_include_models,
                progress_heartbeat_sec=self.config.progress_heartbeat_sec,
                compute_backend=self.config.compute_backend,
                gpu_device=self.config.gpu_device,
                gpu_precision=self.config.gpu_precision,
                gpu_warmup=bool(self.config.gpu_warmup),
                gpu_cache_policy=self.config.gpu_cache_policy,
                task=str(self.config.task),
            )
            self.done.emit(run_offline_train_eval(cfg, log_fn=self.log.emit, progress_fn=self.progress.emit))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class TrainingEvaluationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP 训练评测")
        self.resize(1280, 860)
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TrainEvalWorker] = None
        self._last_report_path: Optional[Path] = None
        self._last_figures_dir: Optional[Path] = None
        self._dataset_scan_rows: list[dict[str, Any]] = []
        self._simple_mode_variant = "standard"
        self._evaluation_mode = str(DEFAULT_EVALUATION_MODE)
        self._quick_screen_top_k = int(DEFAULT_QUICK_SCREEN_TOP_K)
        self._force_include_models = tuple(str(name) for name in DEFAULT_FORCE_INCLUDE_MODELS)
        self._progress_heartbeat_sec = float(DEFAULT_PROGRESS_HEARTBEAT_SEC)
        self._compute_backend = str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND)
        self._gpu_device = int(DEFAULT_GPU_DEVICE_ID)
        self._gpu_precision = str(TRAIN_EVAL_DEFAULT_GPU_PRECISION)
        self._gpu_warmup = True
        self._gpu_cache_policy = str(DEFAULT_GPU_CACHE_MODE)
        self._task = str(DEFAULT_TRAIN_EVAL_TASK)
        self._remote_record: Optional[dict[str, Any]] = None
        self._remote_status_timer = None

        root = QWidget(self)
        self.setCentralWidget(root)
        self._layout = QVBoxLayout(root)
        self._build_ui()

    def _build_ui(self) -> None:
        quick_row = QHBoxLayout()
        self.simple_mode_check = QCheckBox("Simple Mode (Recommended)")
        self.keep_baseline_group_check = QCheckBox("Keep baseline model group")
        self.keep_baseline_group_check.setChecked(True)
        self.simple_mode_check.setChecked(True)
        self.btn_quick_run = QPushButton("Run FBCCA Weight Training (Quick)")
        self.btn_toggle_advanced = QPushButton("鏄剧ず楂樼骇璁剧疆")
        self.simple_mode_check.setText("Simple Mode (Recommended)")
        self.btn_quick_run.setText("FBCCA鏉冮噸璁粌锛堝揩閫燂級")
        self.btn_weighted_compare_run = QPushButton("Train weights + compare all models (Recommended)")
        self.btn_model_compare_run = QPushButton("Generate full model comparison report")
        self.btn_toggle_advanced.setText("鏄剧ず楂樼骇璁剧疆")
        self.remote_mode_check = QCheckBox("Server Remote (default)")
        self.remote_mode_check.setChecked(True)
        self.allow_local_mode_check = QCheckBox("Enable local fallback")
        self.allow_local_mode_check.setChecked(False)
        quick_row.addWidget(self.simple_mode_check)
        quick_row.addWidget(self.keep_baseline_group_check)
        quick_row.addWidget(self.remote_mode_check)
        quick_row.addWidget(self.allow_local_mode_check)
        quick_row.addWidget(self.btn_weighted_compare_run)
        quick_row.addWidget(self.btn_quick_run)
        quick_row.addWidget(self.btn_model_compare_run)
        quick_row.addWidget(self.btn_toggle_advanced)
        quick_row.addStretch(1)
        self._layout.addLayout(quick_row)

        form = QFormLayout()
        self._form_layout = form
        self.dataset_root_edit = QLineEdit(str(DEFAULT_DATASET_ROOT))
        self.session1_edit = QLineEdit("")
        self.session2_edit = QLineEdit("")
        self.output_profile_edit = QLineEdit(str(DEFAULT_PROFILE_PATH))
        self.report_edit = QLineEdit(str(DEFAULT_REPORT_DIR / f"offline_train_eval_{_now_stamp()}.json"))
        self.report_root_edit = QLineEdit(str(DEFAULT_REPORT_ROOT))
        self.organize_report_edit = QLineEdit("1")
        self.quality_min_ratio_edit = QLineEdit("0.90")
        self.quality_max_retry_spin = QSpinBox()
        self.quality_max_retry_spin.setRange(0, 20)
        self.quality_max_retry_spin.setValue(3)
        self.strict_protocol_edit = QLineEdit("1")
        self.strict_subject_edit = QLineEdit("1")
        self.models_edit = QLineEdit(",".join(ModelRegistry.list_models(task="benchmark")))
        self.channel_modes_edit = QLineEdit(",".join(DEFAULT_BENCHMARK_CHANNEL_MODES))
        self.multi_seed_spin = QSpinBox()
        self.multi_seed_spin.setRange(1, 20)
        self.multi_seed_spin.setValue(DEFAULT_BENCHMARK_MULTI_SEED_COUNT)
        self.gate_policy_edit = QLineEdit(DEFAULT_GATE_POLICY)
        self.weight_mode_edit = QLineEdit(str(DEFAULT_CHANNEL_WEIGHT_MODE))
        self.subband_weight_mode_edit = QLineEdit(str(DEFAULT_SUBBAND_WEIGHT_MODE))
        self.spatial_mode_edit = QLineEdit(str(DEFAULT_SPATIAL_FILTER_MODE))
        self.spatial_ranks_edit = QLineEdit(",".join(str(v) for v in DEFAULT_SPATIAL_RANK_CANDIDATES))
        self.joint_iters_edit = QLineEdit(str(int(DEFAULT_JOINT_WEIGHT_ITERS)))
        self.weight_cv_folds_edit = QLineEdit(str(int(DEFAULT_FBCCA_WEIGHT_CV_FOLDS)))
        self.spatial_source_edit = QLineEdit(str(DEFAULT_SPATIAL_SOURCE_MODEL))
        self.metric_scope_edit = QLineEdit(DEFAULT_METRIC_SCOPE)
        self.decision_time_mode_edit = QLineEdit(DEFAULT_PAPER_DECISION_TIME_MODE)
        self.async_decision_time_mode_edit = QLineEdit(DEFAULT_ASYNC_DECISION_TIME_MODE)
        self.data_policy_edit = QLineEdit(DEFAULT_DATA_POLICY)
        self.export_figures_edit = QLineEdit("1" if DEFAULT_EXPORT_FIGURES else "0")
        self.ranking_policy_edit = QLineEdit(DEFAULT_RANKING_POLICY)
        self.dynamic_stop_edit = QLineEdit("1" if DEFAULT_DYNAMIC_STOP_ENABLED else "0")
        self.dynamic_alpha_edit = QLineEdit(f"{DEFAULT_DYNAMIC_STOP_ALPHA:g}")
        self.win_candidates_edit = QLineEdit(",".join(f"{item:g}" for item in DEFAULT_WIN_SEC_CANDIDATES))
        self.seed_edit = QLineEdit("20260410")
        self.compute_backend_combo = QComboBox()
        self.compute_backend_combo.addItems(["cuda", "auto", "cpu"])
        self.compute_backend_combo.setCurrentText(str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND))
        self.gpu_device_edit = QLineEdit(str(DEFAULT_GPU_DEVICE_ID))
        self.gpu_precision_combo = QComboBox()
        self.gpu_precision_combo.addItems(["float32", "float64"])
        self.gpu_precision_combo.setCurrentText(str(TRAIN_EVAL_DEFAULT_GPU_PRECISION))
        self.gpu_warmup_edit = QLineEdit("1")
        self.gpu_cache_combo = QComboBox()
        self.gpu_cache_combo.addItems(["windows", "full"])
        self.gpu_cache_combo.setCurrentText(str(DEFAULT_GPU_CACHE_MODE))
        self.server_host_edit = QLineEdit(DEFAULT_SERVER_HOST)
        self.server_port_edit = QLineEdit(str(DEFAULT_SERVER_PORT))
        self.server_username_edit = QLineEdit(DEFAULT_SERVER_USERNAME)
        self.server_password_edit = QLineEdit("")
        self.server_password_edit.setEchoMode(QLineEdit.Password)

        form.addRow("鏁版嵁闆嗘牴鐩綍", self.dataset_root_edit)
        form.addRow("Session1 Manifest (fallback)", self.session1_edit)
        form.addRow("Session2 娓呭崟锛堝彲閫夛級", self.session2_edit)
        form.addRow("杈撳嚭 Profile", self.output_profile_edit)
        form.addRow("鎶ュ憡 JSON", self.report_edit)
        form.addRow("Report Root Dir", self.report_root_edit)
        form.addRow("Organize Report Dir (1/0)", self.organize_report_edit)
        form.addRow("Quality Min Sample Ratio", self.quality_min_ratio_edit)
        form.addRow("Quality Max Retry Count", self.quality_max_retry_spin)
        form.addRow("Strict Protocol Consistency (1/0)", self.strict_protocol_edit)
        form.addRow("Strict Subject Consistency (1/0)", self.strict_subject_edit)
        form.addRow("妯″瀷鍒楄〃", self.models_edit)
        form.addRow("閫氶亾妯″紡", self.channel_modes_edit)
        form.addRow("Multi-Seed Count", self.multi_seed_spin)
        form.addRow("闂ㄦ帶绛栫暐", self.gate_policy_edit)
        form.addRow("閫氶亾鏉冮噸妯″紡", self.weight_mode_edit)
        form.addRow("绌洪棿婊ゆ尝妯″紡", self.spatial_mode_edit)
        form.addRow("Spatial Rank Candidates", self.spatial_ranks_edit)
        form.addRow("鑱斿悎杩唬杞暟", self.joint_iters_edit)
        form.addRow("Spatial Source Model", self.spatial_source_edit)
        form.addRow("璇勬祴鑼冨洿", self.metric_scope_edit)
        form.addRow("璁烘枃鍙ｅ緞鍐崇瓥鏃堕棿", self.decision_time_mode_edit)
        form.addRow("寮傛鍙ｅ緞鍐崇瓥鏃堕棿", self.async_decision_time_mode_edit)
        form.addRow("鏁版嵁绛栫暐", self.data_policy_edit)
        form.addRow("Export Figures (1/0)", self.export_figures_edit)
        form.addRow("鎺掑簭绛栫暐", self.ranking_policy_edit)
        form.addRow("Dynamic Stop (1/0)", self.dynamic_stop_edit)
        form.addRow("鍔ㄦ€佺疮璁?alpha", self.dynamic_alpha_edit)
        form.addRow("Win Candidates", self.win_candidates_edit)
        form.addRow("闅忔満绉嶅瓙", self.seed_edit)
        form.addRow("璁＄畻鍚庣", self.compute_backend_combo)
        form.addRow("GPU 璁惧", self.gpu_device_edit)
        form.addRow("GPU 绮惧害", self.gpu_precision_combo)
        form.addRow("GPU 棰勭儹(1/0)", self.gpu_warmup_edit)
        form.addRow("GPU 缂撳瓨", self.gpu_cache_combo)
        form.addRow("Server Host", self.server_host_edit)
        form.addRow("Server Port", self.server_port_edit)
        form.addRow("Server Username", self.server_username_edit)
        form.addRow("Server Password", self.server_password_edit)
        form.addRow("瀛愬甫鏉冮噸妯″紡", self.subband_weight_mode_edit)
        form.addRow("鏉冮噸浜ゅ弶楠岃瘉鎶樻暟", self.weight_cv_folds_edit)
        self._layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_pick_dataset_root = QPushButton("閫夋嫨鏁版嵁闆嗘牴鐩綍")
        self.btn_pick_s1 = QPushButton("閫夋嫨 Session1")
        self.btn_pick_s2 = QPushButton("閫夋嫨 Session2")
        self.btn_pick_profile = QPushButton("閫夋嫨杈撳嚭 Profile")
        self.btn_pick_report = QPushButton("閫夋嫨鎶ュ憡 JSON")
        self.btn_pick_report_root = QPushButton("Pick Report Root")
        self.btn_scan_datasets = QPushButton("鎵弿浼氳瘽")
        self.btn_select_all_datasets = QPushButton("Select All")
        self.btn_clear_datasets = QPushButton("娓呯┖閫夋嫨")
        self.btn_run = QPushButton("Start Training/Evaluation")
        self.btn_open_report_dir = QPushButton("鎵撳紑鎶ュ憡鐩綍")
        self.btn_open_figures_dir = QPushButton("鎵撳紑鍥捐〃鐩綍")
        self.btn_open_report_dir.setEnabled(False)
        self.btn_open_figures_dir.setEnabled(False)
        for btn in (
            self.btn_pick_dataset_root,
            self.btn_pick_s1,
            self.btn_pick_s2,
            self.btn_pick_profile,
            self.btn_pick_report,
            self.btn_pick_report_root,
            self.btn_scan_datasets,
            self.btn_select_all_datasets,
            self.btn_clear_datasets,
            self.btn_run,
            self.btn_open_report_dir,
            self.btn_open_figures_dir,
        ):
            btn_row.addWidget(btn)
        self._layout.addLayout(btn_row)

        self.dataset_list_title = QLabel("Available sessions")
        self._layout.addWidget(self.dataset_list_title)
        self.dataset_list = QListWidget()
        self.dataset_list.setAlternatingRowColors(True)
        self.dataset_list.setSelectionMode(QListWidget.NoSelection)
        self.dataset_list.setMinimumHeight(210)
        self._layout.addWidget(self.dataset_list)

        self.status_label = QLabel("绌洪棽")
        self.status_label.setStyleSheet("font-size:16px; font-weight:600;")
        self._layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._layout.addWidget(self.progress_bar)
        self.progress_detail_label = QLabel("Current stage: not started")
        self.eta_label = QLabel("棰勮鍓╀綑锛?-")
        self._layout.addWidget(self.progress_detail_label)
        self._layout.addWidget(self.eta_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self._layout.addWidget(self.log_text, 1)

        self.btn_pick_dataset_root.clicked.connect(self._pick_dataset_root)
        self.btn_pick_s1.clicked.connect(self._pick_session1)
        self.btn_pick_s2.clicked.connect(self._pick_session2)
        self.btn_pick_profile.clicked.connect(self._pick_profile)
        self.btn_pick_report.clicked.connect(self._pick_report)
        self.btn_pick_report_root.clicked.connect(self._pick_report_root)
        self.btn_scan_datasets.clicked.connect(self._scan_dataset_manifests)
        self.btn_select_all_datasets.clicked.connect(self._select_all_dataset_items)
        self.btn_clear_datasets.clicked.connect(self._clear_dataset_selection)
        self.btn_run.clicked.connect(self._start_standard_run)
        self.btn_open_report_dir.clicked.connect(self._open_report_dir)
        self.btn_open_figures_dir.clicked.connect(self._open_figures_dir)
        self.btn_quick_run.clicked.connect(self._quick_auto_run)
        self.btn_weighted_compare_run.clicked.connect(self._weighted_compare_run)
        self.btn_model_compare_run.clicked.connect(self._model_compare_run)
        self.simple_mode_check.toggled.connect(self._on_simple_mode_toggled)
        self.btn_toggle_advanced.clicked.connect(self._toggle_advanced)
        self._remote_status_timer = QTimer(self)
        self._remote_status_timer.setInterval(int(DEFAULT_REMOTE_POLL_INTERVAL_MS))
        self._remote_status_timer.timeout.connect(self._poll_remote_status)

        self._advanced_widgets = [
            self.session1_edit,
            self.session2_edit,
            self.output_profile_edit,
            self.report_edit,
            self.report_root_edit,
            self.organize_report_edit,
            self.quality_min_ratio_edit,
            self.quality_max_retry_spin,
            self.strict_protocol_edit,
            self.strict_subject_edit,
            self.models_edit,
            self.channel_modes_edit,
            self.multi_seed_spin,
            self.gate_policy_edit,
            self.weight_mode_edit,
            self.subband_weight_mode_edit,
            self.spatial_mode_edit,
            self.spatial_ranks_edit,
            self.joint_iters_edit,
            self.weight_cv_folds_edit,
            self.spatial_source_edit,
            self.metric_scope_edit,
            self.decision_time_mode_edit,
            self.async_decision_time_mode_edit,
            self.data_policy_edit,
            self.export_figures_edit,
            self.ranking_policy_edit,
            self.dynamic_stop_edit,
            self.dynamic_alpha_edit,
            self.win_candidates_edit,
            self.seed_edit,
            self.btn_pick_s1,
            self.btn_pick_s2,
            self.btn_pick_profile,
            self.btn_pick_report,
            self.btn_pick_report_root,
            self.btn_scan_datasets,
            self.btn_select_all_datasets,
            self.btn_clear_datasets,
            self.dataset_list_title,
            self.dataset_list,
        ]
        self._scan_dataset_manifests()
        self._on_simple_mode_toggled(True)
        self._apply_localized_texts()

    def _label_for(self, widget: QWidget) -> Optional[QLabel]:
        if not hasattr(self, "_form_layout"):
            return None
        try:
            return self._form_layout.labelForField(widget)
        except Exception:
            return None

    def _set_form_label_text(self, widget: QWidget, text: str) -> None:
        label = self._label_for(widget)
        if label is not None:
            label.setText(str(text))

    def _apply_localized_texts(self) -> None:
        self.setWindowTitle("SSVEP 训练评测")
        self.simple_mode_check.setText("简易模式（推荐）")
        self.keep_baseline_group_check.setText("保留基线模型组")
        self.remote_mode_check.setText("远端模式（默认）")
        self.allow_local_mode_check.setText("启用本地兜底")
        self.btn_weighted_compare_run.setText("训练权重并对比全模型（推荐）")
        self.btn_quick_run.setText("FBCCA 权重训练（快速）")
        self.btn_model_compare_run.setText("全模型对比报告")
        self.btn_toggle_advanced.setText("显示高级设置")

        self.btn_pick_dataset_root.setText("选择数据集根目录")
        self.btn_pick_s1.setText("选择 Session1")
        self.btn_pick_s2.setText("选择 Session2")
        self.btn_pick_profile.setText("选择 Profile 输出")
        self.btn_pick_report.setText("选择报告 JSON")
        self.btn_pick_report_root.setText("选择报告根目录")
        self.btn_scan_datasets.setText("扫描会话")
        self.btn_select_all_datasets.setText("全选")
        self.btn_clear_datasets.setText("清空选择")
        self.btn_run.setText("开始训练/评测")
        self.btn_open_report_dir.setText("打开报告目录")
        self.btn_open_figures_dir.setText("打开图表目录")

        self.dataset_list_title.setText("可选会话")
        self.status_label.setText("空闲")
        self.progress_detail_label.setText("当前阶段：未开始")
        self.eta_label.setText("预计剩余：-")

        self._set_form_label_text(self.dataset_root_edit, "数据集根目录")
        self._set_form_label_text(self.session1_edit, "Session1 Manifest（回退手动）")
        self._set_form_label_text(self.session2_edit, "Session2 Manifest（可选，推荐）")
        self._set_form_label_text(self.output_profile_edit, "输出 Profile")
        self._set_form_label_text(self.report_edit, "报告 JSON")
        self._set_form_label_text(self.report_root_edit, "报告根目录")
        self._set_form_label_text(self.organize_report_edit, "报告按运行整理(1/0)")
        self._set_form_label_text(self.quality_min_ratio_edit, "质量过滤最小样本比例")
        self._set_form_label_text(self.quality_max_retry_spin, "质量过滤最大重采样次数")
        self._set_form_label_text(self.strict_protocol_edit, "严格协议一致(1/0)")
        self._set_form_label_text(self.strict_subject_edit, "严格被试一致(1/0)")
        self._set_form_label_text(self.models_edit, "模型列表")
        self._set_form_label_text(self.channel_modes_edit, "通道模式")
        self._set_form_label_text(self.multi_seed_spin, "多种子次数")
        self._set_form_label_text(self.gate_policy_edit, "门控策略")
        self._set_form_label_text(self.weight_mode_edit, "通道权重模式")
        self._set_form_label_text(self.subband_weight_mode_edit, "子带权重模式")
        self._set_form_label_text(self.spatial_mode_edit, "空间滤波模式")
        self._set_form_label_text(self.spatial_ranks_edit, "空间秩候选")
        self._set_form_label_text(self.joint_iters_edit, "联合迭代轮数")
        self._set_form_label_text(self.weight_cv_folds_edit, "权重交叉验证折数")
        self._set_form_label_text(self.spatial_source_edit, "空间源模型")
        self._set_form_label_text(self.metric_scope_edit, "评估范围")
        self._set_form_label_text(self.decision_time_mode_edit, "论文口径决策时间")
        self._set_form_label_text(self.async_decision_time_mode_edit, "异步口径决策时间")
        self._set_form_label_text(self.data_policy_edit, "数据策略")
        self._set_form_label_text(self.export_figures_edit, "导出图表(1/0)")
        self._set_form_label_text(self.ranking_policy_edit, "排序策略")
        self._set_form_label_text(self.dynamic_stop_edit, "动态停止(1/0)")
        self._set_form_label_text(self.dynamic_alpha_edit, "动态累积 alpha")
        self._set_form_label_text(self.win_candidates_edit, "窗长候选")
        self._set_form_label_text(self.seed_edit, "随机种子")
        self._set_form_label_text(self.compute_backend_combo, "计算后端")
        self._set_form_label_text(self.gpu_device_edit, "GPU 设备")
        self._set_form_label_text(self.gpu_precision_combo, "GPU 精度")
        self._set_form_label_text(self.gpu_warmup_edit, "GPU 预热(1/0)")
        self._set_form_label_text(self.gpu_cache_combo, "GPU 缓存策略")
        self._set_form_label_text(self.server_host_edit, "服务器 Host")
        self._set_form_label_text(self.server_port_edit, "服务器 Port")
        self._set_form_label_text(self.server_username_edit, "服务器用户名")
        self._set_form_label_text(self.server_password_edit, "服务器密码")

    def _set_form_row_visible(self, widget: QWidget, visible: bool) -> None:
        label = self._form_layout.labelForField(widget)
        if label is not None:
            label.setVisible(bool(visible))
        widget.setVisible(bool(visible))

    def _set_advanced_visible(self, visible: bool) -> None:
        for widget in self._advanced_widgets:
            if widget in {
                self.dataset_list_title,
                self.dataset_list,
                self.btn_pick_s1,
                self.btn_pick_s2,
                self.btn_pick_profile,
                self.btn_pick_report,
                self.btn_pick_report_root,
                self.btn_scan_datasets,
                self.btn_select_all_datasets,
                self.btn_clear_datasets,
            }:
                widget.setVisible(bool(visible))
                continue
            self._set_form_row_visible(widget, bool(visible))
        self.btn_toggle_advanced.setText("隐藏高级设置" if visible else "显示高级设置")

    def _legacy_on_simple_mode_toggled_unused_1(self, enabled: bool) -> None:
        if bool(enabled):
            self.btn_quick_run.setVisible(True)
            self.btn_toggle_advanced.setVisible(True)
            self._set_advanced_visible(False)
            self.status_label.setText("Simple Mode")
        else:
            self.btn_quick_run.setVisible(False)
            self.btn_toggle_advanced.setVisible(False)
            self._set_advanced_visible(True)
            self.status_label.setText("楂樼骇妯″紡")

    def _toggle_advanced(self) -> None:
        if not self.simple_mode_check.isChecked():
            return
        currently_visible = bool(self.dataset_list.isVisible())
        self._set_advanced_visible(not currently_visible)

    def _legacy_simple_mode_spec_unused_1(self, *, quick: bool) -> dict[str, Any]:
        if bool(quick):
            return {
                "variant": "quick",
                "model_names": tuple(QUICK_MODE_MODELS),
                "channel_modes": tuple(QUICK_MODE_CHANNEL_MODES),
                "multi_seed_count": int(QUICK_MODE_MULTI_SEED_COUNT),
                "win_candidates": tuple(float(value) for value in QUICK_MODE_WIN_CANDIDATES),
                "joint_weight_iters": int(QUICK_MODE_JOINT_WEIGHT_ITERS),
                "weight_cv_folds": int(QUICK_MODE_WEIGHT_CV_FOLDS),
                "quick_screen_top_k": int(QUICK_MODE_QUICK_SCREEN_TOP_K),
                "force_include_models": tuple(str(name) for name in QUICK_MODE_FORCE_INCLUDE_MODELS),
                "channel_weight_mode": str(QUICK_MODE_CHANNEL_WEIGHT_MODE),
                "subband_weight_mode": str(QUICK_MODE_SUBBAND_WEIGHT_MODE),
                "spatial_filter_mode": str(QUICK_MODE_SPATIAL_FILTER_MODE),
                "compute_backend": str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND),
                "gpu_precision": str(TRAIN_EVAL_DEFAULT_GPU_PRECISION),
            }
        return {
            "variant": "standard",
            "model_names": tuple(SIMPLE_MODE_MODELS),
            "channel_modes": tuple(SIMPLE_MODE_CHANNEL_MODES),
            "multi_seed_count": int(SIMPLE_MODE_MULTI_SEED_COUNT),
            "win_candidates": tuple(float(value) for value in DEFAULT_WIN_SEC_CANDIDATES),
            "joint_weight_iters": int(DEFAULT_JOINT_WEIGHT_ITERS),
            "weight_cv_folds": int(DEFAULT_FBCCA_WEIGHT_CV_FOLDS),
            "quick_screen_top_k": int(DEFAULT_QUICK_SCREEN_TOP_K),
            "force_include_models": tuple(str(name) for name in DEFAULT_FORCE_INCLUDE_MODELS),
            "channel_weight_mode": str(DEFAULT_CHANNEL_WEIGHT_MODE),
            "subband_weight_mode": str(DEFAULT_SUBBAND_WEIGHT_MODE),
            "spatial_filter_mode": str(DEFAULT_SPATIAL_FILTER_MODE),
            "compute_backend": str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND),
            "gpu_precision": str(TRAIN_EVAL_DEFAULT_GPU_PRECISION),
        }

    def _legacy_apply_simple_defaults_unused_1(self, *, quick: bool) -> None:
        spec = self._simple_mode_spec(quick=quick)
        self._simple_mode_variant = str(spec["variant"])
        self.models_edit.setText(",".join(spec["model_names"]))
        self.channel_modes_edit.setText(",".join(spec["channel_modes"]))
        self.multi_seed_spin.setValue(int(spec["multi_seed_count"]))
        self.win_candidates_edit.setText(",".join(f"{float(value):g}" for value in spec["win_candidates"]))
        self.joint_iters_edit.setText(str(int(spec["joint_weight_iters"])))
        self.weight_cv_folds_edit.setText(str(int(spec["weight_cv_folds"])))
        self.weight_mode_edit.setText(str(spec["channel_weight_mode"]))
        self.subband_weight_mode_edit.setText(str(spec["subband_weight_mode"]))
        self.spatial_mode_edit.setText(str(spec["spatial_filter_mode"]))
        self.compute_backend_combo.setCurrentText(str(spec["compute_backend"]))
        self.gpu_precision_combo.setCurrentText(str(spec["gpu_precision"]))
        self._quick_screen_top_k = int(spec["quick_screen_top_k"])
        self._force_include_models = tuple(str(name) for name in spec["force_include_models"])

    def _legacy_simple_mode_run_count_unused_1(self, *, quick: bool) -> int:
        spec = self._simple_mode_spec(quick=quick)
        model_names = tuple(str(name) for name in spec["model_names"])
        channel_modes = tuple(str(name) for name in spec["channel_modes"])
        multi_seed_count = int(spec["multi_seed_count"])
        channel_weight_mode = str(spec["channel_weight_mode"]).strip()
        subband_weight_mode = parse_subband_weight_mode(str(spec["subband_weight_mode"]).strip())
        run_count = 0
        for model_name in model_names:
            for channel_mode in channel_modes:
                if (
                    str(channel_mode) == "auto"
                    and normalize_model_name(str(model_name)) == "fbcca"
                    and (
                        str(channel_weight_mode or "").strip().lower() not in {"", "none"}
                        or str(subband_weight_mode or "").strip().lower() not in {"", "none", "chen_fixed"}
                    )
                ):
                    continue
                run_count += multi_seed_count
        return int(run_count)

    def _legacy_on_simple_mode_toggled_unused_2(self, enabled: bool) -> None:
        if bool(enabled):
            self._apply_simple_defaults(quick=False)
            self.btn_quick_run.setVisible(True)
            self.btn_toggle_advanced.setVisible(True)
            self._set_advanced_visible(False)
            self.status_label.setText(f"Simple Mode: default run {self._simple_mode_run_count(quick=False)} groups")
        else:
            self.btn_quick_run.setVisible(False)
            self.btn_toggle_advanced.setVisible(False)
            self._set_advanced_visible(True)
            self.status_label.setText("楂樼骇妯″紡")

    def _valid_scan_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self._dataset_scan_rows:
            if not isinstance(row, dict):
                continue
            if row.get("error"):
                continue
            manifest = str(row.get("manifest_path", "")).strip()
            if not manifest:
                continue
            path = Path(manifest).expanduser().resolve()
            if not path.exists():
                continue
            rows.append(dict(row))
        return rows

    def _apply_dataset_selection(self, manifests: Sequence[Path]) -> None:
        selected = {str(Path(path).expanduser().resolve()) for path in manifests}
        for idx in range(self.dataset_list.count()):
            item = self.dataset_list.item(idx)
            if item is None:
                continue
            raw = str(item.data(Qt.UserRole) or "").strip()
            target = str(Path(raw).expanduser().resolve()) if raw else ""
            item.setCheckState(Qt.Checked if target in selected else Qt.Unchecked)

    def _auto_choose_simple_sessions(self) -> tuple[Path, Optional[Path], tuple[Path, ...]]:
        if not self._dataset_scan_rows:
            self._scan_dataset_manifests()
        valid_rows = self._valid_scan_rows()
        if not valid_rows:
            raise ValueError("鏁版嵁鐩綍涓病鏈夊彲鐢ㄤ細璇濓紝璇峰厛閲囬泦骞剁敓鎴?session_manifest.json")

        session1_row = valid_rows[0]
        session1_manifest = Path(str(session1_row["manifest_path"])).expanduser().resolve()
        subject = str(session1_row.get("subject_id", "")).strip()
        signature = str(session1_row.get("protocol_signature", "")).strip()

        compatible: list[Path] = [session1_manifest]
        for row in valid_rows[1:]:
            row_subject = str(row.get("subject_id", "")).strip()
            row_signature = str(row.get("protocol_signature", "")).strip()
            if subject and row_subject and row_subject != subject:
                continue
            if signature and row_signature and row_signature != signature:
                continue
            compatible.append(Path(str(row["manifest_path"])).expanduser().resolve())
        compatible = compatible[:8]
        session2_manifest = compatible[1] if len(compatible) > 1 else None
        return session1_manifest, session2_manifest, tuple(compatible)

    def _legacy_quick_auto_run_unused_1(self) -> None:
        if self.worker_thread is not None:
            return
        self.simple_mode_check.setChecked(True)
        try:
            session1_manifest, session2_manifest, selected = self._auto_choose_simple_sessions()
        except Exception as exc:
            self._log(f"Auto selection preparation failed: {exc}")
            return
        self.session1_edit.setText(str(session1_manifest))
        self.session2_edit.setText("" if session2_manifest is None else str(session2_manifest))
        self._apply_dataset_selection(selected)
        self._log(
            f"Simple auto selection: session1={session1_manifest.name}, "
            f"session2={(session2_manifest.name if session2_manifest is not None else 'none')}, "
            f"鍙備笌浼氳瘽鏁?{len(selected)}"
        )
        self._log(
            f"蹇€熻瘎娴嬶細{self._simple_mode_run_count(quick=True)} 缁?| "
            f"models={','.join(QUICK_MODE_MODELS)} | "
            f"channel_modes={','.join(SIMPLE_MODE_CHANNEL_MODES)} | seeds={SIMPLE_MODE_MULTI_SEED_COUNT}"
        )
        self._start_run()

    def _legacy_on_simple_mode_toggled_unused_3(self, enabled: bool) -> None:
        if bool(enabled):
            self._apply_simple_defaults(quick=False)
            self.btn_quick_run.setVisible(True)
            self.btn_toggle_advanced.setVisible(True)
            self._set_advanced_visible(False)
            self.status_label.setText(
                f"绠€鏄撴ā寮忥細鏍囧噯璇勬祴 {self._simple_mode_run_count(quick=False)} 缁勶紝榛樿浣跨敤 GPU"
            )
        else:
            self.btn_quick_run.setVisible(False)
            self.btn_toggle_advanced.setVisible(False)
            self._set_advanced_visible(True)
            self.status_label.setText("楂樼骇妯″紡")

    def _legacy_quick_auto_run_unused_2(self) -> None:
        if self.worker_thread is not None:
            return
        self.simple_mode_check.setChecked(True)
        self._apply_simple_defaults(quick=True)
        try:
            session1_manifest, session2_manifest, selected = self._auto_choose_simple_sessions()
        except Exception as exc:
            self._log(f"Auto selection preparation failed: {exc}")
            return
        self.session1_edit.setText(str(session1_manifest))
        self.session2_edit.setText("" if session2_manifest is None else str(session2_manifest))
        self._apply_dataset_selection(selected)
        self._log(
            f"Quick mode auto selection: session1={session1_manifest.name}, "
            f"session2={(session2_manifest.name if session2_manifest is not None else 'none')}, "
            f"鍙備笌浼氳瘽鏁?{len(selected)}"
        )
        self._log(
            f"蹇€熻瘎娴嬶細{self._simple_mode_run_count(quick=True)} 缁?| "
            f"backend={TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND} | "
            f"models={','.join(QUICK_MODE_MODELS)} | "
            f"channel_modes={','.join(QUICK_MODE_CHANNEL_MODES)} | seeds={QUICK_MODE_MULTI_SEED_COUNT}"
        )
        self._start_run()

    def _start_standard_run(self) -> None:
        if self.simple_mode_check.isChecked():
            self._apply_simple_defaults(quick=False)
        self._start_run()

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {text}")

    def _on_progress(self, payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage", "") or "")
        model_name = str(payload.get("model_name", "") or "")
        run_index = int(payload.get("run_index", 0) or 0)
        run_total = int(payload.get("run_total", 0) or 0)
        config_index = int(payload.get("config_index", 0) or 0)
        config_total = int(payload.get("config_total", 0) or 0)
        elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
        eta_s = payload.get("eta_s", None)
        if stage == "stage_a":
            percent = 0 if run_total <= 0 else min(20, int(20.0 * run_index / max(run_total, 1)))
        elif stage == "stage_b":
            percent = 20 if run_total <= 0 else min(95, 20 + int(75.0 * run_index / max(run_total, 1)))
        elif stage == "complete":
            percent = 100
        else:
            percent = 0
        self.progress_bar.setValue(percent)
        stage_label = {
            "prepare": "准备",
            "stage_a": "阶段A：快速筛选",
            "stage_b": "阶段B：完整评测",
            "complete": "完成",
        }.get(stage, stage or "未知")
        detail = f"当前阶段：{stage_label}"
        if model_name:
            detail += f" | 当前模型：{model_name}"
        if run_total > 0:
            detail += f" | 运行：{run_index}/{run_total}"
        if config_total > 0:
            detail += f" | 配置：{config_index}/{config_total}"
        self.progress_detail_label.setText(detail)
        eta_text = "--" if eta_s is None else f"{float(eta_s):.1f}s"
        self.eta_label.setText(f"已耗时：{elapsed_s:.1f}s | 预计剩余：{eta_text}")
        self.status_label.setText(detail)

    def _pick_json(self, target: QLineEdit, title: str) -> None:
        start_dir = target.text().strip() or str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, title, start_dir, "JSON (*.json)")
        if path:
            target.setText(path)

    def _pick_dir(self, target: QLineEdit, title: str) -> None:
        start_dir = target.text().strip() or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, title, start_dir)
        if path:
            target.setText(path)

    def _pick_dataset_root(self) -> None:
        self._pick_dir(self.dataset_root_edit, "选择数据集根目录")
        self._scan_dataset_manifests()

    def _pick_session1(self) -> None:
        self._pick_json(self.session1_edit, "选择 Session1 Manifest")

    def _pick_session2(self) -> None:
        self._pick_json(self.session2_edit, "选择 Session2 Manifest")

    def _pick_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择 Profile 输出路径",
            self.output_profile_edit.text().strip(),
            "JSON (*.json)",
        )
        if path:
            self.output_profile_edit.setText(path)

    def _pick_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择报告 JSON 路径",
            self.report_edit.text().strip(),
            "JSON (*.json)",
        )
        if path:
            self.report_edit.setText(path)

    def _pick_report_root(self) -> None:
        self._pick_dir(self.report_root_edit, "选择报告根目录")

    def _scan_dataset_manifests(self) -> None:
        root = Path(self.dataset_root_edit.text().strip()).expanduser().resolve()
        rows = discover_collection_manifests(root)
        self._dataset_scan_rows = rows
        self.dataset_list.clear()
        for row in rows:
            created = str(row.get("generated_at", ""))
            subj = str(row.get("subject_id", ""))
            sid = str(row.get("session_id", ""))
            trials = int(row.get("trial_count", 0) or 0)
            shortfall = float(row.get("shortfall_ratio_mean", 0.0) or 0.0)
            preset = str(row.get("preset_name", ""))
            text = (
                f"{created} | 被试={subj} | 会话={sid} | 预设={preset} | "
                f"trial数={trials} | 样本短缺={shortfall:.3f}"
            )
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, str(row.get("manifest_path", "")))
            self.dataset_list.addItem(item)
        self._log(f"会话扫描完成：root={root}，会话数={len(rows)}")

    def _selected_dataset_manifest_paths(self) -> tuple[Path, ...]:
        rows: list[Path] = []
        for idx in range(self.dataset_list.count()):
            item = self.dataset_list.item(idx)
            if item is None or item.checkState() != Qt.Checked:
                continue
            raw = str(item.data(Qt.UserRole) or "").strip()
            if not raw:
                continue
            rows.append(Path(raw).expanduser().resolve())
        dedup: list[Path] = []
        seen: set[str] = set()
        for path in rows:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(path)
        return tuple(dedup)

    def _select_all_dataset_items(self) -> None:
        for idx in range(self.dataset_list.count()):
            item = self.dataset_list.item(idx)
            if item is not None:
                item.setCheckState(Qt.Checked)

    def _clear_dataset_selection(self) -> None:
        for idx in range(self.dataset_list.count()):
            item = self.dataset_list.item(idx)
            if item is not None:
                item.setCheckState(Qt.Unchecked)

    def _open_path(self, path: Optional[Path]) -> None:
        if path is None:
            self._log("Path is not available yet.")
            return
        target = Path(path).expanduser().resolve()
        if not target.exists():
            self._log(f"路径不存在：{target}")
            return
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _open_report_dir(self) -> None:
        if self._last_report_path is not None:
            self._open_path(self._last_report_path.parent)
            return
        self._open_path(Path(self.report_root_edit.text().strip()).expanduser().resolve())

    def _open_figures_dir(self) -> None:
        self._open_path(self._last_figures_dir)

    def _read_config(self) -> TrainEvalUIConfig:
        if self.simple_mode_check.isChecked():
            self._apply_simple_defaults(quick=self._simple_mode_variant == "quick")
            session1_manifest, session2_manifest, selected = self._auto_choose_simple_sessions()
            self.session1_edit.setText(str(session1_manifest))
            self.session2_edit.setText("" if session2_manifest is None else str(session2_manifest))
            self._apply_dataset_selection(selected)
        else:
            selected = self._selected_dataset_manifest_paths()
            if selected:
                session1_manifest = selected[0]
            else:
                raw = self.session1_edit.text().strip()
                if not raw:
                    raise ValueError("请至少选择一个会话，或手动指定 Session1 Manifest")
                session1_manifest = Path(raw).expanduser().resolve()
            if not session1_manifest.exists():
                raise FileNotFoundError(f"Session1 manifest not found: {session1_manifest}")
            raw_s2 = self.session2_edit.text().strip()
            session2_manifest = Path(raw_s2).expanduser().resolve() if raw_s2 else None
            if session2_manifest is not None and not session2_manifest.exists():
                raise FileNotFoundError(f"Session2 manifest not found: {session2_manifest}")
        selection_snapshot = {
            "dataset_root": str(Path(self.dataset_root_edit.text().strip()).expanduser().resolve()),
            "selected_manifest_count": int(len(selected)),
            "selected_manifests": [str(path) for path in selected],
            "quality_min_sample_ratio": float(self.quality_min_ratio_edit.text().strip() or "0.90"),
            "quality_max_retry_count": int(self.quality_max_retry_spin.value()),
            "strict_protocol_consistency": bool(int(self.strict_protocol_edit.text().strip() or "1")),
            "strict_subject_consistency": bool(int(self.strict_subject_edit.text().strip() or "1")),
            "decision_time_mode": str(self.decision_time_mode_edit.text().strip()),
            "async_decision_time_mode": str(self.async_decision_time_mode_edit.text().strip()),
            "data_policy": str(self.data_policy_edit.text().strip()),
            "keep_baseline_group": bool(self.keep_baseline_group_check.isChecked()),
        }
        requested_models = list(parse_model_list(self.models_edit.text().strip()))
        if self.keep_baseline_group_check.isChecked():
            for model_name in BASELINE_COMPARE_MODELS:
                if model_name not in requested_models:
                    requested_models.append(str(model_name))
        return TrainEvalUIConfig(
            session1_manifest=session1_manifest,
            session2_manifest=session2_manifest,
            dataset_manifests=selected,
            dataset_root=Path(self.dataset_root_edit.text().strip()).expanduser().resolve(),
            dataset_selection_snapshot=selection_snapshot,
            quality_min_sample_ratio=float(self.quality_min_ratio_edit.text().strip() or "0.90"),
            quality_max_retry_count=int(self.quality_max_retry_spin.value()),
            strict_protocol_consistency=bool(int(self.strict_protocol_edit.text().strip() or "1")),
            strict_subject_consistency=bool(int(self.strict_subject_edit.text().strip() or "1")),
            output_profile_path=Path(self.output_profile_edit.text().strip()).expanduser().resolve(),
            report_path=Path(self.report_edit.text().strip()).expanduser().resolve(),
            report_root_dir=Path(self.report_root_edit.text().strip()).expanduser().resolve(),
            organize_report_dir=bool(int(self.organize_report_edit.text().strip() or "1")),
            model_names=tuple(requested_models),
            channel_modes=tuple(parse_channel_mode_list(self.channel_modes_edit.text().strip())),
            multi_seed_count=int(self.multi_seed_spin.value()),
            gate_policy=parse_gate_policy(self.gate_policy_edit.text().strip()),
            channel_weight_mode=(
                None if str(self.weight_mode_edit.text()).strip() == "" else str(self.weight_mode_edit.text()).strip()
            ),
            subband_weight_mode=parse_subband_weight_mode(self.subband_weight_mode_edit.text().strip()),
            spatial_filter_mode=parse_spatial_filter_mode(self.spatial_mode_edit.text().strip()),
            spatial_rank_candidates=tuple(parse_spatial_rank_candidates(self.spatial_ranks_edit.text().strip())),
            joint_weight_iters=max(1, int(self.joint_iters_edit.text().strip() or "1")),
            weight_cv_folds=max(2, int(self.weight_cv_folds_edit.text().strip() or str(DEFAULT_FBCCA_WEIGHT_CV_FOLDS))),
            spatial_source_model=parse_spatial_source_model(self.spatial_source_edit.text().strip()),
            metric_scope=parse_metric_scope(self.metric_scope_edit.text().strip()),
            decision_time_mode=parse_decision_time_mode(self.decision_time_mode_edit.text().strip()),
            async_decision_time_mode=parse_decision_time_mode(
                self.async_decision_time_mode_edit.text().strip()
            ),
            data_policy=parse_data_policy(self.data_policy_edit.text().strip()),
            export_figures=bool(int(self.export_figures_edit.text().strip() or "1")),
            ranking_policy=parse_ranking_policy(self.ranking_policy_edit.text().strip()),
            dynamic_stop_enabled=bool(int(self.dynamic_stop_edit.text().strip() or "1")),
            dynamic_stop_alpha=float(self.dynamic_alpha_edit.text().strip()),
            win_candidates=tuple(float(item.strip()) for item in self.win_candidates_edit.text().split(",") if item.strip()),
            seed=int(self.seed_edit.text().strip()),
            evaluation_mode=str(self._evaluation_mode),
            quick_screen_top_k=int(self._quick_screen_top_k),
            force_include_models=tuple(self._force_include_models),
            progress_heartbeat_sec=float(self._progress_heartbeat_sec),
            compute_backend=parse_compute_backend_name(self.compute_backend_combo.currentText().strip()),
            gpu_device=int(self.gpu_device_edit.text().strip() or str(DEFAULT_GPU_DEVICE_ID)),
            gpu_precision=parse_gpu_precision(self.gpu_precision_combo.currentText().strip()),
            gpu_warmup=bool(int(self.gpu_warmup_edit.text().strip() or "1")),
            gpu_cache_policy=parse_gpu_cache_policy(self.gpu_cache_combo.currentText().strip()),
            task=str(self._task),
        )

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        if running:
            self.btn_open_report_dir.setEnabled(False)
            self.btn_open_figures_dir.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_detail_label.setText("褰撳墠闃舵锛氬噯澶囦腑")
            self.eta_label.setText("棰勮鍓╀綑锛?-")

    def _start_run(self) -> None:
        if self.worker_thread is not None:
            return
        use_remote = bool(self.remote_mode_check.isChecked())
        if use_remote:
            self._start_remote_run()
            return
        if not bool(self.allow_local_mode_check.isChecked()):
            self._log("本地训练已禁用。请开启 'Enable local fallback' 后再使用本地模式。")
            return
        self._start_local_run()

    def _start_local_run(self) -> None:
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"Configuration error: {exc}")
            return
        worker = TrainEvalWorker(cfg)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.log.connect(self._log)
        worker.progress.connect(self._on_progress)
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
        self.status_label.setText("本地训练评测运行中")
        thread.start()

    def _server_config(self) -> ServerConfig:
        host = str(self.server_host_edit.text().strip() or DEFAULT_SERVER_HOST)
        username = str(self.server_username_edit.text().strip() or DEFAULT_SERVER_USERNAME)
        password = str(self.server_password_edit.text())
        port = int(self.server_port_edit.text().strip() or str(DEFAULT_SERVER_PORT))
        return ServerConfig(host=host, port=port, username=username, password=password)

    def _remote_gpu_params(self, cfg: TrainEvalUIConfig) -> dict[str, Any]:
        return {
            "compute_backend": str(cfg.compute_backend or DEFAULT_REMOTE_COMPUTE_BACKEND),
            "gpu_device": int(cfg.gpu_device),
            "gpu_precision": str(cfg.gpu_precision or DEFAULT_REMOTE_GPU_PRECISION),
            "gpu_warmup": bool(cfg.gpu_warmup),
            "gpu_cache_policy": str(cfg.gpu_cache_policy or DEFAULT_REMOTE_GPU_CACHE_POLICY),
            "win_candidates": ",".join(f"{float(item):g}" for item in cfg.win_candidates)
            if cfg.win_candidates
            else str(DEFAULT_REMOTE_WIN_CANDIDATES),
            "multi_seed_count": max(1, int(cfg.multi_seed_count or DEFAULT_REMOTE_MULTI_SEED_COUNT)),
        }

    def _start_remote_run(self) -> None:
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"配置错误: {exc}")
            return
        if cfg.session2_manifest is None:
            decision = QMessageBox.question(
                self,
                "Session2 Recommended",
                "未提供 Session2。将继续运行，但报告会标记为 no_session2。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if decision != QMessageBox.Yes:
                self._log("已取消：请补充 Session2 或确认继续。")
                return
        server_cfg = self._server_config()
        if not server_cfg.password:
            self._log("请输入服务器密码后再提交远端任务。")
            QMessageBox.warning(self, "Missing Password", "请先填写 Server Password。")
            return

        task_name = str(cfg.task)
        if task_name not in {"fbcca-weights", "model-compare", "fbcca-weighted-compare"}:
            self._log(f"远端暂不支持该 task: {task_name}")
            QMessageBox.warning(self, "Unsupported Task", f"远端暂不支持任务: {task_name}")
            return

        run_id = now_run_id(task_name.replace("-", "_"))
        gpu_params = self._remote_gpu_params(cfg)
        self._set_running(True)
        self.status_label.setText("远端任务提交中")
        try:
            ssh = SSHClient(server_cfg)
            ssh.connect()
            try:
                dataset_session1 = _find_dataset_by_manifest(cfg.session1_manifest)
                dataset_session2 = (
                    None if cfg.session2_manifest is None else _find_dataset_by_manifest(cfg.session2_manifest)
                )
                remote_s1 = upload_dataset(ssh, dataset_session1)
                remote_s2 = None
                if (
                    dataset_session2 is not None
                    and dataset_session2.manifest_path.resolve() != dataset_session1.manifest_path.resolve()
                ):
                    remote_s2 = upload_dataset(ssh, dataset_session2)
                preflight = preflight_cuda_or_fail(
                    ssh,
                    compute_backend=str(gpu_params["compute_backend"]),
                    gpu_device=int(gpu_params["gpu_device"]),
                )
                command_payload = build_train_command(
                    task=task_name,
                    dataset_manifest_remote=str(remote_s1["manifest"]),
                    dataset_manifest_session2_remote=(
                        None if remote_s2 is None else str(remote_s2["manifest"])
                    ),
                    run_id=run_id,
                    compute_backend=str(gpu_params["compute_backend"]),
                    gpu_device=int(gpu_params["gpu_device"]),
                    gpu_precision=str(gpu_params["gpu_precision"]),
                    gpu_warmup=bool(gpu_params["gpu_warmup"]),
                    gpu_cache_policy=str(gpu_params["gpu_cache_policy"]),
                    win_candidates=str(gpu_params["win_candidates"]),
                    multi_seed_count=int(gpu_params["multi_seed_count"]),
                )
                self._remote_record = start_remote_task(
                    ssh,
                    command_payload,
                    metadata={
                        "session1": str(cfg.session1_manifest),
                        "session2": "" if cfg.session2_manifest is None else str(cfg.session2_manifest),
                        "remote_manifest_paths": {
                            "session1": str(remote_s1.get("manifest", "")),
                            "session2": "" if remote_s2 is None else str(remote_s2.get("manifest", "")),
                        },
                        "gpu_params": dict(gpu_params),
                        "gpu_preflight": preflight,
                        "metrics_source": "no_session2" if cfg.session2_manifest is None else "cross_session",
                    },
                )
            finally:
                ssh.close()
        except Exception as exc:
            self._set_running(False)
            self.status_label.setText("远端任务提交失败")
            self._log(f"远端任务提交失败: {exc}")
            QMessageBox.critical(self, "Remote Submit Failed", str(exc))
            return

        self._log(
            "远端任务已提交: "
            f"run_id={self._remote_record.get('run_id','')} task={task_name} "
            f"session2={'none' if cfg.session2_manifest is None else cfg.session2_manifest.name}"
        )
        if self._remote_status_timer is not None:
            self._remote_status_timer.start()
        self._poll_remote_status()

    def _poll_remote_status(self) -> None:
        if not self.remote_mode_check.isChecked():
            return
        record = self._remote_record
        if not isinstance(record, dict):
            return
        server_cfg = self._server_config()
        if not server_cfg.password:
            return
        try:
            ssh = SSHClient(server_cfg)
            ssh.connect()
            try:
                status = read_remote_status(ssh, record)
            finally:
                ssh.close()
        except Exception as exc:
            self._log(f"远端状态查询失败: {exc}")
            return

        progress = dict(status.get("progress") or {})
        stage = str(progress.get("stage", "")).strip()
        stage_percent = progress.get("progress_percent", progress.get("percent", None))
        if isinstance(stage_percent, (int, float)):
            self.progress_bar.setValue(max(0, min(100, int(float(stage_percent)))))
        detail = f"远端阶段: {stage or 'running'}"
        model_name = str(progress.get("model_name", "")).strip()
        if model_name:
            detail += f" | model={model_name}"
        self.progress_detail_label.setText(detail)
        self.status_label.setText(detail)
        self.eta_label.setText(f"log: {status.get('log_path', '')}")
        tail_text = str(status.get("tail", "") or "").strip()
        if tail_text:
            self.log_text.setPlainText(tail_text)
        process_text = str(status.get("process", "") or "").strip()
        artifacts = dict(status.get("artifacts") or {})
        finished = (not process_text) and bool(artifacts.get("report_json", False))
        if finished:
            if self._remote_status_timer is not None:
                self._remote_status_timer.stop()
            self._download_remote_results()

    def _download_remote_results(self) -> None:
        record = self._remote_record
        if not isinstance(record, dict):
            return
        server_cfg = self._server_config()
        if not server_cfg.password:
            self._log("缺少服务器密码，无法下载结果。")
            self._set_running(False)
            return
        try:
            ssh = SSHClient(server_cfg)
            ssh.connect()
            try:
                result = download_results(ssh, record)
            finally:
                ssh.close()
        except Exception as exc:
            self._log(f"远端结果下载失败: {exc}")
            self._set_running(False)
            return
        local_run_dir = Path(str(result.get("local_run_dir", ""))).expanduser().resolve()
        report_path = local_run_dir / "offline_train_eval.json"
        if report_path.exists():
            self._last_report_path = report_path
        self.btn_open_report_dir.setEnabled(local_run_dir.exists())
        self.btn_open_figures_dir.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_detail_label.setText("远端任务完成")
        self.status_label.setText("远端训练评测完成")
        self._set_running(False)
        self._log(
            "远端任务完成并已下载: "
            f"run_dir={local_run_dir} profile={result.get('local_profile', '')}"
        )

    def _on_done(self, payload: dict[str, Any]) -> None:
        report_path = payload.get("report_path") or self.report_edit.text().strip()
        self._last_report_path = Path(str(report_path)).expanduser().resolve()
        figures = dict(payload.get("figures", {}))
        figures_dir = figures.get("dir")
        self._last_figures_dir = Path(str(figures_dir)).expanduser().resolve() if figures_dir else None
        self.btn_open_report_dir.setEnabled(True)
        self.btn_open_figures_dir.setEnabled(self._last_figures_dir is not None)
        self.progress_bar.setValue(100)
        self.progress_detail_label.setText("当前阶段：完成")
        self.eta_label.setText("预计剩余：0.0s")
        async_metrics = dict(payload.get("chosen_async_metrics", {}))
        metrics_4 = dict(payload.get("chosen_metrics_4class", {}))
        kept_trials = int(payload.get("quality_kept_trials_session1", 0) or 0)
        total_trials = int(payload.get("quality_total_trials_session1", 0) or 0)
        self._log(
            "结果摘要："
            f"保留样本={kept_trials}/{total_trials}, "
            f"数据策略={payload.get('data_policy', '')}, "
            f"idle误触发/分钟={float(async_metrics.get('idle_fp_per_min', float('inf'))):.4f}, "
            f"控制召回={float(async_metrics.get('control_recall', 0.0)):.4f}, "
            f"切换时延={float(async_metrics.get('switch_latency_s', float('inf'))):.4f}s, "
            f"释放时延={float(async_metrics.get('release_latency_s', float('inf'))):.4f}s, "
            f"四分类准确率={float(metrics_4.get('acc', 0.0)):.4f}, "
            f"四分类Macro-F1={float(metrics_4.get('macro_f1', 0.0)):.4f}"
        )
        if bool(payload.get("profile_saved", False)):
            self.status_label.setText("训练评测完成")
            self._log(f"完成。已选模型={payload.get('chosen_model')}，报告={self._last_report_path}")
        else:
            self.status_label.setText("已完成（无达标模型）")
            self._log(f"完成但未保存 profile。推荐模型={payload.get('recommended_model')}")

    def _on_error(self, text: str) -> None:
        self.status_label.setText("训练评测失败")
        self.progress_detail_label.setText("当前阶段：失败")
        self._log(text)

    def _on_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_running(False)

    def _simple_mode_spec(self, *, quick: bool) -> dict[str, Any]:
        if str(getattr(self, "_task", DEFAULT_TRAIN_EVAL_TASK)) == "fbcca-weighted-compare":
            return {
                "variant": "fbcca-weighted-compare",
                "task": "fbcca-weighted-compare",
                "model_names": tuple(WEIGHTED_COMPARE_MODELS),
                "channel_modes": tuple(WEIGHTED_COMPARE_CHANNEL_MODES),
                "multi_seed_count": int(WEIGHTED_COMPARE_MULTI_SEED_COUNT),
                "win_candidates": tuple(float(value) for value in WEIGHTED_COMPARE_WIN_CANDIDATES),
                "joint_weight_iters": int(WEIGHTED_COMPARE_JOINT_WEIGHT_ITERS),
                "weight_cv_folds": int(WEIGHTED_COMPARE_WEIGHT_CV_FOLDS),
                "quick_screen_top_k": int(WEIGHTED_COMPARE_QUICK_SCREEN_TOP_K),
                "force_include_models": tuple(str(name) for name in WEIGHTED_COMPARE_FORCE_INCLUDE_MODELS),
                "channel_weight_mode": str(WEIGHTED_COMPARE_CHANNEL_WEIGHT_MODE),
                "subband_weight_mode": str(WEIGHTED_COMPARE_SUBBAND_WEIGHT_MODE),
                "spatial_filter_mode": str(WEIGHTED_COMPARE_SPATIAL_FILTER_MODE),
                "control_state_mode": "frequency-specific-logistic",
                "compute_backend": str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND),
                "gpu_precision": str(TRAIN_EVAL_DEFAULT_GPU_PRECISION),
            }
        if bool(quick) or str(getattr(self, "_task", DEFAULT_TRAIN_EVAL_TASK)) == "fbcca-weights":
            return {
                "variant": "quick" if bool(quick) else "fbcca-weights",
                "task": "fbcca-weights",
                "model_names": tuple(QUICK_MODE_MODELS),
                "channel_modes": tuple(QUICK_MODE_CHANNEL_MODES),
                "multi_seed_count": int(QUICK_MODE_MULTI_SEED_COUNT),
                "win_candidates": tuple(float(value) for value in QUICK_MODE_WIN_CANDIDATES),
                "joint_weight_iters": int(QUICK_MODE_JOINT_WEIGHT_ITERS),
                "weight_cv_folds": int(QUICK_MODE_WEIGHT_CV_FOLDS),
                "quick_screen_top_k": int(QUICK_MODE_QUICK_SCREEN_TOP_K),
                "force_include_models": tuple(str(name) for name in QUICK_MODE_FORCE_INCLUDE_MODELS),
                "channel_weight_mode": str(QUICK_MODE_CHANNEL_WEIGHT_MODE),
                "subband_weight_mode": str(QUICK_MODE_SUBBAND_WEIGHT_MODE),
                "spatial_filter_mode": str(QUICK_MODE_SPATIAL_FILTER_MODE),
                "control_state_mode": str(DEFAULT_CONTROL_STATE_MODE),
                "compute_backend": str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND),
                "gpu_precision": str(TRAIN_EVAL_DEFAULT_GPU_PRECISION),
            }
        return {
            "variant": "model-compare",
            "task": "model-compare",
            "model_names": tuple(MODEL_COMPARE_MODELS),
            "channel_modes": tuple(MODEL_COMPARE_CHANNEL_MODES),
            "multi_seed_count": int(MODEL_COMPARE_MULTI_SEED_COUNT),
            "win_candidates": tuple(float(value) for value in MODEL_COMPARE_WIN_CANDIDATES),
            "joint_weight_iters": int(MODEL_COMPARE_JOINT_WEIGHT_ITERS),
            "weight_cv_folds": int(MODEL_COMPARE_WEIGHT_CV_FOLDS),
            "quick_screen_top_k": int(MODEL_COMPARE_QUICK_SCREEN_TOP_K),
            "force_include_models": tuple(str(name) for name in MODEL_COMPARE_FORCE_INCLUDE_MODELS),
            "channel_weight_mode": str(MODEL_COMPARE_CHANNEL_WEIGHT_MODE),
            "subband_weight_mode": str(MODEL_COMPARE_SUBBAND_WEIGHT_MODE),
            "spatial_filter_mode": str(MODEL_COMPARE_SPATIAL_FILTER_MODE),
            "control_state_mode": "frequency-specific-logistic",
            "compute_backend": str(TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND),
            "gpu_precision": str(TRAIN_EVAL_DEFAULT_GPU_PRECISION),
        }

    def _apply_simple_defaults(self, *, quick: bool) -> None:
        spec = self._simple_mode_spec(quick=quick)
        self._simple_mode_variant = str(spec["variant"])
        self._task = str(spec["task"])
        self.models_edit.setText(",".join(spec["model_names"]))
        self.channel_modes_edit.setText(",".join(spec["channel_modes"]))
        self.multi_seed_spin.setValue(int(spec["multi_seed_count"]))
        self.win_candidates_edit.setText(",".join(f"{float(value):g}" for value in spec["win_candidates"]))
        self.joint_iters_edit.setText(str(int(spec["joint_weight_iters"])))
        self.weight_cv_folds_edit.setText(str(int(spec["weight_cv_folds"])))
        self.weight_mode_edit.setText(str(spec["channel_weight_mode"]))
        self.subband_weight_mode_edit.setText(str(spec["subband_weight_mode"]))
        self.spatial_mode_edit.setText(str(spec["spatial_filter_mode"]))
        if hasattr(self, "control_state_mode_edit") and self.control_state_mode_edit is not None:
            self.control_state_mode_edit.setText(str(spec.get("control_state_mode", DEFAULT_CONTROL_STATE_MODE)))
        self.compute_backend_combo.setCurrentText(str(spec["compute_backend"]))
        self.gpu_precision_combo.setCurrentText(str(spec["gpu_precision"]))
        self._evaluation_mode = str(DEFAULT_EVALUATION_MODE)
        self._quick_screen_top_k = int(spec["quick_screen_top_k"])
        self._force_include_models = tuple(str(name) for name in spec["force_include_models"])

    def _simple_mode_run_count(self, *, quick: bool) -> int:
        spec = self._simple_mode_spec(quick=quick)
        return int(len(spec["model_names"]) * len(spec["channel_modes"]) * int(spec["multi_seed_count"]))

    def _on_simple_mode_toggled(self, enabled: bool) -> None:
        if bool(enabled):
            self._apply_simple_defaults(quick=self._task == "fbcca-weights")
            self.btn_weighted_compare_run.setVisible(True)
            self.btn_quick_run.setVisible(True)
            self.btn_model_compare_run.setVisible(True)
            self.btn_toggle_advanced.setVisible(True)
            self._set_advanced_visible(False)
            self.status_label.setText("简易模式：建议先运行权重训练+全模型对比")
        else:
            self.btn_weighted_compare_run.setVisible(False)
            self.btn_quick_run.setVisible(False)
            self.btn_model_compare_run.setVisible(False)
            self.btn_toggle_advanced.setVisible(False)
            self._set_advanced_visible(True)
            self.status_label.setText("高级模式：按当前参数运行")

    def _run_auto_task(self, task: str) -> None:
        if self.worker_thread is not None:
            return
        self._task = _parse_task(task)
        self.simple_mode_check.setChecked(True)
        self._apply_simple_defaults(quick=self._task == "fbcca-weights")
        try:
            session1_manifest, session2_manifest, selected = self._auto_choose_simple_sessions()
        except Exception as exc:
            self._log(f"Auto dataset selection failed: {exc}")
            return
        self.session1_edit.setText(str(session1_manifest))
        self.session2_edit.setText("" if session2_manifest is None else str(session2_manifest))
        self._apply_dataset_selection(selected)
        if self._task == "fbcca-weighted-compare":
            self._log(
                f"训练 FBCCA 通道/子带权重并加入全模型对比：cuda/all8/seeds={int(WEIGHTED_COMPARE_MULTI_SEED_COUNT)} | "
                f"models={','.join(WEIGHTED_COMPARE_MODELS)} | sessions={len(selected)}"
            )
        elif self._task == "fbcca-weights":
            self._log(
                "FBCCA 权重实验：cuda/all8/seed=1/win=1.5 | "
                f"models={','.join(QUICK_MODE_MODELS)} | sessions={len(selected)}"
            )
        else:
            self._log(
                f"全模型对比报告：cuda/all8/seeds={int(MODEL_COMPARE_MULTI_SEED_COUNT)} | "
                f"models={','.join(MODEL_COMPARE_MODELS)} | sessions={len(selected)}"
            )
        self._start_run()

    def _quick_auto_run(self) -> None:
        self._run_auto_task("fbcca-weights")

    def _weighted_compare_run(self) -> None:
        self._run_auto_task("fbcca-weighted-compare")

    def _model_compare_run(self) -> None:
        self._run_auto_task("model-compare")

    def _start_standard_run(self) -> None:
        if self.simple_mode_check.isChecked():
            self._task = "model-compare"
            self._apply_simple_defaults(quick=False)
        self._start_run()

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_weighted_compare_run.setEnabled(not running)
        self.btn_quick_run.setEnabled(not running)
        self.btn_model_compare_run.setEnabled(not running)
        if running:
            self.btn_open_report_dir.setEnabled(False)
            self.btn_open_figures_dir.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_detail_label.setText("当前阶段：准备中")
            self.eta_label.setText("预计剩余：-")
        else:
            if self._remote_status_timer is not None:
                self._remote_status_timer.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSVEP 训练评测 UI / CLI")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="session1 manifest path")
    parser.add_argument("--dataset-manifest-session2", type=Path, default=None, help="session2 manifest path")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--include-manifests", type=str, default="", help="comma-separated manifest paths")
    parser.add_argument("--output-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_DIR / f"offline_train_eval_{_now_stamp()}.json")
    parser.add_argument("--report-root-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--organize-report-dir", type=int, default=1)
    parser.add_argument("--quality-min-sample-ratio", type=float, default=0.9)
    parser.add_argument("--quality-max-retry-count", type=int, default=3)
    parser.add_argument("--strict-protocol-consistency", type=int, default=1)
    parser.add_argument("--strict-subject-consistency", type=int, default=1)
    parser.add_argument("--models", type=str, default=",".join(ModelRegistry.list_models(task="benchmark")))
    parser.add_argument("--channel-modes", type=str, default=",".join(DEFAULT_BENCHMARK_CHANNEL_MODES))
    parser.add_argument("--multi-seed-count", type=int, default=DEFAULT_BENCHMARK_MULTI_SEED_COUNT)
    parser.add_argument("--gate-policy", type=str, default=DEFAULT_GATE_POLICY)
    parser.add_argument("--channel-weight-mode", type=str, default=str(DEFAULT_CHANNEL_WEIGHT_MODE))
    parser.add_argument("--subband-weight-mode", type=str, default=str(DEFAULT_SUBBAND_WEIGHT_MODE))
    parser.add_argument("--spatial-filter-mode", type=str, default=str(DEFAULT_SPATIAL_FILTER_MODE))
    parser.add_argument("--spatial-rank-candidates", type=str, default=",".join(str(v) for v in DEFAULT_SPATIAL_RANK_CANDIDATES))
    parser.add_argument("--joint-weight-iters", type=int, default=DEFAULT_JOINT_WEIGHT_ITERS)
    parser.add_argument("--weight-cv-folds", type=int, default=DEFAULT_FBCCA_WEIGHT_CV_FOLDS)
    parser.add_argument("--spatial-source-model", type=str, default=str(DEFAULT_SPATIAL_SOURCE_MODEL))
    parser.add_argument("--metric-scope", type=str, default=DEFAULT_METRIC_SCOPE)
    parser.add_argument("--decision-time-mode", type=str, default=DEFAULT_PAPER_DECISION_TIME_MODE)
    parser.add_argument("--async-decision-time-mode", type=str, default=DEFAULT_ASYNC_DECISION_TIME_MODE)
    parser.add_argument("--data-policy", type=str, default=DEFAULT_DATA_POLICY)
    parser.add_argument("--export-figures", type=int, default=1 if DEFAULT_EXPORT_FIGURES else 0)
    parser.add_argument("--ranking-policy", type=str, default=DEFAULT_RANKING_POLICY)
    parser.add_argument("--dynamic-stop-enabled", type=int, default=1)
    parser.add_argument("--dynamic-stop-alpha", type=float, default=DEFAULT_DYNAMIC_STOP_ALPHA)
    parser.add_argument("--win-candidates", type=str, default=",".join(f"{item:g}" for item in DEFAULT_WIN_SEC_CANDIDATES))
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--evaluation-mode", type=str, default=DEFAULT_EVALUATION_MODE)
    parser.add_argument("--quick-screen-top-k", type=int, default=DEFAULT_QUICK_SCREEN_TOP_K)
    parser.add_argument("--force-include-models", type=str, default=",".join(DEFAULT_FORCE_INCLUDE_MODELS))
    parser.add_argument("--progress-heartbeat-sec", type=float, default=DEFAULT_PROGRESS_HEARTBEAT_SEC)
    parser.add_argument("--compute-backend", type=str, default=TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND)
    parser.add_argument("--gpu-device", type=int, default=DEFAULT_GPU_DEVICE_ID)
    parser.add_argument("--gpu-precision", type=str, default=TRAIN_EVAL_DEFAULT_GPU_PRECISION)
    parser.add_argument("--gpu-warmup", type=int, default=1)
    parser.add_argument("--gpu-cache-policy", type=str, default=DEFAULT_GPU_CACHE_MODE)
    parser.add_argument("--remote-mode", type=int, default=1)
    parser.add_argument("--enable-local-fallback", type=int, default=0)
    parser.add_argument("--server-host", type=str, default=DEFAULT_SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--server-username", type=str, default=DEFAULT_SERVER_USERNAME)
    parser.add_argument("--server-password", type=str, default="")
    parser.add_argument("--quick-mode", type=int, default=0)
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TRAIN_EVAL_TASK,
        choices=["fbcca-weights", "model-compare", "fbcca-weighted-compare"],
    )
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(int(args.quick_mode)) or _parse_task(args.task) == "fbcca-weights":
        _apply_quick_mode_args(args)
    elif _parse_task(args.task) == "fbcca-weighted-compare":
        _apply_weighted_compare_args(args)
    elif _parse_task(args.task) == "model-compare":
        _apply_model_compare_args(args)
    include_manifests = _parse_manifest_csv(args.include_manifests)
    if bool(args.headless):
        s1 = include_manifests[0] if include_manifests else args.dataset_manifest
        if s1 is None:
            raise ValueError("--dataset-manifest or --include-manifests is required in --headless mode")
        config = OfflineTrainEvalConfig(
            dataset_manifest_session1=Path(s1).expanduser().resolve(),
            dataset_manifest_session2=(None if args.dataset_manifest_session2 is None else Path(args.dataset_manifest_session2).expanduser().resolve()),
            dataset_manifests=include_manifests,
            output_profile_path=Path(args.output_profile).expanduser().resolve(),
            report_path=Path(args.report_path).expanduser().resolve(),
            report_root_dir=Path(args.report_root_dir).expanduser().resolve(),
            organize_report_dir=bool(int(args.organize_report_dir)),
            dataset_selection_snapshot={
                "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
                "selected_manifests": [str(path) for path in include_manifests],
                "selected_manifest_count": int(len(include_manifests)),
                "quality_min_sample_ratio": float(args.quality_min_sample_ratio),
                "quality_max_retry_count": int(args.quality_max_retry_count),
                "strict_protocol_consistency": bool(int(args.strict_protocol_consistency)),
                "strict_subject_consistency": bool(int(args.strict_subject_consistency)),
            },
            quality_min_sample_ratio=float(args.quality_min_sample_ratio),
            quality_max_retry_count=int(args.quality_max_retry_count),
            strict_protocol_consistency=bool(int(args.strict_protocol_consistency)),
            strict_subject_consistency=bool(int(args.strict_subject_consistency)),
            model_names=tuple(parse_model_list(args.models)),
            channel_modes=tuple(parse_channel_mode_list(args.channel_modes)),
            multi_seed_count=int(args.multi_seed_count),
            gate_policy=parse_gate_policy(args.gate_policy),
            channel_weight_mode=(None if str(args.channel_weight_mode).strip() == "" else str(args.channel_weight_mode).strip()),
            subband_weight_mode=parse_subband_weight_mode(str(args.subband_weight_mode).strip()),
            spatial_filter_mode=parse_spatial_filter_mode(str(args.spatial_filter_mode).strip()),
            spatial_rank_candidates=tuple(parse_spatial_rank_candidates(str(args.spatial_rank_candidates))),
            joint_weight_iters=max(1, int(args.joint_weight_iters)),
            weight_cv_folds=max(2, int(args.weight_cv_folds)),
            spatial_source_model=parse_spatial_source_model(str(args.spatial_source_model).strip()),
            metric_scope=parse_metric_scope(args.metric_scope),
            decision_time_mode=parse_decision_time_mode(args.decision_time_mode),
            async_decision_time_mode=parse_decision_time_mode(args.async_decision_time_mode),
            data_policy=parse_data_policy(args.data_policy),
            export_figures=bool(int(args.export_figures)),
            ranking_policy=parse_ranking_policy(args.ranking_policy),
            dynamic_stop_enabled=bool(int(args.dynamic_stop_enabled)),
            dynamic_stop_alpha=float(args.dynamic_stop_alpha),
            win_candidates=tuple(float(item.strip()) for item in str(args.win_candidates).split(",") if item.strip()),
            seed=int(args.seed),
            task=_parse_task(args.task),
            evaluation_mode=str(args.evaluation_mode),
            quick_screen_top_k=max(1, int(args.quick_screen_top_k)),
            force_include_models=tuple(parse_model_list(str(args.force_include_models))),
            progress_heartbeat_sec=float(args.progress_heartbeat_sec),
            compute_backend=parse_compute_backend_name(str(args.compute_backend).strip()),
            gpu_device=int(args.gpu_device),
            gpu_precision=parse_gpu_precision(str(args.gpu_precision).strip()),
            gpu_warmup=bool(int(args.gpu_warmup)),
            gpu_cache_policy=parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()),
        )
        run_offline_train_eval(config, log_fn=lambda text: print(text, flush=True))
        return 0

    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = TrainingEvaluationWindow()
    window.dataset_root_edit.setText(str(Path(args.dataset_root).expanduser().resolve()))
    if args.dataset_manifest is not None:
        window.session1_edit.setText(str(Path(args.dataset_manifest).expanduser().resolve()))
    if args.dataset_manifest_session2 is not None:
        window.session2_edit.setText(str(Path(args.dataset_manifest_session2).expanduser().resolve()))
    window.output_profile_edit.setText(str(Path(args.output_profile).expanduser().resolve()))
    window.report_edit.setText(str(Path(args.report_path).expanduser().resolve()))
    window.report_root_edit.setText(str(Path(args.report_root_dir).expanduser().resolve()))
    window.organize_report_edit.setText("1" if bool(int(args.organize_report_dir)) else "0")
    window.quality_min_ratio_edit.setText(f"{float(args.quality_min_sample_ratio):g}")
    window.quality_max_retry_spin.setValue(max(0, int(args.quality_max_retry_count)))
    window.strict_protocol_edit.setText("1" if bool(int(args.strict_protocol_consistency)) else "0")
    window.strict_subject_edit.setText("1" if bool(int(args.strict_subject_consistency)) else "0")
    window.models_edit.setText(str(args.models))
    window.channel_modes_edit.setText(str(args.channel_modes))
    window.multi_seed_spin.setValue(int(args.multi_seed_count))
    window.gate_policy_edit.setText(str(args.gate_policy))
    window.weight_mode_edit.setText(str(args.channel_weight_mode))
    window.subband_weight_mode_edit.setText(str(args.subband_weight_mode))
    window.spatial_mode_edit.setText(str(args.spatial_filter_mode))
    window.spatial_ranks_edit.setText(str(args.spatial_rank_candidates))
    window.joint_iters_edit.setText(str(int(args.joint_weight_iters)))
    window.weight_cv_folds_edit.setText(str(int(args.weight_cv_folds)))
    window.spatial_source_edit.setText(str(args.spatial_source_model))
    window.metric_scope_edit.setText(str(args.metric_scope))
    window.decision_time_mode_edit.setText(str(args.decision_time_mode))
    window.async_decision_time_mode_edit.setText(str(args.async_decision_time_mode))
    window.data_policy_edit.setText(str(args.data_policy))
    window.export_figures_edit.setText("1" if bool(int(args.export_figures)) else "0")
    window.ranking_policy_edit.setText(str(args.ranking_policy))
    window.dynamic_stop_edit.setText("1" if bool(int(args.dynamic_stop_enabled)) else "0")
    window.dynamic_alpha_edit.setText(f"{float(args.dynamic_stop_alpha):g}")
    window.win_candidates_edit.setText(str(args.win_candidates))
    window.seed_edit.setText(str(int(args.seed)))
    window._evaluation_mode = str(args.evaluation_mode)
    window._quick_screen_top_k = max(1, int(args.quick_screen_top_k))
    window._force_include_models = tuple(parse_model_list(str(args.force_include_models)))
    window._progress_heartbeat_sec = float(args.progress_heartbeat_sec)
    window._simple_mode_variant = "quick" if bool(int(args.quick_mode)) else "standard"
    window._task = _parse_task(args.task)
    window.compute_backend_combo.setCurrentText(parse_compute_backend_name(str(args.compute_backend).strip()))
    window.gpu_device_edit.setText(str(int(args.gpu_device)))
    window.gpu_precision_combo.setCurrentText(parse_gpu_precision(str(args.gpu_precision).strip()))
    window.gpu_warmup_edit.setText("1" if bool(int(args.gpu_warmup)) else "0")
    window.gpu_cache_combo.setCurrentText(parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()))
    window.remote_mode_check.setChecked(bool(int(args.remote_mode)))
    window.allow_local_mode_check.setChecked(bool(int(args.enable_local_fallback)))
    window.server_host_edit.setText(str(args.server_host).strip() or DEFAULT_SERVER_HOST)
    window.server_port_edit.setText(str(int(args.server_port)))
    window.server_username_edit.setText(str(args.server_username).strip() or DEFAULT_SERVER_USERNAME)
    window.server_password_edit.setText(str(args.server_password or os.environ.get("SSVEP_SERVER_PASSWORD", "")))
    window._apply_simple_defaults(quick=bool(int(args.quick_mode)) or window._task == "fbcca-weights")
    if include_manifests:
        selected = {str(path) for path in include_manifests}
        for i in range(window.dataset_list.count()):
            item = window.dataset_list.item(i)
            if item is None:
                continue
            raw = str(item.data(Qt.UserRole) or "")
            if raw in selected:
                item.setCheckState(Qt.Checked)
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

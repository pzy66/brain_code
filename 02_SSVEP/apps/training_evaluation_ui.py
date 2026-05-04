from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, replace
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

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from brain_workspace.paths import HYBRID_SSVEP_PROFILE_DIR, SSVEP_DATASET_DIR, SSVEP_PROFILE_DIR
from ssvep_core.async_fbcca_idle_standalone import (
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_BENCHMARK_CHANNEL_MODES,
    DEFAULT_BENCHMARK_MULTI_SEED_COUNT,
    DEFAULT_CHANNEL_WEIGHT_MODE,
    DEFAULT_COMPUTE_BACKEND_NAME,
    DEFAULT_CONTROL_STATE_MODE,
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
from ssvep_core.run_artifacts import make_run_tag, resolve_ssvep_run_artifacts
from ssvep_core.tdca_local_opt import (
    DEFAULT_TDCA_LOCAL_SEARCH_PRESET,
    TDCA_LOCAL_SEARCH_PRESETS,
    TDCALocalOptConfig,
    run_tdca_local_opt,
)
from ssvep_core.fbcca_local_opt import (
    DEFAULT_FBCCA_LOCAL_SEARCH_PRESET,
    FBCCA_LOCAL_SEARCH_PRESETS,
    FBCCALocalOptConfig,
    run_fbcca_local_opt,
)
from ssvep_core.fbcca_threshold_pretrain import (
    DEFAULT_FBCCA_THRESHOLD_TASK,
    FBCCAThresholdPretrainConfig,
    run_fbcca_threshold_pretrain,
)
from ssvep_core.fbcca_external_replay_opt import (
    DEFAULT_FBCCA_EXTERNAL_DATASET_ROOT,
    DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL,
    DEFAULT_FBCCA_EXTERNAL_REPLAY_SPEED,
    DEFAULT_FBCCA_EXTERNAL_SEARCH_PRESET,
    FBCCA_EXTERNAL_OUTER_EVALS,
    FBCCA_EXTERNAL_REPLAY_SPEEDS,
    FBCCA_EXTERNAL_SEARCH_PRESETS,
    FBCCAExternalReplayOptConfig,
    run_fbcca_external_replay_opt,
)
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
from apps.external_replay_viewer import ExternalReplayViewer
from ssvep_core.registry import ModelRegistry
from tools.server_train_client import (
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
    sync_local_code_tree,
    upload_dataset,
)

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = PROJECT_DIR / "artifacts"
DEFAULT_LOCAL_RUN_ROOT = DEFAULT_ARTIFACT_ROOT / "runs" / "local"
DEFAULT_REPORT_DIR = DEFAULT_LOCAL_RUN_ROOT
DEFAULT_DATASET_ROOT = SSVEP_DATASET_DIR
DEFAULT_REPORT_ROOT = DEFAULT_LOCAL_RUN_ROOT
HYBRID_PROFILE_DIR = HYBRID_SSVEP_PROFILE_DIR
HYBRID_CURRENT_PROFILE_PATH = HYBRID_PROFILE_DIR / "current_fbcca_profile.json"
SSVEP_REALTIME_PROFILE_PATH = SSVEP_PROFILE_DIR / "fbcca_profile.json"
SSVEP_REALTIME_PROFILE_V2_PATH = SSVEP_PROFILE_DIR / "fbcca_profile_v2.json"
SSVEP_REALTIME_EXPECTED_FREQS = (8.0, 10.0, 12.0, 15.0)
TRAIN_EVAL_DEFAULT_COMPUTE_BACKEND = "cuda"
TRAIN_EVAL_DEFAULT_GPU_PRECISION = "float32"
TDCA_LOCAL_OPT_MODELS = ("tdca",)
TDCA_LOCAL_OPT_CHANNEL_MODES = ("all8",)
TDCA_LOCAL_OPT_MULTI_SEED_COUNT = 5
TDCA_LOCAL_OPT_WIN_CANDIDATES = (2.0, 2.5, 3.0, 3.5)
TDCA_LOCAL_OPT_COMPUTE_BACKEND = "cpu"
TDCA_LOCAL_OPT_SEARCH_PRESET = DEFAULT_TDCA_LOCAL_SEARCH_PRESET
FBCCA_LOCAL_OPT_MODELS = ("fbcca",)
FBCCA_LOCAL_OPT_CHANNEL_MODES = ("all8",)
FBCCA_LOCAL_OPT_MULTI_SEED_COUNT = 5
FBCCA_LOCAL_OPT_WIN_CANDIDATES = (2.0, 2.5, 3.0, 3.5)
FBCCA_LOCAL_OPT_COMPUTE_BACKEND = "auto"
FBCCA_LOCAL_OPT_SEARCH_PRESET = DEFAULT_FBCCA_LOCAL_SEARCH_PRESET
FBCCA_THRESHOLD_PRETRAIN_MODELS = ("fbcca_fixed_all8",)
FBCCA_THRESHOLD_PRETRAIN_CHANNEL_MODES = ("all8",)
FBCCA_THRESHOLD_PRETRAIN_MULTI_SEED_COUNT = 1
FBCCA_THRESHOLD_PRETRAIN_WIN_CANDIDATES = (3.0,)
FBCCA_THRESHOLD_PRETRAIN_COMPUTE_BACKEND = "cpu"
FBCCA_EXTERNAL_REPLAY_MODELS = ("fbcca",)
FBCCA_EXTERNAL_REPLAY_CHANNEL_MODES = ("all8",)
FBCCA_EXTERNAL_REPLAY_COMPUTE_BACKEND = "auto"
FBCCA_EXTERNAL_REPLAY_SEARCH_PRESET = DEFAULT_FBCCA_EXTERNAL_SEARCH_PRESET
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
        "tdca_local_opt": "tdca-local-opt",
        "tdca-local-opt": "tdca-local-opt",
        "tdca_local": "tdca-local-opt",
        "local_tdca": "tdca-local-opt",
        "fbcca_local_opt": "fbcca-local-opt",
        "fbcca-local-opt": "fbcca-local-opt",
        "local_fbcca": "fbcca-local-opt",
        "fbcca_threshold_pretrain": DEFAULT_FBCCA_THRESHOLD_TASK,
        "fbcca-threshold-pretrain": DEFAULT_FBCCA_THRESHOLD_TASK,
        "fbcca_threshold": DEFAULT_FBCCA_THRESHOLD_TASK,
        "threshold_pretrain": DEFAULT_FBCCA_THRESHOLD_TASK,
        "fbcca_external_replay_opt": "fbcca-external-replay-opt",
        "fbcca-external-replay-opt": "fbcca-external-replay-opt",
        "external_fbcca": "fbcca-external-replay-opt",
        "external-replay": "fbcca-external-replay-opt",
    }
    value = aliases.get(value, value)
    if value not in {
        "fbcca-weights",
        "model-compare",
        "fbcca-weighted-compare",
        "tdca-local-opt",
        "fbcca-local-opt",
        DEFAULT_FBCCA_THRESHOLD_TASK,
        "fbcca-external-replay-opt",
    }:
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


def _argv_has_flag(argv_tokens: Sequence[str], flag: str) -> bool:
    target = str(flag).strip().lower()
    for token in argv_tokens:
        text = str(token).strip()
        if not text:
            continue
        lower = text.lower()
        if lower == target or lower.startswith(target + "="):
            return True
    return False


def _apply_tdca_local_opt_args(args: argparse.Namespace, argv_tokens: Sequence[str]) -> None:
    args.task = "tdca-local-opt"
    if not _argv_has_flag(argv_tokens, "--models"):
        args.models = ",".join(TDCA_LOCAL_OPT_MODELS)
    if not _argv_has_flag(argv_tokens, "--channel-modes"):
        args.channel_modes = ",".join(TDCA_LOCAL_OPT_CHANNEL_MODES)
    if not _argv_has_flag(argv_tokens, "--multi-seed-count"):
        args.multi_seed_count = int(TDCA_LOCAL_OPT_MULTI_SEED_COUNT)
    if not _argv_has_flag(argv_tokens, "--win-candidates"):
        args.win_candidates = ",".join(f"{float(value):g}" for value in TDCA_LOCAL_OPT_WIN_CANDIDATES)
    if not _argv_has_flag(argv_tokens, "--compute-backend"):
        args.compute_backend = str(TDCA_LOCAL_OPT_COMPUTE_BACKEND)
    if not _argv_has_flag(argv_tokens, "--search-preset"):
        args.search_preset = str(TDCA_LOCAL_OPT_SEARCH_PRESET)


def _apply_fbcca_local_opt_args(args: argparse.Namespace, argv_tokens: Sequence[str]) -> None:
    args.task = "fbcca-local-opt"
    if not _argv_has_flag(argv_tokens, "--models"):
        args.models = ",".join(FBCCA_LOCAL_OPT_MODELS)
    if not _argv_has_flag(argv_tokens, "--channel-modes"):
        args.channel_modes = ",".join(FBCCA_LOCAL_OPT_CHANNEL_MODES)
    if not _argv_has_flag(argv_tokens, "--multi-seed-count"):
        args.multi_seed_count = int(FBCCA_LOCAL_OPT_MULTI_SEED_COUNT)
    if not _argv_has_flag(argv_tokens, "--win-candidates"):
        args.win_candidates = ",".join(f"{float(value):g}" for value in FBCCA_LOCAL_OPT_WIN_CANDIDATES)
    if not _argv_has_flag(argv_tokens, "--compute-backend"):
        args.compute_backend = str(FBCCA_LOCAL_OPT_COMPUTE_BACKEND)
    if not _argv_has_flag(argv_tokens, "--search-preset"):
        args.search_preset = str(FBCCA_LOCAL_OPT_SEARCH_PRESET)


def _apply_fbcca_threshold_pretrain_args(args: argparse.Namespace, argv_tokens: Sequence[str]) -> None:
    args.task = DEFAULT_FBCCA_THRESHOLD_TASK
    if not _argv_has_flag(argv_tokens, "--models"):
        args.models = ",".join(FBCCA_THRESHOLD_PRETRAIN_MODELS)
    if not _argv_has_flag(argv_tokens, "--channel-modes"):
        args.channel_modes = ",".join(FBCCA_THRESHOLD_PRETRAIN_CHANNEL_MODES)
    if not _argv_has_flag(argv_tokens, "--multi-seed-count"):
        args.multi_seed_count = int(FBCCA_THRESHOLD_PRETRAIN_MULTI_SEED_COUNT)
    if not _argv_has_flag(argv_tokens, "--win-candidates"):
        args.win_candidates = ",".join(f"{float(value):g}" for value in FBCCA_THRESHOLD_PRETRAIN_WIN_CANDIDATES)
    if not _argv_has_flag(argv_tokens, "--channel-weight-mode"):
        args.channel_weight_mode = "none"
    if not _argv_has_flag(argv_tokens, "--subband-weight-mode"):
        args.subband_weight_mode = "chen_fixed"
    if not _argv_has_flag(argv_tokens, "--spatial-filter-mode"):
        args.spatial_filter_mode = "none"
    if not _argv_has_flag(argv_tokens, "--dynamic-stop-enabled"):
        args.dynamic_stop_enabled = 0
    if not _argv_has_flag(argv_tokens, "--compute-backend"):
        args.compute_backend = str(FBCCA_THRESHOLD_PRETRAIN_COMPUTE_BACKEND)


def _apply_fbcca_external_replay_args(args: argparse.Namespace, argv_tokens: Sequence[str]) -> None:
    args.task = "fbcca-external-replay-opt"
    if not _argv_has_flag(argv_tokens, "--models"):
        args.models = ",".join(FBCCA_EXTERNAL_REPLAY_MODELS)
    if not _argv_has_flag(argv_tokens, "--channel-modes"):
        args.channel_modes = ",".join(FBCCA_EXTERNAL_REPLAY_CHANNEL_MODES)
    if not _argv_has_flag(argv_tokens, "--compute-backend"):
        args.compute_backend = str(FBCCA_EXTERNAL_REPLAY_COMPUTE_BACKEND)
    if not _argv_has_flag(argv_tokens, "--search-preset"):
        args.search_preset = str(FBCCA_EXTERNAL_REPLAY_SEARCH_PRESET)
    if not _argv_has_flag(argv_tokens, "--external-dataset-root"):
        args.external_dataset_root = Path(DEFAULT_FBCCA_EXTERNAL_DATASET_ROOT)
    if not _argv_has_flag(argv_tokens, "--outer-eval"):
        args.outer_eval = str(DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL)
    if not _argv_has_flag(argv_tokens, "--replay-speed"):
        args.replay_speed = str(DEFAULT_FBCCA_EXTERNAL_REPLAY_SPEED)
    if not _argv_has_flag(argv_tokens, "--output-profile"):
        args.output_profile = Path(DEFAULT_PROFILE_PATH).with_name("profile.json")


@dataclass(frozen=True)
class TrainEvalUIConfig:
    session1_manifest: Path
    session2_manifest: Optional[Path]
    dataset_manifests: tuple[Path, ...]
    dataset_root: Path
    external_dataset_root: Path
    external_subject: str
    external_outer_eval: str
    external_replay_speed: str
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
    tdca_search_preset: str
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
            task_name = str(self.config.task)
            if task_name == "tdca-local-opt":
                cfg = TDCALocalOptConfig(
                    dataset_manifest_session1=self.config.session1_manifest,
                    dataset_manifests=self.config.dataset_manifests,
                    output_profile_path=self.config.output_profile_path,
                    report_path=self.config.report_path,
                    report_root_dir=self.config.report_root_dir,
                    organize_report_dir=bool(self.config.organize_report_dir),
                    model_names=self.config.model_names,
                    channel_modes=self.config.channel_modes,
                    multi_seed_count=self.config.multi_seed_count,
                    win_candidates=self.config.win_candidates,
                    search_preset=str(self.config.tdca_search_preset),
                    seed=self.config.seed,
                    compute_backend=self.config.compute_backend,
                    gpu_device=self.config.gpu_device,
                    gpu_precision=self.config.gpu_precision,
                    gpu_warmup=bool(self.config.gpu_warmup),
                    gpu_cache_policy=self.config.gpu_cache_policy,
                    decision_time_mode=self.config.decision_time_mode,
                    async_decision_time_mode=self.config.async_decision_time_mode,
                    progress_heartbeat_sec=self.config.progress_heartbeat_sec,
                )
                self.done.emit(run_tdca_local_opt(cfg, log_fn=self.log.emit, progress_fn=self.progress.emit))
            elif task_name == "fbcca-local-opt":
                cfg = FBCCALocalOptConfig(
                    dataset_manifest_session1=self.config.session1_manifest,
                    dataset_manifests=self.config.dataset_manifests,
                    output_profile_path=self.config.output_profile_path,
                    report_path=self.config.report_path,
                    report_root_dir=self.config.report_root_dir,
                    organize_report_dir=bool(self.config.organize_report_dir),
                    model_names=self.config.model_names,
                    channel_modes=self.config.channel_modes,
                    multi_seed_count=self.config.multi_seed_count,
                    win_candidates=self.config.win_candidates,
                    search_preset=str(self.config.tdca_search_preset),
                    seed=self.config.seed,
                    compute_backend=self.config.compute_backend,
                    gpu_device=self.config.gpu_device,
                    gpu_precision=self.config.gpu_precision,
                    gpu_warmup=bool(self.config.gpu_warmup),
                    gpu_cache_policy=self.config.gpu_cache_policy,
                    decision_time_mode=self.config.decision_time_mode,
                    async_decision_time_mode=self.config.async_decision_time_mode,
                    progress_heartbeat_sec=self.config.progress_heartbeat_sec,
                )
                self.done.emit(run_fbcca_local_opt(cfg, log_fn=self.log.emit, progress_fn=self.progress.emit))
            elif task_name == DEFAULT_FBCCA_THRESHOLD_TASK:
                win_sec = 3.0
                if self.config.win_candidates:
                    win_sec = float(self.config.win_candidates[0])
                cfg = FBCCAThresholdPretrainConfig(
                    dataset_manifest_session1=self.config.session1_manifest,
                    dataset_manifests=self.config.dataset_manifests,
                    output_profile_path=self.config.output_profile_path,
                    report_path=self.config.report_path,
                    report_root_dir=self.config.report_root_dir,
                    organize_report_dir=bool(self.config.organize_report_dir),
                    win_sec=float(win_sec),
                    gate_policy=self.config.gate_policy,
                    dynamic_stop_enabled=False,
                    dynamic_stop_alpha=self.config.dynamic_stop_alpha,
                    seed=self.config.seed,
                    compute_backend=self.config.compute_backend,
                    gpu_device=self.config.gpu_device,
                    gpu_precision=self.config.gpu_precision,
                    gpu_warmup=bool(self.config.gpu_warmup),
                    gpu_cache_policy=self.config.gpu_cache_policy,
                    decision_time_mode=self.config.decision_time_mode,
                    async_decision_time_mode=self.config.async_decision_time_mode,
                    progress_heartbeat_sec=self.config.progress_heartbeat_sec,
                    publish_realtime=True,
                )
                self.done.emit(run_fbcca_threshold_pretrain(cfg, log_fn=self.log.emit, progress_fn=self.progress.emit))
            elif task_name == "fbcca-external-replay-opt":
                cfg = FBCCAExternalReplayOptConfig(
                    external_dataset_root=self.config.external_dataset_root,
                    subject=str(self.config.external_subject),
                    output_profile_path=self.config.output_profile_path,
                    report_path=self.config.report_path,
                    report_root_dir=self.config.report_root_dir,
                    organize_report_dir=bool(self.config.organize_report_dir),
                    model_names=self.config.model_names,
                    channel_modes=self.config.channel_modes,
                    search_preset=str(self.config.tdca_search_preset),
                    outer_eval=str(self.config.external_outer_eval),
                    replay_speed=str(self.config.external_replay_speed),
                    seed=self.config.seed,
                    compute_backend=self.config.compute_backend,
                    gpu_device=self.config.gpu_device,
                    gpu_precision=self.config.gpu_precision,
                    gpu_warmup=bool(self.config.gpu_warmup),
                    gpu_cache_policy=self.config.gpu_cache_policy,
                    decision_time_mode=self.config.decision_time_mode,
                    async_decision_time_mode=self.config.async_decision_time_mode,
                    progress_heartbeat_sec=self.config.progress_heartbeat_sec,
                )
                self.done.emit(run_fbcca_external_replay_opt(cfg, log_fn=self.log.emit, progress_fn=self.progress.emit))
            else:
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
                    task=task_name,
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
        self._last_profile_path: Optional[Path] = None
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
        self._tdca_search_preset = str(TDCA_LOCAL_OPT_SEARCH_PRESET)
        self._task = str(DEFAULT_TRAIN_EVAL_TASK)
        self._remote_record: Optional[dict[str, Any]] = None
        self._remote_status_timer = None
        self._local_monitor_timer: Optional[QTimer] = None
        self._local_monitor_run_dir: Optional[Path] = None
        self._local_monitor_progress_path: Optional[Path] = None
        self._local_monitor_log_path: Optional[Path] = None
        self._local_monitor_last_log_text = ""
        self._external_viewers: list[ExternalReplayViewer] = []

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
        self.btn_fbcca_threshold_pretrain_run = QPushButton("FBCCA Threshold Pretrain (Fast)")
        self.btn_fbcca_local_opt_run = QPushButton("Run FBCCA Async Local Optimization")
        self.btn_fbcca_external_replay_run = QPushButton("Run FBCCA External Replay Optimization")
        self.btn_tdca_local_opt_run = QPushButton("Run TDCA Async Local Optimization")
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
        quick_row.addWidget(self.btn_fbcca_threshold_pretrain_run)
        quick_row.addWidget(self.btn_fbcca_local_opt_run)
        quick_row.addWidget(self.btn_fbcca_external_replay_run)
        quick_row.addWidget(self.btn_tdca_local_opt_run)
        quick_row.addWidget(self.btn_toggle_advanced)
        quick_row.addStretch(1)
        self._layout.addLayout(quick_row)

        form = QFormLayout()
        self._form_layout = form
        self.dataset_root_edit = QLineEdit(str(DEFAULT_DATASET_ROOT))
        self.external_dataset_root_edit = QLineEdit(str(DEFAULT_FBCCA_EXTERNAL_DATASET_ROOT))
        self.external_subject_edit = QLineEdit("")
        self.external_outer_eval_combo = QComboBox()
        self.external_outer_eval_combo.addItems(list(FBCCA_EXTERNAL_OUTER_EVALS))
        self.external_outer_eval_combo.setCurrentText(str(DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL))
        self.external_replay_speed_combo = QComboBox()
        self.external_replay_speed_combo.addItems(list(FBCCA_EXTERNAL_REPLAY_SPEEDS))
        self.external_replay_speed_combo.setCurrentText(str(DEFAULT_FBCCA_EXTERNAL_REPLAY_SPEED))
        self.session1_edit = QLineEdit("")
        self.session2_edit = QLineEdit("")
        self.output_profile_edit = QLineEdit(str(DEFAULT_PROFILE_PATH))
        self.report_edit = QLineEdit(str(DEFAULT_REPORT_DIR / "report.json"))
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
        form.addRow("External Dataset Root", self.external_dataset_root_edit)
        form.addRow("External Subject", self.external_subject_edit)
        form.addRow("External Outer Eval", self.external_outer_eval_combo)
        form.addRow("External Replay Speed", self.external_replay_speed_combo)
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
        self.btn_open_profile = QPushButton("Open Profile")
        self.btn_publish_realtime_profile = QPushButton("发布到实时识别")
        self.btn_publish_hybrid_profile = QPushButton("发布到集成控制器")
        self.btn_open_figures_dir = QPushButton("鎵撳紑鍥捐〃鐩綍")
        self.btn_open_replay_viewer = QPushButton("Open Replay Viewer")
        self.btn_open_report_dir.setEnabled(False)
        self.btn_open_profile.setEnabled(False)
        self.btn_publish_realtime_profile.setEnabled(False)
        self.btn_publish_hybrid_profile.setEnabled(False)
        self.btn_open_figures_dir.setEnabled(False)
        self.btn_open_replay_viewer.setEnabled(False)
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
            self.btn_open_profile,
            self.btn_publish_realtime_profile,
            self.btn_publish_hybrid_profile,
            self.btn_open_figures_dir,
            self.btn_open_replay_viewer,
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

        self.current_task_label = QLabel("当前任务：未开始")
        self.current_run_dir_label = QLabel("运行目录：未开始")
        self.current_report_label = QLabel("报告文件：未开始")
        self.current_profile_label = QLabel("Profile：未开始")
        for label in (
            self.current_task_label,
            self.current_run_dir_label,
            self.current_report_label,
            self.current_profile_label,
        ):
            label.setWordWrap(True)
            self._layout.addWidget(label)

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
        self.btn_open_profile.clicked.connect(self._open_profile_path)
        self.btn_publish_realtime_profile.clicked.connect(self._publish_profile_to_ssvep_realtime)
        self.btn_publish_hybrid_profile.clicked.connect(self._publish_profile_to_hybrid_controller)
        self.btn_open_figures_dir.clicked.connect(self._open_figures_dir)
        self.btn_quick_run.clicked.connect(self._quick_auto_run)
        self.btn_weighted_compare_run.clicked.connect(self._weighted_compare_run)
        self.btn_model_compare_run.clicked.connect(self._model_compare_run)
        self.btn_fbcca_threshold_pretrain_run.clicked.connect(self._fbcca_threshold_pretrain_run)
        self.btn_fbcca_local_opt_run.clicked.connect(self._fbcca_local_opt_run)
        self.btn_fbcca_external_replay_run.clicked.connect(self._fbcca_external_replay_run)
        self.btn_tdca_local_opt_run.clicked.connect(self._tdca_local_opt_run)
        self.btn_open_replay_viewer.clicked.connect(self._open_replay_viewer)
        self.simple_mode_check.toggled.connect(self._on_simple_mode_toggled)
        self.btn_toggle_advanced.clicked.connect(self._toggle_advanced)
        self._remote_status_timer = QTimer(self)
        self._remote_status_timer.setInterval(int(DEFAULT_REMOTE_POLL_INTERVAL_MS))
        self._remote_status_timer.timeout.connect(self._poll_remote_status)
        self._local_monitor_timer = QTimer(self)
        self._local_monitor_timer.setInterval(1000)
        self._local_monitor_timer.timeout.connect(self._poll_local_run_status)

        self._advanced_widgets = [
            self.session1_edit,
            self.session2_edit,
            self.external_dataset_root_edit,
            self.external_subject_edit,
            self.external_outer_eval_combo,
            self.external_replay_speed_combo,
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
        self.btn_fbcca_threshold_pretrain_run.setText("FBCCA 阈值快速预训练")
        self.btn_fbcca_local_opt_run.setText("FBCCA 本地异步优化")
        self.btn_fbcca_external_replay_run.setText("外部数据 FBCCA 回放优化")
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
        self.btn_open_replay_viewer.setText("打开回放 Viewer")

        self.dataset_list_title.setText("可选会话")
        self.status_label.setText("空闲")
        self.progress_detail_label.setText("当前阶段：未开始")
        self.eta_label.setText("预计剩余：-")

        self._set_form_label_text(self.dataset_root_edit, "数据集根目录")
        self._set_form_label_text(self.external_dataset_root_edit, "外部数据集根目录")
        self._set_form_label_text(self.external_subject_edit, "外部 Subject")
        self._set_form_label_text(self.external_outer_eval_combo, "外部 Outer Eval")
        self._set_form_label_text(self.external_replay_speed_combo, "外部 Replay Speed")
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

    @staticmethod
    def _read_json_payload(path: Path) -> dict[str, Any]:
        return dict(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _tail_text_file(path: Path, *, max_lines: int = 120) -> str:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-max(int(max_lines), 1) :]).rstrip()

    def _stop_local_run_monitor(self) -> None:
        if self._local_monitor_timer is not None:
            self._local_monitor_timer.stop()
        self._local_monitor_run_dir = None
        self._local_monitor_progress_path = None
        self._local_monitor_log_path = None
        self._local_monitor_last_log_text = ""

    def attach_local_run_monitor(self, run_dir: Path) -> None:
        resolved_dir = Path(run_dir).expanduser().resolve()
        progress_path = resolved_dir / "progress_snapshot.json"
        log_path = resolved_dir / "run.log"
        if not progress_path.exists():
            raise FileNotFoundError(f"progress snapshot not found: {progress_path}")
        self._stop_local_run_monitor()
        self._local_monitor_run_dir = resolved_dir
        self._local_monitor_progress_path = progress_path
        self._local_monitor_log_path = log_path
        self._set_running(True)
        self._log(f"开始监控本地运行：{resolved_dir}")
        self._poll_local_run_status()
        if self._local_monitor_timer is not None:
            self._local_monitor_timer.start()

    def _poll_local_run_status(self) -> None:
        progress_path = self._local_monitor_progress_path
        run_dir = self._local_monitor_run_dir
        if progress_path is None or run_dir is None:
            return
        if not progress_path.exists():
            self._log(f"本地监控丢失 progress_snapshot：{progress_path}")
            self._stop_local_run_monitor()
            self._set_running(False)
            return
        try:
            payload = self._read_json_payload(progress_path)
        except Exception as exc:
            self._log(f"读取本地进度失败：{exc}")
            return
        payload.setdefault("task", str(self._task))
        payload.setdefault("report_dir", str(run_dir))
        if payload.get("stage"):
            payload["stage_label"] = ""
        self._on_progress(payload)
        log_path = self._local_monitor_log_path
        if log_path is not None and log_path.exists():
            try:
                log_text = self._tail_text_file(log_path, max_lines=120)
            except Exception as exc:
                self._log(f"读取本地日志失败：{exc}")
            else:
                if log_text != self._local_monitor_last_log_text:
                    self._local_monitor_last_log_text = log_text
                    self.log_text.setPlainText(log_text)
        stage = str(payload.get("stage", "") or "").strip().lower()
        if stage == "complete":
            self._stop_local_run_monitor()
            self._set_running(False)

    def _on_progress(self, payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage", "") or "")
        stage_label = str(payload.get("stage_label", "") or "")
        detail_text = str(payload.get("detail", "") or "")
        model_name = str(payload.get("model_name", "") or "")
        run_index = int(payload.get("run_index", 0) or 0)
        run_total = int(payload.get("run_total", 0) or 0)
        config_index = int(payload.get("config_index", 0) or 0)
        config_total = int(payload.get("config_total", 0) or 0)
        elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
        eta_s = payload.get("eta_s", None)
        percent_payload = payload.get("progress_percent", None)
        if isinstance(percent_payload, (int, float)):
            percent = max(0, min(100, int(float(percent_payload))))
        elif stage == "stage_a":
            percent = 0 if run_total <= 0 else min(20, int(20.0 * run_index / max(run_total, 1)))
        elif stage == "stage_b":
            percent = 20 if run_total <= 0 else min(95, 20 + int(75.0 * run_index / max(run_total, 1)))
        elif stage == "complete":
            percent = 100
        else:
            percent = 0
        self.progress_bar.setValue(percent)
        stage_label = stage_label or {
            "prepare": "准备",
            "stage_a": "阶段A：快速筛选",
            "stage_b": "阶段B：完整评测",
            "baseline_opening": "基线开场对比",
            "candidate_search": "TDCA 候选搜索",
            "decision_search": "异步决策搜索",
            "baseline_seal": "基线封板对比",
            "finalize": "保存产物",
            "complete": "完成",
        }.get(stage, stage or "未知")
        detail = f"当前阶段：{stage_label}"
        if detail_text:
            detail += f" | {detail_text}"
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
        self._set_current_artifacts(
            task_name=str(payload.get("task", self._task)),
            report_path=payload.get("report_path"),
            profile_path=payload.get("profile_path"),
            report_dir=payload.get("report_dir"),
        )

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

    def _open_profile_path(self) -> None:
        self._open_path(self._last_profile_path)

    def _resolve_publish_profile_source(self) -> Optional[Path]:
        source = getattr(self, "_last_profile_path", None)
        if source is None:
            edit = getattr(self, "output_profile_edit", None)
            raw = str(edit.text()).strip() if edit is not None else ""
            source = Path(raw).expanduser().resolve() if raw else None
        return Path(source).expanduser().resolve() if source is not None else None

    @staticmethod
    def _validate_fbcca_realtime_profile(source: Path) -> None:
        payload = json.loads(Path(source).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("profile JSON 不是对象")
        model_name = normalize_model_name(str(payload.get("model_name", "")))
        freqs = tuple(float(item) for item in payload.get("freqs", ()))
        if "fbcca" not in model_name:
            raise ValueError(f"profile 不是 FBCCA: model_name={model_name or '<missing>'}")
        expected = tuple(float(item) for item in SSVEP_REALTIME_EXPECTED_FREQS)
        if len(freqs) != len(expected) or any(abs(left - right) > 1e-6 for left, right in zip(freqs, expected)):
            raise ValueError(f"profile 频率不匹配，期望 8/10/12/15Hz，实际 {freqs}")

    def _publish_profile_to_ssvep_realtime(self) -> None:
        source = TrainingEvaluationWindow._resolve_publish_profile_source(self)
        if source is None or not source.exists():
            self._log("没有可发布到实时识别的 profile。请先完成 FBCCA 训练，或在输出 Profile 中选择已有文件。")
            return
        try:
            TrainingEvaluationWindow._validate_fbcca_realtime_profile(source)
            SSVEP_REALTIME_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != SSVEP_REALTIME_PROFILE_PATH.resolve():
                shutil.copy2(source, SSVEP_REALTIME_PROFILE_PATH)
            source_v2 = source.with_name(f"{source.stem}_v2.json")
            copied_v2 = False
            if source_v2.exists():
                if source_v2.resolve() != SSVEP_REALTIME_PROFILE_V2_PATH.resolve():
                    shutil.copy2(source_v2, SSVEP_REALTIME_PROFILE_V2_PATH)
                copied_v2 = True
        except Exception as exc:
            self._log(f"发布到实时识别失败: {exc}")
            return
        v2_text = f" | v2={SSVEP_REALTIME_PROFILE_V2_PATH}" if copied_v2 else ""
        self._log(f"已发布到 SSVEP 实时识别: {SSVEP_REALTIME_PROFILE_PATH}{v2_text}")

    def _publish_profile_to_hybrid_controller(self) -> None:
        source = TrainingEvaluationWindow._resolve_publish_profile_source(self)
        if source is None or not source.exists():
            self._log("没有可发布的 profile。请先完成训练，或在输出 Profile 中选择已有文件。")
            return
        try:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            model_name = str(payload.get("model_name", "")).strip().lower()
            freqs = tuple(float(item) for item in payload.get("freqs", ()))
            if "fbcca" not in model_name:
                raise ValueError(f"profile 不是 FBCCA: model_name={model_name or '<missing>'}")
            if len(freqs) != 4 or any(abs(left - right) > 1e-6 for left, right in zip(freqs, (8.0, 10.0, 12.0, 15.0))):
                raise ValueError(f"profile 频率不匹配，期望 8/10/12/15Hz，实际 {freqs}")
            stamp = _now_stamp()
            HYBRID_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            history_path = HYBRID_PROFILE_DIR / f"ssvep_fbcca_profile_{stamp}.json"
            source_resolved = source.resolve()
            if source_resolved != history_path.resolve():
                shutil.copy2(source, history_path)
            if source_resolved != HYBRID_CURRENT_PROFILE_PATH.resolve():
                shutil.copy2(source, HYBRID_CURRENT_PROFILE_PATH)
        except Exception as exc:
            self._log(f"发布到集成控制器失败: {exc}")
            return
        self._log(
            "已发布到集成控制器: "
            f"current={HYBRID_CURRENT_PROFILE_PATH} | history={history_path}"
        )

    def _open_figures_dir(self) -> None:
        self._open_path(self._last_figures_dir)

    def _open_replay_viewer(self) -> None:
        if self._last_report_path is None or not self._last_report_path.exists():
            self._log("回放 viewer 需要先生成 report.json。")
            return
        try:
            viewer = ExternalReplayViewer(self._last_report_path)
        except Exception as exc:
            self._log(f"打开回放 viewer 失败: {exc}")
            return
        viewer.show()
        self._external_viewers.append(viewer)

    def _set_current_artifacts(
        self,
        *,
        task_name: Optional[str] = None,
        report_path: Any = None,
        profile_path: Any = None,
        report_dir: Any = None,
    ) -> None:
        task_value = str(task_name or self._task or "").strip() or "unknown"
        report_path_obj = None if not report_path else Path(str(report_path)).expanduser().resolve()
        profile_path_obj = None if not profile_path else Path(str(profile_path)).expanduser().resolve()
        if report_path_obj is not None:
            self._last_report_path = report_path_obj
        if profile_path_obj is not None:
            self._last_profile_path = profile_path_obj
        if report_dir:
            run_dir = Path(str(report_dir)).expanduser().resolve()
        elif report_path_obj is not None:
            run_dir = report_path_obj.parent
        else:
            run_dir = None
        self.current_task_label.setText(f"当前任务：{task_value}")
        self.current_run_dir_label.setText(f"运行目录：{run_dir if run_dir is not None else '未开始'}")
        self.current_report_label.setText(f"报告文件：{report_path_obj if report_path_obj is not None else '未开始'}")
        self.current_profile_label.setText(f"Profile：{profile_path_obj if profile_path_obj is not None else '未开始'}")
        self.btn_open_report_dir.setEnabled(run_dir is not None and run_dir.exists())
        self.btn_open_profile.setEnabled(profile_path_obj is not None and profile_path_obj.exists())
        self.btn_publish_realtime_profile.setEnabled(profile_path_obj is not None and profile_path_obj.exists())
        self.btn_publish_hybrid_profile.setEnabled(profile_path_obj is not None and profile_path_obj.exists())
        self.btn_open_replay_viewer.setEnabled(self._last_report_path is not None and self._last_report_path.exists())

    def _resolve_local_artifacts(self, *, task_name: str) -> dict[str, Path]:
        task_token = str(task_name or self._task or DEFAULT_TRAIN_EVAL_TASK)
        report_stub = DEFAULT_LOCAL_RUN_ROOT / "report.json"
        profile_stub = SSVEP_PROFILE_DIR / "profile.json"
        artifacts = resolve_ssvep_run_artifacts(
            task=task_token,
            report_path=report_stub,
            output_profile_path=profile_stub,
            organize_report_dir=True,
            report_root_dir=DEFAULT_LOCAL_RUN_ROOT,
            run_tag=make_run_tag(task=task_token),
        )
        payload = artifacts.to_payload()
        return {key: Path(value).expanduser().resolve() for key, value in payload.items() if key not in {"task", "run_tag"}}

    def _apply_local_artifact_layout(self, cfg: TrainEvalUIConfig) -> TrainEvalUIConfig:
        artifacts = self._resolve_local_artifacts(task_name=str(cfg.task))
        self.output_profile_edit.setText(str(artifacts["output_profile"]))
        self.report_edit.setText(str(artifacts["report_json"]))
        self.report_root_edit.setText(str(artifacts["root_dir"]))
        self.organize_report_edit.setText("1")
        self._set_current_artifacts(
            task_name=str(cfg.task),
            report_path=artifacts["report_json"],
            profile_path=artifacts["output_profile"],
            report_dir=artifacts["run_dir"],
        )
        return replace(
            cfg,
            output_profile_path=artifacts["output_profile"],
            report_path=artifacts["report_json"],
            report_root_dir=artifacts["root_dir"],
            organize_report_dir=True,
        )

    def _read_config(self) -> TrainEvalUIConfig:
        if str(self._task) == "fbcca-external-replay-opt":
            external_dataset_root = Path(self.external_dataset_root_edit.text().strip()).expanduser().resolve()
            subject = str(self.external_subject_edit.text().strip())
            if not subject:
                raise ValueError("请填写 External Subject")
            return TrainEvalUIConfig(
                session1_manifest=external_dataset_root,
                session2_manifest=None,
                dataset_manifests=tuple(),
                dataset_root=Path(self.dataset_root_edit.text().strip()).expanduser().resolve(),
                external_dataset_root=external_dataset_root,
                external_subject=subject,
                external_outer_eval=str(self.external_outer_eval_combo.currentText()).strip().lower(),
                external_replay_speed=str(self.external_replay_speed_combo.currentText()).strip().lower(),
                dataset_selection_snapshot={
                    "external_dataset_root": str(external_dataset_root),
                    "subject": str(subject),
                    "outer_eval": str(self.external_outer_eval_combo.currentText()).strip().lower(),
                    "replay_speed": str(self.external_replay_speed_combo.currentText()).strip().lower(),
                },
                quality_min_sample_ratio=float(self.quality_min_ratio_edit.text().strip() or "0.90"),
                quality_max_retry_count=int(self.quality_max_retry_spin.value()),
                strict_protocol_consistency=bool(int(self.strict_protocol_edit.text().strip() or "1")),
                strict_subject_consistency=bool(int(self.strict_subject_edit.text().strip() or "1")),
                output_profile_path=Path(self.output_profile_edit.text().strip()).expanduser().resolve(),
                report_path=Path(self.report_edit.text().strip()).expanduser().resolve(),
                report_root_dir=Path(self.report_root_edit.text().strip()).expanduser().resolve(),
                organize_report_dir=bool(int(self.organize_report_edit.text().strip() or "1")),
                model_names=tuple(parse_model_list(self.models_edit.text().strip())),
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
                dynamic_stop_enabled=False,
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
                tdca_search_preset=str(getattr(self, "_tdca_search_preset", FBCCA_EXTERNAL_REPLAY_SEARCH_PRESET)).strip().lower(),
                task=str(self._task),
            )
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
        if self.keep_baseline_group_check.isChecked() and str(self._task) not in {
            "tdca-local-opt",
            "fbcca-local-opt",
            DEFAULT_FBCCA_THRESHOLD_TASK,
            "fbcca-external-replay-opt",
        }:
            for model_name in BASELINE_COMPARE_MODELS:
                if model_name not in requested_models:
                    requested_models.append(str(model_name))
        return TrainEvalUIConfig(
            session1_manifest=session1_manifest,
            session2_manifest=session2_manifest,
            dataset_manifests=selected,
            dataset_root=Path(self.dataset_root_edit.text().strip()).expanduser().resolve(),
            external_dataset_root=Path(self.external_dataset_root_edit.text().strip()).expanduser().resolve(),
            external_subject=str(self.external_subject_edit.text().strip()),
            external_outer_eval=str(self.external_outer_eval_combo.currentText()).strip().lower(),
            external_replay_speed=str(self.external_replay_speed_combo.currentText()).strip().lower(),
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
            tdca_search_preset=str(getattr(self, "_tdca_search_preset", TDCA_LOCAL_OPT_SEARCH_PRESET)).strip().lower(),
            task=str(self._task),
        )

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        if running:
            self.btn_open_report_dir.setEnabled(False)
            self.btn_publish_realtime_profile.setEnabled(False)
            self.btn_open_figures_dir.setEnabled(False)
            self.btn_open_replay_viewer.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_detail_label.setText("褰撳墠闃舵锛氬噯澶囦腑")
            self.eta_label.setText("棰勮鍓╀綑锛?-")

    def _start_run(self) -> None:
        if self.worker_thread is not None:
            return
        use_remote = bool(self.remote_mode_check.isChecked())
        if str(self._task) in {"tdca-local-opt", "fbcca-local-opt", DEFAULT_FBCCA_THRESHOLD_TASK, "fbcca-external-replay-opt"}:
            use_remote = False
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
        self._stop_local_run_monitor()
        cfg = self._apply_local_artifact_layout(cfg)
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
            ssh = SSHClient(server_cfg, log_fn=self._log)
            ssh.connect()
            try:
                self._log("正在同步本地 02_SSVEP 代码到服务器...")
                code_sync = sync_local_code_tree(ssh)
                self._log(
                    "代码同步完成: "
                    f"hash={code_sync.get('tree_hash','')} "
                    f"files={code_sync.get('file_count', 0)} "
                    f"uploaded={code_sync.get('uploaded_count', 0)} "
                    f"removed={code_sync.get('removed_count', 0)}"
                )
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
                        "code_sync": code_sync,
                        "requested_config": {
                            "task": str(task_name),
                            "model_names": [str(name) for name in cfg.model_names],
                            "channel_modes": [str(name) for name in cfg.channel_modes],
                            "multi_seed_count": int(cfg.multi_seed_count),
                            "win_candidates": [float(item) for item in cfg.win_candidates],
                            "compute_backend": str(gpu_params["compute_backend"]),
                            "ranking_policy": str(cfg.ranking_policy),
                        },
                        "metrics_source": "no_session2" if cfg.session2_manifest is None else "cross_session",
                    },
                )
            finally:
                ssh.close()
        except Exception as exc:
            self._set_running(False)
            self.status_label.setText("远端任务提交失败")
            message = str(exc)
            if "SSH protocol banner" in message:
                message = (
                    "SSH 握手失败：服务器没有在超时内返回 banner。"
                    "这通常是网络抖动、SSH 服务繁忙或端口未就绪导致。"
                    f"\n\n原始错误：{exc}"
                )
            self._log(f"远端任务提交失败: {message}")
            QMessageBox.critical(self, "Remote Submit Failed", message)
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
            ssh = SSHClient(server_cfg, log_fn=self._log)
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
            ssh = SSHClient(server_cfg, log_fn=self._log)
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
        report_path = local_run_dir / "report.json"
        if report_path.exists():
            self._last_report_path = report_path
        local_profile = str(result.get("local_profile", "")).strip()
        if local_profile:
            profile_path = Path(local_profile).expanduser().resolve()
            if profile_path.exists():
                self._last_profile_path = profile_path
        figures_dir = local_run_dir / "figures"
        self._last_figures_dir = figures_dir if figures_dir.exists() else None
        self.btn_open_report_dir.setEnabled(local_run_dir.exists())
        self.btn_open_profile.setEnabled(self._last_profile_path is not None and self._last_profile_path.exists())
        self.btn_publish_realtime_profile.setEnabled(
            self._last_profile_path is not None and self._last_profile_path.exists()
        )
        self.btn_publish_hybrid_profile.setEnabled(
            self._last_profile_path is not None and self._last_profile_path.exists()
        )
        self.btn_open_figures_dir.setEnabled(self._last_figures_dir is not None)
        self.btn_open_replay_viewer.setEnabled(self._last_report_path is not None and self._last_report_path.exists())
        self.progress_bar.setValue(100)
        self.progress_detail_label.setText("远端任务完成")
        self.status_label.setText("远端训练评测完成")
        self._set_running(False)
        self._set_current_artifacts(
            task_name=str(record.get("task", self._task)),
            report_path=report_path,
            profile_path=self._last_profile_path,
            report_dir=local_run_dir,
        )
        self._log(
            "远端任务完成并已下载: "
            f"run_dir={local_run_dir} profile={result.get('local_profile', '')}"
        )
        if bool(result.get("invalid_run", False)):
            consistency = dict(result.get("config_consistency") or {})
            self.status_label.setText("远端完成，但本次运行无效")
            self._log(
                "警告：本地提交参数与远端实际 run_config 不一致，本次报告已标记 invalid_run。 "
                f"checks={consistency.get('checks', {})}"
            )

    def _on_done(self, payload: dict[str, Any]) -> None:
        report_path = payload.get("report_path") or self.report_edit.text().strip()
        self._last_report_path = Path(str(report_path)).expanduser().resolve()
        profile_path = payload.get("chosen_profile_path") or payload.get("profile_path") or self.output_profile_edit.text().strip()
        self._last_profile_path = Path(str(profile_path)).expanduser().resolve() if str(profile_path).strip() else None
        figures = dict(payload.get("figures", {}))
        figures_dir = figures.get("dir")
        self._last_figures_dir = Path(str(figures_dir)).expanduser().resolve() if figures_dir else None
        self.btn_open_report_dir.setEnabled(True)
        self.btn_open_profile.setEnabled(self._last_profile_path is not None and self._last_profile_path.exists())
        self.btn_publish_realtime_profile.setEnabled(
            self._last_profile_path is not None and self._last_profile_path.exists()
        )
        self.btn_publish_hybrid_profile.setEnabled(
            self._last_profile_path is not None and self._last_profile_path.exists()
        )
        self.btn_open_figures_dir.setEnabled(self._last_figures_dir is not None)
        self.btn_open_replay_viewer.setEnabled(self._last_report_path is not None and self._last_report_path.exists())
        self.progress_bar.setValue(100)
        self.progress_detail_label.setText("当前阶段：完成")
        self.eta_label.setText("预计剩余：0.0s")
        self._set_current_artifacts(
            task_name=str(payload.get("task", self._task)),
            report_path=self._last_report_path,
            profile_path=self._last_profile_path,
            report_dir=payload.get("report_dir"),
        )
        async_metrics = dict(payload.get("chosen_async_metrics", {}))
        metrics_4 = dict(payload.get("chosen_metrics_4class", {}))
        gate_valid = bool(payload.get("gate_calibration_valid", False))
        decision_effective = "decision_search_not_effective" not in [str(item) for item in payload.get("status_reasons", [])]
        run_valid_for_deployment = bool(payload.get("run_valid_for_deployment", False))
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
        self._log(
            "选择口径："
            f"decision search target={payload.get('decision_search_target', '')}, "
            f"final selection target={payload.get('final_selection_target', '')}"
        )
        status = str(payload.get("status", "ok")).strip().lower()
        status_reasons = [str(item) for item in payload.get("status_reasons", [])]
        if status != "ok":
            self.status_label.setText("已完成（本次运行无效）")
            self.progress_detail_label.setText(
                "当前阶段：完成 | "
                f"gate={'valid' if gate_valid else 'invalid'} | "
                f"decision={'effective' if decision_effective else 'ineffective'} | "
                "不可部署"
            )
            self._log(
                "运行无效："
                f"status_reasons={status_reasons or ['unknown']}, "
                f"chosen_model_rationale={payload.get('chosen_model_rationale', '')}, "
                f"profile_saved={bool(payload.get('profile_saved', False))}, "
                f"run_valid_for_deployment={run_valid_for_deployment}"
            )
        elif bool(payload.get("profile_saved", False)):
            self.status_label.setText("训练评测完成")
            self.progress_detail_label.setText(
                "当前阶段：完成 | "
                f"gate={'valid' if gate_valid else 'invalid'} | "
                f"decision={'effective' if decision_effective else 'ineffective'} | "
                f"{'可部署' if run_valid_for_deployment else '仅调试'}"
            )
            self._log(f"完成。已选模型={payload.get('chosen_model')}，报告={self._last_report_path}")
        else:
            self.status_label.setText("已完成（无达标模型）")
            self.progress_detail_label.setText(
                "当前阶段：完成 | "
                f"gate={'valid' if gate_valid else 'invalid'} | "
                f"decision={'effective' if decision_effective else 'ineffective'} | "
                f"{'可部署' if run_valid_for_deployment else '仅调试'}"
            )
            self._log(
                f"完成但未保存 profile。推荐模型={payload.get('recommended_model')} "
                f"rationale={payload.get('chosen_model_rationale', '')} "
                f"run_valid_for_deployment={run_valid_for_deployment}"
            )

    def configure_tdca_local_opt_mode(self, *, auto_start: bool = False) -> None:
        self._task = "tdca-local-opt"
        self._tdca_search_preset = str(TDCA_LOCAL_OPT_SEARCH_PRESET)
        self.simple_mode_check.setChecked(False)
        self.remote_mode_check.setChecked(False)
        self.allow_local_mode_check.setChecked(True)
        self.keep_baseline_group_check.setChecked(False)
        self.models_edit.setText(",".join(TDCA_LOCAL_OPT_MODELS))
        self.channel_modes_edit.setText(",".join(TDCA_LOCAL_OPT_CHANNEL_MODES))
        self.multi_seed_spin.setValue(int(TDCA_LOCAL_OPT_MULTI_SEED_COUNT))
        self.win_candidates_edit.setText(",".join(f"{float(value):g}" for value in TDCA_LOCAL_OPT_WIN_CANDIDATES))
        self.compute_backend_combo.setCurrentText(str(TDCA_LOCAL_OPT_COMPUTE_BACKEND))
        self._log(
            "切换到 TDCA 本地异步优化任务："
            f"all8 / repeated group split / preset={self._tdca_search_preset} / 本地结构化异步链路"
        )
        if bool(auto_start):
            self._start_local_run()

    def configure_fbcca_local_opt_mode(self, *, auto_start: bool = False) -> None:
        self._task = "fbcca-local-opt"
        self._tdca_search_preset = str(FBCCA_LOCAL_OPT_SEARCH_PRESET)
        self.simple_mode_check.setChecked(False)
        self.remote_mode_check.setChecked(False)
        self.allow_local_mode_check.setChecked(True)
        self.keep_baseline_group_check.setChecked(False)
        self.models_edit.setText(",".join(FBCCA_LOCAL_OPT_MODELS))
        self.channel_modes_edit.setText(",".join(FBCCA_LOCAL_OPT_CHANNEL_MODES))
        self.multi_seed_spin.setValue(int(FBCCA_LOCAL_OPT_MULTI_SEED_COUNT))
        self.win_candidates_edit.setText(",".join(f"{float(value):g}" for value in FBCCA_LOCAL_OPT_WIN_CANDIDATES))
        self.compute_backend_combo.setCurrentText(str(FBCCA_LOCAL_OPT_COMPUTE_BACKEND))
        self._log(
            "切换到 FBCCA 本地异步优化任务："
            f"all8 / repeated group split / preset={self._tdca_search_preset} / 本地结构化异步链路"
        )
        if bool(auto_start):
            self._start_local_run()

    def configure_fbcca_threshold_pretrain_mode(self, *, auto_start: bool = False) -> None:
        self._task = DEFAULT_FBCCA_THRESHOLD_TASK
        self.simple_mode_check.setChecked(False)
        self.remote_mode_check.setChecked(False)
        self.allow_local_mode_check.setChecked(True)
        self.keep_baseline_group_check.setChecked(False)
        self.models_edit.setText(",".join(FBCCA_THRESHOLD_PRETRAIN_MODELS))
        self.channel_modes_edit.setText(",".join(FBCCA_THRESHOLD_PRETRAIN_CHANNEL_MODES))
        self.multi_seed_spin.setValue(int(FBCCA_THRESHOLD_PRETRAIN_MULTI_SEED_COUNT))
        self.win_candidates_edit.setText(
            ",".join(f"{float(value):g}" for value in FBCCA_THRESHOLD_PRETRAIN_WIN_CANDIDATES)
        )
        self.weight_mode_edit.setText("none")
        self.subband_weight_mode_edit.setText("chen_fixed")
        self.spatial_mode_edit.setText("none")
        self.joint_iters_edit.setText("1")
        self.weight_cv_folds_edit.setText("2")
        self.dynamic_stop_edit.setText("0")
        self.compute_backend_combo.setCurrentText(str(FBCCA_THRESHOLD_PRETRAIN_COMPUTE_BACKEND))
        self._log(
            "切换到 FBCCA 阈值快速预训练：默认 FBCCA 参数 / 只拟合实时识别阈值 / 自动发布到实时识别 profile"
        )
        if bool(auto_start):
            self._start_local_run()

    def configure_fbcca_external_replay_mode(self, *, auto_start: bool = False) -> None:
        self._task = "fbcca-external-replay-opt"
        current_preset = str(getattr(self, "_tdca_search_preset", "")).strip().lower()
        if current_preset not in set(FBCCA_EXTERNAL_SEARCH_PRESETS):
            current_preset = str(FBCCA_EXTERNAL_REPLAY_SEARCH_PRESET)
        current_dataset_root = str(self.external_dataset_root_edit.text()).strip()
        if not current_dataset_root:
            current_dataset_root = str(DEFAULT_FBCCA_EXTERNAL_DATASET_ROOT)
        current_outer_eval = str(self.external_outer_eval_combo.currentText()).strip().lower()
        if current_outer_eval not in set(FBCCA_EXTERNAL_OUTER_EVALS):
            current_outer_eval = str(DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL)
        current_replay_speed = str(self.external_replay_speed_combo.currentText()).strip().lower()
        if current_replay_speed not in set(FBCCA_EXTERNAL_REPLAY_SPEEDS):
            current_replay_speed = str(DEFAULT_FBCCA_EXTERNAL_REPLAY_SPEED)
        current_output_profile = str(self.output_profile_edit.text()).strip()
        if (not current_output_profile) or Path(current_output_profile).name == "default_profile.json":
            current_output_profile = str(Path(DEFAULT_PROFILE_PATH).with_name("profile.json"))
        current_compute_backend = str(self.compute_backend_combo.currentText()).strip()
        if not current_compute_backend:
            current_compute_backend = str(FBCCA_EXTERNAL_REPLAY_COMPUTE_BACKEND)
        self._tdca_search_preset = current_preset
        self.simple_mode_check.setChecked(False)
        self.remote_mode_check.setChecked(False)
        self.allow_local_mode_check.setChecked(True)
        self.keep_baseline_group_check.setChecked(False)
        self.models_edit.setText(",".join(FBCCA_EXTERNAL_REPLAY_MODELS))
        self.channel_modes_edit.setText(",".join(FBCCA_EXTERNAL_REPLAY_CHANNEL_MODES))
        self.compute_backend_combo.setCurrentText(str(current_compute_backend))
        self.external_dataset_root_edit.setText(str(current_dataset_root))
        self.external_outer_eval_combo.setCurrentText(str(current_outer_eval))
        self.external_replay_speed_combo.setCurrentText(str(current_replay_speed))
        self.output_profile_edit.setText(str(current_output_profile))
        self.dynamic_stop_edit.setText("0")
        self._log(
            "切换到 FBCCA 外部数据回放优化任务："
            f"3 target + rest / 8ch / preset={self._tdca_search_preset} / continuous replay"
        )
        if bool(auto_start):
            self._start_local_run()

    def _fbcca_local_opt_run(self) -> None:
        if self.worker_thread is not None:
            return
        self.configure_fbcca_local_opt_mode(auto_start=True)

    def _fbcca_threshold_pretrain_run(self) -> None:
        if self.worker_thread is not None:
            return
        self.configure_fbcca_threshold_pretrain_mode(auto_start=True)

    def _fbcca_external_replay_run(self) -> None:
        if self.worker_thread is not None:
            return
        self.configure_fbcca_external_replay_mode(auto_start=True)

    def _tdca_local_opt_run(self) -> None:
        if self.worker_thread is not None:
            return
        self.configure_tdca_local_opt_mode(auto_start=True)

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
            self.btn_fbcca_threshold_pretrain_run.setVisible(True)
            self.btn_toggle_advanced.setVisible(True)
            self._set_advanced_visible(False)
            self.status_label.setText("简易模式：建议先运行权重训练+全模型对比")
        else:
            self.btn_weighted_compare_run.setVisible(False)
            self.btn_quick_run.setVisible(False)
            self.btn_model_compare_run.setVisible(False)
            self.btn_fbcca_threshold_pretrain_run.setVisible(True)
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
        self.btn_fbcca_threshold_pretrain_run.setEnabled(not running)
        if running:
            self.btn_open_report_dir.setEnabled(False)
            self.btn_publish_realtime_profile.setEnabled(False)
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
    parser.add_argument("--external-dataset-root", type=Path, default=DEFAULT_FBCCA_EXTERNAL_DATASET_ROOT)
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--include-manifests", type=str, default="", help="comma-separated manifest paths")
    parser.add_argument("--output-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_DIR / "report.json")
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
    parser.add_argument(
        "--search-preset",
        type=str,
        default=str(TDCA_LOCAL_OPT_SEARCH_PRESET),
        choices=sorted(
            set(TDCA_LOCAL_SEARCH_PRESETS)
            | set(FBCCA_LOCAL_SEARCH_PRESETS)
            | set(FBCCA_EXTERNAL_SEARCH_PRESETS)
        ),
    )
    parser.add_argument("--outer-eval", type=str, default=DEFAULT_FBCCA_EXTERNAL_OUTER_EVAL, choices=list(FBCCA_EXTERNAL_OUTER_EVALS))
    parser.add_argument("--replay-speed", type=str, default=DEFAULT_FBCCA_EXTERNAL_REPLAY_SPEED, choices=list(FBCCA_EXTERNAL_REPLAY_SPEEDS))
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
        choices=[
            "fbcca-weights",
            "model-compare",
            "fbcca-weighted-compare",
            "tdca-local-opt",
            "fbcca-local-opt",
            DEFAULT_FBCCA_THRESHOLD_TASK,
            "fbcca-external-replay-opt",
        ],
    )
    parser.add_argument("--monitor-run-dir", type=Path, default=None, help="attach UI to an existing local run dir")
    parser.add_argument("--auto-start", action="store_true", help="start the configured task automatically after the UI opens")
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_tokens = list(argv or sys.argv[1:])
    args = build_parser().parse_args(argv_tokens)
    parsed_task = _parse_task(args.task)
    if bool(int(args.quick_mode)) or parsed_task == "fbcca-weights":
        _apply_quick_mode_args(args)
        parsed_task = _parse_task(args.task)
    elif parsed_task == "fbcca-weighted-compare":
        _apply_weighted_compare_args(args)
        parsed_task = _parse_task(args.task)
    elif parsed_task == "model-compare":
        _apply_model_compare_args(args)
        parsed_task = _parse_task(args.task)
    elif parsed_task == "fbcca-local-opt":
        _apply_fbcca_local_opt_args(args, argv_tokens)
        parsed_task = _parse_task(args.task)
    elif parsed_task == DEFAULT_FBCCA_THRESHOLD_TASK:
        _apply_fbcca_threshold_pretrain_args(args, argv_tokens)
        parsed_task = _parse_task(args.task)
    elif parsed_task == "fbcca-external-replay-opt":
        _apply_fbcca_external_replay_args(args, argv_tokens)
        parsed_task = _parse_task(args.task)
    elif parsed_task == "tdca-local-opt":
        _apply_tdca_local_opt_args(args, argv_tokens)
        parsed_task = _parse_task(args.task)
    include_manifests = _parse_manifest_csv(args.include_manifests)
    if bool(args.headless):
        s1 = include_manifests[0] if include_manifests else args.dataset_manifest
        if parsed_task != "fbcca-external-replay-opt" and s1 is None:
            raise ValueError("--dataset-manifest or --include-manifests is required in --headless mode")
        if parsed_task == "tdca-local-opt":
            config = TDCALocalOptConfig(
                dataset_manifest_session1=Path(s1).expanduser().resolve(),
                dataset_manifests=include_manifests,
                output_profile_path=Path(args.output_profile).expanduser().resolve(),
                report_path=Path(args.report_path).expanduser().resolve(),
                report_root_dir=Path(args.report_root_dir).expanduser().resolve(),
                organize_report_dir=bool(int(args.organize_report_dir)),
                model_names=tuple(parse_model_list(args.models)),
                channel_modes=tuple(parse_channel_mode_list(args.channel_modes)),
                multi_seed_count=int(args.multi_seed_count),
                win_candidates=tuple(float(item.strip()) for item in str(args.win_candidates).split(",") if item.strip()),
                search_preset=str(args.search_preset).strip().lower(),
                seed=int(args.seed),
                compute_backend=parse_compute_backend_name(str(args.compute_backend).strip()),
                gpu_device=int(args.gpu_device),
                gpu_precision=parse_gpu_precision(str(args.gpu_precision).strip()),
                gpu_warmup=bool(int(args.gpu_warmup)),
                gpu_cache_policy=parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()),
                decision_time_mode=parse_decision_time_mode(args.decision_time_mode),
                async_decision_time_mode=parse_decision_time_mode(args.async_decision_time_mode),
                progress_heartbeat_sec=float(args.progress_heartbeat_sec),
            )
            run_tdca_local_opt(config, log_fn=lambda text: print(text, flush=True))
        elif parsed_task == "fbcca-local-opt":
            config = FBCCALocalOptConfig(
                dataset_manifest_session1=Path(s1).expanduser().resolve(),
                dataset_manifests=include_manifests,
                output_profile_path=Path(args.output_profile).expanduser().resolve(),
                report_path=Path(args.report_path).expanduser().resolve(),
                report_root_dir=Path(args.report_root_dir).expanduser().resolve(),
                organize_report_dir=bool(int(args.organize_report_dir)),
                model_names=tuple(parse_model_list(args.models)),
                channel_modes=tuple(parse_channel_mode_list(args.channel_modes)),
                multi_seed_count=int(args.multi_seed_count),
                win_candidates=tuple(float(item.strip()) for item in str(args.win_candidates).split(",") if item.strip()),
                search_preset=str(args.search_preset).strip().lower(),
                seed=int(args.seed),
                compute_backend=parse_compute_backend_name(str(args.compute_backend).strip()),
                gpu_device=int(args.gpu_device),
                gpu_precision=parse_gpu_precision(str(args.gpu_precision).strip()),
                gpu_warmup=bool(int(args.gpu_warmup)),
                gpu_cache_policy=parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()),
                decision_time_mode=parse_decision_time_mode(args.decision_time_mode),
                async_decision_time_mode=parse_decision_time_mode(args.async_decision_time_mode),
                progress_heartbeat_sec=float(args.progress_heartbeat_sec),
            )
            run_fbcca_local_opt(config, log_fn=lambda text: print(text, flush=True))
        elif parsed_task == DEFAULT_FBCCA_THRESHOLD_TASK:
            win_candidates = tuple(float(item.strip()) for item in str(args.win_candidates).split(",") if item.strip())
            config = FBCCAThresholdPretrainConfig(
                dataset_manifest_session1=Path(s1).expanduser().resolve(),
                dataset_manifests=include_manifests,
                output_profile_path=Path(args.output_profile).expanduser().resolve(),
                report_path=Path(args.report_path).expanduser().resolve(),
                report_root_dir=Path(args.report_root_dir).expanduser().resolve(),
                organize_report_dir=bool(int(args.organize_report_dir)),
                win_sec=float(win_candidates[0] if win_candidates else 3.0),
                gate_policy=parse_gate_policy(args.gate_policy),
                dynamic_stop_enabled=False,
                dynamic_stop_alpha=float(args.dynamic_stop_alpha),
                seed=int(args.seed),
                compute_backend=parse_compute_backend_name(str(args.compute_backend).strip()),
                gpu_device=int(args.gpu_device),
                gpu_precision=parse_gpu_precision(str(args.gpu_precision).strip()),
                gpu_warmup=bool(int(args.gpu_warmup)),
                gpu_cache_policy=parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()),
                decision_time_mode=parse_decision_time_mode(args.decision_time_mode),
                async_decision_time_mode=parse_decision_time_mode(args.async_decision_time_mode),
                progress_heartbeat_sec=float(args.progress_heartbeat_sec),
                publish_realtime=True,
            )
            run_fbcca_threshold_pretrain(config, log_fn=lambda text: print(text, flush=True))
        elif parsed_task == "fbcca-external-replay-opt":
            if not str(args.subject or "").strip():
                raise ValueError("--subject is required for fbcca-external-replay-opt")
            config = FBCCAExternalReplayOptConfig(
                external_dataset_root=Path(args.external_dataset_root).expanduser().resolve(),
                subject=str(args.subject).strip(),
                output_profile_path=Path(args.output_profile).expanduser().resolve(),
                report_path=Path(args.report_path).expanduser().resolve(),
                report_root_dir=Path(args.report_root_dir).expanduser().resolve(),
                organize_report_dir=bool(int(args.organize_report_dir)),
                model_names=tuple(parse_model_list(args.models)),
                channel_modes=tuple(parse_channel_mode_list(args.channel_modes)),
                search_preset=str(args.search_preset).strip().lower(),
                outer_eval=str(args.outer_eval).strip().lower(),
                replay_speed=str(args.replay_speed).strip().lower(),
                seed=int(args.seed),
                compute_backend=parse_compute_backend_name(str(args.compute_backend).strip()),
                gpu_device=int(args.gpu_device),
                gpu_precision=parse_gpu_precision(str(args.gpu_precision).strip()),
                gpu_warmup=bool(int(args.gpu_warmup)),
                gpu_cache_policy=parse_gpu_cache_policy(str(args.gpu_cache_policy).strip()),
                decision_time_mode=parse_decision_time_mode(args.decision_time_mode),
                async_decision_time_mode=parse_decision_time_mode(args.async_decision_time_mode),
                progress_heartbeat_sec=float(args.progress_heartbeat_sec),
            )
            run_fbcca_external_replay_opt(config, log_fn=lambda text: print(text, flush=True))
        else:
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
                task=parsed_task,
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
    window.external_dataset_root_edit.setText(str(Path(args.external_dataset_root).expanduser().resolve()))
    window.external_subject_edit.setText(str(args.subject or "").strip())
    window.external_outer_eval_combo.setCurrentText(str(args.outer_eval).strip().lower())
    window.external_replay_speed_combo.setCurrentText(str(args.replay_speed).strip().lower())
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
    window._tdca_search_preset = str(args.search_preset).strip().lower()
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
    if window._task == "tdca-local-opt":
        window.configure_tdca_local_opt_mode(auto_start=False)
    elif window._task == "fbcca-local-opt":
        window.configure_fbcca_local_opt_mode(auto_start=False)
    elif window._task == DEFAULT_FBCCA_THRESHOLD_TASK:
        window.configure_fbcca_threshold_pretrain_mode(auto_start=False)
    elif window._task == "fbcca-external-replay-opt":
        window.configure_fbcca_external_replay_mode(auto_start=False)
    else:
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
    if args.monitor_run_dir is not None:
        window.attach_local_run_monitor(Path(args.monitor_run_dir).expanduser().resolve())
    window.show()
    if bool(args.auto_start) and args.monitor_run_dir is None:
        QTimer.singleShot(0, window._start_run)
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

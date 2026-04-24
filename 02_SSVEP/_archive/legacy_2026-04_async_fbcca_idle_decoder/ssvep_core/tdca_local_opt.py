from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from async_fbcca_idle_standalone import (
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_CONTROL_STATE_MODE,
    DEFAULT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_NH,
    DEFAULT_PAPER_DECISION_TIME_MODE,
    MODEL_FEATURE_NAMES,
    ThresholdProfile,
    TrialSpec,
    _quantile_candidates,
    atomic_write_text,
    build_feature_rows_with_decoder,
    compute_classification_metrics,
    create_decoder,
    extract_window_batch,
    fit_threshold_profile,
    json_dumps,
    json_safe,
    parse_tdca_decoder_variant,
    parse_compute_backend_name,
    parse_control_state_mode,
    parse_decision_time_mode,
    profile_meets_acceptance,
    save_profile,
)

from .compute_backend import resolve_compute_backend
from .dataset import LoadedDataset, infer_trial_role, load_collection_dataset, summarize_trial_roles
from .decision import DecisionEngine, DecisionEngineConfig, EvidenceAccumulatorConfig, StateMachineConfig
from .gating import PerFrequencyLogRegGate, RollingFeatureHistory
from .gating.per_freq_logreg_gate import LogRegFitConfig
from .profile_v2 import DEFAULT_GATE_FEATURES, build_profile_v2


DEFAULT_TDCA_LOCAL_MODEL = "tdca"
DEFAULT_TDCA_LOCAL_CHANNEL_MODE = "all8"
DEFAULT_TDCA_LOCAL_WIN_CANDIDATES = (2.0, 2.5, 3.0, 3.5)
DEFAULT_TDCA_LOCAL_DELAY_STEPS = (2, 3, 4, 5)
DEFAULT_TDCA_LOCAL_N_COMPONENTS = (2, 3, 4)
DEFAULT_TDCA_LOCAL_STEP_SEC = 0.25
DEFAULT_TDCA_LOCAL_REPEAT_COUNT = 5
DEFAULT_TDCA_LOCAL_TOP_K = 8
DEFAULT_DECISION_GRID_CANDIDATE_MIN_WINDOWS = (1, 2)
DEFAULT_DECISION_GRID_ARMED_MIN_WINDOWS = (2, 3)
DEFAULT_DECISION_GRID_LAMBDA = (0.80, 0.85, 0.90)
DEFAULT_DECISION_GRID_UPPER = (1.8, 2.2, 2.6)
DEFAULT_DECISION_GRID_LOWER = (0.2, 0.4, 0.6)
DEFAULT_DECISION_GRID_REFRACTORY = (0.4, 0.8)
DEFAULT_DECISION_COMMIT_CONSISTENCY_TH = 0.6
DEFAULT_DECISION_BETA_CONSISTENCY = 0.5
DEFAULT_LOGREG_FIT_CONFIG = LogRegFitConfig()
DEFAULT_BASELINE_MODELS = ("fbcca", "trca_r")
DEFAULT_BASELINE_WIN_SEC = 3.0
DEFAULT_TDCA_LOCAL_DECODER_VARIANT = "tdca_like_legacy"
DEFAULT_TDCA_LOCAL_DECODER_VARIANTS = ("tdca_like_legacy", "tdca_paper_aligned")
TDCA_DECODER_VARIANT_METADATA = {
    "tdca_like_legacy": {
        "algorithm_alignment": "not-paper-exact",
        "paper_tdca_projection_enabled": False,
    },
    "tdca_paper_aligned": {
        "algorithm_alignment": "paper-aligned",
        "paper_tdca_projection_enabled": True,
    },
}
DEFAULT_REPLAY_BENCHMARK_SHAPE = (256, 8, 64)
DEFAULT_REPLAY_BENCHMARK_REPEATS = 3
DEFAULT_GPU_REPLAY_MIN_WINDOWS = 256
DEFAULT_GPU_REPLAY_MIN_SPEEDUP = 1.5
DEFAULT_BASELINE_DECISION_PARAMS = {
    "candidate_min_windows": 1,
    "armed_min_windows": 2,
    "lambda_decay": 0.85,
    "upper_commit_th": 2.2,
    "lower_idle_th": 0.4,
    "refractory_sec": 0.8,
}


class GateReplayState:
    def __init__(self, profile: ThresholdProfile) -> None:
        self.profile = profile
        self.reset()

    def reset(self) -> None:
        self._candidate_freq: Optional[float] = None
        self._candidate_windows = 0
        self._gate_open_freq: Optional[float] = None
        self._exit_windows = 0
        self._switch_candidate_freq: Optional[float] = None
        self._switch_candidate_windows = 0

    def _threshold_payload(self, *, freq_value: Optional[float], selected_fallback: bool = False) -> dict[str, Any]:
        payload_map = dict(self.profile.frequency_specific_thresholds or {})
        if not payload_map:
            return {}
        resolved_freq = self._gate_open_freq if selected_fallback and self._gate_open_freq is not None else freq_value
        if resolved_freq is None:
            return {}
        payload = payload_map.get(_freq_label(float(resolved_freq)))
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _enter_pass(self, row: Mapping[str, Any], *, pred_freq: Optional[float]) -> bool:
        if pred_freq is None:
            return False
        payload = self._threshold_payload(freq_value=pred_freq)
        score_th = float(payload.get("enter_score_th", self.profile.enter_score_th))
        ratio_th = float(payload.get("enter_ratio_th", self.profile.enter_ratio_th))
        margin_th = float(payload.get("enter_margin_th", self.profile.enter_margin_th))
        enter_log_lr_th = payload.get("enter_log_lr_th", self.profile.enter_log_lr_th)
        legacy_pass = (
            _safe_float(row.get("top1_score", 0.0), 0.0) >= score_th
            and _safe_float(row.get("ratio", 1.0), 1.0) >= ratio_th
            and _safe_float(row.get("margin", 0.0), 0.0) >= margin_th
        )
        if enter_log_lr_th is None:
            return bool(legacy_pass)
        return bool(legacy_pass and _safe_float(row.get("control_log_lr", row.get("gate_score", 0.0)), float("-inf")) >= float(enter_log_lr_th))

    def _switch_pass(self, row: Mapping[str, Any], *, pred_freq: Optional[float]) -> bool:
        if self._gate_open_freq is None or pred_freq is None:
            return False
        if abs(float(pred_freq) - float(self._gate_open_freq)) <= 1e-8:
            return False
        payload = self._threshold_payload(freq_value=pred_freq)
        switch_score_th = float(payload.get("switch_enter_score_th", self.profile.switch_enter_score_th or self.profile.enter_score_th))
        switch_ratio_th = float(payload.get("switch_enter_ratio_th", self.profile.switch_enter_ratio_th or self.profile.enter_ratio_th))
        switch_margin_th = float(payload.get("switch_enter_margin_th", self.profile.switch_enter_margin_th or self.profile.enter_margin_th))
        return bool(
            _safe_float(row.get("top1_score", 0.0), 0.0) >= switch_score_th
            and _safe_float(row.get("ratio", 1.0), 1.0) >= switch_ratio_th
            and _safe_float(row.get("margin", 0.0), 0.0) >= switch_margin_th
        )

    def _exit_fail(self, row: Mapping[str, Any], *, pred_freq: Optional[float]) -> bool:
        if self._gate_open_freq is None:
            return True
        payload = self._threshold_payload(freq_value=pred_freq, selected_fallback=True)
        exit_score_th = float(payload.get("exit_score_th", self.profile.exit_score_th))
        exit_ratio_th = float(payload.get("exit_ratio_th", self.profile.exit_ratio_th))
        exit_log_lr_th = payload.get("exit_log_lr_th", self.profile.exit_log_lr_th)
        if pred_freq is None or abs(float(pred_freq) - float(self._gate_open_freq)) > 1e-8:
            return True
        if _safe_float(row.get("top1_score", 0.0), 0.0) < exit_score_th:
            return True
        if _safe_float(row.get("ratio", 1.0), 1.0) < exit_ratio_th:
            return True
        if exit_log_lr_th is not None and _safe_float(row.get("control_log_lr", row.get("gate_score", 0.0)), float("-inf")) < float(exit_log_lr_th):
            return True
        return False

    def update(self, row_raw: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(row_raw)
        pred_freq = row.get("pred_freq")
        pred_freq_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
        if pred_freq_value is not None and not np.isfinite(pred_freq_value):
            pred_freq_value = None
        enter_pass = self._enter_pass(row, pred_freq=pred_freq_value)
        switch_pass = self._switch_pass(row, pred_freq=pred_freq_value)
        exit_fail = False
        gate_event = "hold"
        gate_switched = False
        if self._gate_open_freq is None:
            min_enter_windows = int(
                self._threshold_payload(freq_value=pred_freq_value).get("min_enter_windows", self.profile.min_enter_windows)
            )
            if enter_pass and pred_freq_value is not None:
                if self._candidate_freq is None or abs(float(pred_freq_value) - float(self._candidate_freq)) > 1e-8:
                    self._candidate_freq = float(pred_freq_value)
                    self._candidate_windows = 1
                else:
                    self._candidate_windows += 1
                if self._candidate_windows >= max(int(min_enter_windows), 1):
                    self._gate_open_freq = float(pred_freq_value)
                    self._exit_windows = 0
                    self._switch_candidate_freq = None
                    self._switch_candidate_windows = 0
                    gate_event = "enter"
            else:
                self._candidate_freq = None
                self._candidate_windows = 0
                self._switch_candidate_freq = None
                self._switch_candidate_windows = 0
        else:
            pred_is_current = (
                pred_freq_value is not None and abs(float(pred_freq_value) - float(self._gate_open_freq)) <= 1e-8
            )
            if switch_pass and pred_freq_value is not None and not pred_is_current:
                min_switch_windows = int(
                    self._threshold_payload(freq_value=pred_freq_value).get(
                        "min_switch_windows",
                        self.profile.min_switch_windows,
                    )
                )
                if self._switch_candidate_freq is None or abs(float(pred_freq_value) - float(self._switch_candidate_freq)) > 1e-8:
                    self._switch_candidate_freq = float(pred_freq_value)
                    self._switch_candidate_windows = 1
                else:
                    self._switch_candidate_windows += 1
                self._exit_windows = 0
                if self._switch_candidate_windows >= max(int(min_switch_windows), 1):
                    self._gate_open_freq = float(pred_freq_value)
                    self._candidate_freq = None
                    self._candidate_windows = 0
                    self._switch_candidate_freq = None
                    self._switch_candidate_windows = 0
                    gate_event = "switch"
                    gate_switched = True
            else:
                self._switch_candidate_freq = None
                self._switch_candidate_windows = 0
                exit_fail = self._exit_fail(row, pred_freq=pred_freq_value)
                min_exit_windows = int(
                    self._threshold_payload(freq_value=pred_freq_value, selected_fallback=True).get(
                        "min_exit_windows",
                        self.profile.min_exit_windows,
                    )
                )
                if exit_fail:
                    self._exit_windows += 1
                    if self._exit_windows >= max(int(min_exit_windows), 1):
                        self._gate_open_freq = None
                        self._candidate_freq = None
                        self._candidate_windows = 0
                        self._exit_windows = 0
                        gate_event = "exit"
                else:
                    self._exit_windows = 0
        row["gate_open_freq"] = None if self._gate_open_freq is None else float(self._gate_open_freq)
        row["gate_is_open"] = bool(self._gate_open_freq is not None)
        row["enter_pass"] = bool(enter_pass)
        row["switch_pass"] = bool(switch_pass)
        row["exit_fail"] = bool(exit_fail)
        row["gate_candidate_windows"] = int(self._candidate_windows)
        row["gate_exit_windows"] = int(self._exit_windows)
        row["gate_switch_candidate_freq"] = None if self._switch_candidate_freq is None else float(self._switch_candidate_freq)
        row["gate_switch_candidate_windows"] = int(self._switch_candidate_windows)
        row["gate_event"] = str(gate_event)
        row["gate_switched"] = bool(gate_switched)
        return row


@dataclass(frozen=True)
class RepeatedGroupSplit:
    repeat_index: int
    train_indices: tuple[int, ...]
    gate_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    fingerprint: str


@dataclass(frozen=True)
class MergedLocalDataset:
    manifest_paths: tuple[Path, ...]
    datasets: tuple[LoadedDataset, ...]
    trial_segments: tuple[tuple[TrialSpec, np.ndarray], ...]
    sampling_rate: int
    freqs: tuple[float, float, float, float]
    board_eeg_channels: tuple[int, ...]
    subject_id: str
    session_ids: tuple[str, ...]
    trial_role_counts: dict[str, int]
    quality_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TDCALocalOptConfig:
    dataset_manifest_session1: Path
    output_profile_path: Path
    report_path: Path
    dataset_manifests: tuple[Path, ...] = ()
    report_root_dir: Optional[Path] = None
    organize_report_dir: bool = False
    model_names: tuple[str, ...] = (DEFAULT_TDCA_LOCAL_MODEL,)
    channel_modes: tuple[str, ...] = (DEFAULT_TDCA_LOCAL_CHANNEL_MODE,)
    multi_seed_count: int = DEFAULT_TDCA_LOCAL_REPEAT_COUNT
    win_candidates: tuple[float, ...] = DEFAULT_TDCA_LOCAL_WIN_CANDIDATES
    tdca_delay_steps: tuple[int, ...] = DEFAULT_TDCA_LOCAL_DELAY_STEPS
    tdca_n_components: tuple[int, ...] = DEFAULT_TDCA_LOCAL_N_COMPONENTS
    step_sec: float = DEFAULT_TDCA_LOCAL_STEP_SEC
    Nh: int = DEFAULT_NH
    seed: int = 20260410
    compute_backend: str = "auto"
    gpu_device: int = DEFAULT_GPU_DEVICE_ID
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME
    gpu_warmup: bool = True
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE
    control_state_mode: str = "frequency-specific-logistic"
    decision_time_mode: str = DEFAULT_PAPER_DECISION_TIME_MODE
    async_decision_time_mode: str = DEFAULT_ASYNC_DECISION_TIME_MODE
    progress_heartbeat_sec: float = 5.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except Exception:
        return float(default)
    return float(output) if np.isfinite(output) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _freq_label(freq: float) -> str:
    return f"{float(freq):g}"


def _tdca_variant_metadata(decoder_variant: Optional[str]) -> dict[str, Any]:
    variant = parse_tdca_decoder_variant(decoder_variant)
    payload = dict(TDCA_DECODER_VARIANT_METADATA.get(variant, {}))
    payload["decoder_variant"] = str(variant)
    return payload


def _tdca_variant_priority(decoder_variant: Optional[str]) -> int:
    variant = parse_tdca_decoder_variant(decoder_variant)
    return 0 if variant == "tdca_paper_aligned" else 1


def _nearest_freq(freq: float, freqs: Sequence[float]) -> float:
    items = [float(item) for item in freqs]
    if not items:
        raise ValueError("frequency list is empty")
    return min(items, key=lambda item: abs(float(item) - float(freq)))


def _latency_or_penalty(value: Any, penalty: float = 1_000_000.0) -> float:
    output = _safe_float(value, penalty)
    return float(output) if np.isfinite(output) else float(penalty)


def _median(values: Sequence[Any], default: float = 0.0) -> float:
    numeric = np.asarray(
        [float(item) for item in values if np.isfinite(_safe_float(item, float("nan")))],
        dtype=float,
    )
    if numeric.size == 0:
        return float(default)
    return float(np.median(numeric))


def _rank_metrics_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float, float, float]:
    control_recall = float(metrics.get("control_recall", 0.0))
    control_recall_at_3s = float(metrics.get("control_recall_at_3s", control_recall))
    return (
        float(metrics.get("idle_fp_per_min", float("inf"))),
        _latency_or_penalty(metrics.get("release_latency_s", float("inf"))),
        _latency_or_penalty(metrics.get("switch_latency_s", float("inf"))),
        -control_recall_at_3s,
        -control_recall,
        float(metrics.get("inference_ms", float("inf"))),
    )


def _fingerprint_for_split(
    *,
    train_indices: Sequence[int],
    gate_indices: Sequence[int],
    holdout_indices: Sequence[int],
) -> str:
    payload = "|".join(
        [
            ",".join(str(int(item)) for item in sorted(train_indices)),
            ",".join(str(int(item)) for item in sorted(gate_indices)),
            ",".join(str(int(item)) for item in sorted(holdout_indices)),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _candidate_key(*, decoder_variant: str, win_sec: float, delay_steps: int, n_components: int) -> str:
    return (
        f"variant={parse_tdca_decoder_variant(decoder_variant)}|win={float(win_sec):g}|"
        f"delay={int(delay_steps)}|n_components={int(n_components)}"
    )


def _default_decision_params() -> dict[str, Any]:
    return dict(DEFAULT_BASELINE_DECISION_PARAMS)


def _normalize_name_list(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item).strip().lower() for item in values if str(item).strip())


def _validate_tdca_local_config(config: TDCALocalOptConfig) -> None:
    model_names = _normalize_name_list(config.model_names)
    channel_modes = _normalize_name_list(config.channel_modes)
    if model_names != (DEFAULT_TDCA_LOCAL_MODEL,):
        raise ValueError(f"tdca-local-opt only supports model_names=('tdca',); got {config.model_names}")
    if channel_modes != (DEFAULT_TDCA_LOCAL_CHANNEL_MODE,):
        raise ValueError(f"tdca-local-opt only supports channel_modes=('all8',); got {config.channel_modes}")


def _estimate_window_count(
    *,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    sampling_rate: int,
    win_sec: float,
    step_sec: float,
) -> int:
    win_samples = max(int(round(float(win_sec) * int(sampling_rate))), 1)
    step_samples = max(int(round(float(step_sec) * int(sampling_rate))), 1)
    total = 0
    for _trial, segment in trial_segments:
        matrix = np.asarray(segment, dtype=float)
        if matrix.shape[0] < win_samples:
            continue
        total += 1 + max(int((matrix.shape[0] - win_samples) // step_samples), 0)
    return int(total)


def _microbenchmark_replay_backend(
    backend: Any,
    *,
    sample_shape: tuple[int, int, int] = DEFAULT_REPLAY_BENCHMARK_SHAPE,
    repeats: int = DEFAULT_REPLAY_BENCHMARK_REPEATS,
) -> dict[str, Any]:
    shape = tuple(max(int(item), 1) for item in sample_shape)
    repeat_count = max(int(repeats), 1)
    host_array = np.linspace(0.0, 1.0, int(np.prod(shape, dtype=np.int64)), dtype=np.float32).reshape(shape)
    warmup_ms = float(getattr(backend, "benchmark_warmup", lambda: 0.0)())
    total_ms_values: list[float] = []
    compute_ms_values: list[float] = []
    for _ in range(repeat_count):
        device_value, upload_ms = backend.to_device(host_array)
        t0 = time.perf_counter()
        reduced = backend.xp.sum(device_value * device_value, axis=-1)
        sync_ms = float(backend.synchronize())
        _host_value, download_ms = backend.to_host(reduced)
        t1 = time.perf_counter()
        total_ms_values.append(float((t1 - t0) * 1000.0 + upload_ms))
        compute_ms_values.append(float(max((t1 - t0) * 1000.0 - download_ms - sync_ms, 0.0)))
    return {
        "backend_name": str(getattr(backend, "backend_name", "cpu")),
        "sample_shape": [int(item) for item in shape],
        "repeats": int(repeat_count),
        "warmup_overhead_ms": float(warmup_ms),
        "total_ms": _median(total_ms_values, default=float("inf")),
        "compute_ms": _median(compute_ms_values, default=float("inf")),
    }


def _resolve_replay_backend_policy(
    *,
    env_preflight: Mapping[str, Any],
    estimated_window_count: int,
) -> dict[str, Any]:
    effective_backend = str(env_preflight.get("effective_backend", "cpu"))
    gpu_speedup = _safe_float(env_preflight.get("gpu_replay_speedup", 0.0), 0.0)
    gpu_ready = bool(env_preflight.get("gpu_backend_available", False))
    estimated_count = max(int(estimated_window_count), 0)
    if effective_backend != "cuda":
        reason = "preflight_effective_backend_not_cuda"
        chosen_backend = "cpu"
    elif not gpu_ready:
        reason = "gpu_backend_not_available"
        chosen_backend = "cpu"
    elif estimated_count < int(DEFAULT_GPU_REPLAY_MIN_WINDOWS):
        reason = "estimated_windows_below_threshold"
        chosen_backend = "cpu"
    elif gpu_speedup < float(DEFAULT_GPU_REPLAY_MIN_SPEEDUP):
        reason = "gpu_speedup_below_threshold"
        chosen_backend = "cpu"
    else:
        reason = "batched_replay_not_implemented"
        chosen_backend = "cpu"
    return {
        "effective_replay_backend": str(chosen_backend),
        "gpu_replay_speedup": float(gpu_speedup),
        "gpu_replay_eligible": bool(chosen_backend == "cuda"),
        "gpu_replay_reason": str(reason),
        "estimated_window_count": int(estimated_count),
    }


def _default_model_params(
    *,
    model_name: str,
    Nh: int,
    delay_steps: Optional[int] = None,
    n_components: Optional[int] = None,
    decoder_variant: Optional[str] = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"Nh": int(Nh)}
    if str(model_name).strip().lower() == DEFAULT_TDCA_LOCAL_MODEL:
        params["delay_steps"] = int(delay_steps if delay_steps is not None else DEFAULT_TDCA_LOCAL_DELAY_STEPS[0])
        params["n_components"] = int(n_components if n_components is not None else DEFAULT_TDCA_LOCAL_N_COMPONENTS[0])
        params["decoder_variant"] = parse_tdca_decoder_variant(decoder_variant or DEFAULT_TDCA_LOCAL_DECODER_VARIANT)
    return params


def _baseline_candidate(model_name: str, *, Nh: int) -> dict[str, Any]:
    return {
        "model_name": str(model_name),
        "channel_mode": DEFAULT_TDCA_LOCAL_CHANNEL_MODE,
        "win_sec": float(DEFAULT_BASELINE_WIN_SEC),
        "model_params": _default_model_params(model_name=str(model_name), Nh=int(Nh)),
    }


def _candidate_rank_tuple(row: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in row.get("rank_key", []))


def _tdca_board_sort_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    decoder_variant = (
        row.get("decoder_variant")
        or dict(row.get("candidate", {})).get("decoder_variant")
        or DEFAULT_TDCA_LOCAL_DECODER_VARIANT
    )
    return (*_candidate_rank_tuple(row), float(_tdca_variant_priority(str(decoder_variant))))


def _resolve_report_paths(config: TDCALocalOptConfig) -> dict[str, Path]:
    now = datetime.now()
    if bool(config.organize_report_dir):
        root_dir = (
            Path(config.report_root_dir).expanduser().resolve()
            if config.report_root_dir is not None
            else Path(config.report_path).expanduser().resolve().parent
        )
        report_dir = root_dir / now.strftime("%Y%m%d") / f"run_{now.strftime('%Y%m%d_%H%M%S')}_tdca_local"
        report_json = report_dir / "offline_train_eval.json"
    else:
        report_json = Path(config.report_path).expanduser().resolve()
        report_dir = report_json.parent
    return {
        "report_dir": report_dir,
        "report_json": report_json,
        "report_md": report_json.with_suffix(".md"),
        "run_config": report_dir / "run_config.json",
        "selection_snapshot": report_dir / "selection_snapshot.json",
        "run_log": report_dir / "run.log",
        "progress_snapshot": report_dir / "progress_snapshot.json",
    }


def _append_run_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {text}\n")


def _tdca_progress_label(stage: str) -> str:
    return {
        "prepare": "准备",
        "baseline_opening": "基线开场对比",
        "candidate_search": "TDCA 候选搜索",
        "decision_search": "异步决策搜索",
        "baseline_seal": "基线封板对比",
        "finalize": "保存产物",
        "complete": "完成",
    }.get(str(stage).strip().lower(), str(stage or "处理中"))


def _tdca_progress_percent(stage: str, *, run_index: int = 0, run_total: int = 0) -> int:
    ranges = {
        "prepare": (0.0, 12.0),
        "baseline_opening": (12.0, 18.0),
        "candidate_search": (18.0, 72.0),
        "decision_search": (72.0, 92.0),
        "baseline_seal": (92.0, 96.0),
        "finalize": (96.0, 99.0),
        "complete": (100.0, 100.0),
    }
    stage_name = str(stage).strip().lower()
    start, end = ranges.get(stage_name, (0.0, 0.0))
    if stage_name == "complete":
        return 100
    if run_total <= 0:
        fraction = 0.0
    else:
        fraction = min(max(float(run_index) / max(float(run_total), 1.0), 0.0), 1.0)
    return int(round(start + (end - start) * fraction))


def preflight_tdca_local_env(
    *,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
) -> dict[str, Any]:
    requested_backend = parse_compute_backend_name(compute_backend)
    try:
        backend = resolve_compute_backend(
            requested_backend,
            gpu_device=int(gpu_device),
            precision=str(gpu_precision),
        )
        backend_error = None
    except Exception as exc:
        backend = None
        backend_error = str(exc)
    if requested_backend == "cuda" and backend is None:
        raise RuntimeError(
            f"explicit compute_backend=cuda failed preflight before search started: {backend_error}"
        )
    if backend is None:
        backend = resolve_compute_backend("cpu", gpu_device=int(gpu_device), precision=str(gpu_precision))
    transfer_benchmark: dict[str, Any]
    try:
        transfer_benchmark = backend.microbenchmark_transfer(sample_shape=(256, 8), repeats=2, use_pinned=True)
    except Exception as exc:
        transfer_benchmark = {"error": str(exc)}
    cpu_backend = resolve_compute_backend("cpu", gpu_device=int(gpu_device), precision=str(gpu_precision))
    cpu_replay_benchmark = _microbenchmark_replay_backend(cpu_backend)
    try:
        gpu_backend = resolve_compute_backend("cuda", gpu_device=int(gpu_device), precision=str(gpu_precision))
        gpu_backend_available = True
        gpu_replay_benchmark = _microbenchmark_replay_backend(gpu_backend)
        gpu_backend_error = None
    except Exception as exc:
        gpu_backend = None
        gpu_backend_available = False
        gpu_replay_benchmark = {"backend_name": "cuda", "error": str(exc)}
        gpu_backend_error = str(exc)
    cpu_total_ms = _safe_float(cpu_replay_benchmark.get("total_ms"), float("inf"))
    gpu_total_ms = _safe_float(gpu_replay_benchmark.get("total_ms"), float("inf"))
    if gpu_backend_available and np.isfinite(cpu_total_ms) and np.isfinite(gpu_total_ms) and gpu_total_ms > 1e-12:
        gpu_replay_speedup = float(cpu_total_ms / gpu_total_ms)
    else:
        gpu_replay_speedup = 0.0
    payload = {
        "requested_backend": str(requested_backend),
        "effective_backend": str(getattr(backend, "backend_name", "cpu")),
        "gpu_device": int(gpu_device),
        "gpu_precision": str(gpu_precision),
        "backend_description": dict(backend.describe()),
        "transfer_benchmark": transfer_benchmark,
        "cpu_replay_benchmark": cpu_replay_benchmark,
        "gpu_replay_benchmark": gpu_replay_benchmark,
        "gpu_backend_available": bool(gpu_backend_available),
        "gpu_backend_error": gpu_backend_error,
        "gpu_replay_speedup": float(gpu_replay_speedup),
        "preflight_error": backend_error,
        "notes": [
            "explicit cuda requests fail during preflight instead of during replay search",
            "auto is allowed to fall back to cpu",
            "gpu replay is only enabled for 256+ estimated windows and >=1.5x speedup",
        ],
    }
    return payload


def backfill_manifest_trial_roles(manifest_path: Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    raw = Path(path).read_text(encoding="utf-8")
    manifest = dict(json.loads(raw))
    trials = [dict(row) for row in list(manifest.get("trials", [])) if isinstance(row, Mapping)]
    changed = False
    for row in trials:
        inferred = infer_trial_role(
            label=str(row.get("label", "")),
            expected_freq=None if row.get("expected_freq") is None else float(row.get("expected_freq")),
        )
        if str(row.get("trial_role", "")).strip().lower() != str(inferred):
            row["trial_role"] = str(inferred)
            changed = True
    quality_summary = dict(manifest.get("quality_summary", {}))
    role_counts = summarize_trial_roles(trials)
    if dict(quality_summary.get("trial_role_counts", {})) != role_counts:
        quality_summary["trial_role_counts"] = role_counts
        changed = True
    manifest["trials"] = trials
    manifest["quality_summary"] = quality_summary
    if changed:
        atomic_write_text(path, json_dumps(json_safe(manifest)) + "\n")
    return {
        "manifest_path": str(path),
        "changed": bool(changed),
        "trial_count": int(len(trials)),
        "trial_role_counts": role_counts,
    }


def _build_quality_row(dataset: LoadedDataset) -> dict[str, Any]:
    quality_summary = dict(dataset.manifest.get("quality_summary", {}))
    trials = [dict(row) for row in list(dataset.manifest.get("trials", [])) if isinstance(row, Mapping)]
    role_counts = summarize_trial_roles(trials)
    return {
        "manifest_path": str(dataset.manifest_path),
        "session_id": str(dataset.session_id),
        "subject_id": str(dataset.subject_id),
        "trial_count": int(len(trials)),
        "trial_role_counts": role_counts,
        "quality_summary": quality_summary,
    }


def _load_merged_dataset(config: TDCALocalOptConfig) -> MergedLocalDataset:
    manifest_paths = tuple(
        dict.fromkeys(
            Path(path).expanduser().resolve()
            for path in (config.dataset_manifests or (config.dataset_manifest_session1,))
        )
    )
    if not manifest_paths:
        raise ValueError("at least one dataset manifest is required")

    datasets: list[LoadedDataset] = []
    merged_segments: list[tuple[TrialSpec, np.ndarray]] = []
    quality_rows: list[dict[str, Any]] = []
    subject_id = ""
    sampling_rate: Optional[int] = None
    freqs: Optional[tuple[float, float, float, float]] = None
    board_channels: Optional[tuple[int, ...]] = None
    global_trial_id = 0
    block_offset = 0

    for path in manifest_paths:
        backfill_manifest_trial_roles(path)
        dataset = load_collection_dataset(path)
        datasets.append(dataset)
        quality_rows.append(_build_quality_row(dataset))
        if subject_id and str(dataset.subject_id) != subject_id:
            raise ValueError(
                f"subject mismatch across manifests: {subject_id} vs {dataset.subject_id} ({path})"
            )
        subject_id = str(dataset.subject_id)
        if sampling_rate is None:
            sampling_rate = int(dataset.sampling_rate)
            freqs = tuple(float(item) for item in dataset.freqs)  # type: ignore[assignment]
            board_channels = tuple(int(item) for item in dataset.board_eeg_channels)
        else:
            if int(dataset.sampling_rate) != int(sampling_rate):
                raise ValueError(f"sampling_rate mismatch in {path}")
            if tuple(float(item) for item in dataset.freqs) != tuple(freqs or ()):
                raise ValueError(f"frequency set mismatch in {path}")
            if tuple(int(item) for item in dataset.board_eeg_channels) != tuple(board_channels or ()):
                raise ValueError(f"board_eeg_channels mismatch in {path}")
        for trial, segment in dataset.trial_segments:
            merged_segments.append(
                (
                    TrialSpec(
                        label=str(trial.label),
                        expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
                        trial_id=int(global_trial_id),
                        block_index=int(block_offset + max(int(trial.block_index), 0)),
                    ),
                    np.ascontiguousarray(np.asarray(segment, dtype=np.float64)),
                )
            )
            global_trial_id += 1
        block_offset += len(dataset.trial_segments) + 1

    trial_role_counts = summarize_trial_roles(
        [
            {
                "label": str(trial.label),
                "expected_freq": None if trial.expected_freq is None else float(trial.expected_freq),
                "trial_role": infer_trial_role(
                    label=str(trial.label),
                    expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
                ),
            }
            for trial, _segment in merged_segments
        ]
    )
    if sampling_rate is None or freqs is None or board_channels is None:
        raise RuntimeError("merged dataset resolution failed")
    return MergedLocalDataset(
        manifest_paths=manifest_paths,
        datasets=tuple(datasets),
        trial_segments=tuple(merged_segments),
        sampling_rate=int(sampling_rate),
        freqs=tuple(freqs),  # type: ignore[arg-type]
        board_eeg_channels=tuple(board_channels),
        subject_id=str(subject_id),
        session_ids=tuple(str(dataset.session_id) for dataset in datasets),
        trial_role_counts=trial_role_counts,
        quality_rows=tuple(dict(row) for row in quality_rows),
    )


def _allocate_split_counts(count: int) -> tuple[int, int, int]:
    total = max(int(count), 0)
    if total <= 1:
        return total, 0, 0
    if total == 2:
        return 1, 0, 1
    if total == 3:
        return 1, 1, 1
    train_count = max(1, int(round(total * 0.6)))
    gate_count = max(1, int(round(total * 0.2)))
    holdout_count = total - train_count - gate_count
    if holdout_count <= 0:
        holdout_count = 1
        if gate_count > 1:
            gate_count -= 1
        else:
            train_count = max(1, train_count - 1)
    return train_count, gate_count, holdout_count


def build_repeated_group_splits(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    repeats: int,
    seed: int,
) -> list[RepeatedGroupSplit]:
    indexed = list(enumerate(trial_segments))
    grouped: dict[str, list[int]] = {}
    for index, (trial, _segment) in indexed:
        role = infer_trial_role(
            label=str(trial.label),
            expected_freq=None if trial.expected_freq is None else float(trial.expected_freq),
        )
        group_key = f"{role}|{str(trial.label).strip().lower()}"
        grouped.setdefault(group_key, []).append(int(index))

    output: list[RepeatedGroupSplit] = []
    for repeat_index in range(max(int(repeats), 1)):
        rng = np.random.default_rng(int(seed) + repeat_index * 1009)
        train_indices: list[int] = []
        gate_indices: list[int] = []
        holdout_indices: list[int] = []
        for group_key in sorted(grouped):
            values = list(grouped[group_key])
            if len(values) > 1:
                values = [int(item) for item in rng.permutation(values).tolist()]
            train_count, gate_count, holdout_count = _allocate_split_counts(len(values))
            train_indices.extend(values[:train_count])
            gate_indices.extend(values[train_count : train_count + gate_count])
            holdout_indices.extend(values[train_count + gate_count : train_count + gate_count + holdout_count])
        if not holdout_indices and train_indices:
            holdout_indices.append(train_indices.pop())
        fingerprint = _fingerprint_for_split(
            train_indices=train_indices,
            gate_indices=gate_indices,
            holdout_indices=holdout_indices,
        )
        output.append(
            RepeatedGroupSplit(
                repeat_index=int(repeat_index),
                train_indices=tuple(sorted(int(item) for item in train_indices)),
                gate_indices=tuple(sorted(int(item) for item in gate_indices)),
                holdout_indices=tuple(sorted(int(item) for item in holdout_indices)),
                fingerprint=str(fingerprint),
            )
        )
    return output


def _subset_segments(
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    indices: Sequence[int],
) -> list[tuple[TrialSpec, np.ndarray]]:
    keep = {int(item) for item in indices}
    return [trial_segments[index] for index in range(len(trial_segments)) if index in keep]


def _feature_stats(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    if not rows:
        means = {str(name): 0.0 for name in MODEL_FEATURE_NAMES}
        stds = {str(name): 1.0 for name in MODEL_FEATURE_NAMES}
        return means, stds
    matrix = np.asarray(
        [[_safe_float(row.get(name, 0.0), 0.0) for name in MODEL_FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    mean_values = np.mean(matrix, axis=0)
    std_values = np.maximum(np.std(matrix, axis=0), 1e-6)
    return (
        {str(name): float(value) for name, value in zip(MODEL_FEATURE_NAMES, mean_values)},
        {str(name): float(value) for name, value in zip(MODEL_FEATURE_NAMES, std_values)},
    )


def _score_rows_with_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate: PerFrequencyLogRegGate,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row_raw in rows:
        row = dict(row_raw)
        pred_freq = row.get("pred_freq")
        pred_freq_value = None if pred_freq is None else _safe_float(pred_freq, float("nan"))
        if pred_freq_value is None or not np.isfinite(pred_freq_value):
            pred_freq_value = None
        gate_out = gate.predict(row, pred_freq_value)
        row["pred_freq"] = pred_freq_value
        row["p_control"] = float(gate_out.p_control)
        row["gate_score"] = float(gate_out.gate_score)
        row["control_log_lr"] = float(gate_out.gate_score)
        scored.append(row)
    return scored


def _measure_decoder_inference_ms(
    decoder: Any,
    trial_segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    max_windows: int = 64,
) -> float:
    latencies_ms: list[float] = []
    for _trial, segment in trial_segments:
        if len(latencies_ms) >= int(max_windows):
            break
        matrix = np.asarray(segment, dtype=float)
        if matrix.shape[0] < int(decoder.win_samples):
            continue
        windows = extract_window_batch(
            matrix,
            win_samples=int(decoder.win_samples),
            step_samples=int(decoder.step_samples),
        )
        for window in windows:
            t0 = time.perf_counter()
            decoder.analyze_window(np.asarray(window, dtype=np.float64))
            t1 = time.perf_counter()
            latencies_ms.append(float((t1 - t0) * 1000.0))
            if len(latencies_ms) >= int(max_windows):
                break
    return _median(latencies_ms, default=float("inf"))


def _gate_score_partitions(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    control: list[float] = []
    idle: list[float] = []
    freq_value = float(freq)
    for row in scored_rows:
        pred_freq = row.get("pred_freq")
        if pred_freq is None or abs(_safe_float(pred_freq, float("nan")) - freq_value) > 1e-8:
            continue
        expected = row.get("expected_freq")
        role = str(row.get("trial_role", "")).strip().lower()
        score = _safe_float(row.get("gate_score", 0.0), 0.0)
        if expected is not None and role == "control" and abs(_safe_float(expected, float("nan")) - freq_value) <= 1e-8:
            control.append(float(score))
        else:
            idle.append(float(score))
    return np.asarray(control, dtype=float), np.asarray(idle, dtype=float)


def _select_enter_exit_logit(
    *,
    control_scores: np.ndarray,
    idle_scores: np.ndarray,
    enter_fallback: Optional[float],
    exit_fallback: Optional[float],
) -> tuple[float, float, dict[str, Any]]:
    enter_candidates = sorted(
        {
            *(_quantile_candidates(control_scores, (0.05, 0.10, 0.20, 0.30, 0.40), floor=-1_000_000.0)),
            *(_quantile_candidates(idle_scores, (0.80, 0.90, 0.95, 0.98), floor=-1_000_000.0)),
            float(enter_fallback if enter_fallback is not None else 0.0),
        }
    )
    exit_candidates = sorted(
        {
            *(_quantile_candidates(control_scores, (0.02, 0.05, 0.10, 0.20, 0.30), floor=-1_000_000.0)),
            *(_quantile_candidates(idle_scores, (0.50, 0.60, 0.70, 0.80, 0.90), floor=-1_000_000.0)),
            float(exit_fallback if exit_fallback is not None else 0.0),
        }
    )
    best_enter = float(enter_fallback if enter_fallback is not None else 0.0)
    best_exit = float(exit_fallback if exit_fallback is not None else best_enter)
    enter_objective: Optional[tuple[float, float, float]] = None
    for candidate in enter_candidates:
        idle_fp = float(np.mean(idle_scores >= candidate)) if idle_scores.size else 0.0
        control_recall = float(np.mean(control_scores >= candidate)) if control_scores.size else 0.0
        objective = (idle_fp, -control_recall, float(candidate))
        if enter_objective is None or objective < enter_objective:
            enter_objective = objective
            best_enter = float(candidate)
    exit_objective: Optional[tuple[float, float, float]] = None
    for candidate in exit_candidates:
        control_drop_rate = float(np.mean(control_scores < candidate)) if control_scores.size else 0.0
        idle_clear_rate = float(np.mean(idle_scores < candidate)) if idle_scores.size else 0.0
        objective = (control_drop_rate, -idle_clear_rate, float(candidate))
        if exit_objective is None or objective < exit_objective:
            exit_objective = objective
            best_exit = float(candidate)
    return (
        float(best_enter),
        float(best_exit),
        {
            "enter_candidates": [float(item) for item in enter_candidates],
            "exit_candidates": [float(item) for item in exit_candidates],
            "enter_objective": None if enter_objective is None else [float(item) for item in enter_objective],
            "exit_objective": None if exit_objective is None else [float(item) for item in exit_objective],
        },
    )


def _build_frequency_specific_thresholds(
    *,
    base_profile: ThresholdProfile,
    scored_rows: Sequence[Mapping[str, Any]],
    freqs: Sequence[float],
    min_enter_windows: int,
    min_exit_windows: int,
    min_switch_windows: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    output: dict[str, dict[str, Any]] = {}
    board: list[dict[str, Any]] = []
    base_payload = dict(base_profile.frequency_specific_thresholds or {})
    for freq in freqs:
        freq_key = _freq_label(float(freq))
        payload = dict(base_payload.get(freq_key, {}))
        control_scores, idle_scores = _gate_score_partitions(scored_rows, freq=float(freq))
        enter_th, exit_th, detail = _select_enter_exit_logit(
            control_scores=control_scores,
            idle_scores=idle_scores,
            enter_fallback=payload.get("enter_log_lr_th", base_profile.enter_log_lr_th),
            exit_fallback=payload.get("exit_log_lr_th", base_profile.exit_log_lr_th),
        )
        payload.setdefault("enter_score_th", float(base_profile.enter_score_th))
        payload.setdefault("enter_ratio_th", float(base_profile.enter_ratio_th))
        payload.setdefault("enter_margin_th", float(base_profile.enter_margin_th))
        payload.setdefault("exit_score_th", float(base_profile.exit_score_th))
        payload.setdefault("exit_ratio_th", float(base_profile.exit_ratio_th))
        payload.setdefault("switch_enter_score_th", float(base_profile.switch_enter_score_th or base_profile.enter_score_th))
        payload.setdefault("switch_enter_ratio_th", float(base_profile.switch_enter_ratio_th or base_profile.enter_ratio_th))
        payload.setdefault("switch_enter_margin_th", float(base_profile.switch_enter_margin_th or base_profile.enter_margin_th))
        payload["enter_log_lr_th"] = float(enter_th)
        payload["exit_log_lr_th"] = float(exit_th)
        payload["min_enter_windows"] = int(min_enter_windows)
        payload["min_exit_windows"] = int(min_exit_windows)
        payload["min_switch_windows"] = int(min_switch_windows)
        output[freq_key] = payload
        board.append(
            {
                "freq": float(freq),
                "enter_logit_th": float(enter_th),
                "exit_logit_th": float(exit_th),
                "min_switch_windows": int(min_switch_windows),
                "n_control_rows": int(control_scores.size),
                "n_idle_rows": int(idle_scores.size),
                "detail": detail,
            }
        )
    return output, board


def _search_gate_profile(
    *,
    base_profile: ThresholdProfile,
    scored_gate_rows: Sequence[Mapping[str, Any]],
    freqs: Sequence[float],
    inference_ms: float,
    decision_time_mode: str,
    async_decision_time_mode: str,
) -> tuple[ThresholdProfile, list[dict[str, Any]], list[dict[str, Any]]]:
    best_profile = base_profile
    best_objective: Optional[tuple[float, float, float, float, float, float]] = None
    gate_board: list[dict[str, Any]] = []
    threshold_board: list[dict[str, Any]] = []
    for min_enter_windows, min_exit_windows, min_switch_windows in product((1, 2), (1, 2), (1, 2)):
        per_freq_thresholds, freq_board = _build_frequency_specific_thresholds(
            base_profile=base_profile,
            scored_rows=scored_gate_rows,
            freqs=freqs,
            min_enter_windows=int(min_enter_windows),
            min_exit_windows=int(min_exit_windows),
            min_switch_windows=int(min_switch_windows),
        )
        enter_values = [float(item.get("enter_log_lr_th", 0.0)) for item in per_freq_thresholds.values()]
        exit_values = [float(item.get("exit_log_lr_th", 0.0)) for item in per_freq_thresholds.values()]
        candidate_profile = replace(
            base_profile,
            min_enter_windows=int(min_enter_windows),
            min_exit_windows=int(min_exit_windows),
            min_switch_windows=int(min_switch_windows),
            enter_log_lr_th=_median(enter_values, default=base_profile.enter_log_lr_th or 0.0),
            exit_log_lr_th=_median(exit_values, default=base_profile.exit_log_lr_th or 0.0),
            frequency_specific_thresholds=per_freq_thresholds,
            control_state_mode=parse_control_state_mode("frequency-specific-logistic"),
        )
        metrics = dict(
            _evaluate_structured_rows(
                scored_rows=scored_gate_rows,
                profile=candidate_profile,
                freqs=freqs,
                decision_params=_default_decision_params(),
                inference_ms=float(inference_ms),
                decision_time_mode=str(decision_time_mode),
                async_decision_time_mode=str(async_decision_time_mode),
            ).get("async_metrics", {})
        )
        objective = _rank_metrics_key(metrics)
        gate_board.append(
            {
                "min_enter_windows": int(min_enter_windows),
                "min_exit_windows": int(min_exit_windows),
                "min_switch_windows": int(min_switch_windows),
                "metrics": dict(metrics),
                "rank_key": [float(item) for item in objective],
            }
        )
        for item in freq_board:
            threshold_board.append(
                {
                    "min_enter_windows": int(min_enter_windows),
                    "min_exit_windows": int(min_exit_windows),
                    "min_switch_windows": int(min_switch_windows),
                    **dict(item),
                }
            )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_profile = candidate_profile
    gate_board.sort(key=lambda item: tuple(float(value) for value in item.get("rank_key", [])))
    return best_profile, gate_board, threshold_board


def _make_decision_params_key(params: Mapping[str, Any]) -> str:
    return "|".join(
        [
            f"candidate_min_windows={int(params.get('candidate_min_windows', 0))}",
            f"armed_min_windows={int(params.get('armed_min_windows', 0))}",
            f"lambda_decay={float(params.get('lambda_decay', 0.0)):.2f}",
            f"upper_commit_th={float(params.get('upper_commit_th', 0.0)):.1f}",
            f"lower_idle_th={float(params.get('lower_idle_th', 0.0)):.1f}",
            f"refractory_sec={float(params.get('refractory_sec', 0.0)):.1f}",
        ]
    )


def _make_decision_engine_config(
    *,
    profile: ThresholdProfile,
    decision_params: Mapping[str, Any],
) -> DecisionEngineConfig:
    return DecisionEngineConfig(
        evidence=EvidenceAccumulatorConfig(
            lambda_decay=float(decision_params.get("lambda_decay", 0.85)),
            beta_consistency=float(DEFAULT_DECISION_BETA_CONSISTENCY),
            upper_commit_th=float(decision_params.get("upper_commit_th", 2.2)),
            lower_idle_th=float(decision_params.get("lower_idle_th", 0.4)),
        ),
        state=StateMachineConfig(
            candidate_min_windows=int(decision_params.get("candidate_min_windows", 1)),
            armed_min_windows=int(decision_params.get("armed_min_windows", 2)),
            commit_consistency_th=float(DEFAULT_DECISION_COMMIT_CONSISTENCY_TH),
            enter_gate_th=float(profile.enter_log_lr_th or 0.0),
            exit_gate_th=float(profile.exit_log_lr_th or 0.0),
            refractory_sec=float(decision_params.get("refractory_sec", 0.8)),
        ),
    )


def _evaluate_structured_rows(
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    profile: ThresholdProfile,
    freqs: Sequence[float],
    decision_params: Mapping[str, Any],
    inference_ms: float,
    decision_time_mode: str,
    async_decision_time_mode: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in scored_rows]
    if not rows:
        empty_metrics = {
            "idle_fp_per_min": 0.0,
            "control_recall": 0.0,
            "control_recall_at_2s": 0.0,
            "control_recall_at_3s": 0.0,
            "switch_detect_rate": 0.0,
            "switch_detect_rate_at_2.8s": 0.0,
            "release_detect_rate": 0.0,
            "switch_latency_s": float("inf"),
            "release_latency_s": float("inf"),
            "detection_latency_s": float("inf"),
            "inference_ms": float(inference_ms),
        }
        return {
            "async_metrics": empty_metrics,
            "metrics_4class": {},
            "metrics_2class": {},
            "metrics_5class": None,
            "paper_lens_metrics_4class": {},
            "paper_lens_metrics_2class": {},
            "async_lens_metrics_4class": {},
            "async_lens_metrics_2class": {},
            "decision_params": dict(decision_params),
            "trial_events": [],
        }

    gate = GateReplayState(profile)
    history = RollingFeatureHistory(window_size=4)
    engine = DecisionEngine(_make_decision_engine_config(profile=profile, decision_params=decision_params))
    gate.reset()
    history.reset()
    engine.reset()

    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    first_seen: dict[tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        key = (str(row.get("label", "")), _safe_int(row.get("trial_id", -1), -1))
        grouped_rows.setdefault(key, []).append(dict(row))
        first_seen.setdefault(key, int(index))
    ordered_trials = sorted(grouped_rows.keys(), key=lambda item: int(first_seen.get(item, 0)))

    control_trials = 0
    control_detected_trials = 0
    control_detected_trials_at_2s = 0
    control_detected_trials_at_3s = 0
    idle_selected_events = 0
    idle_duration_sec = 0.0
    switch_trials = 0
    switch_detected_trials = 0
    switch_detected_trials_at_2p8s = 0
    release_trials = 0
    release_detected_trials = 0
    detection_latencies: list[float] = []
    switch_latencies: list[float] = []
    release_latencies: list[float] = []
    y4_true: list[str] = []
    y4_pred: list[str] = []
    y2_true: list[str] = []
    y2_pred: list[str] = []
    times4: list[float] = []
    times2: list[float] = []
    last_control_freq: Optional[float] = None
    previous_trial_expected: Optional[float] = None
    trial_events: list[dict[str, Any]] = []
    selected_active_prev = False
    stream_index = 0

    for key in ordered_trials:
        trial_rows = sorted(grouped_rows[key], key=lambda item: _safe_int(item.get("window_index", 0), 0))
        if not trial_rows:
            continue
        label_text = str(trial_rows[0].get("label", ""))
        expected = trial_rows[0].get("expected_freq")
        expected_freq = None if expected is None else _safe_float(expected, float("nan"))
        if expected_freq is not None and not np.isfinite(expected_freq):
            expected_freq = None
        trial_duration = float(
            profile.win_sec
            + max(_safe_int(trial_rows[-1].get("window_index", len(trial_rows) - 1), len(trial_rows) - 1), 0)
            * float(profile.step_sec)
        )
        penalty_latency = float(trial_duration + profile.win_sec)
        first_correct_latency: Optional[float] = None
        first_any_latency: Optional[float] = None
        first_any_freq: Optional[float] = None
        first_release_latency: Optional[float] = None
        tracked_freq_first_seen: Optional[float] = None
        commit_freq_first_seen: Optional[float] = None
        last_pred_freq: Optional[float] = None
        idle_commit_seen = False
        gate_switch_count = 0
        gate_event = "hold"
        previous_gate_open_freq: Optional[float] = None

        for row in trial_rows:
            raw_pred_freq = row.get("pred_freq")
            pred_freq_raw = None if raw_pred_freq is None else _safe_float(raw_pred_freq, float("nan"))
            if pred_freq_raw is not None and np.isfinite(pred_freq_raw):
                last_pred_freq = float(pred_freq_raw)
            gate_row = gate.update(dict(row))
            hist = history.update(
                pred_freq=pred_freq_raw,
                margin=_safe_float(gate_row.get("margin", 0.0), 0.0),
                ratio=_safe_float(gate_row.get("ratio", 1.0), 1.0),
            )
            timestamp_s = float(stream_index) * float(profile.step_sec)
            decision = engine.step(
                pred_freq_raw,
                _safe_float(gate_row.get("control_log_lr", gate_row.get("gate_score", 0.0)), 0.0),
                float(hist["consistency"]),
                gate_open_freq=gate_row.get("gate_open_freq"),
                timestamp_s=timestamp_s,
            )
            stream_index += 1
            commit = bool(decision.get("commit", False))
            committed_freq = decision.get("commit_freq")
            tracked_freq = decision.get("tracked_freq")
            decision_selected = decision.get("selected_freq")
            window_index = _safe_int(row.get("window_index", 0), 0)
            latency_value = float(profile.win_sec + window_index * profile.step_sec)
            gate_open_freq = gate_row.get("gate_open_freq")
            if tracked_freq is not None and tracked_freq_first_seen is None:
                tracked_freq_first_seen = float(latency_value)
            if commit and first_any_latency is None:
                first_any_latency = float(latency_value)
                first_any_freq = None if committed_freq is None else _safe_float(committed_freq, float("nan"))
            if commit and commit_freq_first_seen is None:
                commit_freq_first_seen = float(latency_value)
            if expected_freq is None:
                if commit and committed_freq is not None and not idle_commit_seen:
                    idle_selected_events += 1
                    idle_commit_seen = True
                if (
                    previous_trial_expected is not None
                    and first_release_latency is None
                    and selected_active_prev
                    and decision_selected is None
                ):
                    first_release_latency = float(latency_value)
            else:
                if (
                    commit
                    and committed_freq is not None
                    and abs(_safe_float(committed_freq, float("nan")) - float(expected_freq)) <= 1e-8
                    and first_correct_latency is None
                ):
                    first_correct_latency = float(latency_value)
            if gate_row.get("gate_event") == "switch":
                gate_switch_count += 1
            if gate_row.get("gate_event") not in {"", "hold", None}:
                gate_event = str(gate_row.get("gate_event"))
            elif previous_gate_open_freq is None and gate_open_freq is not None:
                gate_event = "enter"
            elif previous_gate_open_freq is not None and gate_open_freq is None:
                gate_event = "exit"
            previous_gate_open_freq = None if gate_open_freq is None else float(gate_open_freq)
            selected_active_prev = decision_selected is not None

        if expected_freq is None:
            idle_duration_sec += float(max(trial_duration, 0.0))
            y2_true.append("idle")
            y2_pred.append("control" if first_any_latency is not None else "idle")
            times2.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
            if previous_trial_expected is not None:
                release_trials += 1
                if first_release_latency is None:
                    release_latencies.append(float(penalty_latency))
                else:
                    release_detected_trials += 1
                    release_latencies.append(float(first_release_latency))
            previous_trial_expected = None
            trial_events.append(
                {
                    "label": label_text,
                    "expected_freq": None,
                    "first_any_latency_s": first_any_latency,
                    "first_release_latency_s": first_release_latency,
                    "tracked_freq_first_seen_s": tracked_freq_first_seen,
                    "commit_freq_first_seen_s": commit_freq_first_seen,
                    "gate_event": str(gate_event),
                    "gate_switch_count": int(gate_switch_count),
                    "trial_duration_s": float(trial_duration),
                }
            )
            continue

        control_trials += 1
        y2_true.append("control")
        y2_pred.append("control" if first_any_latency is not None else "idle")
        times2.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
        if first_correct_latency is not None:
            control_detected_trials += 1
            detection_latencies.append(float(first_correct_latency))
            if float(first_correct_latency) <= 2.0:
                control_detected_trials_at_2s += 1
            if float(first_correct_latency) <= 3.0:
                control_detected_trials_at_3s += 1
        is_switch_trial = last_control_freq is not None and abs(float(last_control_freq) - float(expected_freq)) > 1e-8
        if is_switch_trial:
            switch_trials += 1
            if first_correct_latency is None:
                switch_latencies.append(float(penalty_latency))
            else:
                switch_detected_trials += 1
                switch_latencies.append(float(first_correct_latency))
                if float(first_correct_latency) <= 2.8:
                    switch_detected_trials_at_2p8s += 1

        pred4_freq = first_any_freq
        if pred4_freq is None and last_pred_freq is not None:
            pred4_freq = last_pred_freq
        if pred4_freq is None:
            pred4_freq = float(expected_freq)
        y4_true.append(_freq_label(float(expected_freq)))
        y4_pred.append(_freq_label(_nearest_freq(float(pred4_freq), freqs)))
        times4.append(float(first_any_latency if first_any_latency is not None else penalty_latency))
        last_control_freq = float(expected_freq)
        previous_trial_expected = float(expected_freq)
        trial_events.append(
            {
                "label": label_text,
                "expected_freq": float(expected_freq),
                "first_correct_latency_s": first_correct_latency,
                "first_any_latency_s": first_any_latency,
                "tracked_freq_first_seen_s": tracked_freq_first_seen,
                "commit_freq_first_seen_s": commit_freq_first_seen,
                "gate_event": str(gate_event),
                "gate_switch_count": int(gate_switch_count),
                "trial_duration_s": float(trial_duration),
                "switch_trial": bool(is_switch_trial),
            }
        )

    idle_minutes = float(max(idle_duration_sec, 0.0)) / 60.0
    async_metrics = {
        "idle_fp_per_min": float(idle_selected_events / idle_minutes) if idle_minutes > 1e-12 else 0.0,
        "control_recall": float(control_detected_trials / control_trials) if control_trials else 0.0,
        "control_recall_at_2s": float(control_detected_trials_at_2s / control_trials) if control_trials else 0.0,
        "control_recall_at_3s": float(control_detected_trials_at_3s / control_trials) if control_trials else 0.0,
        "switch_detect_rate": float(switch_detected_trials / switch_trials) if switch_trials else 0.0,
        "switch_detect_rate_at_2.8s": float(switch_detected_trials_at_2p8s / switch_trials) if switch_trials else 0.0,
        "release_detect_rate": float(release_detected_trials / release_trials) if release_trials else 0.0,
        "switch_latency_s": _median(switch_latencies, default=float("inf")),
        "release_latency_s": _median(release_latencies, default=float("inf")),
        "detection_latency_s": _median(detection_latencies, default=float("inf")),
        "control_trials": float(control_trials),
        "switch_trials": float(switch_trials),
        "release_trials": float(release_trials),
        "idle_fp_event_count": float(idle_selected_events),
        "idle_time_sec": float(idle_duration_sec),
        "inference_ms": float(inference_ms),
    }
    metrics_4class = compute_classification_metrics(
        y_true=y4_true,
        y_pred=y4_pred,
        labels=[_freq_label(float(freq)) for freq in freqs],
        decision_time_samples_s=times4,
        itr_class_count=4,
        decision_time_fallback_s=float(profile.win_sec),
    )
    metrics_2class = compute_classification_metrics(
        y_true=y2_true,
        y_pred=y2_pred,
        labels=["idle", "control"],
        decision_time_samples_s=times2,
        itr_class_count=2,
        decision_time_fallback_s=float(profile.win_sec),
    )
    decision_mode = parse_decision_time_mode(decision_time_mode)
    async_mode = parse_decision_time_mode(async_decision_time_mode)
    return {
        "async_metrics": async_metrics,
        "metrics_4class": metrics_4class,
        "metrics_2class": metrics_2class,
        "metrics_5class": None,
        "paper_lens_metrics_4class": dict(metrics_4class),
        "paper_lens_metrics_2class": dict(metrics_2class),
        "async_lens_metrics_4class": dict(metrics_4class),
        "async_lens_metrics_2class": dict(metrics_2class),
        "paper_lens_decision_time_mode": str(decision_mode),
        "async_lens_decision_time_mode": str(async_mode),
        "decision_params": dict(decision_params),
        "trial_events": trial_events,
    }


def _build_candidate_context(
    *,
    merged_dataset: MergedLocalDataset,
    split: RepeatedGroupSplit,
    model_name: str,
    win_sec: float,
    model_params: Mapping[str, Any],
    decoder_compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    gpu_warmup: bool,
    gpu_cache_policy: str,
    control_state_mode: str,
    decision_time_mode: str,
    async_decision_time_mode: str,
    replay_policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    train_segments = _subset_segments(merged_dataset.trial_segments, split.train_indices)
    gate_segments = _subset_segments(merged_dataset.trial_segments, split.gate_indices)
    holdout_segments = _subset_segments(merged_dataset.trial_segments, split.holdout_indices)
    if not train_segments:
        raise ValueError("train split is empty")
    decoder = create_decoder(
        str(model_name),
        sampling_rate=int(merged_dataset.sampling_rate),
        freqs=merged_dataset.freqs,
        win_sec=float(win_sec),
        step_sec=float(DEFAULT_TDCA_LOCAL_STEP_SEC),
        model_params=dict(model_params),
        decoder_compute_backend=str(decoder_compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=bool(gpu_warmup),
        gpu_cache_policy=str(gpu_cache_policy),
    )
    if bool(getattr(decoder, "requires_fit", False)):
        decoder.fit(train_segments)
    train_rows = build_feature_rows_with_decoder(decoder, train_segments)
    gate_rows = build_feature_rows_with_decoder(decoder, gate_segments or train_segments)
    holdout_rows = build_feature_rows_with_decoder(decoder, holdout_segments or gate_segments or train_segments)
    gate_model = PerFrequencyLogRegGate()
    gate_fit_summary = gate_model.fit(
        rows=train_rows,
        freqs=merged_dataset.freqs,
        fit_config=DEFAULT_LOGREG_FIT_CONFIG,
    )
    scored_gate_rows = _score_rows_with_gate(gate_rows, gate=gate_model)
    scored_holdout_rows = _score_rows_with_gate(holdout_rows, gate=gate_model)
    control_rows = [
        row
        for row in train_rows
        if str(
            row.get(
                "trial_role",
                infer_trial_role(label=str(row.get("label", "")), expected_freq=row.get("expected_freq")),
            )
        )
        .strip()
        .lower()
        == "control"
    ]
    idle_rows = [
        row
        for row in train_rows
        if str(
            row.get(
                "trial_role",
                infer_trial_role(label=str(row.get("label", "")), expected_freq=row.get("expected_freq")),
            )
        )
        .strip()
        .lower()
        != "control"
    ]
    control_feature_means, control_feature_stds = _feature_stats(control_rows)
    idle_feature_means, idle_feature_stds = _feature_stats(idle_rows)
    base_profile = fit_threshold_profile(
        train_rows,
        freqs=merged_dataset.freqs,
        win_sec=float(win_sec),
        step_sec=float(DEFAULT_TDCA_LOCAL_STEP_SEC),
        min_enter_windows=1,
        min_exit_windows=1,
        gate_policy=DEFAULT_GATE_POLICY,
        evaluation_rows=gate_rows,
        dynamic_stop_enabled=False,
        control_state_mode=parse_control_state_mode(control_state_mode),
    )
    base_profile = replace(
        base_profile,
        model_name=str(model_name),
        model_params=dict(model_params),
        eeg_channels=tuple(int(item) for item in merged_dataset.board_eeg_channels),
        control_feature_means=control_feature_means,
        control_feature_stds=control_feature_stds,
        idle_feature_means=idle_feature_means,
        idle_feature_stds=idle_feature_stds,
        control_state_mode=parse_control_state_mode(control_state_mode),
        runtime_backend_preference=str(decoder_compute_backend),
        runtime_precision_preference=str(gpu_precision),
    )
    inference_ms = _measure_decoder_inference_ms(decoder, holdout_segments or gate_segments or train_segments)
    gate_profile, gate_board, threshold_board = _search_gate_profile(
        base_profile=base_profile,
        scored_gate_rows=scored_gate_rows,
        freqs=merged_dataset.freqs,
        inference_ms=float(inference_ms),
        decision_time_mode=str(decision_time_mode),
        async_decision_time_mode=str(async_decision_time_mode),
    )
    default_bundle = _evaluate_structured_rows(
        scored_rows=scored_holdout_rows,
        profile=gate_profile,
        freqs=merged_dataset.freqs,
        decision_params=_default_decision_params(),
        inference_ms=float(inference_ms),
        decision_time_mode=str(decision_time_mode),
        async_decision_time_mode=str(async_decision_time_mode),
    )
    state_payload = None
    try:
        state_payload = decoder.get_state()
    except Exception:
        state_payload = None
    return {
        "decoder": decoder,
        "model_name": str(model_name),
        "model_params": dict(model_params),
        "decoder_variant": str(dict(model_params).get("decoder_variant", DEFAULT_TDCA_LOCAL_DECODER_VARIANT)),
        "state_payload": state_payload,
        "train_segments": train_segments,
        "gate_segments": gate_segments,
        "holdout_segments": holdout_segments,
        "train_rows": train_rows,
        "gate_rows": gate_rows,
        "holdout_rows": holdout_rows,
        "scored_gate_rows": scored_gate_rows,
        "scored_holdout_rows": scored_holdout_rows,
        "gate_model": gate_model,
        "gate_fit_summary": gate_fit_summary,
        "gate_profile": gate_profile,
        "gate_search_board": gate_board,
        "gate_exit_threshold_board": threshold_board,
        "default_holdout_bundle": default_bundle,
        "inference_ms": float(inference_ms),
        "replay_backend_policy": dict(replay_policy or {}),
    }


def _aggregate_metric_bundle(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    async_rows = [dict(item.get("async_metrics", {})) for item in bundles]
    metrics4_rows = [dict(item.get("metrics_4class", {})) for item in bundles]
    metrics2_rows = [dict(item.get("metrics_2class", {})) for item in bundles]
    async_keys = sorted({key for row in async_rows for key in row.keys()})
    metrics4_keys = sorted({key for row in metrics4_rows for key in row.keys() if key not in {"confusion_matrix", "labels", "y_true", "y_pred", "decision_time_samples_s"}})
    metrics2_keys = sorted({key for row in metrics2_rows for key in row.keys() if key not in {"confusion_matrix", "labels", "y_true", "y_pred", "decision_time_samples_s"}})
    return {
        "async_metrics": {key: _median([row.get(key) for row in async_rows], default=0.0) for key in async_keys},
        "metrics_4class": {key: _median([row.get(key) for row in metrics4_rows], default=0.0) for key in metrics4_keys},
        "metrics_2class": {key: _median([row.get(key) for row in metrics2_rows], default=0.0) for key in metrics2_keys},
    }


def _decision_param_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for values in product(
        DEFAULT_DECISION_GRID_CANDIDATE_MIN_WINDOWS,
        DEFAULT_DECISION_GRID_ARMED_MIN_WINDOWS,
        DEFAULT_DECISION_GRID_LAMBDA,
        DEFAULT_DECISION_GRID_UPPER,
        DEFAULT_DECISION_GRID_LOWER,
        DEFAULT_DECISION_GRID_REFRACTORY,
    ):
        candidate_min_windows, armed_min_windows, lambda_decay, upper_commit_th, lower_idle_th, refractory_sec = values
        grid.append(
            {
                "candidate_min_windows": int(candidate_min_windows),
                "armed_min_windows": int(armed_min_windows),
                "lambda_decay": float(lambda_decay),
                "upper_commit_th": float(upper_commit_th),
                "lower_idle_th": float(lower_idle_th),
                "refractory_sec": float(refractory_sec),
            }
        )
    return grid


def _render_markdown(report_payload: Mapping[str, Any]) -> str:
    async_metrics = dict(report_payload.get("chosen_async_metrics", {}) or {})
    metrics4 = dict(report_payload.get("chosen_metrics_4class", {}) or {})
    lines = [
        "# TDCA Local Opt",
        "",
        f"- Generated at: `{report_payload.get('generated_at', '')}`",
        f"- Task: `{report_payload.get('task', '')}`",
        f"- Chosen model: `{report_payload.get('chosen_model', '')}`",
        f"- Decoder variant: `{report_payload.get('decoder_variant', '')}`",
        f"- Profile saved: `{report_payload.get('profile_saved', False)}`",
        f"- Profile path: `{report_payload.get('chosen_profile_path', '')}`",
        f"- Report status: `{report_payload.get('status', 'ok')}`",
        "",
        "## Async Metrics",
        "",
        f"- idle_fp_per_min: `{async_metrics.get('idle_fp_per_min', '')}`",
        f"- control_recall: `{async_metrics.get('control_recall', '')}`",
        f"- control_recall_at_3s: `{async_metrics.get('control_recall_at_3s', '')}`",
        f"- switch_latency_s: `{async_metrics.get('switch_latency_s', '')}`",
        f"- release_latency_s: `{async_metrics.get('release_latency_s', '')}`",
        f"- inference_ms: `{async_metrics.get('inference_ms', '')}`",
        "",
        "## 4-Class",
        "",
        f"- acc: `{metrics4.get('acc', '')}`",
        f"- macro_f1: `{metrics4.get('macro_f1', '')}`",
        f"- itr_bpm: `{metrics4.get('itr_bpm', '')}`",
    ]
    return "\n".join(lines).strip() + "\n"


def _split_replay_policy(
    *,
    merged_dataset: MergedLocalDataset,
    split: RepeatedGroupSplit,
    win_sec: float,
    env_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    all_indices = tuple(
        sorted(
            {
                *(int(item) for item in split.train_indices),
                *(int(item) for item in split.gate_indices),
                *(int(item) for item in split.holdout_indices),
            }
        )
    )
    estimated_window_count = _estimate_window_count(
        trial_segments=_subset_segments(merged_dataset.trial_segments, all_indices),
        sampling_rate=int(merged_dataset.sampling_rate),
        win_sec=float(win_sec),
        step_sec=float(DEFAULT_TDCA_LOCAL_STEP_SEC),
    )
    return _resolve_replay_backend_policy(
        env_preflight=env_preflight,
        estimated_window_count=int(estimated_window_count),
    )


def _run_baseline_suite(
    *,
    merged_dataset: MergedLocalDataset,
    splits: Sequence[RepeatedGroupSplit],
    env_preflight: Mapping[str, Any],
    config: TDCALocalOptConfig,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    decision_params = _default_decision_params()
    for model_name in DEFAULT_BASELINE_MODELS:
        candidate = _baseline_candidate(model_name, Nh=int(config.Nh))
        holdout_bundles: list[dict[str, Any]] = []
        inference_values: list[float] = []
        replay_policies: list[dict[str, Any]] = []
        for split in splits:
            replay_policy = _split_replay_policy(
                merged_dataset=merged_dataset,
                split=split,
                win_sec=float(candidate["win_sec"]),
                env_preflight=env_preflight,
            )
            context = _build_candidate_context(
                merged_dataset=merged_dataset,
                split=split,
                model_name=str(model_name),
                win_sec=float(candidate["win_sec"]),
                model_params=dict(candidate["model_params"]),
                decoder_compute_backend=str(config.compute_backend),
                gpu_device=int(config.gpu_device),
                gpu_precision=str(config.gpu_precision),
                gpu_warmup=bool(config.gpu_warmup),
                gpu_cache_policy=str(config.gpu_cache_policy),
                control_state_mode=str(config.control_state_mode),
                decision_time_mode=str(config.decision_time_mode),
                async_decision_time_mode=str(config.async_decision_time_mode),
                replay_policy=replay_policy,
            )
            bundle = _evaluate_structured_rows(
                scored_rows=context["scored_holdout_rows"],
                profile=context["gate_profile"],
                freqs=merged_dataset.freqs,
                decision_params=decision_params,
                inference_ms=float(context["inference_ms"]),
                decision_time_mode=str(config.decision_time_mode),
                async_decision_time_mode=str(config.async_decision_time_mode),
            )
            holdout_bundles.append(bundle)
            inference_values.append(float(context["inference_ms"]))
            replay_policies.append(dict(replay_policy))
        aggregated = _aggregate_metric_bundle(holdout_bundles)
        async_metrics = dict(aggregated.get("async_metrics", {}))
        async_metrics["inference_ms"] = _median(inference_values, default=float("inf"))
        policy = replay_policies[0] if replay_policies else {}
        row = {
            "model_name": str(model_name),
            "decoder_variant": "n/a",
            "algorithm_alignment": "not_applicable",
            "paper_tdca_projection_enabled": False,
            "candidate": dict(candidate),
            "split_fingerprints": [str(split.fingerprint) for split in splits],
            "decision_params": dict(decision_params),
            "metrics_median": async_metrics,
            "metrics_4class_median": dict(aggregated.get("metrics_4class", {})),
            "metrics_2class_median": dict(aggregated.get("metrics_2class", {})),
            "inference_ms": float(async_metrics.get("inference_ms", float("inf"))),
            "effective_replay_backend": str(policy.get("effective_replay_backend", "cpu")),
            "gpu_replay_speedup": float(policy.get("gpu_replay_speedup", 0.0)),
            "gpu_replay_eligible": bool(policy.get("gpu_replay_eligible", False)),
            "gpu_replay_reason": str(policy.get("gpu_replay_reason", "")),
        }
        row["rank_key"] = [float(item) for item in _rank_metrics_key(async_metrics)]
        results.append(row)
    results.sort(key=lambda item: _candidate_rank_tuple(item))
    return results


def run_tdca_local_opt(
    config: TDCALocalOptConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    report_paths = _resolve_report_paths(config)
    report_dir = Path(report_paths["report_dir"]).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = Path(report_paths["run_log"]).expanduser().resolve()
    progress_snapshot_path = Path(report_paths["progress_snapshot"]).expanduser().resolve()

    def log(text: str) -> None:
        _append_run_log(run_log_path, text)
        if log_fn is not None:
            log_fn(text)

    started_at = time.perf_counter()
    progress_state: dict[str, Any] = {
        "task": "tdca-local-opt",
        "stage": "prepare",
        "stage_label": _tdca_progress_label("prepare"),
        "detail": "准备运行目录",
        "run_index": 0,
        "run_total": 0,
        "config_index": 0,
        "config_total": 0,
        "report_dir": str(report_dir),
        "report_path": str(Path(report_paths["report_json"]).expanduser().resolve()),
        "profile_path": str(Path(config.output_profile_path).expanduser().resolve()),
        "progress_percent": 0,
        "elapsed_s": 0.0,
        "eta_s": None,
    }

    def emit_progress(*, force: bool = False, **updates: Any) -> None:
        progress_state.update(updates)
        stage_name = str(progress_state.get("stage", "") or "prepare")
        run_index = int(progress_state.get("run_index", 0) or 0)
        run_total = int(progress_state.get("run_total", 0) or 0)
        elapsed_s = float(max(time.perf_counter() - started_at, 0.0))
        progress_percent = progress_state.get("progress_percent")
        if not isinstance(progress_percent, (int, float)):
            progress_percent = _tdca_progress_percent(stage_name, run_index=run_index, run_total=run_total)
        progress_state["stage_label"] = str(progress_state.get("stage_label") or _tdca_progress_label(stage_name))
        progress_state["progress_percent"] = int(max(0, min(100, int(progress_percent))))
        progress_state["elapsed_s"] = float(elapsed_s)
        if progress_state["progress_percent"] > 0 and progress_state["progress_percent"] < 100:
            eta_s = elapsed_s * ((100.0 / float(progress_state["progress_percent"])) - 1.0)
            progress_state["eta_s"] = float(max(eta_s, 0.0))
        elif progress_state["progress_percent"] >= 100:
            progress_state["eta_s"] = 0.0
        else:
            progress_state["eta_s"] = None
        atomic_write_text(progress_snapshot_path, json_dumps(json_safe(progress_state)) + "\n")
        if progress_fn is not None:
            progress_fn(dict(progress_state))

    _validate_tdca_local_config(config)
    atomic_write_text(Path(report_paths["run_config"]).expanduser().resolve(), json_dumps(json_safe(asdict(config))) + "\n")
    log(f"[tdca-local-opt] report directory prepared: {report_dir}")
    emit_progress(
        force=True,
        stage="prepare",
        stage_label=_tdca_progress_label("prepare"),
        detail=f"运行目录已准备：{report_dir}",
        run_index=1,
        run_total=4,
    )
    env_preflight = preflight_tdca_local_env(
        compute_backend=str(config.compute_backend),
        gpu_device=int(config.gpu_device),
        gpu_precision=str(config.gpu_precision),
    )
    log(
        "[tdca-local-opt] compute backend prepared: "
        f"requested={config.compute_backend} effective={env_preflight.get('effective_backend','cpu')}"
    )
    emit_progress(
        force=True,
        stage="prepare",
        stage_label=_tdca_progress_label("prepare"),
        detail="后端预检完成，开始载入数据",
        run_index=2,
        run_total=4,
    )
    merged_dataset = _load_merged_dataset(config)
    emit_progress(
        force=True,
        stage="prepare",
        stage_label=_tdca_progress_label("prepare"),
        detail=f"数据已载入：trial={len(merged_dataset.trial_segments)}",
        run_index=3,
        run_total=4,
    )
    splits = build_repeated_group_splits(
        merged_dataset.trial_segments,
        repeats=int(config.multi_seed_count),
        seed=int(config.seed),
    )
    split_fingerprints = [str(item.fingerprint) for item in splits]
    unique_fingerprint_count = len(set(split_fingerprints))
    seed_effective = {
        "requested_repeats": int(config.multi_seed_count),
        "generated_repeats": int(len(splits)),
        "unique_split_fingerprints": int(unique_fingerprint_count),
        "effective_repeats": int(unique_fingerprint_count),
        "invalid": bool(int(config.multi_seed_count) > 1 and unique_fingerprint_count < 2),
    }
    status = "invalid" if bool(seed_effective["invalid"]) else "ok"
    status_reasons: list[str] = []
    if bool(seed_effective["invalid"]):
        status_reasons.append("split_fingerprints_not_effective")

    emit_progress(
        force=True,
        stage="prepare",
        stage_label=_tdca_progress_label("prepare"),
        detail=f"分组完成：repeat={len(splits)}",
        run_index=4,
        run_total=4,
    )
    emit_progress(
        force=True,
        stage="baseline_opening",
        stage_label=_tdca_progress_label("baseline_opening"),
        detail="开始基线开场对比",
        run_index=0,
        run_total=len(DEFAULT_BASELINE_MODELS),
    )
    baseline_opening = _run_baseline_suite(
        merged_dataset=merged_dataset,
        splits=splits,
        env_preflight=env_preflight,
        config=config,
    )
    emit_progress(
        force=True,
        stage="baseline_opening",
        stage_label=_tdca_progress_label("baseline_opening"),
        detail="基线开场对比完成",
        run_index=len(DEFAULT_BASELINE_MODELS),
        run_total=len(DEFAULT_BASELINE_MODELS),
    )

    candidate_results: dict[str, list[dict[str, Any]]] = {}
    gate_search_by_candidate: dict[str, list[dict[str, Any]]] = {}
    gate_exit_by_candidate: dict[str, list[dict[str, Any]]] = {}
    candidate_context_cache: dict[tuple[str, int], dict[str, Any]] = {}
    candidate_grid = [
        {
            "decoder_variant": str(decoder_variant),
            "win_sec": float(win_sec),
            "delay_steps": int(delay_steps),
            "n_components": int(n_components),
        }
        for decoder_variant, win_sec, delay_steps, n_components in product(
            tuple(str(item) for item in DEFAULT_TDCA_LOCAL_DECODER_VARIANTS),
            tuple(float(item) for item in config.win_candidates),
            tuple(int(item) for item in config.tdca_delay_steps),
            tuple(int(item) for item in config.tdca_n_components),
        )
    ]
    total_candidates = len(candidate_grid) * max(len(splits), 1)
    progress_index = 0
    for split in splits:
        for candidate in candidate_grid:
            progress_index += 1
            key = _candidate_key(
                decoder_variant=str(candidate["decoder_variant"]),
                win_sec=float(candidate["win_sec"]),
                delay_steps=int(candidate["delay_steps"]),
                n_components=int(candidate["n_components"]),
            )
            log(
                f"[tdca-local-opt] repeat={split.repeat_index + 1}/{len(splits)} "
                f"candidate={progress_index}/{total_candidates} {key}"
            )
            emit_progress(
                stage="candidate_search",
                stage_label=_tdca_progress_label("candidate_search"),
                detail=(
                    f"repeat {split.repeat_index + 1}/{len(splits)} | "
                    f"candidate {progress_index}/{total_candidates} | {key}"
                ),
                model_name="tdca",
                run_index=progress_index,
                run_total=total_candidates,
            )
            replay_policy = _split_replay_policy(
                merged_dataset=merged_dataset,
                split=split,
                win_sec=float(candidate["win_sec"]),
                env_preflight=env_preflight,
            )
            context = _build_candidate_context(
                merged_dataset=merged_dataset,
                split=split,
                model_name=DEFAULT_TDCA_LOCAL_MODEL,
                win_sec=float(candidate["win_sec"]),
                model_params=_default_model_params(
                    model_name=DEFAULT_TDCA_LOCAL_MODEL,
                    Nh=int(config.Nh),
                    delay_steps=int(candidate["delay_steps"]),
                    n_components=int(candidate["n_components"]),
                    decoder_variant=str(candidate["decoder_variant"]),
                ),
                decoder_compute_backend=str(config.compute_backend),
                gpu_device=int(config.gpu_device),
                gpu_precision=str(config.gpu_precision),
                gpu_warmup=bool(config.gpu_warmup),
                gpu_cache_policy=str(config.gpu_cache_policy),
                control_state_mode=str(config.control_state_mode),
                decision_time_mode=str(config.decision_time_mode),
                async_decision_time_mode=str(config.async_decision_time_mode),
                replay_policy=replay_policy,
            )
            candidate_context_cache[(key, int(split.repeat_index))] = context
            candidate_results.setdefault(key, []).append(
                {
                    "repeat_index": int(split.repeat_index),
                    "candidate": dict(candidate),
                    "holdout_bundle": dict(context["default_holdout_bundle"]),
                    "inference_ms": float(context["inference_ms"]),
                    "replay_backend_policy": dict(replay_policy),
                }
            )
            gate_search_by_candidate.setdefault(key, []).extend(
                [
                    {
                        "repeat_index": int(split.repeat_index),
                        "decoder_variant": str(candidate["decoder_variant"]),
                        **dict(row),
                    }
                    for row in list(context["gate_search_board"])
                ]
            )
            gate_exit_by_candidate.setdefault(key, []).extend(
                [
                    {
                        "repeat_index": int(split.repeat_index),
                        "decoder_variant": str(candidate["decoder_variant"]),
                        **dict(row),
                    }
                    for row in list(context["gate_exit_threshold_board"])
                ]
            )

    tdca_search_board: list[dict[str, Any]] = []
    for key, rows in candidate_results.items():
        bundles = [dict(item.get("holdout_bundle", {})) for item in rows]
        aggregated = _aggregate_metric_bundle(bundles)
        async_metrics = dict(aggregated.get("async_metrics", {}))
        async_metrics["inference_ms"] = _median([item.get("inference_ms") for item in rows], default=float("inf"))
        rank_key = _rank_metrics_key(async_metrics)
        sample_candidate = dict(rows[0].get("candidate", {}))
        variant_metadata = _tdca_variant_metadata(sample_candidate.get("decoder_variant"))
        tdca_search_board.append(
            {
                "candidate_key": str(key),
                "candidate": sample_candidate,
                "decoder_variant": str(variant_metadata["decoder_variant"]),
                "algorithm_alignment": str(variant_metadata["algorithm_alignment"]),
                "paper_tdca_projection_enabled": bool(variant_metadata["paper_tdca_projection_enabled"]),
                "repeat_count": int(len(rows)),
                "metrics_median": async_metrics,
                "metrics_4class_median": dict(aggregated.get("metrics_4class", {})),
                "metrics_2class_median": dict(aggregated.get("metrics_2class", {})),
                "rank_key": [float(item) for item in rank_key],
            }
        )
    tdca_search_board.sort(key=_tdca_board_sort_key)
    top_candidates: list[dict[str, Any]] = []
    selected_candidate_keys: set[str] = set()
    for decoder_variant in DEFAULT_TDCA_LOCAL_DECODER_VARIANTS:
        variant_rows = [
            dict(item)
            for item in tdca_search_board
            if str(item.get("decoder_variant", "")) == str(decoder_variant)
        ]
        for row in variant_rows[:4]:
            candidate_key = str(row.get("candidate_key", ""))
            if candidate_key and candidate_key not in selected_candidate_keys:
                top_candidates.append(dict(row))
                selected_candidate_keys.add(candidate_key)
    if len(top_candidates) < DEFAULT_TDCA_LOCAL_TOP_K:
        for row in tdca_search_board:
            candidate_key = str(row.get("candidate_key", ""))
            if candidate_key and candidate_key not in selected_candidate_keys:
                top_candidates.append(dict(row))
                selected_candidate_keys.add(candidate_key)
            if len(top_candidates) >= DEFAULT_TDCA_LOCAL_TOP_K:
                break
    top_candidates.sort(key=_tdca_board_sort_key)
    variant_summary: list[dict[str, Any]] = []
    selected_for_decision = {str(row.get("candidate_key", "")) for row in top_candidates}
    for decoder_variant in DEFAULT_TDCA_LOCAL_DECODER_VARIANTS:
        variant_rows = [
            dict(item)
            for item in tdca_search_board
            if str(item.get("decoder_variant", "")) == str(decoder_variant)
        ]
        if not variant_rows:
            continue
        best_variant_row = dict(variant_rows[0])
        variant_summary.append(
            {
                "decoder_variant": str(decoder_variant),
                "algorithm_alignment": str(best_variant_row.get("algorithm_alignment", "")),
                "paper_tdca_projection_enabled": bool(best_variant_row.get("paper_tdca_projection_enabled", False)),
                "best_candidate": dict(best_variant_row.get("candidate", {})),
                "metrics_median": dict(best_variant_row.get("metrics_median", {})),
                "rank_key": [float(item) for item in best_variant_row.get("rank_key", [])],
                "selected_for_decision_search": bool(str(best_variant_row.get("candidate_key", "")) in selected_for_decision),
            }
        )

    decision_grid = _decision_param_grid()
    total_decision_runs = max(len(top_candidates) * len(decision_grid), 1)
    decision_progress_index = 0
    decision_aggregate_rows: list[dict[str, Any]] = []
    final_candidate_rows: list[dict[str, Any]] = []
    candidate_best_global_params: dict[str, dict[str, Any]] = {}
    for candidate_row in top_candidates:
        candidate_key = str(candidate_row.get("candidate_key", ""))
        gate_eval_by_param: dict[str, list[dict[str, Any]]] = {}
        param_payload_by_key: dict[str, dict[str, Any]] = {}
        for split in splits:
            context = candidate_context_cache[(candidate_key, int(split.repeat_index))]
            for params in decision_grid:
                param_key = _make_decision_params_key(params)
                param_payload_by_key[param_key] = dict(params)
                bundle = _evaluate_structured_rows(
                    scored_rows=context["scored_gate_rows"],
                    profile=context["gate_profile"],
                    freqs=merged_dataset.freqs,
                    decision_params=params,
                    inference_ms=float(context["inference_ms"]),
                    decision_time_mode=str(config.decision_time_mode),
                    async_decision_time_mode=str(config.async_decision_time_mode),
                )
                gate_eval_by_param.setdefault(param_key, []).append(bundle)
        candidate_decision_board: list[dict[str, Any]] = []
        for param_key, bundles in gate_eval_by_param.items():
            decision_progress_index += 1
            aggregated = _aggregate_metric_bundle(bundles)
            async_metrics = dict(aggregated.get("async_metrics", {}))
            rank_key = _rank_metrics_key(async_metrics)
            variant_metadata = _tdca_variant_metadata(dict(candidate_row.get("candidate", {})).get("decoder_variant"))
            emit_progress(
                stage="decision_search",
                stage_label=_tdca_progress_label("decision_search"),
                detail=(
                    f"candidate {decision_progress_index}/{total_decision_runs} | "
                    f"{candidate_key} | params={param_key}"
                ),
                model_name="tdca",
                run_index=decision_progress_index,
                run_total=total_decision_runs,
            )
            row = {
                "candidate_key": str(candidate_key),
                "candidate": dict(candidate_row.get("candidate", {})),
                "decoder_variant": str(variant_metadata["decoder_variant"]),
                "algorithm_alignment": str(variant_metadata["algorithm_alignment"]),
                "paper_tdca_projection_enabled": bool(variant_metadata["paper_tdca_projection_enabled"]),
                "decision_params": dict(param_payload_by_key[param_key]),
                "metrics_median": async_metrics,
                "metrics_4class_median": dict(aggregated.get("metrics_4class", {})),
                "metrics_2class_median": dict(aggregated.get("metrics_2class", {})),
                "rank_key": [float(item) for item in rank_key],
            }
            candidate_decision_board.append(row)
            decision_aggregate_rows.append(dict(row))
        candidate_decision_board.sort(key=_tdca_board_sort_key)
        best_global_params = dict(candidate_decision_board[0]["decision_params"])
        candidate_best_global_params[candidate_key] = best_global_params
        holdout_bundles: list[dict[str, Any]] = []
        for split in splits:
            context = candidate_context_cache[(candidate_key, int(split.repeat_index))]
            holdout_bundles.append(
                _evaluate_structured_rows(
                    scored_rows=context["scored_holdout_rows"],
                    profile=context["gate_profile"],
                    freqs=merged_dataset.freqs,
                    decision_params=best_global_params,
                    inference_ms=float(context["inference_ms"]),
                    decision_time_mode=str(config.decision_time_mode),
                    async_decision_time_mode=str(config.async_decision_time_mode),
                )
            )
        aggregated_holdout = _aggregate_metric_bundle(holdout_bundles)
        async_metrics = dict(aggregated_holdout.get("async_metrics", {}))
        async_metrics["inference_ms"] = _median(
            [bundle.get("async_metrics", {}).get("inference_ms") for bundle in holdout_bundles],
            default=float("inf"),
        )
        rank_key = _rank_metrics_key(async_metrics)
        variant_metadata = _tdca_variant_metadata(dict(candidate_row.get("candidate", {})).get("decoder_variant"))
        final_candidate_rows.append(
            {
                "candidate_key": str(candidate_key),
                "candidate": dict(candidate_row.get("candidate", {})),
                "decoder_variant": str(variant_metadata["decoder_variant"]),
                "algorithm_alignment": str(variant_metadata["algorithm_alignment"]),
                "paper_tdca_projection_enabled": bool(variant_metadata["paper_tdca_projection_enabled"]),
                "decision_params": best_global_params,
                "repeat_count": int(len(holdout_bundles)),
                "metrics_median": async_metrics,
                "metrics_4class_median": dict(aggregated_holdout.get("metrics_4class", {})),
                "metrics_2class_median": dict(aggregated_holdout.get("metrics_2class", {})),
                "rank_key": [float(item) for item in rank_key],
            }
        )

    decision_aggregate_rows.sort(key=_tdca_board_sort_key)
    final_candidate_rows.sort(key=_tdca_board_sort_key)
    decision_release_values = {
        round(_safe_float(dict(row.get("metrics_median", {})).get("release_latency_s", float("inf")), float("inf")), 6)
        for row in decision_aggregate_rows
        if np.isfinite(_safe_float(dict(row.get("metrics_median", {})).get("release_latency_s", float("inf")), float("inf")))
    }
    decision_switch_values = {
        round(_safe_float(dict(row.get("metrics_median", {})).get("switch_latency_s", float("inf")), float("inf")), 6)
        for row in decision_aggregate_rows
        if np.isfinite(_safe_float(dict(row.get("metrics_median", {})).get("switch_latency_s", float("inf")), float("inf")))
    }
    if len(decision_aggregate_rows) > 1 and len(decision_release_values) < 2 and len(decision_switch_values) < 2:
        status = "invalid"
        status_reasons.append("decision_search_not_effective")
    chosen_row = dict(final_candidate_rows[0]) if final_candidate_rows else {}
    emit_progress(
        force=True,
        stage="baseline_seal",
        stage_label=_tdca_progress_label("baseline_seal"),
        detail="开始基线封板对比",
        run_index=0,
        run_total=len(DEFAULT_BASELINE_MODELS),
    )
    baseline_seal = _run_baseline_suite(
        merged_dataset=merged_dataset,
        splits=splits,
        env_preflight=env_preflight,
        config=config,
    )
    emit_progress(
        force=True,
        stage="baseline_seal",
        stage_label=_tdca_progress_label("baseline_seal"),
        detail="基线封板对比完成",
        run_index=len(DEFAULT_BASELINE_MODELS),
        run_total=len(DEFAULT_BASELINE_MODELS),
    )
    chosen_model_rationale = ""
    if chosen_row and baseline_seal:
        best_baseline_rank = min((_candidate_rank_tuple(row) for row in baseline_seal), default=tuple())
        chosen_rank_tuple = _candidate_rank_tuple(chosen_row)
        chosen_model_rationale = (
            "tdca_not_clearly_superior"
            if best_baseline_rank and chosen_rank_tuple >= best_baseline_rank
            else "tdca_superior_on_primary_ranking"
        )

    profile_saved = False
    profile_v2_saved = False
    chosen_profile_path: Optional[str] = None
    profile_v2_path: Optional[str] = None
    chosen_profile: Optional[ThresholdProfile] = None
    chosen_replay_policy: dict[str, Any] = {}
    chosen_variant_metadata = _tdca_variant_metadata(
        dict(chosen_row.get("candidate", {})).get("decoder_variant", DEFAULT_TDCA_LOCAL_DECODER_VARIANT)
    )
    if chosen_row and status == "ok":
        emit_progress(
            force=True,
            stage="finalize",
            stage_label=_tdca_progress_label("finalize"),
            detail="基于全量数据重建最终 profile",
            run_index=1,
            run_total=3,
        )
        chosen_candidate = dict(chosen_row.get("candidate", {}))
        full_split = RepeatedGroupSplit(
            repeat_index=0,
            train_indices=tuple(range(len(merged_dataset.trial_segments))),
            gate_indices=tuple(range(len(merged_dataset.trial_segments))),
            holdout_indices=tuple(range(len(merged_dataset.trial_segments))),
            fingerprint="full-data",
        )
        full_replay_policy = _split_replay_policy(
            merged_dataset=merged_dataset,
            split=full_split,
            win_sec=float(chosen_candidate.get("win_sec", 3.0)),
            env_preflight=env_preflight,
        )
        chosen_replay_policy = dict(full_replay_policy)
        full_context = _build_candidate_context(
            merged_dataset=merged_dataset,
            split=full_split,
            model_name=DEFAULT_TDCA_LOCAL_MODEL,
            win_sec=float(chosen_candidate.get("win_sec", 3.0)),
            model_params=_default_model_params(
                model_name=DEFAULT_TDCA_LOCAL_MODEL,
                Nh=int(config.Nh),
                delay_steps=int(chosen_candidate.get("delay_steps", 2)),
                n_components=int(chosen_candidate.get("n_components", 2)),
                decoder_variant=str(chosen_candidate.get("decoder_variant", DEFAULT_TDCA_LOCAL_DECODER_VARIANT)),
            ),
            decoder_compute_backend=str(config.compute_backend),
            gpu_device=int(config.gpu_device),
            gpu_precision=str(config.gpu_precision),
            gpu_warmup=bool(config.gpu_warmup),
            gpu_cache_policy=str(config.gpu_cache_policy),
            control_state_mode=str(config.control_state_mode),
            decision_time_mode=str(config.decision_time_mode),
            async_decision_time_mode=str(config.async_decision_time_mode),
            replay_policy=full_replay_policy,
        )
        final_profile = replace(
            full_context["gate_profile"],
            model_name=DEFAULT_TDCA_LOCAL_MODEL,
            model_params={
                **dict(full_context.get("model_params", {})),
                "state": full_context.get("state_payload"),
            },
            benchmark_metrics=dict(chosen_row.get("metrics_median", {})),
            metadata={
                "task": "tdca-local-opt",
                "env_preflight": env_preflight,
                "split_fingerprints": split_fingerprints,
                "seed_effective": seed_effective,
                "decision_params": dict(chosen_row.get("decision_params", {})),
                "candidate": dict(chosen_candidate),
                "decoder_variant": str(chosen_variant_metadata["decoder_variant"]),
                "algorithm_alignment": str(chosen_variant_metadata["algorithm_alignment"]),
                "paper_tdca_projection_enabled": bool(chosen_variant_metadata["paper_tdca_projection_enabled"]),
                "effective_replay_backend": str(full_replay_policy.get("effective_replay_backend", "cpu")),
                "gpu_replay_speedup": float(full_replay_policy.get("gpu_replay_speedup", 0.0)),
                "gpu_replay_eligible": bool(full_replay_policy.get("gpu_replay_eligible", False)),
                "gpu_replay_reason": str(full_replay_policy.get("gpu_replay_reason", "")),
                "chosen_model_rationale": str(chosen_model_rationale),
            },
            recommended_for_realtime=True,
        )
        save_profile(final_profile, Path(config.output_profile_path).expanduser().resolve())
        chosen_profile = final_profile
        chosen_profile_path = str(Path(config.output_profile_path).expanduser().resolve())
        profile_saved = True
        emit_progress(
            force=True,
            stage="finalize",
            stage_label=_tdca_progress_label("finalize"),
            detail="主 profile 已保存，开始导出 profile_v2",
            run_index=2,
            run_total=3,
            profile_path=chosen_profile_path,
        )

        gate_payload = dict(full_context["gate_model"].to_payload())
        per_freq_gate = {}
        per_freq_thresholds = dict(final_profile.frequency_specific_thresholds or {})
        for freq_key, payload in dict(gate_payload.get("per_freq", {})).items():
            merged_payload = dict(payload)
            merged_payload.update(dict(per_freq_thresholds.get(str(freq_key), {})))
            per_freq_gate[str(freq_key)] = merged_payload
        profile_v2 = build_profile_v2(
            base_profile=final_profile,
            per_freq_gate=per_freq_gate,
            metrics=dict(chosen_row.get("metrics_median", {})),
            feature_names=tuple(getattr(full_context["gate_model"], "feature_names", DEFAULT_GATE_FEATURES)),
            evidence=dict(chosen_row.get("decision_params", {})),
            refractory_sec=float(dict(chosen_row.get("decision_params", {})).get("refractory_sec", 0.8)),
        )
        profile_v2_output = Path(config.output_profile_path).expanduser().resolve().with_name(
            f"{Path(config.output_profile_path).stem}_v2.json"
        )
        atomic_write_text(profile_v2_output, json_dumps(json_safe(profile_v2.to_payload())) + "\n")
        profile_v2_saved = True
        profile_v2_path = str(profile_v2_output)
        emit_progress(
            force=True,
            stage="finalize",
            stage_label=_tdca_progress_label("finalize"),
            detail="profile_v2 已保存，开始生成报告",
            run_index=3,
            run_total=3,
            profile_path=chosen_profile_path,
        )
    else:
        emit_progress(
            force=True,
            stage="finalize",
            stage_label=_tdca_progress_label("finalize"),
            detail="本次运行未保存 profile，仍将输出报告",
            run_index=3,
            run_total=3,
        )

    chosen_async_metrics = dict(chosen_row.get("metrics_median", {}))
    chosen_metrics_4class = dict(chosen_row.get("metrics_4class_median", {}))
    chosen_metrics_2class = dict(chosen_row.get("metrics_2class_median", {}))
    report_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "tdca_local_opt",
        "task": "tdca-local-opt",
        "status": status,
        "status_reasons": list(status_reasons),
        "dataset_manifest_session1": str(Path(config.dataset_manifest_session1).expanduser().resolve()),
        "selected_dataset_manifests_session1": [str(path) for path in merged_dataset.manifest_paths],
        "selected_dataset_count_session1": int(len(merged_dataset.manifest_paths)),
        "report_path": str(report_paths["report_json"]),
        "report_dir": str(report_paths["report_dir"]),
        "run_log_path": str(report_paths["run_log"]),
        "progress_snapshot_path": str(report_paths["progress_snapshot"]),
        "sampling_rate": int(merged_dataset.sampling_rate),
        "freqs": [float(item) for item in merged_dataset.freqs],
        "board_eeg_channels": [int(item) for item in merged_dataset.board_eeg_channels],
        "quality_summary_session1": [dict(item) for item in merged_dataset.quality_rows],
        "quality_total_trials_session1": int(len(merged_dataset.trial_segments)),
        "quality_kept_trials_session1": int(len(merged_dataset.trial_segments)),
        "quality_filter": {},
        "split_fingerprints": split_fingerprints,
        "seed_effective": seed_effective,
        "env_preflight": env_preflight,
        "decoder_variant": str(chosen_variant_metadata["decoder_variant"]),
        "algorithm_alignment": str(chosen_variant_metadata["algorithm_alignment"]),
        "paper_tdca_projection_enabled": bool(chosen_variant_metadata["paper_tdca_projection_enabled"]),
        "effective_replay_backend": str(chosen_replay_policy.get("effective_replay_backend", "cpu")),
        "gpu_replay_speedup": float(chosen_replay_policy.get("gpu_replay_speedup", env_preflight.get("gpu_replay_speedup", 0.0))),
        "gpu_replay_eligible": bool(chosen_replay_policy.get("gpu_replay_eligible", False)),
        "gpu_replay_reason": str(chosen_replay_policy.get("gpu_replay_reason", "")),
        "tdca_search_board": tdca_search_board,
        "gate_exit_search_board": gate_exit_by_candidate.get(str(chosen_row.get("candidate_key", "")), []),
        "decision_search_board": decision_aggregate_rows[:128],
        "variant_summary": variant_summary,
        "ranking_boards": {
            "end_to_end": final_candidate_rows,
            "classifier_only": [],
        },
        "model_results": final_candidate_rows,
        "chosen_model": DEFAULT_TDCA_LOCAL_MODEL if chosen_row else "",
        "chosen_rank": 1 if chosen_row else 0,
        "chosen_metrics": chosen_async_metrics,
        "chosen_async_metrics": chosen_async_metrics,
        "chosen_metrics_4class": chosen_metrics_4class,
        "chosen_metrics_2class": chosen_metrics_2class,
        "chosen_metrics_5class": None,
        "chosen_profile_path": chosen_profile_path,
        "chosen_meets_acceptance": bool(profile_meets_acceptance(chosen_async_metrics)) if chosen_row else False,
        "profile_saved": bool(profile_saved),
        "profile_v2_saved": bool(profile_v2_saved),
        "profile_v2_path": profile_v2_path,
        "recommended_model": DEFAULT_TDCA_LOCAL_MODEL if chosen_row else "",
        "gate_policy": DEFAULT_GATE_POLICY,
        "control_state_mode": str(config.control_state_mode),
        "decision_time_mode": str(config.decision_time_mode),
        "async_decision_time_mode": str(config.async_decision_time_mode),
        "chosen_model_rationale": str(chosen_model_rationale),
        "baseline_opening": baseline_opening,
        "baseline_seal": baseline_seal,
    }

    report_paths["report_dir"].mkdir(parents=True, exist_ok=True)
    atomic_write_text(report_paths["run_config"], json_dumps(json_safe(asdict(config))) + "\n")
    atomic_write_text(
        report_paths["selection_snapshot"],
        json_dumps(
            json_safe(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "selected_dataset_manifests_session1": [str(path) for path in merged_dataset.manifest_paths],
                    "split_fingerprints": split_fingerprints,
                    "seed_effective": seed_effective,
                    "decoder_variant": str(chosen_variant_metadata["decoder_variant"]),
                    "algorithm_alignment": str(chosen_variant_metadata["algorithm_alignment"]),
                    "paper_tdca_projection_enabled": bool(chosen_variant_metadata["paper_tdca_projection_enabled"]),
                    "variant_summary": variant_summary,
                    "chosen_model_rationale": str(chosen_model_rationale),
                }
            )
        )
        + "\n",
    )
    atomic_write_text(report_paths["report_json"], json_dumps(json_safe(report_payload)) + "\n")
    atomic_write_text(report_paths["report_md"], _render_markdown(report_payload))
    log(f"[tdca-local-opt] report saved: {report_paths['report_json']}")
    if chosen_profile is not None:
        log(f"[tdca-local-opt] profile saved: {chosen_profile_path}")
    emit_progress(
        force=True,
        stage="complete",
        stage_label=_tdca_progress_label("complete"),
        detail="TDCA 本地异步优化完成",
        run_index=1,
        run_total=1,
        report_path=str(report_paths["report_json"]),
        profile_path=str(chosen_profile_path or Path(config.output_profile_path).expanduser().resolve()),
        progress_percent=100,
    )
    return report_payload

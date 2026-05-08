from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from brain_workspace.paths import SSVEP_PROFILE_DIR

from .async_fbcca_idle_standalone import (
    DEFAULT_ASYNC_DECISION_TIME_MODE,
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_COMPUTE_BACKEND_NAME,
    DEFAULT_DYNAMIC_STOP_ALPHA,
    DEFAULT_GATE_POLICY,
    DEFAULT_GPU_CACHE_MODE,
    DEFAULT_GPU_DEVICE_ID,
    DEFAULT_GPU_PRECISION_NAME,
    DEFAULT_NH,
    DEFAULT_PAPER_DECISION_TIME_MODE,
    DEFAULT_PROFILE_PATH,
    DEFAULT_STEP_SEC,
    DEFAULT_WIN_SEC,
    ThresholdProfile,
    TrialSpec,
    atomic_copy_text_file,
    atomic_write_text,
    build_feature_rows_with_decoder,
    create_decoder,
    evaluate_profile_on_feature_rows,
    fit_threshold_profile,
    json_dumps,
    json_safe,
    parse_compute_backend_name,
    parse_control_state_mode,
    parse_decision_time_mode,
    parse_gate_policy,
    parse_gpu_cache_policy,
    parse_gpu_precision,
    save_profile,
    summarize_profile_quality,
)
from .dataset import LoadedDataset, load_collection_dataset
from .profile_deployment_audit import (
    DEFAULT_FAST_CONTROL_PRETRAIN_TASK,
    FAST_CONTROL_RELEASE_THRESHOLDS,
    fast_control_release_failures,
    logreg_coefficients_all_zero,
)
from .profile_v2 import DEFAULT_GATE_FEATURES, build_profile_v2
from .run_artifacts import make_run_tag, publish_deployed_profile, resolve_ssvep_run_artifacts


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = PROJECT_DIR / "artifacts" / "runs" / "local"
DEFAULT_REALTIME_PROFILE_PATH = SSVEP_PROFILE_DIR / "fbcca_profile.json"
DEFAULT_REALTIME_PROFILE_V2_PATH = SSVEP_PROFILE_DIR / "fbcca_profile_v2.json"
DEFAULT_FBCCA_THRESHOLD_TASK = "fbcca-threshold-pretrain"
DEFAULT_FAST_CONTROL_WIN_SEC_CANDIDATES = (1.5, 2.0, 2.5, 3.0)
DEFAULT_FAST_CONTROL_GATE_POLICY_CANDIDATES = ("balanced", "speed")
DEFAULT_FAST_CONTROL_MIN_ENTER_CANDIDATES = (1, 2)
DEFAULT_FAST_CONTROL_MIN_EXIT_CANDIDATES = (1, 2)
DEFAULT_FAST_CONTROL_STATE_MODE_CANDIDATES = ("unified", "frequency-specific-threshold")
DEFAULT_FAST_CONTROL_RELEASE_IDLE_FP_MAX = float(FAST_CONTROL_RELEASE_THRESHOLDS["idle_fp_per_min_max"])
DEFAULT_FAST_CONTROL_RELEASE_CONTROL_RECALL_MIN = float(FAST_CONTROL_RELEASE_THRESHOLDS["control_recall_min"])
DEFAULT_FAST_CONTROL_RELEASE_CONTROL_RECALL_AT_3S_MIN = float(
    FAST_CONTROL_RELEASE_THRESHOLDS["control_recall_at_3s_min"]
)
DEFAULT_FAST_CONTROL_RELEASE_SWITCH_DETECT_RATE_MIN = float(FAST_CONTROL_RELEASE_THRESHOLDS["switch_detect_rate_min"])
DEFAULT_FAST_CONTROL_RELEASE_RELEASE_LATENCY_MAX_S = float(FAST_CONTROL_RELEASE_THRESHOLDS["release_latency_s_max"])
DEFAULT_FAST_CONTROL_RELEASE_SWITCH_LATENCY_MAX_S = float(FAST_CONTROL_RELEASE_THRESHOLDS["switch_latency_s_max"])


@dataclass(frozen=True)
class FBCCAThresholdPretrainConfig:
    dataset_manifest_session1: Path
    output_profile_path: Path = DEFAULT_PROFILE_PATH.with_name("fbcca_profile.json")
    report_path: Path = DEFAULT_REPORT_ROOT / "report.json"
    dataset_manifests: tuple[Path, ...] = ()
    report_root_dir: Optional[Path] = DEFAULT_REPORT_ROOT
    organize_report_dir: bool = True
    win_sec: float = DEFAULT_WIN_SEC
    step_sec: float = DEFAULT_STEP_SEC
    gate_policy: str = DEFAULT_GATE_POLICY
    min_enter_windows: int = 1
    min_exit_windows: int = 2
    fast_control_grid_search: bool = True
    win_sec_candidates: tuple[float, ...] = DEFAULT_FAST_CONTROL_WIN_SEC_CANDIDATES
    gate_policy_candidates: tuple[str, ...] = DEFAULT_FAST_CONTROL_GATE_POLICY_CANDIDATES
    min_enter_windows_candidates: tuple[int, ...] = DEFAULT_FAST_CONTROL_MIN_ENTER_CANDIDATES
    min_exit_windows_candidates: tuple[int, ...] = DEFAULT_FAST_CONTROL_MIN_EXIT_CANDIDATES
    control_state_mode_candidates: tuple[str, ...] = DEFAULT_FAST_CONTROL_STATE_MODE_CANDIDATES
    dynamic_stop_enabled: bool = False
    dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA
    seed: int = DEFAULT_CALIBRATION_SEED
    compute_backend: str = "cpu"
    gpu_device: int = DEFAULT_GPU_DEVICE_ID
    gpu_precision: str = DEFAULT_GPU_PRECISION_NAME
    gpu_warmup: bool = False
    gpu_cache_policy: str = DEFAULT_GPU_CACHE_MODE
    decision_time_mode: str = DEFAULT_PAPER_DECISION_TIME_MODE
    async_decision_time_mode: str = DEFAULT_ASYNC_DECISION_TIME_MODE
    progress_heartbeat_sec: float = 1.0
    publish_realtime: bool = True


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(Path(path), json_dumps(json_safe(payload)) + "\n")


def _dedup_manifest_paths(config: FBCCAThresholdPretrainConfig) -> tuple[Path, ...]:
    candidates = list(config.dataset_manifests or ())
    if not candidates:
        candidates.append(config.dataset_manifest_session1)
    else:
        first = Path(config.dataset_manifest_session1).expanduser().resolve()
        if all(Path(item).expanduser().resolve() != first for item in candidates):
            candidates.insert(0, first)
    dedup: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        path = Path(item).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return tuple(dedup)


def _load_compatible_datasets(manifests: Sequence[Path]) -> list[LoadedDataset]:
    datasets = [load_collection_dataset(Path(path)) for path in manifests]
    if not datasets:
        raise ValueError("threshold pretrain requires at least one dataset manifest")
    base = datasets[0]
    for dataset in datasets[1:]:
        if int(dataset.sampling_rate) != int(base.sampling_rate):
            raise ValueError(
                f"sampling_rate mismatch: {dataset.manifest_path} has {dataset.sampling_rate}, "
                f"expected {base.sampling_rate}"
            )
        if tuple(float(item) for item in dataset.freqs) != tuple(float(item) for item in base.freqs):
            raise ValueError(f"freqs mismatch: {dataset.manifest_path}")
        if tuple(int(item) for item in dataset.board_eeg_channels) != tuple(int(item) for item in base.board_eeg_channels):
            raise ValueError(f"board_eeg_channels mismatch: {dataset.manifest_path}")
        if str(dataset.subject_id).strip() != str(base.subject_id).strip():
            raise ValueError(
                f"subject_id mismatch: {dataset.manifest_path} has {dataset.subject_id}, "
                f"expected {base.subject_id}"
            )
    return datasets


def _merge_trial_segments(datasets: Sequence[LoadedDataset]) -> list[tuple[TrialSpec, np.ndarray]]:
    merged: list[tuple[TrialSpec, np.ndarray]] = []
    next_trial_id = 0
    for dataset_index, dataset in enumerate(datasets):
        for trial, segment in dataset.trial_segments:
            merged_trial = replace(
                trial,
                trial_id=int(next_trial_id),
                block_index=int(dataset_index * 100000 + max(int(trial.block_index), 0)),
            )
            merged.append((merged_trial, np.ascontiguousarray(np.asarray(segment, dtype=np.float64))))
            next_trial_id += 1
    return merged


def _count_segments(segments: Sequence[tuple[TrialSpec, np.ndarray]]) -> dict[str, int]:
    control = sum(1 for trial, _segment in segments if trial.expected_freq is not None)
    idle = int(len(segments) - control)
    return {
        "total": int(len(segments)),
        "control": int(control),
        "idle": int(idle),
    }


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            output[str(key)] = float(value)
    return output


def _metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            output[str(key)] = {
                str(child_key): (
                    float(child_value)
                    if isinstance(child_value, (int, float)) and np.isfinite(float(child_value))
                    else child_value
                )
                for child_key, child_value in value.items()
            }
        elif isinstance(value, (int, float)) and np.isfinite(float(value)):
            output[str(key)] = float(value)
        elif isinstance(value, (str, bool)):
            output[str(key)] = value
    return output


def _all_zero_logreg_gate(per_freq_gate: dict[str, dict[str, Any]]) -> bool:
    return logreg_coefficients_all_zero(per_freq_gate)


def _profile_v2_payload(profile: ThresholdProfile, metrics: dict[str, Any]) -> dict[str, Any]:
    per_freq_gate: dict[str, dict[str, Any]] = {}
    freq_specific = profile.frequency_specific_thresholds
    if isinstance(freq_specific, dict):
        for key, payload in freq_specific.items():
            item = dict(payload or {})
            if str(getattr(profile, "control_state_mode", "")) == "frequency-specific-threshold":
                per_freq_gate[str(key)] = {
                    "enter_score_th": float(item.get("enter_score_th", profile.enter_score_th)),
                    "enter_ratio_th": float(item.get("enter_ratio_th", profile.enter_ratio_th)),
                    "enter_margin_th": float(item.get("enter_margin_th", profile.enter_margin_th)),
                    "exit_score_th": float(item.get("exit_score_th", profile.exit_score_th)),
                    "exit_ratio_th": float(item.get("exit_ratio_th", profile.exit_ratio_th)),
                    "switch_enter_score_th": float(
                        item.get("switch_enter_score_th", profile.switch_enter_score_th or profile.enter_score_th)
                    ),
                    "switch_enter_ratio_th": float(
                        item.get("switch_enter_ratio_th", profile.switch_enter_ratio_th or profile.enter_ratio_th)
                    ),
                    "switch_enter_margin_th": float(
                        item.get("switch_enter_margin_th", profile.switch_enter_margin_th or profile.enter_margin_th)
                    ),
                    "enter_log_lr_th": item.get("enter_log_lr_th", profile.enter_log_lr_th),
                    "exit_log_lr_th": item.get("exit_log_lr_th", profile.exit_log_lr_th),
                }
            else:
                per_freq_gate[str(key)] = {
                    "coef": list(item.get("coef", [0.0] * len(DEFAULT_GATE_FEATURES))),
                    "intercept": float(item.get("intercept", 0.0)),
                    "enter_logit_th": float(item.get("enter_logit_th", item.get("enter_log_lr_th", 0.35))),
                    "exit_logit_th": float(item.get("exit_logit_th", item.get("exit_log_lr_th", 0.05))),
                }
    if not per_freq_gate:
        enter_llr = profile.enter_log_lr_th
        exit_llr = profile.exit_log_lr_th
        for freq in tuple(float(item) for item in profile.freqs):
            per_freq_gate[f"{freq:g}"] = {
                "coef": [0.0] * len(DEFAULT_GATE_FEATURES),
                "intercept": 0.0,
                "enter_logit_th": float(enter_llr if enter_llr is not None else 0.35),
                "exit_logit_th": float(exit_llr if exit_llr is not None else 0.05),
            }
    if str(getattr(profile, "control_state_mode", "")) == "frequency-specific-threshold" and profile.frequency_specific_thresholds:
        gate_type = "frequency_specific_threshold"
    else:
        gate_type = "threshold_only_global_gate" if _all_zero_logreg_gate(per_freq_gate) else "frequency_specific_logreg"
    profile_v2 = build_profile_v2(
        base_profile=profile,
        per_freq_gate=per_freq_gate,
        metrics=dict(metrics),
        feature_names=tuple(DEFAULT_GATE_FEATURES),
        gate_type=gate_type,
        evidence={
            "lambda": 0.85,
            "beta_consistency": 0.5,
            "upper_commit_th": 2.2,
            "lower_idle_th": 0.4,
        },
        refractory_sec=0.8,
    )
    return dict(profile_v2.to_payload())


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = dict(candidate.get("async_metrics", {}))
    idle_fp = float(metrics.get("idle_fp_per_min", float("inf")))
    idle_selected = float(metrics.get("idle_selected_windows_per_min", float("inf")))
    control_recall = float(metrics.get("control_recall", 0.0))
    control_recall_25 = float(metrics.get("control_recall_at_2.5s", metrics.get("control_recall_at_3s", 0.0)))
    control_recall_3 = float(metrics.get("control_recall_at_3s", 0.0))
    switch_rate_28 = float(metrics.get("switch_detect_rate_at_2.8s", 0.0))
    release_latency = float(metrics.get("release_latency_s", float("inf")))
    switch_latency = float(metrics.get("switch_latency_s", float("inf")))
    gate_policy = str(candidate.get("gate_policy", "balanced"))
    speed_penalty = 0.0 if gate_policy == "speed" else 0.01
    zero_idle_penalty = 0.0 if idle_fp <= 1e-12 else 1000.0 + idle_fp
    return (
        0.0 if _candidate_release_valid(candidate) else 1.0,
        zero_idle_penalty,
        idle_selected,
        -control_recall_25,
        -control_recall_3,
        -control_recall,
        release_latency,
        switch_latency,
        -switch_rate_28,
        speed_penalty,
        float(candidate.get("win_sec", float("inf"))),
        float(candidate.get("min_enter_windows", 99)),
        float(candidate.get("min_exit_windows", 99)),
    )


def _candidate_allowed(candidate: dict[str, Any]) -> bool:
    metrics = dict(candidate.get("async_metrics", {}))
    if str(candidate.get("gate_policy", "balanced")) != "speed":
        return True
    if float(metrics.get("idle_fp_per_min", float("inf"))) > 1e-12:
        return False
    if float(metrics.get("idle_selected_windows_per_min", float("inf"))) > 12.0:
        return False
    return True


def _fast_control_release_failures(metrics: dict[str, Any]) -> list[str]:
    return fast_control_release_failures(metrics)


def _candidate_release_valid(candidate: dict[str, Any]) -> bool:
    if not _candidate_allowed(candidate):
        return False
    return not _fast_control_release_failures(dict(candidate.get("async_metrics", {})))


def _dedup_float_candidates(values: Sequence[float], *, fallback: float) -> tuple[float, ...]:
    output: list[float] = []
    for value in tuple(values or ()) + (float(fallback),):
        item = float(value)
        if not np.isfinite(item) or item <= 0.0:
            continue
        if all(abs(item - existing) > 1e-9 for existing in output):
            output.append(item)
    return tuple(output)


def _dedup_int_candidates(values: Sequence[int], *, fallback: int) -> tuple[int, ...]:
    output: list[int] = []
    for value in tuple(values or ()) + (int(fallback),):
        item = max(1, int(value))
        if item not in output:
            output.append(item)
    return tuple(output)


def _dedup_control_state_mode_candidates(values: Sequence[str], *, fallback: str) -> tuple[str, ...]:
    output: list[str] = []
    for value in tuple(values or ()) + (str(fallback),):
        item = parse_control_state_mode(str(value))
        if item not in output:
            output.append(item)
    return tuple(output)


def _copy_if_different(source: Path, destination: Path) -> bool:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        shutil.copy2(src, dst)
    return True


def _publish_realtime_outputs(
    *,
    profile_path: Path,
    profile_v2_path: Path,
    artifacts_run_dir: Path,
    report_json: Path,
    run_tag: str,
) -> dict[str, Any]:
    deployed = publish_deployed_profile(
        source_profile=profile_path,
        source_profile_v2=profile_v2_path if profile_v2_path.exists() else None,
        run_dir=artifacts_run_dir,
        task=DEFAULT_FBCCA_THRESHOLD_TASK,
        run_tag=run_tag,
        report_json=report_json,
        extra_metadata={"profile_kind": "fbcca_threshold_only"},
    )
    realtime_profile_copied = _copy_if_different(profile_path, DEFAULT_REALTIME_PROFILE_PATH)
    realtime_v2_copied = _copy_if_different(profile_v2_path, DEFAULT_REALTIME_PROFILE_V2_PATH)
    return {
        **deployed,
        "realtime_profile_json": str(DEFAULT_REALTIME_PROFILE_PATH) if realtime_profile_copied else "",
        "realtime_profile_v2_json": str(DEFAULT_REALTIME_PROFILE_V2_PATH) if realtime_v2_copied else "",
    }


def _render_markdown_report(payload: dict[str, Any]) -> str:
    metrics = dict(payload.get("chosen_async_metrics", {}))
    validation = dict(payload.get("profile_validation_status", {}))
    warnings = [str(item) for item in validation.get("warnings", [])]
    lines = [
        "# FBCCA Threshold Pretrain",
        "",
        f"- Task: `{payload.get('task', DEFAULT_FBCCA_THRESHOLD_TASK)}`",
        f"- Model: `{payload.get('chosen_model', 'fbcca')}`",
        f"- Profile: `{payload.get('profile_path', '')}`",
        f"- Profile v2: `{payload.get('profile_v2_path', '')}`",
        f"- Realtime profile: `{dict(payload.get('published_paths', {})).get('realtime_profile_json', '')}`",
        "",
        "## Metrics",
        "",
        f"- Control recall: {float(metrics.get('control_recall', 0.0)):.4f}",
        f"- Control recall @2.5s: {float(metrics.get('control_recall_at_2.5s', 0.0)):.4f}",
        f"- Control recall @3s: {float(metrics.get('control_recall_at_3s', 0.0)):.4f}",
        f"- Idle false positives/min: {float(metrics.get('idle_fp_per_min', 0.0)):.4f}",
        f"- Idle selected windows/min: {float(metrics.get('idle_selected_windows_per_min', 0.0)):.4f}",
        f"- Detection latency: {float(metrics.get('detection_latency_s', float('inf'))):.4f}s",
        f"- Switch detect rate: {float(metrics.get('switch_detect_rate', 0.0)):.4f}",
        f"- Release latency: {float(metrics.get('release_latency_s', float('inf'))):.4f}s",
        "",
        "## Notes",
        "",
        "- FBCCA decoder parameters were kept at the fixed all-channel default.",
        "- This run fitted only decision/gate thresholds from the selected SSVEP collection data.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines).rstrip() + "\n"


def run_fbcca_threshold_pretrain(
    config: FBCCAThresholdPretrainConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    task_name = DEFAULT_FBCCA_THRESHOLD_TASK
    manifests = _dedup_manifest_paths(config)
    run_tag = make_run_tag(task=task_name)
    artifacts = resolve_ssvep_run_artifacts(
        task=task_name,
        report_path=Path(config.report_path).expanduser().resolve(),
        output_profile_path=Path(config.output_profile_path).expanduser().resolve(),
        organize_report_dir=bool(config.organize_report_dir),
        report_root_dir=(
            Path(config.report_root_dir).expanduser().resolve()
            if config.report_root_dir is not None
            else None
        ),
        run_tag=run_tag,
    )
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    log_lines: list[str] = []
    last_emit = 0.0

    def log(message: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        log_lines.append(line)
        atomic_write_text(artifacts.run_log, "\n".join(log_lines).rstrip() + "\n")
        if log_fn is not None:
            log_fn(message)

    def emit(stage: str, percent: float, detail: str, *, force: bool = False) -> None:
        nonlocal last_emit
        now = time.perf_counter()
        if not force and (now - last_emit) < max(float(config.progress_heartbeat_sec), 0.1):
            return
        last_emit = now
        payload = {
            "task": task_name,
            "stage": str(stage),
            "stage_label": str(stage),
            "detail": str(detail),
            "progress_percent": float(max(0.0, min(100.0, percent))),
            "elapsed_s": float(now - start_time),
            "eta_s": None,
            "report_path": str(artifacts.report_json),
            "profile_path": str(artifacts.output_profile),
            "report_dir": str(artifacts.run_dir),
        }
        _write_json_atomic(artifacts.progress_snapshot, payload)
        if progress_fn is not None:
            progress_fn(dict(payload))

    config_payload = {
        **json_safe(asdict(config)),
        "task": task_name,
        "dataset_manifests_resolved": [str(path) for path in manifests],
        "artifacts": artifacts.to_payload(),
    }
    _write_json_atomic(artifacts.run_config, config_payload)
    _write_json_atomic(
        artifacts.selection_snapshot,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "selected_manifests": [str(path) for path in manifests],
            "selected_manifest_count": int(len(manifests)),
            "task": task_name,
        },
    )

    emit("prepare", 5.0, "loading dataset manifests", force=True)
    log(f"Threshold-only FBCCA pretrain start | manifests={len(manifests)}")
    datasets = _load_compatible_datasets(manifests)
    base_dataset = datasets[0]
    segments = _merge_trial_segments(datasets)
    counts = _count_segments(segments)
    if counts["control"] <= 0:
        raise ValueError("threshold pretrain requires at least one control trial")
    if counts["idle"] <= 0:
        raise ValueError("threshold pretrain requires at least one idle trial")
    log(
        "Dataset loaded | "
        f"subject={base_dataset.subject_id} sessions={len(datasets)} "
        f"trials={counts['total']} control={counts['control']} idle={counts['idle']}"
    )

    fs = int(base_dataset.sampling_rate)
    freqs = tuple(float(item) for item in base_dataset.freqs)
    max_segment_sec = min(np.asarray(segment).shape[0] / max(float(fs), 1.0) for _trial, segment in segments)
    compute_backend = parse_compute_backend_name(config.compute_backend or DEFAULT_COMPUTE_BACKEND_NAME)
    gpu_precision = parse_gpu_precision(config.gpu_precision)
    gpu_cache_policy = parse_gpu_cache_policy(config.gpu_cache_policy)

    raw_win_candidates = (
        config.win_sec_candidates if bool(config.fast_control_grid_search) else (float(config.win_sec),)
    )
    win_candidates = tuple(
        min(float(win), float(max_segment_sec)) for win in _dedup_float_candidates(raw_win_candidates, fallback=config.win_sec)
    )
    win_candidates = _dedup_float_candidates(
        tuple(win for win in win_candidates if float(win) > 0.25),
        fallback=min(float(config.win_sec), float(max_segment_sec)),
    )
    if not win_candidates:
        raise ValueError("threshold pretrain has no valid window candidates")
    gate_policy_candidates = (
        tuple(parse_gate_policy(item) for item in config.gate_policy_candidates)
        if bool(config.fast_control_grid_search)
        else (parse_gate_policy(config.gate_policy),)
    )
    min_enter_candidates = _dedup_int_candidates(
        config.min_enter_windows_candidates if bool(config.fast_control_grid_search) else (config.min_enter_windows,),
        fallback=config.min_enter_windows,
    )
    min_exit_candidates = _dedup_int_candidates(
        config.min_exit_windows_candidates if bool(config.fast_control_grid_search) else (config.min_exit_windows,),
        fallback=config.min_exit_windows,
    )
    control_state_mode_candidates = _dedup_control_state_mode_candidates(
        (
            config.control_state_mode_candidates
            if bool(config.fast_control_grid_search)
            else (DEFAULT_FAST_CONTROL_STATE_MODE_CANDIDATES[0],)
        ),
        fallback=DEFAULT_FAST_CONTROL_STATE_MODE_CANDIDATES[0],
    )
    emit("feature_extract", 25.0, f"default FBCCA feature extraction, wins={list(win_candidates)}", force=True)
    candidate_rows: list[dict[str, Any]] = []
    decoder_backend_used = str(compute_backend)
    decoder_model_params: dict[str, Any] = {}
    best_payload: Optional[dict[str, Any]] = None
    for win_index, win_sec in enumerate(win_candidates):
        log(
            "Default FBCCA feature extraction | "
            f"win={float(win_sec):g}s step={float(config.step_sec):g}s backend={compute_backend}"
        )
        decoder = create_decoder(
            "fbcca_fixed_all8",
            sampling_rate=fs,
            freqs=freqs,
            win_sec=float(win_sec),
            step_sec=float(config.step_sec),
            model_params={"Nh": DEFAULT_NH},
            compute_backend=compute_backend,
            gpu_device=int(config.gpu_device),
            gpu_precision=gpu_precision,
            gpu_warmup=bool(config.gpu_warmup),
            gpu_cache_policy=gpu_cache_policy,
        )
        if decoder.requires_fit:
            decoder.fit(segments)
        feature_rows_for_win = build_feature_rows_with_decoder(decoder, segments)
        if not feature_rows_for_win:
            log(f"Feature rows empty for win={float(win_sec):g}s; skipping")
            continue
        decoder_backend_used = str(decoder.compute_backend_used)
        decoder_model_params = dict(getattr(decoder, "model_params", {}) or {})
        log(
            f"Feature rows ready | win={float(win_sec):g}s rows={len(feature_rows_for_win)} "
            f"backend_used={decoder.compute_backend_used}"
        )
        emit(
            "threshold_fit",
            45.0 + 30.0 * float(win_index + 1) / max(len(win_candidates), 1),
            f"fitting gate grid for win={float(win_sec):g}s",
            force=True,
        )
        for gate_policy in gate_policy_candidates:
            for min_enter in min_enter_candidates:
                for min_exit in min_exit_candidates:
                    for control_state_mode in control_state_mode_candidates:
                        profile_candidate = fit_threshold_profile(
                            feature_rows_for_win,
                            freqs=freqs,
                            win_sec=float(win_sec),
                            step_sec=float(config.step_sec),
                            min_enter_windows=max(1, int(min_enter)),
                            min_exit_windows=max(1, int(min_exit)),
                            gate_policy=parse_gate_policy(gate_policy),
                            evaluation_rows=feature_rows_for_win,
                            dynamic_stop_enabled=bool(config.dynamic_stop_enabled),
                            dynamic_stop_alpha=float(config.dynamic_stop_alpha),
                            control_state_mode=parse_control_state_mode(control_state_mode),
                        )
                        profile_quality_candidate = summarize_profile_quality(feature_rows_for_win, profile_candidate)
                        async_metrics_candidate = evaluate_profile_on_feature_rows(feature_rows_for_win, profile_candidate)
                        candidate = {
                            "win_sec": float(win_sec),
                            "step_sec": float(config.step_sec),
                            "gate_policy": parse_gate_policy(gate_policy),
                            "min_enter_windows": int(min_enter),
                            "min_exit_windows": int(min_exit),
                            "control_state_mode": parse_control_state_mode(control_state_mode),
                            "frequency_specific_threshold_count": int(
                                len(profile_candidate.frequency_specific_thresholds or {})
                            ),
                            "feature_rows": int(len(feature_rows_for_win)),
                            "async_metrics": _metric_payload(async_metrics_candidate),
                            "profile_quality": _finite_metrics(profile_quality_candidate),
                            "allowed_for_release": bool(
                                _candidate_allowed(
                                    {
                                        "gate_policy": parse_gate_policy(gate_policy),
                                        "async_metrics": async_metrics_candidate,
                                    }
                                )
                            ),
                        }
                        candidate_rows.append(candidate)
                        comparable = {
                            **candidate,
                            "profile": profile_candidate,
                            "feature_rows_raw": feature_rows_for_win,
                        }
                        if best_payload is None:
                            best_payload = comparable
                        else:
                            best_allowed = _candidate_allowed(best_payload)
                            new_allowed = _candidate_allowed(comparable)
                            if (new_allowed and not best_allowed) or (
                                new_allowed == best_allowed
                                and _candidate_sort_key(comparable) < _candidate_sort_key(best_payload)
                            ):
                                best_payload = comparable
    if best_payload is None:
        raise RuntimeError("default FBCCA did not produce any feature rows")

    profile = best_payload["profile"]
    feature_rows = list(best_payload["feature_rows_raw"])
    win_sec = float(best_payload["win_sec"])
    profile_quality = summarize_profile_quality(feature_rows, profile)
    async_metrics = evaluate_profile_on_feature_rows(feature_rows, profile)
    chosen_candidate = {
        key: value for key, value in best_payload.items() if key not in {"profile", "feature_rows_raw"}
    }
    chosen_candidate["async_metrics"] = _metric_payload(async_metrics)
    chosen_candidate["profile_quality"] = _finite_metrics(profile_quality)
    warnings: list[str] = []
    if float(async_metrics.get("control_recall", 0.0)) <= 0.0:
        warnings.append("control_recall is 0; realtime detection may not trigger")
    if float(profile_quality.get("raw_accuracy", 0.0)) < 0.25:
        warnings.append("raw 4-class accuracy is below chance-level guidance")
    if float(async_metrics.get("idle_fp_per_min", 0.0)) > 10.0:
        warnings.append("idle false positive rate is high")
    if str(chosen_candidate.get("gate_policy", "")) == "speed" and not _candidate_allowed(chosen_candidate):
        warnings.append("speed gate was not allowed by idle safety checks")
    release_failures = _fast_control_release_failures(async_metrics)
    warnings.extend(release_failures)
    recommended = not warnings
    validation_status = {
        "mode": "threshold_only_default_fbcca",
        "passed": bool(recommended),
        "warnings": warnings,
        "release_failures": list(release_failures),
        "feature_rows": int(len(feature_rows)),
        "segment_counts": counts,
        "selection_policy": DEFAULT_FAST_CONTROL_PRETRAIN_TASK if bool(config.fast_control_grid_search) else "single_candidate",
        "chosen_candidate": _metric_payload(chosen_candidate),
        "candidate_count": int(len(candidate_rows)),
        "release_thresholds": {
            "idle_fp_per_min_max": float(DEFAULT_FAST_CONTROL_RELEASE_IDLE_FP_MAX),
            "control_recall_min": float(DEFAULT_FAST_CONTROL_RELEASE_CONTROL_RECALL_MIN),
            "control_recall_at_3s_min": float(DEFAULT_FAST_CONTROL_RELEASE_CONTROL_RECALL_AT_3S_MIN),
            "switch_detect_rate_min": float(DEFAULT_FAST_CONTROL_RELEASE_SWITCH_DETECT_RATE_MIN),
            "release_latency_s_max": float(DEFAULT_FAST_CONTROL_RELEASE_RELEASE_LATENCY_MAX_S),
            "switch_latency_s_max": float(DEFAULT_FAST_CONTROL_RELEASE_SWITCH_LATENCY_MAX_S),
        },
    }
    model_params = dict(decoder_model_params or {})
    model_params.setdefault("Nh", DEFAULT_NH)
    model_params.setdefault("_decoder_model_name", "fbcca_fixed_all8")
    model_params.setdefault("subband_weight_mode", "chen_fixed")
    model_params["threshold_pretrain_only"] = True
    final_profile = replace(
        profile,
        model_name="fbcca",
        model_params=json_safe(model_params),
        calibration_split_seed=int(config.seed),
        benchmark_metrics=_finite_metrics(async_metrics),
        eeg_channels=tuple(int(item) for item in base_dataset.board_eeg_channels),
        gate_policy=parse_gate_policy(str(chosen_candidate.get("gate_policy", config.gate_policy))),
        control_state_mode=parse_control_state_mode(
            str(chosen_candidate.get("control_state_mode", DEFAULT_FAST_CONTROL_STATE_MODE_CANDIDATES[0]))
        ),
        frequency_specific_thresholds=profile.frequency_specific_thresholds,
        dynamic_stop=dict(profile.dynamic_stop or {}),
        channel_weight_mode=None,
        channel_weights=None,
        subband_weight_mode="chen_fixed",
        subband_weights=None,
        subband_weight_params=None,
        spatial_filter_mode=None,
        spatial_filter_rank=None,
        spatial_filter_state=None,
        joint_weight_training=None,
        profile_validation_status=validation_status,
        recommended_for_realtime=bool(recommended),
        runtime_backend_preference=str(decoder_backend_used),
        runtime_precision_preference=str(gpu_precision),
        training_window_policy=DEFAULT_FAST_CONTROL_PRETRAIN_TASK if bool(config.fast_control_grid_search) else "threshold_only_fixed_default_fbcca",
        metadata={
            "source": DEFAULT_FAST_CONTROL_PRETRAIN_TASK if bool(config.fast_control_grid_search) else "fbcca_threshold_pretrain",
            "profile_kind": "threshold_only_default_fbcca",
            "decoder_variant": "fbcca_fixed_all8",
            "fixed_fbcca_params": {
                "Nh": DEFAULT_NH,
                "channel_weights": "none",
                "subband_weight_mode": "chen_fixed",
                "spatial_filter": "none",
            },
            "dataset_manifests": [str(path) for path in manifests],
            "feature_rows": int(len(feature_rows)),
            "profile_quality": _finite_metrics(profile_quality),
            "async_metrics": _metric_payload(async_metrics),
            "chosen_candidate": _metric_payload(chosen_candidate),
            "candidate_count": int(len(candidate_rows)),
            "compute_backend_requested": str(compute_backend),
            "compute_backend_used": str(decoder_backend_used),
            "gpu_precision": str(gpu_precision),
        },
    )
    log(
        "Thresholds fitted | "
        f"score={final_profile.enter_score_th:.6f} ratio={final_profile.enter_ratio_th:.6f} "
        f"margin={final_profile.enter_margin_th:.6f} recall={float(async_metrics.get('control_recall', 0.0)):.4f} "
        f"idle_fp/min={float(async_metrics.get('idle_fp_per_min', 0.0)):.4f}"
    )

    emit("save", 80.0, "saving profile and report", force=True)
    save_profile(final_profile, artifacts.output_profile)
    profile_v2_payload = _profile_v2_payload(final_profile, async_metrics)
    _write_json_atomic(artifacts.profile_v2, profile_v2_payload)
    published_paths: dict[str, Any] = {}
    if bool(config.publish_realtime):
        published_paths = _publish_realtime_outputs(
            profile_path=artifacts.output_profile,
            profile_v2_path=artifacts.profile_v2,
            artifacts_run_dir=artifacts.run_dir,
            report_json=artifacts.report_json,
            run_tag=run_tag,
        )
        log(f"Realtime profile published | {published_paths.get('realtime_profile_json', '')}")

    report_payload = {
        "task": task_name,
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_path": str(artifacts.report_json),
        "report_dir": str(artifacts.run_dir),
        "profile_path": str(artifacts.output_profile),
        "chosen_profile_path": str(artifacts.output_profile),
        "profile_saved": True,
        "profile_v2_saved": True,
        "profile_v2_path": str(artifacts.profile_v2),
        "published_paths": published_paths,
        "chosen_model": "fbcca",
        "recommended_model": "fbcca",
        "chosen_model_rationale": DEFAULT_FAST_CONTROL_PRETRAIN_TASK if bool(config.fast_control_grid_search) else "threshold_only_default_fbcca",
        "chosen_candidate": _metric_payload(chosen_candidate),
        "candidate_grid": [_metric_payload(candidate) for candidate in candidate_rows],
        "chosen_async_metrics": _metric_payload(async_metrics),
        "chosen_metrics": _metric_payload(async_metrics),
        "chosen_metrics_4class": {
            "acc": float(profile_quality.get("raw_accuracy", 0.0)),
            "macro_f1": float(profile_quality.get("raw_accuracy", 0.0)),
        },
        "profile_validation_status": validation_status,
        "gate_calibration_valid": True,
        "run_valid_for_deployment": bool(recommended),
        "status_reasons": list(warnings),
        "quality_kept_trials_session1": int(counts["total"]),
        "quality_total_trials_session1": int(counts["total"]),
        "data_policy": "threshold_only_all_selected_sessions",
        "decision_search_target": DEFAULT_FAST_CONTROL_PRETRAIN_TASK if bool(config.fast_control_grid_search) else "threshold_only",
        "final_selection_target": "all_feature_rows",
        "dataset_manifests": [str(path) for path in manifests],
        "dataset_summary": {
            "subject_id": str(base_dataset.subject_id),
            "session_ids": [str(dataset.session_id) for dataset in datasets],
            "sampling_rate": int(fs),
            "freqs": [float(item) for item in freqs],
            "board_eeg_channels": [int(item) for item in base_dataset.board_eeg_channels],
            "segments": counts,
        },
        "config": config_payload,
        "artifacts": artifacts.to_payload(),
    }
    _write_json_atomic(artifacts.report_json, report_payload)
    atomic_write_text(artifacts.report_md, _render_markdown_report(report_payload))
    if artifacts.output_profile.resolve() != Path(config.output_profile_path).expanduser().resolve():
        requested_profile = Path(config.output_profile_path).expanduser().resolve()
        if not bool(config.organize_report_dir):
            atomic_copy_text_file(artifacts.output_profile, requested_profile)

    emit("complete", 100.0, "threshold-only FBCCA profile ready", force=True)
    log(f"Threshold-only FBCCA pretrain complete | profile={artifacts.output_profile}")
    return report_payload

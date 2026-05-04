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
    parse_decision_time_mode,
    parse_gate_policy,
    parse_gpu_cache_policy,
    parse_gpu_precision,
    save_profile,
    summarize_profile_quality,
)
from .dataset import LoadedDataset, load_collection_dataset
from .profile_v2 import DEFAULT_GATE_FEATURES, build_profile_v2
from .run_artifacts import make_run_tag, publish_deployed_profile, resolve_ssvep_run_artifacts


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = PROJECT_DIR / "artifacts" / "runs" / "local"
DEFAULT_REALTIME_PROFILE_PATH = SSVEP_PROFILE_DIR / "fbcca_profile.json"
DEFAULT_REALTIME_PROFILE_V2_PATH = SSVEP_PROFILE_DIR / "fbcca_profile_v2.json"
DEFAULT_FBCCA_THRESHOLD_TASK = "fbcca-threshold-pretrain"


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


def _profile_v2_payload(profile: ThresholdProfile, metrics: dict[str, Any]) -> dict[str, Any]:
    per_freq_gate: dict[str, dict[str, Any]] = {}
    freq_specific = profile.frequency_specific_thresholds
    if isinstance(freq_specific, dict):
        for key, payload in freq_specific.items():
            item = dict(payload or {})
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
    profile_v2 = build_profile_v2(
        base_profile=profile,
        per_freq_gate=per_freq_gate,
        metrics=dict(metrics),
        feature_names=tuple(DEFAULT_GATE_FEATURES),
        evidence={
            "lambda": 0.85,
            "beta_consistency": 0.5,
            "upper_commit_th": 2.2,
            "lower_idle_th": 0.4,
        },
        refractory_sec=0.8,
    )
    return dict(profile_v2.to_payload())


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
        f"- Idle false positives/min: {float(metrics.get('idle_fp_per_min', 0.0)):.4f}",
        f"- Detection latency: {float(metrics.get('detection_latency_s', float('inf'))):.4f}s",
        f"- Switch detect rate: {float(metrics.get('switch_detect_rate', 0.0)):.4f}",
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
    win_sec = min(float(config.win_sec), min(np.asarray(segment).shape[0] / max(float(fs), 1.0) for _trial, segment in segments))
    if win_sec <= 0.25:
        raise ValueError(f"invalid threshold pretrain window length: {win_sec:g}s")
    compute_backend = parse_compute_backend_name(config.compute_backend or DEFAULT_COMPUTE_BACKEND_NAME)
    gpu_precision = parse_gpu_precision(config.gpu_precision)
    gpu_cache_policy = parse_gpu_cache_policy(config.gpu_cache_policy)

    emit("feature_extract", 25.0, f"default FBCCA feature extraction, win={win_sec:g}s", force=True)
    log(
        "Default FBCCA feature extraction | "
        f"win={win_sec:g}s step={float(config.step_sec):g}s backend={compute_backend}"
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
    feature_rows = build_feature_rows_with_decoder(decoder, segments)
    if not feature_rows:
        raise RuntimeError("default FBCCA did not produce any feature rows")
    log(f"Feature rows ready | rows={len(feature_rows)} backend_used={decoder.compute_backend_used}")

    emit("threshold_fit", 55.0, "fitting realtime decision thresholds", force=True)
    profile = fit_threshold_profile(
        feature_rows,
        freqs=freqs,
        win_sec=float(win_sec),
        step_sec=float(config.step_sec),
        min_enter_windows=max(1, int(config.min_enter_windows)),
        min_exit_windows=max(1, int(config.min_exit_windows)),
        gate_policy=parse_gate_policy(config.gate_policy),
        evaluation_rows=feature_rows,
        dynamic_stop_enabled=bool(config.dynamic_stop_enabled),
        dynamic_stop_alpha=float(config.dynamic_stop_alpha),
        control_state_mode="unified",
    )
    profile_quality = summarize_profile_quality(feature_rows, profile)
    async_metrics = evaluate_profile_on_feature_rows(feature_rows, profile)
    warnings: list[str] = []
    if float(async_metrics.get("control_recall", 0.0)) <= 0.0:
        warnings.append("control_recall is 0; realtime detection may not trigger")
    if float(profile_quality.get("raw_accuracy", 0.0)) < 0.25:
        warnings.append("raw 4-class accuracy is below chance-level guidance")
    if float(async_metrics.get("idle_fp_per_min", 0.0)) > 10.0:
        warnings.append("idle false positive rate is high")
    recommended = not warnings
    validation_status = {
        "mode": "threshold_only_default_fbcca",
        "passed": bool(recommended),
        "warnings": warnings,
        "feature_rows": int(len(feature_rows)),
        "segment_counts": counts,
    }
    model_params = dict(getattr(decoder, "model_params", {}) or {})
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
        gate_policy=parse_gate_policy(config.gate_policy),
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
        runtime_backend_preference=str(decoder.compute_backend_used),
        runtime_precision_preference=str(gpu_precision),
        training_window_policy="threshold_only_fixed_default_fbcca",
        metadata={
            "source": "fbcca_threshold_pretrain",
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
            "async_metrics": _finite_metrics(async_metrics),
            "compute_backend_requested": str(compute_backend),
            "compute_backend_used": str(decoder.compute_backend_used),
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
        "chosen_model_rationale": "threshold_only_default_fbcca",
        "chosen_async_metrics": _finite_metrics(async_metrics),
        "chosen_metrics": _finite_metrics(async_metrics),
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
        "decision_search_target": "threshold_only",
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

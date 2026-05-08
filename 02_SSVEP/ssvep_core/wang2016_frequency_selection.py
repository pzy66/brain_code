from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .async_fbcca_idle_standalone import (
    DEFAULT_NH,
    DEFAULT_STEP_SEC,
    DEFAULT_WIN_SEC,
    FBCCAEngine,
    TrialSpec,
    build_feature_rows_from_score_matrix,
    evaluate_profile_on_feature_rows,
    extract_window_batch,
    fit_threshold_profile,
    json_dumps,
    json_safe,
    parse_gate_policy,
)
from .external_wang2016_dataset import (
    WANG2016_BLOCKS,
    WANG2016_REQUIRED_CHANNELS,
    WANG2016_SAMPLING_RATE,
    WANG2016_TARGET_FREQUENCIES,
    WANG2016_TARGET_COUNT,
    build_wang2016_segments,
    load_wang2016_subject,
    resolve_wang2016_command_frequencies,
)
from .stimulus_profiles import frame_lock_frequency_report


DEFAULT_SELECTION_WIN_SEC = 2.0
DEFAULT_SELECTION_STEP_SEC = 0.25
DEFAULT_MIN_FREQ_SPACING_HZ = 1.0
DEFAULT_MIN_SELECTED_FREQ_HZ = 9.5
DEFAULT_MAX_COMBINATIONS_REPORT = 50
DEFAULT_MAX_EVALUATED_COMBINATIONS = 40
DEFAULT_GATE_POLICIES = ("balanced",)
DEFAULT_CONTROL_STATE_MODES = ("unified", "frequency-specific-threshold")
DEFAULT_FRAME_LOCK_REFRESH_RATE_HZ = 240.0


@dataclass(frozen=True)
class Wang2016FrequencySelectionConfig:
    mat_path: Path
    channel_loc_path: Path
    report_path: Path
    win_sec: float = DEFAULT_SELECTION_WIN_SEC
    step_sec: float = DEFAULT_SELECTION_STEP_SEC
    min_freq_spacing_hz: float = DEFAULT_MIN_FREQ_SPACING_HZ
    min_selected_freq_hz: float = DEFAULT_MIN_SELECTED_FREQ_HZ
    gate_policies: tuple[str, ...] = DEFAULT_GATE_POLICIES
    control_state_modes: tuple[str, ...] = DEFAULT_CONTROL_STATE_MODES
    min_enter_windows: int = 1
    min_exit_windows: int = 1
    max_combinations_report: int = DEFAULT_MAX_COMBINATIONS_REPORT
    max_evaluated_combinations: int = DEFAULT_MAX_EVALUATED_COMBINATIONS
    compute_backend: str = "cpu"
    gpu_precision: str = "float32"


def _freq_lookup_key(value: float) -> float:
    return round(float(value), 6)


def _freq_key(value: float) -> str:
    return f"{_freq_lookup_key(value):g}"


def _freq_token(freqs: Sequence[float]) -> str:
    return "_".join(_freq_key(float(freq)).replace(".", "p") for freq in freqs)


def _sanitize_freqs(freqs: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(freq) for freq in freqs)
    if len(values) != 4:
        raise ValueError(f"frequency combination must contain exactly 4 values; got {len(values)}")
    return resolve_wang2016_command_frequencies(values)[0]


def _parse_csv_tuple(raw: str, *, cast: Any = str) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in str(raw).split(",") if item.strip())


def _extract_all_windows(
    segments: Sequence[tuple[TrialSpec, np.ndarray]],
    *,
    win_samples: int,
    step_samples: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    windows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for trial, segment in segments:
        batch = extract_window_batch(
            np.asarray(segment, dtype=np.float64),
            win_samples=int(win_samples),
            step_samples=int(step_samples),
        )
        for window_index, window in enumerate(batch):
            windows.append(np.ascontiguousarray(window, dtype=np.float64))
            metadata.append(
                {
                    "label": str(trial.label),
                    "expected_freq": None if trial.expected_freq is None else float(trial.expected_freq),
                    "trial_id": int(trial.trial_id),
                    "block_index": int(trial.block_index),
                    "window_index": int(window_index),
                }
            )
    if not windows:
        raise RuntimeError("no Wang2016 windows were extracted for frequency selection")
    return np.ascontiguousarray(np.stack(windows, axis=0), dtype=np.float64), metadata


def _score_all_targets(config: Wang2016FrequencySelectionConfig) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    subject = load_wang2016_subject(config.mat_path, config.channel_loc_path)
    all_freqs = tuple(float(freq) for freq in WANG2016_TARGET_FREQUENCIES)
    all_segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    from .external_wang2016_dataset import _stimulus_segment  # local import keeps the public API strict.

    for block_index in range(WANG2016_BLOCKS):
        for target_index, freq in enumerate(all_freqs, start=1):
            all_segments.append(
                (
                    TrialSpec(
                        label=f"target{int(target_index):02d}_{float(freq):g}Hz",
                        expected_freq=float(freq),
                        trial_id=int(trial_id),
                        block_index=int(block_index),
                    ),
                    _stimulus_segment(
                        subject.eeg,
                        target_index_1based=int(target_index),
                        block_index=int(block_index),
                        channel_indices=subject.selected_channel_indices,
                    ),
                )
            )
            trial_id += 1
    engine = FBCCAEngine(
        sampling_rate=WANG2016_SAMPLING_RATE,
        freqs=all_freqs,
        win_sec=float(config.win_sec),
        step_sec=float(config.step_sec),
        Nh=DEFAULT_NH,
        compute_backend=str(config.compute_backend),
        gpu_precision=str(config.gpu_precision),
        gpu_warmup=False,
    )
    windows, metadata = _extract_all_windows(
        all_segments,
        win_samples=int(engine.win_samples),
        step_samples=int(engine.step_samples),
    )
    score_matrix = engine.score_windows_batch(windows)
    provenance = {
        "dataset": "Wang2016",
        "source_mat": str(Path(config.mat_path).expanduser().resolve()),
        "source_channels_loc": str(Path(config.channel_loc_path).expanduser().resolve()),
        "required_channel_names": [str(name) for name in WANG2016_REQUIRED_CHANNELS],
        "selected_channel_names": [str(name) for name in subject.selected_channel_names],
        "selected_channel_indices_zero_based": [int(index) for index in subject.selected_channel_indices],
        "selected_channel_indices_one_based": [int(index) + 1 for index in subject.selected_channel_indices],
        "only_required_channels_used": True,
        "excluded_channel_count": int(len(subject.channel_names) - len(subject.selected_channel_names)),
        "all_target_frequencies": [float(freq) for freq in all_freqs],
        "target_count": int(WANG2016_TARGET_COUNT),
        "blocks": int(WANG2016_BLOCKS),
        "sampling_rate": int(WANG2016_SAMPLING_RATE),
        "win_sec": float(config.win_sec),
        "step_sec": float(config.step_sec),
        "window_count": int(score_matrix.shape[0]),
        "score_shape": [int(item) for item in score_matrix.shape],
    }
    return np.asarray(score_matrix, dtype=np.float64), metadata, provenance


def _single_frequency_stats(
    score_matrix: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    *,
    all_freqs: Sequence[float],
) -> dict[str, dict[str, float]]:
    scores = np.asarray(score_matrix, dtype=np.float64)
    freq_to_index = {_freq_lookup_key(freq): int(index) for index, freq in enumerate(all_freqs)}
    expected = np.asarray([float(row["expected_freq"]) for row in metadata], dtype=np.float64)
    stats: dict[str, dict[str, float]] = {}
    for freq in tuple(float(item) for item in all_freqs):
        index = int(freq_to_index[_freq_lookup_key(freq)])
        own_mask = np.isclose(expected, float(freq), atol=1e-6)
        idle_mask = ~own_mask
        own_scores = scores[own_mask, index]
        idle_scores = scores[idle_mask, index]
        if not own_scores.size or not idle_scores.size:
            continue
        other_scores = np.delete(scores[own_mask, :], index, axis=1)
        top_other = np.max(other_scores, axis=1) if other_scores.size else np.zeros_like(own_scores)
        all_top = np.argmax(scores, axis=1)
        stats[_freq_key(freq)] = {
            "freq": float(freq),
            "self_score_p50": float(np.median(own_scores)),
            "self_score_p20": float(np.quantile(own_scores, 0.20)),
            "self_margin_p50": float(np.median(own_scores - top_other)),
            "self_top1_rate_all40": float(np.mean(all_top[own_mask] == index)),
            "hard_idle_score_p95": float(np.quantile(idle_scores, 0.95)),
            "headroom_p50": float(np.median(own_scores) - np.quantile(idle_scores, 0.95)),
        }
    return stats


def _rows_for_combination(
    score_matrix: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    *,
    all_freqs: Sequence[float],
    command_freqs: Sequence[float],
) -> list[dict[str, Any]]:
    command_freqs = _sanitize_freqs(command_freqs)
    freq_to_index = {_freq_lookup_key(freq): int(index) for index, freq in enumerate(all_freqs)}
    indices = [int(freq_to_index[_freq_lookup_key(freq)]) for freq in command_freqs]
    subset = np.asarray(score_matrix[:, indices], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    command_by_key = {_freq_lookup_key(freq): float(freq) for freq in command_freqs}
    for source_row, feature_row in zip(
        metadata,
        build_feature_rows_from_score_matrix(
            subset,
            freqs=command_freqs,
            expected_freq=None,
            label="",
            trial_id=-1,
            block_index=-1,
        ),
    ):
        original_expected = float(source_row["expected_freq"])
        original_key = _freq_lookup_key(original_expected)
        expected_canonical = command_by_key.get(original_key)
        row = dict(feature_row)
        row["expected_freq"] = expected_canonical
        if row["expected_freq"] is None:
            row["label"] = f"hard_idle_wang2016_{original_expected:g}Hz"
            row["correct"] = False
            row["trial_role"] = "hard_idle"
        else:
            row["label"] = f"{float(expected_canonical):g}Hz"
            row["correct"] = bool(np.isclose(float(row["pred_freq"]), float(expected_canonical), atol=1e-6))
            row["trial_role"] = "control"
        row["trial_id"] = int(source_row["trial_id"])
        row["block_index"] = int(source_row["block_index"])
        row["window_index"] = int(source_row["window_index"])
        row["source_expected_freq"] = float(original_expected)
        rows.append(row)
    return rows


def _combination_spacing_ok(freqs: Sequence[float], *, min_spacing_hz: float, min_freq_hz: float) -> bool:
    values = tuple(sorted(float(freq) for freq in freqs))
    if any(float(freq) < float(min_freq_hz) for freq in values):
        return False
    return all(abs(float(b) - float(a)) >= float(min_spacing_hz) - 1e-9 for a, b in zip(values, values[1:]))


def _combination_proxy_summary(
    score_matrix: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    *,
    all_freqs: Sequence[float],
    command_freqs: Sequence[float],
    single_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    command_freqs = _sanitize_freqs(command_freqs)
    freq_to_index = {_freq_lookup_key(freq): int(index) for index, freq in enumerate(all_freqs)}
    indices = [int(freq_to_index[_freq_lookup_key(freq)]) for freq in command_freqs]
    subset = np.asarray(score_matrix[:, indices], dtype=np.float64)
    expected = np.asarray([float(row["expected_freq"]) for row in metadata], dtype=np.float64)
    command_set = {_freq_lookup_key(freq) for freq in command_freqs}
    command_mask = np.asarray([_freq_lookup_key(value) in command_set for value in expected], dtype=bool)
    idle_mask = ~command_mask
    pred_indices = np.argmax(subset, axis=1)
    pred_freqs = np.asarray([float(command_freqs[int(index)]) for index in pred_indices], dtype=np.float64)
    top_sorted = np.sort(subset, axis=1)[:, ::-1]
    top1 = top_sorted[:, 0]
    top2 = top_sorted[:, 1] if subset.shape[1] > 1 else np.zeros_like(top1)
    margins = top1 - top2
    per_freq_window_acc: dict[str, float] = {}
    per_freq_margin_p20: dict[str, float] = {}
    for freq in command_freqs:
        mask = np.isclose(expected, float(freq), atol=1e-6)
        if not np.any(mask):
            per_freq_window_acc[_freq_key(freq)] = 0.0
            per_freq_margin_p20[_freq_key(freq)] = 0.0
            continue
        per_freq_window_acc[_freq_key(freq)] = float(np.mean(np.isclose(pred_freqs[mask], float(freq), atol=1e-6)))
        per_freq_margin_p20[_freq_key(freq)] = float(np.quantile(margins[mask], 0.20))
    idle_top1 = top1[idle_mask]
    headrooms = [
        float(dict(single_stats.get(_freq_key(freq), {})).get("headroom_p50", -999.0))
        for freq in command_freqs
    ]
    return {
        "freqs": [float(freq) for freq in command_freqs],
        "frame_lock_240hz": frame_lock_frequency_report(
            command_freqs,
            refresh_rate_hz=DEFAULT_FRAME_LOCK_REFRESH_RATE_HZ,
        ),
        "raw_control_window_acc": float(
            np.mean(np.isclose(pred_freqs[command_mask], expected[command_mask], atol=1e-6))
        )
        if np.any(command_mask)
        else 0.0,
        "min_per_freq_window_acc": float(min(per_freq_window_acc.values(), default=0.0)),
        "control_margin_p20": float(np.quantile(margins[command_mask], 0.20)) if np.any(command_mask) else 0.0,
        "min_per_freq_margin_p20": float(min(per_freq_margin_p20.values(), default=0.0)),
        "hard_idle_top1_score_p95": float(np.quantile(idle_top1, 0.95)) if idle_top1.size else 1.0,
        "hard_idle_top1_score_p99": float(np.quantile(idle_top1, 0.99)) if idle_top1.size else 1.0,
        "min_single_headroom_p50": float(min(headrooms, default=-999.0)),
        "mean_single_headroom_p50": float(np.mean(headrooms)) if headrooms else -999.0,
        "per_frequency_window_acc": per_freq_window_acc,
        "per_frequency_margin_p20": per_freq_margin_p20,
    }


def _proxy_sort_key(proxy: dict[str, Any]) -> tuple[float, ...]:
    freqs = tuple(float(freq) for freq in proxy.get("freqs", ()))
    return (
        -float(proxy.get("min_per_freq_window_acc", 0.0)),
        -float(proxy.get("raw_control_window_acc", 0.0)),
        float(proxy.get("hard_idle_top1_score_p95", float("inf"))),
        float(proxy.get("hard_idle_top1_score_p99", float("inf"))),
        -float(proxy.get("min_per_freq_margin_p20", 0.0)),
        -float(proxy.get("control_margin_p20", 0.0)),
        -float(proxy.get("min_single_headroom_p50", -999.0)),
        -float(proxy.get("mean_single_headroom_p50", -999.0)),
        -float(np.mean(freqs)) if freqs else 0.0,
    )


def _combo_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = dict(row.get("best_async_metrics", {}))
    single = dict(row.get("single_frequency_summary", {}))
    min_headroom = min((float(item.get("headroom_p50", -999.0)) for item in single.values()), default=-999.0)
    mean_headroom = float(np.mean([float(item.get("headroom_p50", -999.0)) for item in single.values()])) if single else -999.0
    min_recall = min(
        (float(value) for value in dict(metrics.get("per_frequency_recall", {})).values()),
        default=0.0,
    )
    idle_fp = float(metrics.get("idle_fp_per_min", float("inf")))
    idle_selected = float(metrics.get("idle_selected_windows_per_min", float("inf")))
    return (
        0.0 if idle_fp <= 1e-12 else 1000.0 + idle_fp,
        idle_selected,
        -float(metrics.get("control_recall_at_2.5s", 0.0)),
        -float(metrics.get("control_recall_at_3s", 0.0)),
        -float(metrics.get("control_recall", 0.0)),
        -min_recall,
        float(metrics.get("release_latency_s", float("inf"))),
        float(metrics.get("switch_latency_s", float("inf"))),
        -min_headroom,
        -mean_headroom,
        float(np.mean(row.get("freqs", [99.0]))),
    )


def evaluate_frequency_combination(
    score_matrix: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    *,
    all_freqs: Sequence[float],
    command_freqs: Sequence[float],
    win_sec: float,
    step_sec: float,
    gate_policies: Sequence[str],
    control_state_modes: Sequence[str],
    min_enter_windows: int,
    min_exit_windows: int,
    single_stats: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, Any]:
    command_freqs = _sanitize_freqs(command_freqs)
    rows = _rows_for_combination(
        score_matrix,
        metadata,
        all_freqs=tuple(float(freq) for freq in all_freqs),
        command_freqs=command_freqs,
    )
    best: Optional[dict[str, Any]] = None
    candidates: list[dict[str, Any]] = []
    for gate_policy in tuple(gate_policies):
        policy = parse_gate_policy(str(gate_policy))
        for control_state_mode in tuple(control_state_modes):
            profile = fit_threshold_profile(
                rows,
                freqs=command_freqs,
                win_sec=float(win_sec),
                step_sec=float(step_sec),
                min_enter_windows=max(1, int(min_enter_windows)),
                min_exit_windows=max(1, int(min_exit_windows)),
                gate_policy=policy,
                evaluation_rows=rows,
                control_state_mode=str(control_state_mode),
            )
            metrics = evaluate_profile_on_feature_rows(rows, profile)
            candidate = {
                "gate_policy": policy,
                "control_state_mode": str(control_state_mode),
                "frequency_specific_threshold_count": int(len(profile.frequency_specific_thresholds or {})),
                "async_metrics": metrics,
            }
            candidates.append(candidate)
            comparable = {
                **candidate,
                "best_async_metrics": metrics,
                "single_frequency_summary": {
                    _freq_key(freq): dict((single_stats or {}).get(_freq_key(freq), {})) for freq in command_freqs
                },
            }
            if best is None or _combo_sort_key(comparable) < _combo_sort_key(best):
                best = comparable
    if best is None:
        raise RuntimeError(f"no frequency-selection candidate evaluated for {command_freqs}")
    return {
        "freqs": [float(freq) for freq in command_freqs],
        "freq_token": _freq_token(command_freqs),
        "frame_lock_240hz": frame_lock_frequency_report(
            command_freqs,
            refresh_rate_hz=DEFAULT_FRAME_LOCK_REFRESH_RATE_HZ,
        ),
        "best_gate_policy": str(best["gate_policy"]),
        "best_control_state_mode": str(best["control_state_mode"]),
        "frequency_specific_threshold_count": int(best["frequency_specific_threshold_count"]),
        "best_async_metrics": best["best_async_metrics"],
        "candidate_count": int(len(candidates)),
        "single_frequency_summary": dict(best.get("single_frequency_summary", {})),
    }


def run_wang2016_frequency_selection(
    config: Wang2016FrequencySelectionConfig,
    *,
    log_fn: Optional[Any] = None,
) -> dict[str, Any]:
    def log(message: str) -> None:
        if log_fn is not None:
            log_fn(str(message))

    all_freqs = tuple(float(freq) for freq in WANG2016_TARGET_FREQUENCIES)
    log("Scoring Wang2016 SSVEP Benchmark SSVEP windows with strict 8 channels")
    score_matrix, metadata, provenance = _score_all_targets(config)
    single_stats = _single_frequency_stats(score_matrix, metadata, all_freqs=all_freqs)
    combos = [
        tuple(float(freq) for freq in combo)
        for combo in combinations(all_freqs, 4)
        if _combination_spacing_ok(
            combo,
            min_spacing_hz=float(config.min_freq_spacing_hz),
            min_freq_hz=float(config.min_selected_freq_hz),
        )
    ]
    log(f"Proxy-ranking {len(combos)} four-frequency combinations")
    proxy_rows = [
        _combination_proxy_summary(
            score_matrix,
            metadata,
            all_freqs=all_freqs,
            command_freqs=combo,
            single_stats=single_stats,
        )
        for combo in combos
    ]
    proxy_ranked = sorted(proxy_rows, key=_proxy_sort_key)
    max_eval = max(1, int(config.max_evaluated_combinations))
    selected_combos = [tuple(float(freq) for freq in row["freqs"]) for row in proxy_ranked[:max_eval]]
    original_freqs = (8.0, 10.0, 12.0, 15.0)
    original_result = evaluate_frequency_combination(
        score_matrix,
        metadata,
        all_freqs=all_freqs,
        command_freqs=original_freqs,
        win_sec=float(config.win_sec),
        step_sec=float(config.step_sec),
        gate_policies=tuple(config.gate_policies),
        control_state_modes=tuple(config.control_state_modes),
        min_enter_windows=int(config.min_enter_windows),
        min_exit_windows=int(config.min_exit_windows),
        single_stats=single_stats,
    )
    results: list[dict[str, Any]] = []
    log(f"Fully evaluating top {len(selected_combos)} proxy-ranked combinations")
    proxy_by_token = {_freq_token(row["freqs"]): dict(row) for row in proxy_ranked}
    for index, combo in enumerate(selected_combos, start=1):
        result = evaluate_frequency_combination(
            score_matrix,
            metadata,
            all_freqs=all_freqs,
            command_freqs=combo,
            win_sec=float(config.win_sec),
            step_sec=float(config.step_sec),
            gate_policies=tuple(config.gate_policies),
            control_state_modes=tuple(config.control_state_modes),
            min_enter_windows=int(config.min_enter_windows),
            min_exit_windows=int(config.min_exit_windows),
            single_stats=single_stats,
        )
        result["proxy_summary"] = dict(proxy_by_token.get(_freq_token(combo), {}))
        results.append(result)
        if index % 25 == 0:
            log(f"Evaluated {index}/{len(selected_combos)} prefiltered combinations")
    ranked = sorted(results, key=_combo_sort_key)
    best = ranked[0] if ranked else original_result
    payload = {
        "task": "wang2016-frequency-selection",
        "status": "ok",
        "config": asdict(config),
        "provenance": provenance,
        "selection_constraints": {
            "min_freq_spacing_hz": float(config.min_freq_spacing_hz),
            "min_selected_freq_hz": float(config.min_selected_freq_hz),
            "combination_count": int(len(combos)),
            "proxy_prefilter_count": int(len(selected_combos)),
            "proxy_prefilter_policy": "rank by per-frequency raw accuracy, hard-idle top score, margins, headroom, then comfort",
            "frame_lock_refresh_rate_hz": float(DEFAULT_FRAME_LOCK_REFRESH_RATE_HZ),
        },
        "original_8_10_12_15": original_result,
        "best": best,
        "top_combinations": ranked[: max(1, int(config.max_combinations_report))],
        "top_proxy_combinations": proxy_ranked[: max(1, int(config.max_combinations_report))],
        "single_frequency_stats": single_stats,
    }
    report_path = Path(config.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json_safe(payload)
    report_path.write_text(json_dumps(safe_payload) + "\n", encoding="utf-8")
    return json.loads(json_dumps(json_safe({**payload, "report_path": str(report_path)})))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select four Wang2016 SSVEP frequencies using strict required 8-channel FBCCA/gate metrics."
    )
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--channels-loc", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--win-sec", type=float, default=DEFAULT_SELECTION_WIN_SEC)
    parser.add_argument("--step-sec", type=float, default=DEFAULT_SELECTION_STEP_SEC)
    parser.add_argument("--min-spacing", type=float, default=DEFAULT_MIN_FREQ_SPACING_HZ)
    parser.add_argument("--min-freq", type=float, default=DEFAULT_MIN_SELECTED_FREQ_HZ)
    parser.add_argument("--max-report", type=int, default=DEFAULT_MAX_COMBINATIONS_REPORT)
    parser.add_argument("--max-eval", type=int, default=DEFAULT_MAX_EVALUATED_COMBINATIONS)
    parser.add_argument("--gate-policies", type=str, default=",".join(DEFAULT_GATE_POLICIES))
    parser.add_argument("--control-state-modes", type=str, default=",".join(DEFAULT_CONTROL_STATE_MODES))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_wang2016_frequency_selection(
        Wang2016FrequencySelectionConfig(
            mat_path=args.mat,
            channel_loc_path=args.channels_loc,
            report_path=args.report_path,
            win_sec=float(args.win_sec),
            step_sec=float(args.step_sec),
            min_freq_spacing_hz=float(args.min_spacing),
            min_selected_freq_hz=float(args.min_freq),
            max_combinations_report=int(args.max_report),
            max_evaluated_combinations=int(args.max_eval),
            gate_policies=_parse_csv_tuple(str(args.gate_policies), cast=str),
            control_state_modes=_parse_csv_tuple(str(args.control_state_modes), cast=str),
        ),
        log_fn=print,
    )
    print(json_dumps({"report_path": payload["report_path"], "best": payload["best"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .async_fbcca_idle_standalone import (
    DEFAULT_NH,
    FBCCAEngine,
    TrialSpec,
    json_dumps,
    json_safe,
)
from .external_beta_dataset import (
    BETA_BLOCKS,
    BETA_REQUIRED_CHANNELS,
    BETA_SAMPLING_RATE,
    BETA_TARGET_COUNT,
    _stimulus_segment,
    load_beta_subject,
)
from .wang2016_frequency_selection import (
    DEFAULT_CONTROL_STATE_MODES,
    DEFAULT_FRAME_LOCK_REFRESH_RATE_HZ,
    DEFAULT_GATE_POLICIES,
    DEFAULT_MAX_COMBINATIONS_REPORT,
    DEFAULT_MAX_EVALUATED_COMBINATIONS,
    DEFAULT_MIN_FREQ_SPACING_HZ,
    DEFAULT_MIN_SELECTED_FREQ_HZ,
    DEFAULT_SELECTION_STEP_SEC,
    DEFAULT_SELECTION_WIN_SEC,
    _combo_sort_key,
    _combination_proxy_summary,
    _combination_spacing_ok,
    _extract_all_windows,
    _parse_csv_tuple,
    _proxy_sort_key,
    _single_frequency_stats,
    evaluate_frequency_combination,
)


@dataclass(frozen=True)
class BetaFrequencySelectionConfig:
    mat_path: Path
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


def _score_all_targets(config: BetaFrequencySelectionConfig) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    subject = load_beta_subject(config.mat_path)
    all_freqs = tuple(float(freq) for freq in subject.target_frequencies)
    all_segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    for block_index in range(BETA_BLOCKS):
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
                        subject,
                        target_index_1based=int(target_index),
                        block_index=int(block_index),
                        channel_indices=subject.selected_channel_indices,
                    ),
                )
            )
            trial_id += 1
    engine = FBCCAEngine(
        sampling_rate=BETA_SAMPLING_RATE,
        freqs=all_freqs,
        win_sec=float(min(float(config.win_sec), float(subject.stimulus_sec))),
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
        "dataset": "BETA",
        "source_mat": str(Path(config.mat_path).expanduser().resolve()),
        "required_channel_names": [str(name) for name in BETA_REQUIRED_CHANNELS],
        "selected_channel_names": [str(name) for name in subject.selected_channel_names],
        "selected_channel_indices_zero_based": [int(index) for index in subject.selected_channel_indices],
        "selected_channel_indices_one_based": [int(index) + 1 for index in subject.selected_channel_indices],
        "only_required_channels_used": True,
        "excluded_channel_count": int(len(subject.channel_names) - len(subject.selected_channel_names)),
        "all_target_frequencies": [float(freq) for freq in all_freqs],
        "target_count": int(BETA_TARGET_COUNT),
        "blocks": int(BETA_BLOCKS),
        "sampling_rate": int(BETA_SAMPLING_RATE),
        "stimulus_sec": float(subject.stimulus_sec),
        "trial_sec": float(subject.trial_sec),
        "win_sec": float(min(float(config.win_sec), float(subject.stimulus_sec))),
        "step_sec": float(config.step_sec),
        "window_count": int(score_matrix.shape[0]),
        "score_shape": [int(item) for item in score_matrix.shape],
    }
    return np.asarray(score_matrix, dtype=np.float64), metadata, provenance


def run_beta_frequency_selection(
    config: BetaFrequencySelectionConfig,
    *,
    log_fn: Optional[Any] = None,
) -> dict[str, Any]:
    def log(message: str) -> None:
        if log_fn is not None:
            log_fn(str(message))

    log("Scoring BETA SSVEP windows with strict 8 channels")
    score_matrix, metadata, provenance = _score_all_targets(config)
    all_freqs = tuple(float(freq) for freq in provenance["all_target_frequencies"])
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
    log(f"Proxy-ranking {len(combos)} BETA four-frequency combinations")
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
    original_result = evaluate_frequency_combination(
        score_matrix,
        metadata,
        all_freqs=all_freqs,
        command_freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=float(min(float(config.win_sec), float(provenance["stimulus_sec"]))),
        step_sec=float(config.step_sec),
        gate_policies=tuple(config.gate_policies),
        control_state_modes=tuple(config.control_state_modes),
        min_enter_windows=int(config.min_enter_windows),
        min_exit_windows=int(config.min_exit_windows),
        single_stats=single_stats,
    )
    results: list[dict[str, Any]] = []
    log(f"Fully evaluating top {len(selected_combos)} BETA proxy-ranked combinations")
    proxy_by_token = {"_".join(f"{float(freq):g}" for freq in row["freqs"]): dict(row) for row in proxy_ranked}
    for index, combo in enumerate(selected_combos, start=1):
        result = evaluate_frequency_combination(
            score_matrix,
            metadata,
            all_freqs=all_freqs,
            command_freqs=combo,
            win_sec=float(min(float(config.win_sec), float(provenance["stimulus_sec"]))),
            step_sec=float(config.step_sec),
            gate_policies=tuple(config.gate_policies),
            control_state_modes=tuple(config.control_state_modes),
            min_enter_windows=int(config.min_enter_windows),
            min_exit_windows=int(config.min_exit_windows),
            single_stats=single_stats,
        )
        result["proxy_summary"] = dict(proxy_by_token.get("_".join(f"{float(freq):g}" for freq in combo), {}))
        results.append(result)
        if index % 25 == 0:
            log(f"Evaluated {index}/{len(selected_combos)} BETA prefiltered combinations")
    ranked = sorted(results, key=_combo_sort_key)
    best = ranked[0] if ranked else original_result
    payload = {
        "task": "beta-frequency-selection",
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
    report_path.write_text(json_dumps(json_safe(payload)) + "\n", encoding="utf-8")
    return json.loads(json_dumps(json_safe({**payload, "report_path": str(report_path)})))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select four BETA SSVEP frequencies using strict required 8-channel FBCCA/gate metrics."
    )
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--win-sec", type=float, default=DEFAULT_SELECTION_WIN_SEC)
    parser.add_argument("--step-sec", type=float, default=DEFAULT_SELECTION_STEP_SEC)
    parser.add_argument("--min-spacing", type=float, default=DEFAULT_MIN_FREQ_SPACING_HZ)
    parser.add_argument("--min-freq", type=float, default=DEFAULT_MIN_SELECTED_FREQ_HZ)
    parser.add_argument("--max-report", type=int, default=DEFAULT_MAX_COMBINATIONS_REPORT)
    parser.add_argument("--max-eval", type=int, default=DEFAULT_MAX_EVALUATED_COMBINATIONS)
    parser.add_argument("--gate-policies", type=str, default=",".join(DEFAULT_GATE_POLICIES))
    parser.add_argument("--control-state-modes", type=str, default=",".join(DEFAULT_CONTROL_STATE_MODES))
    parser.add_argument("--compute-backend", type=str, default="cpu")
    parser.add_argument("--gpu-precision", type=str, default="float32")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_beta_frequency_selection(
        BetaFrequencySelectionConfig(
            mat_path=args.mat,
            report_path=args.report_path,
            win_sec=float(args.win_sec),
            step_sec=float(args.step_sec),
            min_freq_spacing_hz=float(args.min_spacing),
            min_selected_freq_hz=float(args.min_freq),
            max_combinations_report=int(args.max_report),
            max_evaluated_combinations=int(args.max_eval),
            gate_policies=_parse_csv_tuple(str(args.gate_policies), cast=str),
            control_state_modes=_parse_csv_tuple(str(args.control_state_modes), cast=str),
            compute_backend=str(args.compute_backend),
            gpu_precision=str(args.gpu_precision),
        ),
        log_fn=print,
    )
    print(json_dumps({"report_path": payload["report_path"], "best": payload["best"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import json_dumps, json_safe
from ssvep_core.beta_frequency_selection import BetaFrequencySelectionConfig, run_beta_frequency_selection
from ssvep_core.external_beta_dataset import convert_beta_subject_to_collection
from ssvep_core.external_wang2016_dataset import convert_wang2016_subject_to_collection
from ssvep_core.fbcca_threshold_pretrain import FBCCAThresholdPretrainConfig, run_fbcca_threshold_pretrain
from ssvep_core.stimulus_profiles import frame_lock_frequency_report
from ssvep_core.wang2016_frequency_selection import Wang2016FrequencySelectionConfig, run_wang2016_frequency_selection


DEFAULT_CURRENT_FREQS = (9.8, 12.0, 14.8, 15.8)
DEFAULT_ORIGINAL_FREQS = (8.0, 10.0, 12.0, 15.0)
ALLOWED_DATASETS = ("wang2016", "beta")
DEFAULT_240HZ_EXACT_FREQS = (
    (8.0, 9.6, 10.0, 12.0),
    (8.0, 9.6, 10.0, 15.0),
    (8.0, 9.6, 12.0, 15.0),
    (9.6, 10.0, 12.0, 15.0),
)


def _freq_token(freqs: Sequence[float]) -> str:
    return "_".join(f"{float(freq):g}".replace(".", "p") for freq in freqs)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(json_safe(payload)) + "\n", encoding="utf-8")


def _csv_str_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(str(item).strip() for item in str(raw or "").split(",") if str(item).strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def _csv_dataset_tuple(raw: str | None) -> tuple[str, ...]:
    values = tuple(str(item).strip().lower() for item in str(raw or "").split(",") if str(item).strip())
    if not values:
        return tuple(ALLOWED_DATASETS)
    invalid = [value for value in values if value not in ALLOWED_DATASETS]
    if invalid:
        raise ValueError(f"datasets must be drawn from {','.join(ALLOWED_DATASETS)}; got {','.join(invalid)}")
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _csv_float_tuple(raw: str, *, default: Sequence[float]) -> tuple[float, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(float(value) for value in default)
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated float value")
    return values


def _csv_int_tuple(raw: str, *, default: Sequence[int]) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(int(value) for value in default)
    values = tuple(int(float(item.strip())) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated integer value")
    return values


def _parse_frequency_sets(raw: str | None) -> list[tuple[float, ...]]:
    text = str(raw or "").strip()
    if not text:
        return []
    sets: list[tuple[float, ...]] = []
    for set_text in text.split(";"):
        values = tuple(float(item.strip()) for item in set_text.split(",") if item.strip())
        if not values:
            continue
        if len(values) != 4:
            raise ValueError(f"formal frequency set must contain exactly 4 values: {set_text!r}")
        if any(freq <= 0.0 for freq in values):
            raise ValueError(f"formal frequency set values must be positive: {set_text!r}")
        sets.append(values)
    return _unique_frequency_sets(sets)


def _unique_frequency_sets(freq_sets: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    output: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for freqs in freq_sets:
        normalized = tuple(float(freq) for freq in freqs)
        key = tuple(round(float(freq), 6) for freq in normalized)
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "idle_fp_per_min",
        "idle_selected_windows_per_min",
        "control_recall",
        "control_recall_at_2s",
        "control_recall_at_2.5s",
        "control_recall_at_3s",
        "switch_detect_rate",
        "switch_detect_rate_at_2.8s",
        "switch_latency_s",
        "release_latency_s",
        "per_frequency_recall",
        "per_frequency_gate_pass_rate",
        "reference_headroom_p50",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = dict(row.get("metrics", {}))
    per_recall = dict(metrics.get("per_frequency_recall", {}) or {})
    min_recall = min((float(value) for value in per_recall.values()), default=0.0)
    frame = dict(row.get("frame_lock_240hz", {}) or {})
    return (
        0.0 if float(metrics.get("idle_fp_per_min", 999.0)) <= 1e-12 else 1000.0,
        float(metrics.get("idle_selected_windows_per_min", 999.0)),
        -float(metrics.get("control_recall_at_3s", 0.0)),
        -float(metrics.get("control_recall_at_2.5s", 0.0)),
        -float(metrics.get("control_recall", 0.0)),
        -float(metrics.get("switch_detect_rate", 0.0)),
        -float(min_recall),
        float(metrics.get("switch_latency_s", 999.0)),
        float(metrics.get("release_latency_s", 999.0)),
        0.0 if bool(frame.get("all_integer_frames_per_cycle", False)) else 0.2,
        float(frame.get("max_frame_sequence_repeat_sec", 999.0)),
    )


def _candidate_freqs(selection: dict[str, Any], *, max_top: int) -> list[tuple[float, ...]]:
    candidates: list[tuple[float, ...]] = [
        DEFAULT_CURRENT_FREQS,
        DEFAULT_ORIGINAL_FREQS,
        *DEFAULT_240HZ_EXACT_FREQS,
    ]
    for key in ("best",):
        row = dict(selection.get(key, {}) or {})
        freqs = row.get("freqs")
        if freqs:
            candidates.append(tuple(float(freq) for freq in freqs))
    for row in list(selection.get("top_combinations", []) or [])[: max(0, int(max_top))]:
        freqs = dict(row).get("freqs")
        if freqs:
            candidates.append(tuple(float(freq) for freq in freqs))
    return _unique_frequency_sets(candidates)


def _formal_candidate_freqs(
    selection: dict[str, Any],
    *,
    max_top: int,
    explicit_sets: Sequence[Sequence[float]] = (),
    skip_selection: bool = False,
) -> list[tuple[float, ...]]:
    if explicit_sets and skip_selection:
        return _unique_frequency_sets(explicit_sets)
    candidates: list[Sequence[float]] = [*explicit_sets]
    candidates.extend(_candidate_freqs(selection, max_top=max_top))
    return _unique_frequency_sets(candidates)


def _run_formal_pretrain(
    *,
    dataset_name: str,
    subject_label: str,
    freqs: Sequence[float],
    convert_fn: Callable[..., dict[str, Any]],
    convert_kwargs: dict[str, Any],
    dataset_root: Path,
    report_root: Path,
    compute_backend: str,
    gpu_device: int,
    gpu_precision: str,
    formal_win_sec_candidates: Sequence[float],
    formal_gate_policy_candidates: Sequence[str],
    formal_min_enter_windows_candidates: Sequence[int],
    formal_min_exit_windows_candidates: Sequence[int],
    formal_control_state_mode_candidates: Sequence[str],
    log: Callable[[str], None],
) -> dict[str, Any]:
    token = _freq_token(freqs)
    session_id = f"external_{dataset_name}_{subject_label}_{token}_strict8_hardidle"
    dataset_payload = convert_fn(
        **convert_kwargs,
        dataset_root=dataset_root,
        session_id=session_id,
        subject_id=f"external_{dataset_name}_{subject_label}",
        freqs=tuple(float(freq) for freq in freqs),
        include_hard_idle=True,
        include_pre_stim_idle=False,
    )
    manifest = Path(dataset_payload["dataset_manifest"]).expanduser().resolve()
    run_dir = report_root / "formal_pretrain" / dataset_name / session_id
    cfg = FBCCAThresholdPretrainConfig(
        dataset_manifest_session1=manifest,
        output_profile_path=run_dir / "profile.json",
        report_path=run_dir / "report.json",
        report_root_dir=run_dir,
        organize_report_dir=False,
        win_sec=2.0,
        step_sec=0.25,
        win_sec_candidates=tuple(float(value) for value in formal_win_sec_candidates),
        gate_policy_candidates=tuple(str(value) for value in formal_gate_policy_candidates),
        min_enter_windows_candidates=tuple(int(value) for value in formal_min_enter_windows_candidates),
        min_exit_windows_candidates=tuple(int(value) for value in formal_min_exit_windows_candidates),
        control_state_mode_candidates=tuple(str(value) for value in formal_control_state_mode_candidates),
        compute_backend=str(compute_backend),
        gpu_device=int(gpu_device),
        gpu_precision=str(gpu_precision),
        gpu_warmup=True,
        publish_realtime=False,
        progress_heartbeat_sec=30.0,
    )
    log(f"formal pretrain {dataset_name} {token} manifest={manifest}")
    report = run_fbcca_threshold_pretrain(cfg, log_fn=log)
    metrics = dict(report.get("chosen_async_metrics", {}) or {})
    return {
        "dataset": dataset_name,
        "subject": subject_label,
        "freqs": [float(freq) for freq in freqs],
        "freq_token": token,
        "dataset_manifest": str(manifest),
        "report_path": str(report.get("report_path", "")),
        "profile_path": str(report.get("profile_path", "")),
        "run_valid_for_deployment": bool(report.get("run_valid_for_deployment", False)),
        "status_reasons": list(report.get("status_reasons", []) or []),
        "chosen_candidate": dict(report.get("chosen_candidate", {}) or {}),
        "metrics": _metric_subset(metrics),
        "frame_lock_240hz": frame_lock_frequency_report(freqs, refresh_rate_hz=240.0),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    run_id = str(args.run_id or f"external_freq_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    report_root = Path(args.output_root).expanduser().resolve() / run_id
    dataset_root = Path(args.dataset_root).expanduser().resolve() / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    log_path = report_root / "sweep.log"

    def log(message: str) -> None:
        text = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    log(f"start run_id={run_id}")
    formal_win_sec_candidates = _csv_float_tuple(
        args.formal_win_sec_candidates,
        default=(1.5, 2.0, 2.5, 3.0),
    )
    formal_gate_policy_candidates = _csv_str_tuple(args.formal_gate_policies or "balanced,speed")
    formal_min_enter_windows_candidates = _csv_int_tuple(
        args.formal_min_enter_windows_candidates,
        default=(1, 2),
    )
    formal_min_exit_windows_candidates = _csv_int_tuple(
        args.formal_min_exit_windows_candidates,
        default=(1, 2),
    )
    formal_control_state_mode_candidates = _csv_str_tuple(
        args.formal_control_state_modes or "unified,frequency-specific-threshold"
    )
    selected_datasets = _csv_dataset_tuple(args.datasets)
    wang_report_path = report_root / "selection" / "wang2016_s1_frequency_selection.json"
    beta_report_path = report_root / "selection" / "beta_s16_frequency_selection.json"
    explicit_formal_sets = _parse_frequency_sets(args.formal_frequency_sets)
    if bool(args.skip_selection):
        if not explicit_formal_sets:
            raise ValueError("--skip-selection requires --formal-frequency-sets")
        log(
            f"selection skipped for datasets={','.join(selected_datasets)}; "
            f"using {len(explicit_formal_sets)} explicit formal frequency sets"
        )
        wang_selection: dict[str, Any] = {}
        beta_selection: dict[str, Any] = {}
    else:
        wang_selection = {}
        beta_selection = {}
        if "wang2016" in selected_datasets:
            wang_selection = run_wang2016_frequency_selection(
                Wang2016FrequencySelectionConfig(
                    mat_path=Path(args.wang_mat),
                    channel_loc_path=Path(args.wang_channels_loc),
                    report_path=wang_report_path,
                    win_sec=float(args.selection_win_sec),
                    step_sec=float(args.step_sec),
                    min_freq_spacing_hz=float(args.min_spacing),
                    min_selected_freq_hz=float(args.min_freq),
                    max_combinations_report=int(args.max_report),
                    max_evaluated_combinations=int(args.max_eval),
                    gate_policies=_csv_str_tuple(args.selection_gate_policies),
                    control_state_modes=_csv_str_tuple(args.selection_control_state_modes),
                    compute_backend=str(args.compute_backend),
                    gpu_precision=str(args.gpu_precision),
                ),
                log_fn=log,
            )
        if "beta" in selected_datasets:
            beta_selection = run_beta_frequency_selection(
                BetaFrequencySelectionConfig(
                    mat_path=Path(args.beta_mat),
                    report_path=beta_report_path,
                    win_sec=float(args.selection_win_sec),
                    step_sec=float(args.step_sec),
                    min_freq_spacing_hz=float(args.min_spacing),
                    min_selected_freq_hz=float(args.min_freq),
                    max_combinations_report=int(args.max_report),
                    max_evaluated_combinations=int(args.max_eval),
                    gate_policies=_csv_str_tuple(args.selection_gate_policies),
                    control_state_modes=_csv_str_tuple(args.selection_control_state_modes),
                    compute_backend=str(args.compute_backend),
                    gpu_precision=str(args.gpu_precision),
                ),
                log_fn=log,
            )
    formal_rows: list[dict[str, Any]] = []
    if "wang2016" in selected_datasets:
        for freqs in _formal_candidate_freqs(
            wang_selection,
            max_top=int(args.max_formal_top),
            explicit_sets=explicit_formal_sets,
            skip_selection=bool(args.skip_selection),
        ):
            formal_rows.append(
                _run_formal_pretrain(
                    dataset_name="wang2016",
                    subject_label="s1",
                    freqs=freqs,
                    convert_fn=convert_wang2016_subject_to_collection,
                    convert_kwargs={
                        "mat_path": Path(args.wang_mat),
                        "channel_loc_path": Path(args.wang_channels_loc),
                    },
                    dataset_root=dataset_root,
                    report_root=report_root,
                    compute_backend=str(args.compute_backend),
                    gpu_device=int(args.gpu_device),
                    gpu_precision=str(args.gpu_precision),
                    formal_win_sec_candidates=formal_win_sec_candidates,
                    formal_gate_policy_candidates=formal_gate_policy_candidates,
                    formal_min_enter_windows_candidates=formal_min_enter_windows_candidates,
                    formal_min_exit_windows_candidates=formal_min_exit_windows_candidates,
                    formal_control_state_mode_candidates=formal_control_state_mode_candidates,
                    log=log,
                )
            )
            _write_json(report_root / "partial_summary.json", {"formal_rows": formal_rows})
    if "beta" in selected_datasets:
        for freqs in _formal_candidate_freqs(
            beta_selection,
            max_top=int(args.max_formal_top),
            explicit_sets=explicit_formal_sets,
            skip_selection=bool(args.skip_selection),
        ):
            formal_rows.append(
                _run_formal_pretrain(
                    dataset_name="beta",
                    subject_label="s16",
                    freqs=freqs,
                    convert_fn=convert_beta_subject_to_collection,
                    convert_kwargs={"mat_path": Path(args.beta_mat)},
                    dataset_root=dataset_root,
                    report_root=report_root,
                    compute_backend=str(args.compute_backend),
                    gpu_device=int(args.gpu_device),
                    gpu_precision=str(args.gpu_precision),
                    formal_win_sec_candidates=formal_win_sec_candidates,
                    formal_gate_policy_candidates=formal_gate_policy_candidates,
                    formal_min_enter_windows_candidates=formal_min_enter_windows_candidates,
                    formal_min_exit_windows_candidates=formal_min_exit_windows_candidates,
                    formal_control_state_mode_candidates=formal_control_state_mode_candidates,
                    log=log,
                )
            )
            _write_json(report_root / "partial_summary.json", {"formal_rows": formal_rows})
    ranked = sorted(formal_rows, key=_rank_key)
    summary = {
        "task": "external-frequency-server-sweep",
        "status": "ok",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "datasets": list(selected_datasets),
        "report_root": str(report_root),
        "dataset_root": str(dataset_root),
        "selection_reports": {
            "wang2016": str(wang_report_path) if (not bool(args.skip_selection) and "wang2016" in selected_datasets) else "",
            "beta": str(beta_report_path) if (not bool(args.skip_selection) and "beta" in selected_datasets) else "",
        },
        "selection_best": {
            "wang2016": dict(wang_selection.get("best", {}) or {}),
            "beta": dict(beta_selection.get("best", {}) or {}),
        },
        "formal_rows": formal_rows,
        "ranked_formal_rows": ranked,
        "recommended_frequency_set": ranked[0] if ranked else {},
    }
    _write_json(report_root / "summary.json", summary)
    log(f"complete summary={report_root / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a server-side external SSVEP frequency sweep.")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--wang-mat", type=Path, required=True)
    parser.add_argument("--wang-channels-loc", type=Path, required=True)
    parser.add_argument("--beta-mat", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--compute-backend", type=str, default="cuda")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-precision", type=str, default="float32")
    parser.add_argument("--selection-win-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.25)
    parser.add_argument("--min-spacing", type=float, default=1.0)
    parser.add_argument("--min-freq", type=float, default=9.5)
    parser.add_argument("--max-eval", type=int, default=120)
    parser.add_argument("--max-report", type=int, default=80)
    parser.add_argument("--max-formal-top", type=int, default=4)
    parser.add_argument(
        "--datasets",
        type=str,
        default="wang2016,beta",
        help="Comma-separated datasets to evaluate: wang2016,beta.",
    )
    parser.add_argument(
        "--formal-frequency-sets",
        type=str,
        default="",
        help="Semicolon-separated four-frequency sets, for example '9.8,12,14.8,15.8;11,12,14.6,15.8'.",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Skip selection sweeps and formally evaluate only --formal-frequency-sets.",
    )
    parser.add_argument(
        "--formal-win-sec-candidates",
        type=str,
        default="1.5,2.0,2.5,3.0",
        help="Comma-separated formal-pretrain window candidates.",
    )
    parser.add_argument(
        "--formal-gate-policies",
        type=str,
        default="balanced,speed",
        help="Comma-separated formal-pretrain gate policies.",
    )
    parser.add_argument(
        "--formal-min-enter-windows-candidates",
        type=str,
        default="1,2",
        help="Comma-separated formal-pretrain min-enter window candidates.",
    )
    parser.add_argument(
        "--formal-min-exit-windows-candidates",
        type=str,
        default="1,2",
        help="Comma-separated formal-pretrain min-exit window candidates.",
    )
    parser.add_argument(
        "--formal-control-state-modes",
        type=str,
        default="unified,frequency-specific-threshold",
        help="Comma-separated formal-pretrain control-state modes.",
    )
    parser.add_argument("--selection-gate-policies", type=str, default="conservative")
    parser.add_argument("--selection-control-state-modes", type=str, default="unified")
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = run_sweep(build_parser().parse_args(argv))
    print(json_dumps({"summary_path": str(Path(summary["report_root"]) / "summary.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

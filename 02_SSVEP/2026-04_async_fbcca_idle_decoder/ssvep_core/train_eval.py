from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from async_fbcca_idle_standalone import (
    DEFAULT_BENCHMARK_MODELS,
    DEFAULT_BENCHMARK_MULTI_SEED_COUNT,
    DEFAULT_BENCHMARK_SEED_STEP,
    DEFAULT_BENCHMARK_CHANNEL_MODES,
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_CHANNEL_WEIGHT_MODE,
    DEFAULT_DECISION_TIME_MODE,
    DEFAULT_DYNAMIC_STOP_ALPHA,
    DEFAULT_DYNAMIC_STOP_ENABLED,
    DEFAULT_EXPORT_FIGURES,
    DEFAULT_GATE_POLICY,
    DEFAULT_METRIC_SCOPE,
    DEFAULT_RANKING_POLICY,
    DEFAULT_WIN_SEC_CANDIDATES,
    BenchmarkRunner,
    benchmark_metric_definition_payload,
    benchmark_rank_key,
    evaluate_decoder_on_trials,
    evaluate_decoder_on_trials_v2,
    json_dumps,
    json_safe,
    load_decoder_from_profile,
    model_implementation_level,
    model_method_note,
    normalize_model_name,
    parse_channel_mode_list,
    parse_decision_time_mode,
    parse_gate_policy,
    parse_metric_scope,
    parse_model_list,
    parse_ranking_policy,
    pack_evaluation_metrics_for_ranking,
    profile_meets_acceptance,
    save_profile,
    select_auto_eeg_channels_for_model,
    split_trial_segments_for_benchmark,
    summarize_benchmark_robustness,
    _subset_trial_segments_by_positions,
)

from .dataset import LoadedDataset, load_collection_dataset
from .reporting import export_evaluation_figures


@dataclass(frozen=True)
class OfflineTrainEvalConfig:
    dataset_manifest_session1: Path
    dataset_manifest_session2: Optional[Path]
    output_profile_path: Path
    report_path: Path
    model_names: tuple[str, ...] = tuple(DEFAULT_BENCHMARK_MODELS)
    channel_modes: tuple[str, ...] = tuple(DEFAULT_BENCHMARK_CHANNEL_MODES)
    multi_seed_count: int = DEFAULT_BENCHMARK_MULTI_SEED_COUNT
    seed_step: int = DEFAULT_BENCHMARK_SEED_STEP
    win_candidates: tuple[float, ...] = tuple(DEFAULT_WIN_SEC_CANDIDATES)
    gate_policy: str = DEFAULT_GATE_POLICY
    channel_weight_mode: Optional[str] = DEFAULT_CHANNEL_WEIGHT_MODE
    metric_scope: str = DEFAULT_METRIC_SCOPE
    decision_time_mode: str = DEFAULT_DECISION_TIME_MODE
    export_figures: bool = DEFAULT_EXPORT_FIGURES
    ranking_policy: str = DEFAULT_RANKING_POLICY
    dynamic_stop_enabled: bool = DEFAULT_DYNAMIC_STOP_ENABLED
    dynamic_stop_alpha: float = DEFAULT_DYNAMIC_STOP_ALPHA
    seed: int = DEFAULT_CALIBRATION_SEED


def _split_session_for_train_eval(dataset: LoadedDataset, *, seed: int) -> tuple[list[Any], list[Any], list[Any]]:
    train_segments, gate_segments, holdout_segments = split_trial_segments_for_benchmark(
        dataset.trial_segments,
        seed=int(seed),
    )
    if not train_segments or not gate_segments or not holdout_segments:
        raise RuntimeError(
            "session1 split invalid; need enough trials for train/gate/holdout "
            f"(got train={len(train_segments)}, gate={len(gate_segments)}, holdout={len(holdout_segments)})"
        )
    return train_segments, gate_segments, holdout_segments


def _render_training_eval_markdown(report_payload: dict[str, Any]) -> str:
    chosen_async = dict(report_payload.get("chosen_async_metrics", {}))
    chosen_4 = dict(report_payload.get("chosen_metrics_4class", {}))
    chosen_2 = dict(report_payload.get("chosen_metrics_2class", {}))
    chosen_5_raw = report_payload.get("chosen_metrics_5class")
    chosen_5 = dict(chosen_5_raw) if isinstance(chosen_5_raw, dict) else None
    lines = [
        "# SSVEP Offline Training-Evaluation Report",
        "",
        f"- Generated at: `{report_payload.get('generated_at', '')}`",
        f"- Session1 manifest: `{report_payload.get('dataset_manifest_session1', '')}`",
        f"- Session2 manifest: `{report_payload.get('dataset_manifest_session2', '')}`",
        f"- Chosen model: `{report_payload.get('chosen_model', '')}`",
        f"- Chosen rank: `{report_payload.get('chosen_rank', '')}`",
        f"- Chosen meets acceptance: `{report_payload.get('chosen_meets_acceptance', False)}`",
        f"- Profile saved: `{report_payload.get('profile_saved', False)}`",
        f"- Profile path: `{report_payload.get('chosen_profile_path', '')}`",
        f"- Metric scope: `{report_payload.get('metric_scope', '')}`",
        f"- Decision-time mode: `{report_payload.get('decision_time_mode', '')}`",
        f"- Ranking policy: `{report_payload.get('ranking_policy', '')}`",
        "",
        "## 6.2.1 SSVEP分类准确率",
        "",
        "| 口径 | Acc_SSVEP | Macro-F1 | Mean Decision Time(s) | ITR(bits/min) |",
        "|---|---:|---:|---:|---:|",
        "| 四分类 (8/10/12/15Hz) | {acc4:.4f} | {f14:.4f} | {t4:.4f} | {itr4:.4f} |".format(
            acc4=float(chosen_4.get("acc", 0.0)),
            f14=float(chosen_4.get("macro_f1", 0.0)),
            t4=float(chosen_4.get("mean_decision_time_s", float("inf"))),
            itr4=float(chosen_4.get("itr_bpm", 0.0)),
        ),
        "| 二分类 (control vs idle) | {acc2:.4f} | {f12:.4f} | {t2:.4f} | {itr2:.4f} |".format(
            acc2=float(chosen_2.get("acc", 0.0)),
            f12=float(chosen_2.get("macro_f1", 0.0)),
            t2=float(chosen_2.get("mean_decision_time_s", float("inf"))),
            itr2=float(chosen_2.get("itr_bpm", 0.0)),
        ),
    ]
    if isinstance(chosen_5, dict):
        lines.append(
            "| 五分类 (idle+4freq) | {acc5:.4f} | {f15:.4f} | {t5:.4f} | {itr5:.4f} |".format(
                acc5=float(chosen_5.get("acc", 0.0)),
                f15=float(chosen_5.get("macro_f1", 0.0)),
                t5=float(chosen_5.get("mean_decision_time_s", float("inf"))),
                itr5=float(chosen_5.get("itr_bpm", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## 6.2.2 异步可用性评测",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            "| idle_fp_per_min | {idle:.4f} |".format(idle=float(chosen_async.get("idle_fp_per_min", float("inf")))),
            "| control_recall | {recall:.4f} |".format(recall=float(chosen_async.get("control_recall", 0.0))),
            "| switch_latency_s | {switch_lat:.4f} |".format(
                switch_lat=float(chosen_async.get("switch_latency_s", float("inf")))
            ),
            "| release_latency_s | {release_lat:.4f} |".format(
                release_lat=float(chosen_async.get("release_latency_s", float("inf")))
            ),
            "| inference_ms | {infer:.4f} |".format(
                infer=float(chosen_async.get("inference_ms", float("inf")))
            ),
            "",
            "## Ranked Models (Session1 Holdout)",
            "",
            "| Rank | Model | Impl | idle_fp_per_min | control_recall | switch_latency_s | release_latency_s | Acc_4class | MacroF1_4class | ITR_4class | inference_ms | Accept |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for item in report_payload.get("model_results", []):
        if "metrics" not in item:
            continue
        metrics = dict(item.get("metrics", {}))
        lines.append(
            "| {rank} | {model} | {impl} | {idle:.4f} | {recall:.4f} | {switch_lat:.4f} | {release_lat:.4f} | {acc4:.4f} | {f14:.4f} | {itr4:.4f} | {infer:.4f} | {ok} |".format(
                rank=int(item.get("rank", 0) or 0),
                model=str(item.get("model_name", "")),
                impl=str(item.get("implementation_level", "")),
                idle=float(metrics.get("idle_fp_per_min", float("inf"))),
                recall=float(metrics.get("control_recall", 0.0)),
                switch_lat=float(metrics.get("switch_latency_s", float("inf"))),
                release_lat=float(metrics.get("release_latency_s", float("inf"))),
                acc4=float(metrics.get("acc_4class", 0.0)),
                f14=float(metrics.get("macro_f1_4class", 0.0)),
                itr4=float(metrics.get("itr_bpm_4class", 0.0)),
                infer=float(metrics.get("inference_ms", float("inf"))),
                ok="Y" if bool(item.get("meets_acceptance")) else "N",
            )
        )
    if any("cross_session_metrics" in item for item in report_payload.get("model_results", [])):
        lines.extend(
            [
                "",
                "## 会话内 vs 跨会话（Session2）",
                "",
                "| Model | Session2 idle_fp_per_min | Session2 control_recall | Session2 switch_latency_s | Session2 release_latency_s | Session2 Acc_4class | Session2 MacroF1_4class |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in report_payload.get("model_results", []):
            metrics = item.get("cross_session_metrics")
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "| {model} | {idle:.4f} | {recall:.4f} | {switch_lat:.4f} | {release_lat:.4f} | {acc4:.4f} | {f14:.4f} |".format(
                    model=str(item.get("model_name", "")),
                    idle=float(metrics.get("idle_fp_per_min", float("inf"))),
                    recall=float(metrics.get("control_recall", 0.0)),
                    switch_lat=float(metrics.get("switch_latency_s", float("inf"))),
                    release_lat=float(metrics.get("release_latency_s", float("inf"))),
                    acc4=float(metrics.get("acc_4class", 0.0)),
                    f14=float(metrics.get("macro_f1_4class", 0.0)),
                )
            )
    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            f"- confusion_4class: `{dict(report_payload.get('figures', {})).get('confusion_4class', '')}`",
            f"- confusion_2class: `{dict(report_payload.get('figures', {})).get('confusion_2class', '')}`",
            f"- decision_time_hist: `{dict(report_payload.get('figures', {})).get('decision_time_hist', '')}`",
            f"- model_radar_async_vs_cls: `{dict(report_payload.get('figures', {})).get('model_radar_async_vs_cls', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def run_offline_train_eval(
    config: OfflineTrainEvalConfig,
    *,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    log = log_fn if log_fn is not None else (lambda _msg: None)
    metric_scope = parse_metric_scope(config.metric_scope)
    decision_time_mode = parse_decision_time_mode(config.decision_time_mode)
    ranking_policy = parse_ranking_policy(config.ranking_policy)
    export_figures = bool(config.export_figures)
    dataset_session1 = load_collection_dataset(config.dataset_manifest_session1)
    dataset_session2 = (
        load_collection_dataset(config.dataset_manifest_session2)
        if config.dataset_manifest_session2 is not None
        else None
    )
    if dataset_session2 is not None and int(dataset_session2.sampling_rate) != int(dataset_session1.sampling_rate):
        raise RuntimeError("session2 sampling rate differs from session1")
    if dataset_session2 is not None and tuple(dataset_session2.freqs) != tuple(dataset_session1.freqs):
        raise RuntimeError("session2 freqs differ from session1")

    fs = int(dataset_session1.sampling_rate)
    eeg_channels = tuple(int(channel) for channel in dataset_session1.board_eeg_channels)
    train_base, gate_base, holdout_base = _split_session_for_train_eval(dataset_session1, seed=config.seed)
    log(
        "Offline split ready: "
        f"train={len(train_base)} gate={len(gate_base)} holdout={len(holdout_base)}"
    )

    helper = BenchmarkRunner(
        serial_port="offline",
        board_id=0,
        freqs=dataset_session1.freqs,
        output_profile_path=config.output_profile_path,
        report_path=config.report_path,
        dataset_dir=config.report_path.parent,
        sampling_rate=fs,
        prepare_sec=float(dataset_session1.protocol_config.get("prepare_sec", 1.0)),
        active_sec=float(dataset_session1.protocol_config.get("active_sec", 4.0)),
        rest_sec=float(dataset_session1.protocol_config.get("rest_sec", 1.0)),
        calibration_target_repeats=1,
        calibration_idle_repeats=1,
        eval_target_repeats=1,
        eval_idle_repeats=1,
        eval_switch_trials=1,
        step_sec=float(dataset_session1.protocol_config.get("step_sec", 0.25)),
        model_names=parse_model_list(",".join(config.model_names)),
        channel_modes=parse_channel_mode_list(",".join(config.channel_modes)),
        multi_seed_count=max(1, int(config.multi_seed_count)),
        seed_step=max(1, int(config.seed_step)),
        win_candidates=tuple(float(value) for value in config.win_candidates),
        gate_policy=parse_gate_policy(config.gate_policy),
        channel_weight_mode=config.channel_weight_mode,
        metric_scope=metric_scope,
        decision_time_mode=decision_time_mode,
        ranking_policy=ranking_policy,
        dynamic_stop_enabled=bool(config.dynamic_stop_enabled),
        dynamic_stop_alpha=float(config.dynamic_stop_alpha),
        seed=int(config.seed),
    )

    model_results: list[dict[str, Any]] = []
    best_profiles: dict[str, Any] = {}
    robustness_runs: list[dict[str, Any]] = []
    primary_mode = "auto" if "auto" in helper.channel_modes else str(helper.channel_modes[0])
    primary_seed = int(helper.eval_seeds[0]) if helper.eval_seeds else int(config.seed)

    for channel_mode in helper.channel_modes:
        for eval_seed in helper.eval_seeds:
            seed_mode_success: list[dict[str, Any]] = []
            seed_mode_all: list[dict[str, Any]] = []
            for model_name in helper.model_names:
                model_name = normalize_model_name(model_name)
                include_details = bool(channel_mode == primary_mode and int(eval_seed) == int(primary_seed))
                try:
                    if channel_mode == "auto":
                        selected_channels, channel_scores = select_auto_eeg_channels_for_model(
                            train_base,
                            model_name=model_name,
                            available_board_channels=eeg_channels,
                            sampling_rate=fs,
                            freqs=dataset_session1.freqs,
                            win_sec=max(helper.win_candidates),
                            step_sec=helper.step_sec,
                            model_params={"Nh": 3},
                            seed=int(eval_seed),
                        )
                    else:
                        selected_channels = tuple(int(channel) for channel in eeg_channels)
                        channel_scores = []

                    selected_positions = [list(eeg_channels).index(channel) for channel in selected_channels]
                    train_segments = _subset_trial_segments_by_positions(train_base, selected_positions)
                    gate_segments = _subset_trial_segments_by_positions(gate_base, selected_positions)
                    holdout_segments = _subset_trial_segments_by_positions(holdout_base, selected_positions)

                    profile, result = helper._benchmark_single_model(
                        model_name=model_name,
                        fs=fs,
                        train_segments=train_segments,
                        gate_segments=gate_segments,
                        eval_segments=holdout_segments,
                        eeg_channels=selected_channels,
                    )
                    result["selected_eeg_channels"] = [int(channel) for channel in selected_channels]
                    result["channel_mode"] = str(channel_mode)
                    result["eval_seed"] = int(eval_seed)
                    result["split_counts"] = {
                        "train_segments": int(len(train_segments)),
                        "gate_segments": int(len(gate_segments)),
                        "holdout_segments": int(len(holdout_segments)),
                    }
                    if include_details:
                        result["channel_selection"] = channel_scores
                    if dataset_session2 is not None:
                        positions_s2 = [list(dataset_session2.board_eeg_channels).index(channel) for channel in selected_channels]
                        session2_subset = _subset_trial_segments_by_positions(dataset_session2.trial_segments, positions_s2)
                        decoder_s2 = load_decoder_from_profile(profile, sampling_rate=dataset_session2.sampling_rate)
                        cross_session_bundle = evaluate_decoder_on_trials_v2(
                            decoder_s2,
                            profile,
                            session2_subset,
                            metric_scope=metric_scope,
                            decision_time_mode=decision_time_mode,
                        )
                        result["cross_session_metrics"] = pack_evaluation_metrics_for_ranking(
                            cross_session_bundle,
                            metric_scope=metric_scope,
                        )
                        result["cross_session_async_metrics"] = dict(cross_session_bundle.get("async_metrics", {}))
                        result["cross_session_metrics_4class"] = dict(cross_session_bundle.get("metrics_4class", {}))
                        result["cross_session_metrics_2class"] = dict(cross_session_bundle.get("metrics_2class", {}))
                        result["cross_session_metrics_5class"] = (
                            None
                            if cross_session_bundle.get("metrics_5class") is None
                            else dict(cross_session_bundle.get("metrics_5class", {}))
                        )

                    if include_details:
                        best_profiles[str(model_name)] = profile
                        model_results.append(dict(result))
                    seed_mode_all.append(dict(result))
                    seed_mode_success.append(dict(result))
                    log(
                        f"Model done: mode={channel_mode} seed={eval_seed} model={model_name} "
                        f"idle_fp={result['metrics'].get('idle_fp_per_min', float('inf')):.4f} "
                        f"recall={result['metrics'].get('control_recall', 0.0):.4f}"
                    )
                except Exception as exc:
                    failed = {
                        "model_name": str(model_name),
                        "implementation_level": model_implementation_level(model_name),
                        "method_note": model_method_note(model_name),
                        "channel_mode": str(channel_mode),
                        "eval_seed": int(eval_seed),
                        "error": str(exc),
                        "meets_acceptance": False,
                    }
                    if include_details:
                        model_results.append(dict(failed))
                    seed_mode_all.append(dict(failed))
                    log(f"Model failed: mode={channel_mode} seed={eval_seed} model={model_name} error={exc}")

            seed_mode_success.sort(
                key=lambda item: benchmark_rank_key(dict(item["metrics"]), ranking_policy=ranking_policy)
            )
            rank_by_model = {str(item["model_name"]): int(rank) for rank, item in enumerate(seed_mode_success, start=1)}
            for item in seed_mode_all:
                item["run_rank"] = rank_by_model.get(str(item.get("model_name", "")))
                robustness_runs.append(item)

    successful = [item for item in model_results if "metrics" in item]
    if not successful:
        raise RuntimeError("offline train-eval failed: all models failed")
    successful.sort(key=lambda item: benchmark_rank_key(dict(item["metrics"]), ranking_policy=ranking_policy))
    for rank, item in enumerate(successful, start=1):
        item["rank"] = int(rank)
        item["meets_acceptance"] = bool(profile_meets_acceptance(dict(item["metrics"])))
    accepted = [item for item in successful if bool(item.get("meets_acceptance"))]
    recommended = successful[0]
    chosen = accepted[0] if accepted else recommended
    profile_saved = False
    chosen_profile_path: Optional[str] = None
    if accepted:
        chosen_profile = best_profiles[str(chosen["model_name"])]
        chosen_profile = replace(
            chosen_profile,
            benchmark_metrics={
                key: float(value)
                for key, value in dict(chosen["metrics"]).items()
                if isinstance(value, (float, int)) and np.isfinite(float(value))
            },
        )
        save_profile(chosen_profile, config.output_profile_path)
        profile_saved = True
        chosen_profile_path = str(config.output_profile_path)
    else:
        log(
            "No model meets acceptance thresholds. "
            f"Top candidate={recommended.get('model_name')} (rank=1) is kept for reference only; profile not saved."
        )

    robustness_summary = summarize_benchmark_robustness(robustness_runs, ranking_policy=ranking_policy)
    metric_definition = benchmark_metric_definition_payload(
        ranking_policy=ranking_policy,
        metric_scope=metric_scope,
        decision_time_mode=decision_time_mode,
    )
    report_payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "offline_train_eval",
        "report_path": str(config.report_path),
        "dataset_manifest_session1": str(config.dataset_manifest_session1),
        "dataset_manifest_session2": None if config.dataset_manifest_session2 is None else str(config.dataset_manifest_session2),
        "session1_id": dataset_session1.session_id,
        "session2_id": None if dataset_session2 is None else dataset_session2.session_id,
        "sampling_rate": fs,
        "freqs": list(dataset_session1.freqs),
        "board_eeg_channels": list(dataset_session1.board_eeg_channels),
        "split_counts_session1": {
            "train": int(len(train_base)),
            "gate": int(len(gate_base)),
            "holdout": int(len(holdout_base)),
        },
        "model_results": model_results,
        "chosen_model": str(chosen["model_name"]),
        "chosen_rank": int(chosen.get("rank", 0) or 0),
        "chosen_metrics": dict(chosen.get("metrics", {})),
        "chosen_async_metrics": dict(chosen.get("async_metrics", {})),
        "chosen_metrics_4class": dict(chosen.get("metrics_4class", {})),
        "chosen_metrics_2class": dict(chosen.get("metrics_2class", {})),
        "chosen_metrics_5class": (
            None
            if chosen.get("metrics_5class") is None
            else dict(chosen.get("metrics_5class", {}))
        ),
        "chosen_fixed_window_metrics": dict(chosen.get("fixed_window_metrics", {})),
        "chosen_fixed_window_async_metrics": dict(chosen.get("fixed_window_async_metrics", {})),
        "chosen_fixed_window_metrics_4class": dict(chosen.get("fixed_window_metrics_4class", {})),
        "chosen_fixed_window_metrics_2class": dict(chosen.get("fixed_window_metrics_2class", {})),
        "chosen_fixed_window_metrics_5class": (
            None
            if chosen.get("fixed_window_metrics_5class") is None
            else dict(chosen.get("fixed_window_metrics_5class", {}))
        ),
        "chosen_dynamic_delta": dict(chosen.get("dynamic_delta", {})),
        "chosen_profile_path": chosen_profile_path,
        "chosen_meets_acceptance": bool(chosen.get("meets_acceptance", False)),
        "accepted_model_count": int(len(accepted)),
        "recommended_model": str(recommended.get("model_name", "")),
        "recommended_rank": int(recommended.get("rank", 0) or 0),
        "recommended_metrics": dict(recommended.get("metrics", {})),
        "profile_saved": bool(profile_saved),
        "metric_scope": str(metric_scope),
        "decision_time_mode": str(decision_time_mode),
        "ranking_policy": str(ranking_policy),
        "export_figures": bool(export_figures),
        "gate_policy": str(helper.gate_policy),
        "channel_weight_mode": helper.channel_weight_mode,
        "dynamic_stop_enabled": bool(helper.dynamic_stop_enabled),
        "dynamic_stop_alpha": float(helper.dynamic_stop_alpha),
        "metric_definition": metric_definition,
        "formula_definitions": dict(metric_definition.get("formula_definitions", {})),
        "method_references": list(metric_definition.get("method_references", [])),
        "robustness": robustness_summary,
    }
    report_payload["async_metrics"] = dict(report_payload.get("chosen_async_metrics", {}))
    report_payload["metrics_4class"] = dict(report_payload.get("chosen_metrics_4class", {}))
    report_payload["metrics_2class"] = dict(report_payload.get("chosen_metrics_2class", {}))
    report_payload["metrics_5class"] = report_payload.get("chosen_metrics_5class")
    if export_figures:
        report_payload["figures"] = export_evaluation_figures(report_payload, output_dir=config.report_path.parent)
    else:
        report_payload["figures"] = {}
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json_dumps(json_safe(report_payload)) + "\n", encoding="utf-8")
    markdown_path = config.report_path.with_suffix(".md")
    markdown_path.write_text(_render_training_eval_markdown(report_payload), encoding="utf-8")
    log(f"Report saved: {config.report_path}")
    log(f"Markdown saved: {markdown_path}")
    if profile_saved:
        log(f"Profile saved: {config.output_profile_path}")
    else:
        log("Profile not saved because no model met acceptance thresholds.")
    return report_payload

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


THIS_FILE = Path(__file__).resolve()
SSVEP_ROOT = THIS_FILE.parents[1]
REPO_ROOT = SSVEP_ROOT.parent
MNE_HOME_DIR = SSVEP_ROOT / "runtime" / "mne_home"
MNE_HOME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(MNE_HOME_DIR))
if str(SSVEP_ROOT) not in sys.path:
    sys.path.insert(0, str(SSVEP_ROOT))

import mne  # noqa: E402
from ssvep_core.async_fbcca_idle_standalone import TrialSpec, create_decoder  # noqa: E402


LABEL_TO_FREQ = {
    "33025": 13.0,
    "33027": 17.0,
    "33026": 21.0,
}
REST_LABEL = "33024"
VISUAL_START = "32779"
VISUAL_STOP = "32780"


@dataclass(frozen=True)
class SessionTrial:
    subject_id: str
    session_id: str
    session_index: int
    trial_index: int
    label_code: str
    expected_freq: Optional[float]
    segment: np.ndarray


def _round_sample(value: float, sfreq: float) -> int:
    return int(round(float(value) * float(sfreq)))


def discover_sessions(dataset_root: Path) -> dict[str, list[Path]]:
    subject_sessions: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(dataset_root.rglob("*.gdf")):
        subject_sessions[path.parent.name].append(path)
    return {subject: sorted(paths) for subject, paths in sorted(subject_sessions.items())}


def extract_trials_from_gdf(path: Path, *, session_index: int) -> tuple[int, list[SessionTrial]]:
    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
    sfreq = int(round(float(raw.info["sfreq"])))
    data = np.asarray(raw.get_data().T, dtype=np.float64)
    annotations = list(raw.annotations)
    session_id = str(path.stem)
    subject_id = str(path.parent.name)
    trials: list[SessionTrial] = []
    current_label: Optional[str] = None
    trial_index = 0

    for index, annotation in enumerate(annotations):
        desc = str(annotation["description"]).strip()
        if desc in LABEL_TO_FREQ or desc == REST_LABEL:
            current_label = desc
            continue
        if desc != VISUAL_START or current_label is None:
            continue

        stop_onset: Optional[float] = None
        for follow in annotations[index + 1 :]:
            follow_desc = str(follow["description"]).strip()
            if follow_desc == VISUAL_STOP:
                stop_onset = float(follow["onset"])
                break
            if follow_desc in LABEL_TO_FREQ or follow_desc == REST_LABEL or follow_desc == VISUAL_START:
                break
        if stop_onset is None:
            stop_onset = float(annotation["onset"]) + 5.0

        start_sample = _round_sample(float(annotation["onset"]), sfreq)
        stop_sample = _round_sample(stop_onset, sfreq)
        if stop_sample <= start_sample:
            current_label = None
            continue

        segment = np.ascontiguousarray(data[start_sample:stop_sample, :], dtype=np.float64)
        trials.append(
            SessionTrial(
                subject_id=subject_id,
                session_id=session_id,
                session_index=int(session_index),
                trial_index=int(trial_index),
                label_code=current_label,
                expected_freq=LABEL_TO_FREQ.get(current_label),
                segment=segment,
            )
        )
        trial_index += 1
        current_label = None

    return sfreq, trials


def crop_fixed_window(
    trial: SessionTrial,
    *,
    fs: int,
    win_sec: float,
    latency_sec: float,
) -> np.ndarray:
    start = int(round(float(latency_sec) * float(fs)))
    win_samples = int(round(float(win_sec) * float(fs)))
    stop = start + win_samples
    if trial.segment.shape[0] < stop:
        raise ValueError(
            f"trial too short for requested window: samples={trial.segment.shape[0]} required={stop} "
            f"subject={trial.subject_id} session={trial.session_id} trial={trial.trial_index}"
        )
    return np.ascontiguousarray(trial.segment[start:stop, :], dtype=np.float64)


def to_training_trials(
    trials: list[SessionTrial],
    *,
    fs: int,
    win_sec: float,
    latency_sec: float,
) -> list[tuple[TrialSpec, np.ndarray]]:
    payload: list[tuple[TrialSpec, np.ndarray]] = []
    for global_trial_id, trial in enumerate(trials):
        if trial.expected_freq is None:
            continue
        payload.append(
            (
                TrialSpec(
                    label=f"{float(trial.expected_freq):g}Hz",
                    expected_freq=float(trial.expected_freq),
                    trial_id=int(global_trial_id),
                    block_index=int(trial.session_index),
                ),
                crop_fixed_window(trial, fs=fs, win_sec=win_sec, latency_sec=latency_sec),
            )
        )
    return payload


def build_model_specs() -> list[dict[str, Any]]:
    return [
        {
            "display_name": "fbcca_fixed_all8",
            "model_name": "fbcca_fixed_all8",
            "model_params": {
                "Nh": 3,
                "compute_backend": "cpu",
            },
        },
        {
            "display_name": "tdca_like_legacy",
            "model_name": "tdca",
            "model_params": {
                "Nh": 3,
                "delay_steps": 3,
                "n_components": 2,
                "decoder_variant": "tdca_like_legacy",
                "compute_backend": "cpu",
            },
        },
    ]


def evaluate_subject_loso(
    subject_id: str,
    sessions: list[Path],
    *,
    win_sec: float,
    latency_sec: float,
) -> dict[str, Any]:
    session_trials: list[list[SessionTrial]] = []
    session_fs: Optional[int] = None
    for session_index, session_path in enumerate(sessions):
        fs, trials = extract_trials_from_gdf(session_path, session_index=session_index)
        if session_fs is None:
            session_fs = int(fs)
        elif int(fs) != int(session_fs):
            raise ValueError(f"inconsistent sampling rate for {subject_id}: {session_fs} vs {fs}")
        session_trials.append(trials)
    if session_fs is None:
        raise ValueError(f"no sessions found for subject {subject_id}")

    freqs = tuple(sorted(LABEL_TO_FREQ.values()))
    active_trials_per_session = [sum(1 for trial in trials if trial.expected_freq is not None) for trials in session_trials]
    model_specs = build_model_specs()
    fold_rows: list[dict[str, Any]] = []

    for holdout_index in range(len(session_trials)):
        train_trials_raw: list[SessionTrial] = []
        test_trials_raw: list[SessionTrial] = []
        for session_index, trials in enumerate(session_trials):
            active = [trial for trial in trials if trial.expected_freq is not None]
            if session_index == holdout_index:
                test_trials_raw.extend(active)
            else:
                train_trials_raw.extend(active)

        training_trials = to_training_trials(
            train_trials_raw,
            fs=int(session_fs),
            win_sec=float(win_sec),
            latency_sec=float(latency_sec),
        )
        testing_trials = to_training_trials(
            test_trials_raw,
            fs=int(session_fs),
            win_sec=float(win_sec),
            latency_sec=float(latency_sec),
        )

        for spec in model_specs:
            decoder = create_decoder(
                spec["model_name"],
                sampling_rate=int(session_fs),
                freqs=freqs,
                win_sec=float(win_sec),
                step_sec=float(win_sec),
                model_params=dict(spec["model_params"]),
            )
            if decoder.requires_fit:
                decoder.fit(training_trials)

            correct_count = 0
            class_totals: Counter[str] = Counter()
            class_correct: Counter[str] = Counter()
            confusion: Counter[str] = Counter()
            inference_ms_values: list[float] = []

            for trial_spec, window in testing_trials:
                start_time = time.perf_counter()
                result = decoder.analyze_window(window)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                inference_ms_values.append(float(elapsed_ms))
                pred_freq = float(result["pred_freq"])
                expected_freq = float(trial_spec.expected_freq) if trial_spec.expected_freq is not None else math.nan
                true_key = f"{expected_freq:g}"
                pred_key = f"{pred_freq:g}"
                class_totals[true_key] += 1
                confusion[f"{true_key}->{pred_key}"] += 1
                if abs(pred_freq - expected_freq) < 1e-8:
                    correct_count += 1
                    class_correct[true_key] += 1

            total_count = len(testing_trials)
            fold_rows.append(
                {
                    "subject_id": subject_id,
                    "win_sec": float(win_sec),
                    "holdout_session_index": int(holdout_index),
                    "holdout_session_name": str(sessions[holdout_index].stem),
                    "model": str(spec["display_name"]),
                    "train_trials": len(training_trials),
                    "test_trials": int(total_count),
                    "correct_trials": int(correct_count),
                    "accuracy": float(correct_count / total_count) if total_count else 0.0,
                    "mean_inference_ms": float(np.mean(inference_ms_values)) if inference_ms_values else 0.0,
                    "median_inference_ms": float(np.median(inference_ms_values)) if inference_ms_values else 0.0,
                    "class_accuracy": {
                        key: float(class_correct[key] / class_totals[key]) if class_totals[key] else 0.0
                        for key in sorted(class_totals)
                    },
                    "confusion": dict(sorted(confusion.items())),
                }
            )

    return {
        "subject_id": subject_id,
        "sampling_rate": int(session_fs),
        "session_count": len(session_trials),
        "trial_count_per_session": [len(trials) for trials in session_trials],
        "active_trials_per_session": active_trials_per_session,
        "fold_results": fold_rows,
    }


def summarize_results(subject_reports: list[dict[str, Any]]) -> dict[str, Any]:
    model_by_win: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_subject_summary: list[dict[str, Any]] = []

    for report in subject_reports:
        subject_id = str(report["subject_id"])
        grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        for row in report["fold_results"]:
            key = (str(row["model"]), float(row["win_sec"]))
            grouped[key].append(row)
            model_by_win[f"{row['model']}|{row['win_sec']}"].append(row)
        for (model, win_sec), rows in sorted(grouped.items()):
            accuracies = np.asarray([float(row["accuracy"]) for row in rows], dtype=np.float64)
            latencies = np.asarray([float(row["mean_inference_ms"]) for row in rows], dtype=np.float64)
            per_subject_summary.append(
                {
                    "subject_id": subject_id,
                    "model": model,
                    "win_sec": float(win_sec),
                    "mean_accuracy": float(np.mean(accuracies)),
                    "std_accuracy": float(np.std(accuracies)),
                    "mean_inference_ms": float(np.mean(latencies)),
                }
            )

    aggregate_rows: list[dict[str, Any]] = []
    for key, rows in sorted(model_by_win.items()):
        model, win_sec_raw = key.split("|", 1)
        win_sec = float(win_sec_raw)
        accuracies = np.asarray([float(row["accuracy"]) for row in rows], dtype=np.float64)
        inference_ms = np.asarray([float(row["mean_inference_ms"]) for row in rows], dtype=np.float64)
        correct = int(sum(int(row["correct_trials"]) for row in rows))
        total = int(sum(int(row["test_trials"]) for row in rows))
        aggregate_rows.append(
            {
                "model": model,
                "win_sec": float(win_sec),
                "mean_fold_accuracy": float(np.mean(accuracies)),
                "std_fold_accuracy": float(np.std(accuracies)),
                "overall_accuracy": float(correct / total) if total else 0.0,
                "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms.size else 0.0,
                "n_folds": len(rows),
                "n_trials": total,
            }
        )

    return {
        "per_subject_summary": per_subject_summary,
        "aggregate_rows": aggregate_rows,
    }


def write_report_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# External SSVEP FBCCA vs TDCA")
    lines.append("")
    lines.append(f"- Dataset: `{report['dataset_root']}`")
    lines.append(f"- Subjects: `{', '.join(report['subjects'])}`")
    lines.append(f"- Window seconds: `{', '.join(str(item) for item in report['win_secs'])}`")
    lines.append(f"- Latency trim: `{report['latency_sec']:.2f}s`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Model | Win (s) | Overall Acc | Mean Fold Acc | Std | Mean Inference (ms) | Trials |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["summary"]["aggregate_rows"]:
        lines.append(
            f"| {row['model']} | {row['win_sec']:.1f} | {row['overall_accuracy']:.4f} | "
            f"{row['mean_fold_accuracy']:.4f} | {row['std_fold_accuracy']:.4f} | "
            f"{row['mean_inference_ms']:.3f} | {row['n_trials']} |"
        )
    lines.append("")
    lines.append("## Per Subject")
    lines.append("")
    lines.append("| Subject | Model | Win (s) | Mean Acc | Std | Mean Inference (ms) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in report["summary"]["per_subject_summary"]:
        lines.append(
            f"| {row['subject_id']} | {row['model']} | {row['win_sec']:.1f} | "
            f"{row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} | {row['mean_inference_ms']:.3f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This comparison uses only stimulation trials (13/17/21 Hz).")
    lines.append("- Rest trials are present in the dataset, but not included in the primary metric because FBCCA and TDCA here are evaluated as frequency classifiers, not full async rejectors.")
    lines.append("- Split protocol is leave-one-session-out within each subject.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare FBCCA and TDCA on the public dataset-ssvep-led GDF files.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SSVEP_ROOT / "artifacts" / "datasets" / "external" / "dataset_ssvep_led_github",
    )
    parser.add_argument("--win-secs", type=str, default="2.0,3.0")
    parser.add_argument("--latency-sec", type=float, default=0.14)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SSVEP_ROOT / "artifacts" / "runs" / "local" / "external-compare",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    win_secs = tuple(float(part.strip()) for part in str(args.win_secs).split(",") if part.strip())
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if not win_secs:
        raise ValueError("at least one win_sec is required")

    sessions_by_subject = discover_sessions(dataset_root)
    if not sessions_by_subject:
        raise RuntimeError(f"no .gdf sessions found under {dataset_root}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / time.strftime("%Y%m%d") / f"run_{stamp}_ssvep_led_fbcca_vs_tdca"
    run_dir.mkdir(parents=True, exist_ok=True)

    subject_reports: list[dict[str, Any]] = []
    for subject_id, sessions in sessions_by_subject.items():
        for win_sec in win_secs:
            subject_reports.append(
                evaluate_subject_loso(
                    subject_id,
                    sessions,
                    win_sec=float(win_sec),
                    latency_sec=float(args.latency_sec),
                )
            )

    summary = summarize_results(subject_reports)
    report = {
        "dataset_root": str(dataset_root),
        "run_dir": str(run_dir),
        "subjects": list(sorted(sessions_by_subject)),
        "win_secs": [float(item) for item in win_secs],
        "latency_sec": float(args.latency_sec),
        "comparison_scope": "leave-one-session-out active-trial classification",
        "model_specs": build_model_specs(),
        "subject_reports": subject_reports,
        "summary": summary,
    }

    report_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report_markdown(report, markdown_path)

    print(f"run_dir={run_dir}")
    for row in summary["aggregate_rows"]:
        print(
            f"{row['model']} win={row['win_sec']:.1f}s overall_acc={row['overall_accuracy']:.4f} "
            f"mean_fold_acc={row['mean_fold_accuracy']:.4f} mean_inference_ms={row['mean_inference_ms']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

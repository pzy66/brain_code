from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(converted):
        return float(default)
    return float(converted)


def _plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    mat = np.asarray(matrix, dtype=float)
    if mat.size == 0:
        mat = np.zeros((max(1, len(labels)), max(1, len(labels))), dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(mat, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    tick_labels = [str(item) for item in labels] if labels else [str(i) for i in range(mat.shape[0])]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_yticklabels(tick_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = int(round(float(mat[i, j])))
            color = "white" if float(mat[i, j]) > float(np.max(mat) * 0.45) else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_decision_time_hist(
    samples: Sequence[float],
    title: str,
    output_path: Path,
) -> None:
    values = np.asarray([float(item) for item in samples if np.isfinite(float(item))], dtype=float)
    if values.size == 0:
        values = np.asarray([0.0], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bins = min(20, max(5, int(np.sqrt(values.size))))
    ax.hist(values, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Decision time (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_model_radar(
    model_rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = [dict(item) for item in model_rows if isinstance(item.get("metrics"), dict)]
    if not rows:
        return
    rows.sort(key=lambda item: int(item.get("rank", 10_000)))
    rows = rows[: min(len(rows), 5)]

    categories = ("IdleSafe", "Recall", "Acc4", "MacroF1", "SwitchFast")
    theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])

    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111, polar=True)
    for row in rows:
        metrics = dict(row.get("metrics", {}))
        idle_fp = _as_float(metrics.get("idle_fp_per_min"), default=10.0)
        recall = _as_float(metrics.get("control_recall"), default=0.0)
        acc4 = _as_float(metrics.get("acc_4class"), default=0.0)
        macro_f1 = _as_float(metrics.get("macro_f1_4class"), default=0.0)
        switch_lat = _as_float(metrics.get("switch_latency_s"), default=10.0)
        values = np.asarray(
            [
                1.0 / (1.0 + max(idle_fp, 0.0)),
                np.clip(recall, 0.0, 1.0),
                np.clip(acc4, 0.0, 1.0),
                np.clip(macro_f1, 0.0, 1.0),
                1.0 / (1.0 + max(switch_lat, 0.0)),
            ],
            dtype=float,
        )
        values = np.concatenate([values, [values[0]]])
        ax.plot(theta, values, linewidth=1.8, label=str(row.get("model_name", "")))
        ax.fill(theta, values, alpha=0.08)

    ax.set_thetagrids(theta[:-1] * 180 / np.pi, categories)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model Radar: Async Utility vs Classification", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def export_evaluation_figures(report_payload: dict[str, Any], *, output_dir: Path) -> dict[str, str]:
    target_dir = Path(output_dir).expanduser().resolve() / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics_4 = dict(report_payload.get("chosen_metrics_4class", {}))
    metrics_2 = dict(report_payload.get("chosen_metrics_2class", {}))

    out_conf4 = target_dir / "confusion_4class.png"
    _plot_confusion_matrix(
        matrix=metrics_4.get("confusion_matrix", []),
        labels=metrics_4.get("labels", []),
        title="Confusion Matrix (4-class: 8/10/12/15 Hz)",
        output_path=out_conf4,
    )

    out_conf2 = target_dir / "confusion_2class.png"
    _plot_confusion_matrix(
        matrix=metrics_2.get("confusion_matrix", []),
        labels=metrics_2.get("labels", []),
        title="Confusion Matrix (2-class: control vs idle)",
        output_path=out_conf2,
    )

    out_hist = target_dir / "decision_time_hist.png"
    _plot_decision_time_hist(
        samples=metrics_4.get("decision_time_samples_s", []),
        title="Decision Time Distribution (4-class)",
        output_path=out_hist,
    )

    out_radar = target_dir / "model_radar_async_vs_cls.png"
    _plot_model_radar(report_payload.get("model_results", []), output_path=out_radar)

    return {
        "dir": str(target_dir),
        "confusion_4class": str(out_conf4),
        "confusion_2class": str(out_conf2),
        "decision_time_hist": str(out_hist),
        "model_radar_async_vs_cls": str(out_radar),
    }

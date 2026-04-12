from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.train_eval import _build_ab_comparisons, _render_training_eval_markdown
from ssvep_training_evaluation_ui import WEIGHTED_COMPARE_MODELS, _parse_task, build_parser


def test_weighted_compare_task_alias_and_default() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.task == "fbcca-weighted-compare"
    assert _parse_task("weighted_compare") == "fbcca-weighted-compare"
    assert "fbcca_cw_sw_all8" in WEIGHTED_COMPARE_MODELS
    assert "fbcca_fixed_all8" in WEIGHTED_COMPARE_MODELS


def test_ab_comparisons_cover_plain_and_weighted_fbcca_variants() -> None:
    rows = [
        {"model_name": "legacy_fbcca_202603", "metrics": {"idle_fp_per_min": 1.0, "control_recall": 0.50, "switch_latency_s": 3.0}},
        {"model_name": "fbcca_fixed_all8", "metrics": {"idle_fp_per_min": 0.9, "control_recall": 0.60, "switch_latency_s": 2.8}},
        {"model_name": "fbcca_cw_all8", "metrics": {"idle_fp_per_min": 0.8, "control_recall": 0.70, "switch_latency_s": 2.1}},
        {"model_name": "fbcca_sw_all8", "metrics": {"idle_fp_per_min": 0.85, "control_recall": 0.68, "switch_latency_s": 2.4}},
        {"model_name": "fbcca_cw_sw_all8", "metrics": {"idle_fp_per_min": 0.7, "control_recall": 0.76, "switch_latency_s": 1.9}},
    ]
    comparisons = _build_ab_comparisons(rows)
    names = {str(item.get("comparison", "")) for item in comparisons}
    assert "legacy_fbcca_202603 -> fbcca_plain_all8" in names
    assert "fbcca_plain_all8 -> fbcca_channel_weighted" in names
    assert "fbcca_plain_all8 -> fbcca_subband_weighted" in names
    assert "fbcca_plain_all8 -> fbcca_channel_subband_weighted" in names


def test_markdown_renderer_explains_global_subband_weights() -> None:
    payload = {
        "generated_at": "2026-04-12T12:00:00",
        "task": "fbcca-weighted-compare",
        "recommended_model": "fbcca_cw_sw_all8",
        "chosen_model": "fbcca_cw_sw_all8",
        "chosen_meets_acceptance": True,
        "default_profile_saved": False,
        "best_candidate_profile_path": "cand.json",
        "best_fbcca_weighted_profile_path": "weighted.json",
        "dataset_manifest_session1": "s1.json",
        "dataset_manifest_session2": "",
        "quality_kept_trials_session1": 70,
        "quality_total_trials_session1": 74,
        "quality_filter": {"min_sample_ratio": 0.9},
        "data_policy": "new-only",
        "protocol_signature_expected": "sig",
        "chosen_async_metrics": {
            "idle_fp_per_min": 0.5,
            "control_recall": 0.8,
            "switch_latency_s": 2.0,
            "release_latency_s": 1.1,
            "inference_ms": 5.0,
        },
        "chosen_metrics_4class": {
            "acc": 0.88,
            "macro_f1": 0.86,
            "mean_decision_time_s": 1.5,
            "itr_bpm": 35.0,
        },
        "chosen_metrics_2class": {
            "acc": 0.91,
            "macro_f1": 0.90,
            "mean_decision_time_s": 1.4,
            "itr_bpm": 28.0,
        },
        "weight_definition_notes": {
            "channel_weights": "8通道权重",
            "subband_weights": "5个全局子带权重",
            "separable_weighting": "8+5可分离",
            "spatial_filter_state": "TRCA对照",
        },
        "fbcca_weight_table": [
            {
                "rank": 1,
                "model_name": "fbcca_cw_sw_all8",
                "weight_strategy": "separable_channel_x_subband",
                "acc_4class": 0.88,
                "macro_f1_4class": 0.86,
                "itr_bpm_4class": 35.0,
                "metrics": {"idle_fp_per_min": 0.5, "control_recall": 0.8},
                "channel_weights": [1.0] * 8,
                "subband_weights": [0.3, 0.25, 0.2, 0.15, 0.1],
                "meets_acceptance": True,
            }
        ],
        "model_compare_table": [],
        "ab_comparisons": [],
        "figures": {},
    }
    text = _render_training_eval_markdown(payload)
    assert "全局子带权重" in text
    assert "可分离" in text

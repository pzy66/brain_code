from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.profile_deployment_audit import (
    audit_profile_files,
    audit_profile_payload,
    effective_profile_v2_gate_type,
    fast_control_release_failures,
    format_profile_audit_summary,
)


def _passing_metrics() -> dict[str, float]:
    return {
        "idle_fp_per_min": 0.0,
        "idle_selected_windows_per_min": 1.0,
        "control_recall": 0.98,
        "control_recall_at_3s": 0.90,
        "switch_detect_rate": 0.95,
        "switch_latency_s": 2.0,
        "release_latency_s": 1.2,
    }


def test_fast_control_release_failures_match_deployment_gate() -> None:
    assert fast_control_release_failures(_passing_metrics()) == []

    failures = fast_control_release_failures(
        {
            "idle_fp_per_min": 0.0,
            "control_recall": 1.0,
            "control_recall_at_3s": 0.20,
            "switch_detect_rate": 1.0,
            "switch_latency_s": 3.5,
            "release_latency_s": 3.25,
        }
    )

    assert any("control_recall_at_3s" in item for item in failures)
    assert any("release_latency_s" in item for item in failures)
    assert any("switch_latency_s" in item for item in failures)


def test_deployed_profile_audit_rejects_old_slow_threshold_profile() -> None:
    profile = {
        "model_name": "fbcca",
        "win_sec": 3.0,
        "step_sec": 0.25,
        "training_window_policy": "threshold_only_fixed_default_fbcca",
        "recommended_for_realtime": True,
        "benchmark_metrics": {
            "idle_fp_per_min": 0.0,
            "idle_selected_windows_per_min": 9.6,
            "control_recall": 1.0,
            "control_recall_at_2s": 0.0,
            "control_recall_at_3s": 0.20588235294117646,
            "switch_detect_rate": 1.0,
            "switch_detect_rate_at_2.8s": 0.0,
            "switch_latency_s": 3.5,
            "release_latency_s": 3.25,
        },
        "metadata": {"source": "fbcca_threshold_pretrain"},
    }
    profile_v2 = {
        "version": "2.0",
        "gate": {
            "type": "frequency_specific_logreg",
            "per_freq": {
                "8": {"coef": [0.0, 0.0], "intercept": 0.0},
                "10": {"coef": [0.0, 0.0], "intercept": 0.0},
            },
        },
    }

    audit = audit_profile_payload(profile, profile_v2_payload=profile_v2)

    assert audit["status"] == "fail"
    assert audit["run_valid_for_deployment"] is False
    assert audit["fast_control_release_valid"] is False
    assert audit["policy_valid"] is False
    assert audit["profile_v2_semantics_valid"] is False
    assert audit["profile_v2_effective_gate_type"] == "threshold_only_global_gate"
    assert any("recommended_for_realtime" in item for item in audit["warnings"])
    assert any("training_window_policy" in item for item in audit["warnings"])
    assert "FAIL" in format_profile_audit_summary(audit)


def test_deployed_profile_audit_accepts_fast_control_profile_files(tmp_path: Path) -> None:
    profile_path = tmp_path / "fbcca_profile.json"
    profile_v2_path = tmp_path / "fbcca_profile_v2.json"
    report_path = tmp_path / "report.json"
    index_path = tmp_path / "profile_index.json"
    profile_path.write_text(
        json.dumps(
            {
                "model_name": "fbcca",
                "win_sec": 2.0,
                "step_sec": 0.25,
                "training_window_policy": "fast-control-pretrain-v1",
                "recommended_for_realtime": True,
                "benchmark_metrics": _passing_metrics(),
                "metadata": {"source": "fast-control-pretrain-v1"},
            }
        ),
        encoding="utf-8",
    )
    profile_v2_path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "gate": {
                    "type": "threshold_only_global_gate",
                    "per_freq": {
                        "8": {"coef": [0.0, 0.0], "intercept": 0.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps({"run_valid_for_deployment": True}), encoding="utf-8")
    index_path.write_text(json.dumps({"report_json": str(report_path), "task": "fbcca-threshold-pretrain"}), encoding="utf-8")

    audit = audit_profile_files(profile_path, profile_index_path=index_path)

    assert audit["status"] == "pass"
    assert audit["run_valid_for_deployment"] is True
    assert audit["warnings"] == []
    assert audit["profile_v2_effective_gate_type"] == "threshold_only_global_gate"


def test_effective_profile_v2_gate_type_preserves_real_logreg() -> None:
    assert (
        effective_profile_v2_gate_type(
            {
                "type": "frequency_specific_logreg",
                "per_freq": {"8": {"coef": [0.1, 0.0], "intercept": 0.0}},
            }
        )
        == "frequency_specific_logreg"
    )

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.benchmark_suite import enrich_primary_metrics
from ssvep_core.profile_v2 import PROFILE_V2_VERSION, build_profile_v2, is_profile_v2_payload
from ssvep_core.registry import ModelRegistry


@dataclass(frozen=True)
class _DummyProfile:
    freqs: tuple[float, float, float, float] = (8.0, 10.0, 12.0, 15.0)
    model_name: str = "tdca"
    model_params: dict[str, object] = None  # type: ignore[assignment]
    eeg_channels: tuple[int, ...] = (0, 1, 2, 3)
    win_sec: float = 3.0
    step_sec: float = 0.25
    metadata: dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_params", dict(self.model_params or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


def test_registry_benchmark_models_include_tdca_and_etrca() -> None:
    models = ModelRegistry.list_models(task="benchmark")
    assert "tdca" in models
    assert "trca_r" in models
    assert "etrca_r" in models
    assert ModelRegistry.normalize("tdca_v2") == "tdca_v2"
    assert ModelRegistry.decoder_name("tdca_v2") == "tdca"
    assert ModelRegistry.normalize("etrca-r") == "etrca_r"


def test_primary_metric_enrichment_uses_idle_fp_and_precision_fallback() -> None:
    metrics = enrich_primary_metrics(
        {
            "idle_fp_per_min": 0.37,
            "control_precision": 0.91,
            "median_detection_latency_sec": 1.48,
        }
    )
    assert metrics["false_trigger_per_min"] == 0.37
    assert abs(metrics["wrong_action_rate"] - 0.09) < 1e-8
    assert metrics["median_commit_latency"] == 1.48


def test_build_profile_v2_payload_has_required_sections() -> None:
    profile = _DummyProfile(model_params={"Nh": 3, "state": {"mock": 1}}, metadata={"source": "unit-test"})
    v2 = build_profile_v2(
        base_profile=profile,
        per_freq_gate={
            "8": {"coef": [0.1, 0.2, 0.3, 0.4, 0.5], "intercept": 0.0, "enter_logit_th": 0.3, "exit_logit_th": 0.1}
        },
        metrics={"idle_fp_per_min": 0.2},
    )
    payload = v2.to_payload()
    assert payload["version"] == PROFILE_V2_VERSION
    assert set(payload.keys()) >= {"decoder", "gate", "evidence", "runtime", "metrics"}
    assert payload["decoder"]["name"] == "tdca"
    assert payload["gate"]["type"] == "frequency_specific_logreg"
    assert is_profile_v2_payload(payload)


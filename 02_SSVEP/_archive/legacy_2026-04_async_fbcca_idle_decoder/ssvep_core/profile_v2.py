from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PROFILE_V2_VERSION = "2.0"
DEFAULT_GATE_FEATURES = ("top1_score", "ratio", "margin", "normalized_top1", "score_entropy")


@dataclass(frozen=True)
class DecoderProfileV2:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    channels: tuple[int, ...] = ()
    templates_path: str = ""


@dataclass(frozen=True)
class GateProfileV2:
    type: str = "frequency_specific_logreg"
    feature_names: tuple[str, ...] = DEFAULT_GATE_FEATURES
    per_freq: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceProfileV2:
    lambda_decay: float = 0.85
    beta_consistency: float = 0.5
    upper_commit_th: float = 2.2
    lower_idle_th: float = 0.4


@dataclass(frozen=True)
class RuntimeProfileV2:
    win_sec: float = 3.0
    step_sec: float = 0.25
    refractory_sec: float = 0.8


@dataclass(frozen=True)
class ProfileV2:
    version: str
    freqs: tuple[float, float, float, float]
    decoder: DecoderProfileV2
    gate: GateProfileV2
    evidence: EvidenceProfileV2
    runtime: RuntimeProfileV2
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["freqs"] = [float(item) for item in self.freqs]
        payload["decoder"]["channels"] = [int(item) for item in self.decoder.channels]
        payload["gate"]["feature_names"] = [str(item) for item in self.gate.feature_names]
        payload["evidence"]["lambda"] = float(payload["evidence"].pop("lambda_decay"))
        return payload


def build_profile_v2(
    *,
    base_profile: Any,
    per_freq_gate: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    feature_names: tuple[str, ...] = DEFAULT_GATE_FEATURES,
    evidence: dict[str, Any] | None = None,
    refractory_sec: float = 0.8,
) -> ProfileV2:
    evidence_cfg = dict(evidence or {})
    model_params = dict(getattr(base_profile, "model_params", None) or {})
    channels = tuple(int(item) for item in (getattr(base_profile, "eeg_channels", None) or ()))
    return ProfileV2(
        version=PROFILE_V2_VERSION,
        freqs=tuple(float(item) for item in getattr(base_profile, "freqs")),  # type: ignore[arg-type]
        decoder=DecoderProfileV2(
            name=str(getattr(base_profile, "model_name", "tdca")),
            params=model_params,
            channels=channels,
        ),
        gate=GateProfileV2(
            type="frequency_specific_logreg",
            feature_names=tuple(str(name) for name in feature_names),
            per_freq=dict(per_freq_gate),
        ),
        evidence=EvidenceProfileV2(
            lambda_decay=float(evidence_cfg.get("lambda", 0.85)),
            beta_consistency=float(evidence_cfg.get("beta_consistency", 0.5)),
            upper_commit_th=float(evidence_cfg.get("upper_commit_th", 2.2)),
            lower_idle_th=float(evidence_cfg.get("lower_idle_th", 0.4)),
        ),
        runtime=RuntimeProfileV2(
            win_sec=float(getattr(base_profile, "win_sec", 3.0)),
            step_sec=float(getattr(base_profile, "step_sec", 0.25)),
            refractory_sec=float(refractory_sec),
        ),
        metrics=dict(metrics),
        metadata=dict(getattr(base_profile, "metadata", None) or {}),
    )


def is_profile_v2_payload(payload: dict[str, Any]) -> bool:
    return str(payload.get("version", "")).strip() == PROFILE_V2_VERSION and isinstance(payload.get("decoder"), dict)


from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import ThresholdProfile, json_dumps, load_decoder_from_profile, save_profile


def _float_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        if isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        metrics[str(key)] = float(numeric)
    return metrics


def threshold_profile_from_classifier_candidate(
    artifact: Mapping[str, Any],
    *,
    source_path: Optional[Path] = None,
) -> ThresholdProfile:
    payload = dict(artifact or {})
    if str(payload.get("artifact_schema_version", "")) != "external_fbcca_classifier_candidate_v1":
        raise ValueError("candidate artifact must use external_fbcca_classifier_candidate_v1 schema")
    if str(payload.get("model_name", "")) != "fbcca_score_ridge_5class":
        raise ValueError("only fbcca_score_ridge_5class artifacts are runtime-loadable")
    if not bool(payload.get("runtime_loadable", False)):
        raise ValueError("candidate artifact is not marked runtime_loadable")
    runtime_params = dict(payload.get("runtime_profile_model_params") or {})
    state = dict(runtime_params.get("state") or payload.get("state") or {})
    if not state:
        raise ValueError("runtime-loadable candidate artifact is missing classifier state")
    provenance = dict(payload.get("training_provenance") or {})
    feature_contract = dict(payload.get("feature_contract") or {})
    windowing = dict(payload.get("windowing") or {})
    command_freqs = state.get("freqs") or provenance.get("command_freqs") or []
    freqs = tuple(float(freq) for freq in command_freqs)
    if len(freqs) != 4:
        raise ValueError(f"runtime profile expects exactly 4 command frequencies, got {len(freqs)}")
    model_params = {
        "state": state,
        "score_source_name": runtime_params.get("score_source_name", provenance.get("score_source_name", "fbcca")),
        "score_bank_mode": runtime_params.get("score_bank_mode", feature_contract.get("score_bank_mode", "command_only")),
        "feature_names": list(runtime_params.get("feature_names", feature_contract.get("feature_names", [])) or []),
        "decoder_name": runtime_params.get("decoder_name", provenance.get("decoder_name", "fbcca_fixed_all8")),
        "decoder_model_params": dict(
            runtime_params.get("decoder_model_params", provenance.get("decoder_model_params", {})) or {}
        ),
        "full_reference_bank_freqs": list(
            runtime_params.get("full_reference_bank_freqs", provenance.get("full_reference_bank_freqs", [])) or []
        ),
        "max_gap_windows": int(runtime_params.get("max_gap_windows", windowing.get("max_gap_windows", 0)) or 0),
    }
    min_enter_windows = int(windowing.get("min_enter_windows", dict(state.get("fit_summary", {}) or {}).get("min_enter_windows", 1)) or 1)
    gate_policy = str(state.get("gate_policy", "confidence_threshold")).strip().lower()
    required_channels = list(provenance.get("required_channel_names", []) or [])
    eeg_channels = tuple(range(len(required_channels))) if required_channels else None
    metadata = {
        "source": "external_classifier_candidate_artifact",
        "candidate_artifact_path": "" if source_path is None else str(Path(source_path).expanduser().resolve()),
        "artifact_schema_version": str(payload.get("artifact_schema_version", "")),
        "training_provenance": provenance,
        "feature_contract": feature_contract,
        "holdout_eval": dict(payload.get("holdout_eval") or {}),
        "runtime_loadable": True,
        "requires_calibration": False,
        "has_stat_model": True,
    }
    return ThresholdProfile(
        freqs=(freqs[0], freqs[1], freqs[2], freqs[3]),
        win_sec=float(windowing.get("win_sec", 2.0)),
        step_sec=float(windowing.get("step_sec", 0.25)),
        enter_score_th=0.0,
        enter_ratio_th=1.0,
        enter_margin_th=0.0,
        exit_score_th=0.0,
        exit_ratio_th=1.0,
        min_enter_windows=max(1, int(min_enter_windows)),
        min_exit_windows=1,
        model_name="fbcca_score_ridge_5class",
        model_params=model_params,
        calibration_split_seed=provenance.get("seed"),
        benchmark_metrics=_float_metrics(payload.get("summary_metrics") or {}),
        eeg_channels=eeg_channels,
        gate_policy=gate_policy,
        channel_weight_mode=None,
        channel_weights=None,
        recommended_for_realtime=True,
        metadata=metadata,
    )


def export_classifier_candidate_profile(
    candidate_artifact_path: Path,
    output_profile_path: Path,
    *,
    validate_sampling_rate: Optional[int] = None,
) -> ThresholdProfile:
    source = Path(candidate_artifact_path).expanduser().resolve()
    artifact = json.loads(source.read_text(encoding="utf-8"))
    profile = threshold_profile_from_classifier_candidate(artifact, source_path=source)
    if validate_sampling_rate is not None:
        load_decoder_from_profile(profile, sampling_rate=int(validate_sampling_rate), compute_backend="cpu", gpu_warmup=False)
    save_profile(profile, Path(output_profile_path).expanduser().resolve())
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a runtime ThresholdProfile from an external classifier candidate artifact.")
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument(
        "--validate-sampling-rate",
        type=int,
        default=0,
        help="Optionally instantiate the runtime decoder with this sampling rate after export.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_rate = int(args.validate_sampling_rate)
    profile = export_classifier_candidate_profile(
        args.candidate_artifact,
        args.output_profile,
        validate_sampling_rate=validate_rate if validate_rate > 0 else None,
    )
    print(json_dumps({"status": "ok", "output_profile": str(Path(args.output_profile).expanduser().resolve()), "model_name": profile.model_name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

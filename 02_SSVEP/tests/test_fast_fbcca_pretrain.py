from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ssvep_core.async_fbcca_idle_standalone import ThresholdProfile, TrialSpec, create_decoder, load_profile, save_profile
from ssvep_core.fast_fbcca_pretrain import (
    DEFAULT_FAST_FBCCA_ACTIVE_SEC,
    DEFAULT_FAST_FBCCA_IDLE_REPEATS,
    DEFAULT_FAST_FBCCA_PREPARE_SEC,
    DEFAULT_FAST_FBCCA_REST_SEC,
    DEFAULT_FAST_FBCCA_TARGET_REPEATS,
    FastFBCCAPretrainConfig,
    fast_fbcca_estimated_collection_seconds,
    fast_fbcca_trial_count,
    run_fast_fbcca_personalization,
)


def _base_profile() -> ThresholdProfile:
    return ThresholdProfile(
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=2.0,
        step_sec=0.25,
        enter_score_th=0.05,
        enter_ratio_th=1.05,
        enter_margin_th=0.005,
        exit_score_th=0.03,
        exit_ratio_th=1.01,
        min_enter_windows=1,
        min_exit_windows=1,
        model_name="fbcca",
        model_params={"Nh": 3, "fbcca_variant": "fbcca_fixed_all8", "_decoder_model_name": "fbcca_fixed_all8"},
        eeg_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        subband_weight_mode="chen_fixed",
        metadata={"source": "test_base"},
    )


def _synth_segment(freq: float | None, *, fs: int = 250, seconds: float = 3.0, channels: int = 8) -> np.ndarray:
    samples = int(round(float(fs) * float(seconds)))
    t = np.arange(samples, dtype=np.float64) / float(fs)
    rng = np.random.default_rng(1234 + int(0 if freq is None else freq * 10))
    data = 0.01 * rng.standard_normal((samples, channels))
    if freq is None:
        return np.ascontiguousarray(data, dtype=np.float64)
    phases = np.linspace(0.0, np.pi / 5.0, channels, dtype=np.float64)
    for channel, phase in enumerate(phases):
        data[:, channel] += np.sin(2.0 * np.pi * float(freq) * t + phase)
        data[:, channel] += 0.35 * np.sin(2.0 * np.pi * float(freq) * 2.0 * t + phase * 0.5)
    return np.ascontiguousarray(data, dtype=np.float64)


def _fast_segments() -> list[tuple[TrialSpec, np.ndarray]]:
    segments: list[tuple[TrialSpec, np.ndarray]] = []
    trial_id = 0
    for _repeat in range(2):
        for freq in (8.0, 10.0, 12.0, 15.0):
            segments.append(
                (
                    TrialSpec(label=f"{freq:g}Hz", expected_freq=float(freq), trial_id=trial_id, block_index=trial_id),
                    _synth_segment(float(freq)),
                )
            )
            trial_id += 1
    for _index in range(4):
        segments.append(
            (
                TrialSpec(label="idle", expected_freq=None, trial_id=trial_id, block_index=trial_id),
                _synth_segment(None),
            )
        )
        trial_id += 1
    return segments


def test_fast_pretrain_defaults_are_about_one_minute() -> None:
    assert fast_fbcca_trial_count(
        target_repeats=DEFAULT_FAST_FBCCA_TARGET_REPEATS,
        idle_repeats=DEFAULT_FAST_FBCCA_IDLE_REPEATS,
    ) == 12
    assert fast_fbcca_estimated_collection_seconds(
        target_repeats=DEFAULT_FAST_FBCCA_TARGET_REPEATS,
        idle_repeats=DEFAULT_FAST_FBCCA_IDLE_REPEATS,
        prepare_sec=DEFAULT_FAST_FBCCA_PREPARE_SEC,
        active_sec=DEFAULT_FAST_FBCCA_ACTIVE_SEC,
        rest_sec=DEFAULT_FAST_FBCCA_REST_SEC,
    ) == 51.0


def test_fast_personalization_profile_roundtrip_and_decoder(tmp_path: Path) -> None:
    base_path = tmp_path / "fbcca_base_profile.json"
    output_path = tmp_path / "fbcca_profile.json"
    history_path = tmp_path / "history_profile.json"
    save_profile(_base_profile(), base_path)

    profile, payload = run_fast_fbcca_personalization(
        FastFBCCAPretrainConfig(
            base_profile_path=base_path,
            fallback_profile_path=tmp_path / "missing_fbcca_profile.json",
            output_profile_path=output_path,
            history_profile_path=history_path,
            compute_backend="cpu",
            gpu_warmup=False,
        ),
        trial_segments=_fast_segments(),
        sampling_rate=250,
        available_board_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        collection_duration_sec=51.0,
    )

    assert output_path.exists()
    assert output_path.with_name("fbcca_profile_v2.json").exists()
    profile_v2 = json.loads(output_path.with_name("fbcca_profile_v2.json").read_text(encoding="utf-8"))
    assert profile_v2["gate"]["type"] == "global_threshold"
    assert history_path.exists()
    assert payload["status"] == "ok"
    assert payload["template_enabled"] is True
    assert payload["gate_feature_source"] == "fbcca_template_fused"
    assert profile.model_name == "fbcca"
    assert profile.model_params is not None
    assert "fast_personalization" in profile.model_params

    loaded = load_profile(output_path, require_exists=True)
    assert loaded.model_name == "fbcca"
    assert loaded.model_params is not None
    assert dict(loaded.metadata or {})["fast_pretrain"]["template_enabled"] is True

    decoder = create_decoder(
        "fbcca",
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=2.0,
        step_sec=0.25,
        model_params=dict(loaded.model_params),
        compute_backend="cpu",
        gpu_warmup=False,
    )
    result = decoder.analyze_window(_synth_segment(8.0, seconds=2.0))
    assert result["fast_personalization_template_enabled"] is True
    assert np.asarray(result["scores"]).shape == (4,)
    assert np.all(np.isfinite(np.asarray(result["scores"], dtype=float)))


def test_fbcca_decoder_ignores_invalid_fast_templates() -> None:
    decoder = create_decoder(
        "fbcca",
        sampling_rate=250,
        freqs=(8.0, 10.0, 12.0, 15.0),
        win_sec=2.0,
        step_sec=0.25,
        model_params={
            "Nh": 3,
            "fast_personalization": {
                "version": 1,
                "template_weight": 0.25,
                "template_win_sec": 2.0,
                "templates": {"8": np.zeros((500, 8), dtype=float).tolist()},
            },
        },
        compute_backend="cpu",
        gpu_warmup=False,
    )
    result = decoder.analyze_window(_synth_segment(8.0, seconds=2.0))
    assert result["fast_personalization_template_enabled"] is False
    assert "fast_personalization_warnings" in result


def test_fast_fallback_strips_old_session_templates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ssvep_core.fast_fbcca_pretrain as module

    fallback_path = tmp_path / "fbcca_profile.json"
    output_path = tmp_path / "out" / "fbcca_profile.json"
    old_profile = _base_profile()
    old_params = dict(old_profile.model_params or {})
    old_params["fast_personalization"] = {
        "version": 1,
        "template_weight": 0.25,
        "template_win_sec": 2.0,
        "templates": {"8": np.zeros((500, 8), dtype=float).tolist()},
    }
    save_profile(
        ThresholdProfile(
            **{
                **old_profile.__dict__,
                "model_params": old_params,
                "metadata": {"source": "old_session"},
            }
        ),
        fallback_path,
    )
    monkeypatch.setattr(
        module,
        "_should_fallback_to_base",
        lambda **_kwargs: (True, ["forced fallback for test"]),
    )

    profile, payload = run_fast_fbcca_personalization(
        FastFBCCAPretrainConfig(
            base_profile_path=tmp_path / "missing_base.json",
            fallback_profile_path=fallback_path,
            output_profile_path=output_path,
            history_profile_path=tmp_path / "history.json",
            compute_backend="cpu",
            gpu_warmup=False,
        ),
        trial_segments=_fast_segments(),
        sampling_rate=250,
        available_board_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        collection_duration_sec=51.0,
    )

    assert payload["status"] == "fallback_to_base"
    assert payload["template_enabled"] is False
    assert "fast_personalization" not in dict(profile.model_params or {})
    loaded = load_profile(output_path, require_exists=True)
    assert "fast_personalization" not in dict(loaded.model_params or {})
    assert dict(loaded.metadata or {})["fast_pretrain"]["template_enabled"] is False


def test_gate_calibration_uses_template_fused_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ssvep_core.fast_fbcca_pretrain as module

    base_path = tmp_path / "fbcca_base_profile.json"
    save_profile(_base_profile(), base_path)
    captured: dict[str, object] = {}
    original_fit = module.fit_threshold_profile

    def spy_fit_threshold_profile(feature_rows, *args, **kwargs):
        rows = list(feature_rows)
        captured["template_flags"] = [bool(row.get("fast_personalization_template_enabled", False)) for row in rows]
        return original_fit(rows, *args, **kwargs)

    monkeypatch.setattr(module, "fit_threshold_profile", spy_fit_threshold_profile)

    profile, payload = run_fast_fbcca_personalization(
        FastFBCCAPretrainConfig(
            base_profile_path=base_path,
            fallback_profile_path=tmp_path / "missing_fbcca_profile.json",
            output_profile_path=tmp_path / "fbcca_profile.json",
            history_profile_path=tmp_path / "history.json",
            compute_backend="cpu",
            gpu_warmup=False,
        ),
        trial_segments=_fast_segments(),
        sampling_rate=250,
        available_board_channels=(0, 1, 2, 3, 4, 5, 6, 7),
        collection_duration_sec=51.0,
    )

    assert payload["status"] == "ok"
    assert payload["gate_feature_source"] == "fbcca_template_fused"
    assert dict(profile.metadata or {})["fast_pretrain"]["gate_feature_source"] == "fbcca_template_fused"
    assert captured["template_flags"]
    assert all(captured["template_flags"])

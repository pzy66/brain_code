from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Sequence

import numpy as np


STIMULUS_PROFILE_COMFORT_FBCCA_V1 = "comfort_fbcca_v1"
STIMULUS_PROFILE_COMFORT_FBCCA_WANG_FAST_V1 = "comfort_fbcca_wang_fast_v1"
STIMULUS_PROFILE_LEGACY_FULL_CONTRAST = "legacy_full_contrast"
DEFAULT_STIMULUS_PROFILE_ID = STIMULUS_PROFILE_COMFORT_FBCCA_V1

DEFAULT_COMFORT_FREQS = (9.8, 12.0, 14.8, 15.8)
DEFAULT_WANG_FAST_FREQS = (11.0, 12.0, 14.8, 15.8)
DEFAULT_COMFORT_REFRESH_RATE_HZ = 240.0
DEFAULT_SCREEN_BRIGHTNESS_NOTE = "not_recorded"
DEFAULT_COMFORT_RATING = None
STIMULUS_MODE_ELAPSED_TIME_SINE = "elapsed_time_sine"
STIMULUS_MODE_FRAME_LOCKED_SINE = "frame_locked_sine"
FRAME_LOCK_FRACTION_MAX_DENOMINATOR = 1_000_000


@dataclass(frozen=True)
class StimulusProfile:
    profile_id: str
    freqs: tuple[float, float, float, float]
    preferred_mode: str
    fallback_mode: str
    refresh_rate_hz: float
    mean: float
    amp: float
    phi: float = 0.0
    ramp_sec: float = 0.0
    screen_brightness_note: str = DEFAULT_SCREEN_BRIGHTNESS_NOTE
    comfort_rating: int | None = DEFAULT_COMFORT_RATING
    description: str = ""

    @property
    def luminance_min(self) -> float:
        return max(0.0, float(self.mean) - abs(float(self.amp)))

    @property
    def luminance_max(self) -> float:
        return min(1.0, float(self.mean) + abs(float(self.amp)))

    @property
    def michelson_contrast(self) -> float:
        denom = float(self.luminance_max + self.luminance_min)
        if denom <= 1e-12:
            return 0.0
        return float((self.luminance_max - self.luminance_min) / denom)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["freqs"] = [float(freq) for freq in self.freqs]
        payload["luminance_min"] = float(self.luminance_min)
        payload["luminance_max"] = float(self.luminance_max)
        payload["michelson_contrast"] = float(self.michelson_contrast)
        return payload


STIMULUS_PROFILES: dict[str, StimulusProfile] = {
    STIMULUS_PROFILE_COMFORT_FBCCA_V1: StimulusProfile(
        profile_id=STIMULUS_PROFILE_COMFORT_FBCCA_V1,
        freqs=DEFAULT_COMFORT_FREQS,
        preferred_mode=STIMULUS_MODE_FRAME_LOCKED_SINE,
        fallback_mode=STIMULUS_MODE_ELAPSED_TIME_SINE,
        refresh_rate_hz=DEFAULT_COMFORT_REFRESH_RATE_HZ,
        mean=0.40,
        amp=0.20,
        phi=0.0,
        ramp_sec=0.30,
        description="Comfort-first FBCCA SSVEP stimulus for 9.8/12/14.8/15.8Hz async control.",
    ),
    STIMULUS_PROFILE_COMFORT_FBCCA_WANG_FAST_V1: StimulusProfile(
        profile_id=STIMULUS_PROFILE_COMFORT_FBCCA_WANG_FAST_V1,
        freqs=DEFAULT_WANG_FAST_FREQS,
        preferred_mode=STIMULUS_MODE_FRAME_LOCKED_SINE,
        fallback_mode=STIMULUS_MODE_ELAPSED_TIME_SINE,
        refresh_rate_hz=DEFAULT_COMFORT_REFRESH_RATE_HZ,
        mean=0.40,
        amp=0.20,
        phi=0.0,
        ramp_sec=0.30,
        description="Experimental comfort FBCCA stimulus for 11/12/14.8/15.8Hz async control.",
    ),
    STIMULUS_PROFILE_LEGACY_FULL_CONTRAST: StimulusProfile(
        profile_id=STIMULUS_PROFILE_LEGACY_FULL_CONTRAST,
        freqs=DEFAULT_COMFORT_FREQS,
        preferred_mode=STIMULUS_MODE_ELAPSED_TIME_SINE,
        fallback_mode=STIMULUS_MODE_ELAPSED_TIME_SINE,
        refresh_rate_hz=DEFAULT_COMFORT_REFRESH_RATE_HZ,
        mean=0.50,
        amp=0.50,
        phi=0.0,
        ramp_sec=0.0,
        description="Legacy full-range luminance sine stimulus.",
    ),
}


def validate_stimulus_profile_id(value: str | None) -> str:
    profile_id = str(value or DEFAULT_STIMULUS_PROFILE_ID).strip().lower()
    if profile_id not in STIMULUS_PROFILES:
        joined = "|".join(sorted(STIMULUS_PROFILES))
        raise ValueError(f"stimulus_profile_id must be one of: {joined}")
    return profile_id


def get_stimulus_profile(profile_id: str | None = None) -> StimulusProfile:
    return STIMULUS_PROFILES[validate_stimulus_profile_id(profile_id)]


def refresh_rate_is_stable_240hz(refresh_rate_hz: float, *, tolerance_hz: float = 2.0) -> bool:
    hz = float(refresh_rate_hz)
    return bool(np.isfinite(hz) and abs(hz - 240.0) <= float(tolerance_hz))


def _positive_finite_fraction(value: float, *, name: str) -> Fraction:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return Fraction(str(numeric)).limit_denominator(FRAME_LOCK_FRACTION_MAX_DENOMINATOR)


def frame_lock_frequency_entry(freq: float, *, refresh_rate_hz: float) -> dict[str, Any]:
    freq_fraction = _positive_finite_fraction(float(freq), name="freq")
    refresh_fraction = _positive_finite_fraction(float(refresh_rate_hz), name="refresh_rate_hz")
    phase_increment = freq_fraction / refresh_fraction
    frames_per_cycle = refresh_fraction / freq_fraction
    repeat_frames = int(phase_increment.denominator)
    repeat_sec = Fraction(repeat_frames, 1) / refresh_fraction
    return {
        "frequency_hz": float(freq_fraction),
        "refresh_rate_hz": float(refresh_fraction),
        "phase_increment_cycles_per_frame": float(phase_increment),
        "frames_per_cycle": float(frames_per_cycle),
        "integer_frames_per_cycle": bool(frames_per_cycle.denominator == 1),
        "frame_sequence_repeat_frames": int(repeat_frames),
        "frame_sequence_repeat_sec": float(repeat_sec),
        "cycles_per_repeat": int(phase_increment.numerator),
        "frame_locked_sine_exact_sampled_sequence": True,
    }


def frame_lock_frequency_report(freqs: Sequence[float], *, refresh_rate_hz: float) -> dict[str, Any]:
    rows = [
        frame_lock_frequency_entry(float(freq), refresh_rate_hz=float(refresh_rate_hz))
        for freq in tuple(freqs or ())
    ]
    repeat_frames = [int(row["frame_sequence_repeat_frames"]) for row in rows]
    repeat_sec = [float(row["frame_sequence_repeat_sec"]) for row in rows]
    return {
        "stimulus_model": STIMULUS_MODE_FRAME_LOCKED_SINE,
        "refresh_rate_hz": float(refresh_rate_hz),
        "frequency_count": int(len(rows)),
        "all_integer_frames_per_cycle": bool(rows and all(bool(row["integer_frames_per_cycle"]) for row in rows)),
        "all_frame_sequences_repeat_exactly": bool(rows and all(bool(row["frame_locked_sine_exact_sampled_sequence"]) for row in rows)),
        "max_frame_sequence_repeat_frames": int(max(repeat_frames)) if repeat_frames else 0,
        "max_frame_sequence_repeat_sec": float(max(repeat_sec)) if repeat_sec else 0.0,
        "frequencies": rows,
    }


def select_stimulus_mode_for_profile(
    profile_id: str | None,
    *,
    refresh_rate_hz: float,
    requested_mode: str | None = None,
) -> tuple[str, str]:
    profile = get_stimulus_profile(profile_id)
    requested = str(requested_mode or "").strip().lower()
    if requested and requested != "auto":
        return requested, "manual"
    if (
        profile.preferred_mode == STIMULUS_MODE_FRAME_LOCKED_SINE
        and profile.fallback_mode == STIMULUS_MODE_ELAPSED_TIME_SINE
    ):
        if refresh_rate_is_stable_240hz(refresh_rate_hz):
            return profile.preferred_mode, "stable_240hz_frame_locked"
        return profile.fallback_mode, "fallback_refresh_not_confirmed_240hz"
    return profile.preferred_mode, "profile_default"


def stimulus_profile_metadata(
    profile_id: str | None,
    *,
    stimulus_mode: str,
    refresh_rate_hz: float,
    freqs: Sequence[float] | None = None,
    mode_selection_reason: str = "",
    comfort_rating: int | None = None,
    screen_brightness_note: str | None = None,
    frame_interval_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = get_stimulus_profile(profile_id)
    note = str(screen_brightness_note or profile.screen_brightness_note or DEFAULT_SCREEN_BRIGHTNESS_NOTE)
    rating = profile.comfort_rating if comfort_rating is None else comfort_rating
    actual_freqs = tuple(float(freq) for freq in (profile.freqs if freqs is None else tuple(freqs)))
    return {
        "stimulus_profile_id": str(profile.profile_id),
        "stimulus_profile": profile.to_payload(),
        "stimulus_mode": str(stimulus_mode),
        "stimulus_mode_selection_reason": str(mode_selection_reason or ""),
        "stim_refresh_rate_hz": float(refresh_rate_hz),
        "stim_mean": float(profile.mean),
        "stim_amp": float(profile.amp),
        "stim_phi": float(profile.phi),
        "stim_luminance_min": float(profile.luminance_min),
        "stim_luminance_max": float(profile.luminance_max),
        "stim_michelson_contrast": float(profile.michelson_contrast),
        "ramp_sec": float(profile.ramp_sec),
        "ramp_included_in_saved_window": bool(float(profile.ramp_sec) > 0.0),
        "frame_interval_stats": dict(frame_interval_stats or {}),
        "frame_lock_frequency_report": frame_lock_frequency_report(
            actual_freqs,
            refresh_rate_hz=float(refresh_rate_hz),
        ),
        "comfort_rating": rating,
        "screen_brightness_note": note,
    }


def profile_matches_freqs(profile_id: str | None, freqs: Sequence[float]) -> bool:
    profile_freqs = tuple(float(freq) for freq in get_stimulus_profile(profile_id).freqs)
    values = tuple(float(freq) for freq in freqs)
    return len(values) == len(profile_freqs) and all(abs(left - right) <= 1e-6 for left, right in zip(values, profile_freqs))


def find_matching_stimulus_profile_id(freqs: Sequence[float]) -> str | None:
    values = tuple(float(freq) for freq in freqs)
    for profile_id in STIMULUS_PROFILES:
        if profile_matches_freqs(profile_id, values):
            return profile_id
    return None

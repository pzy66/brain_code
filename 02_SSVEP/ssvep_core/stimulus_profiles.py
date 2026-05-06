from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


STIMULUS_PROFILE_COMFORT_FBCCA_V1 = "comfort_fbcca_v1"
STIMULUS_PROFILE_LEGACY_FULL_CONTRAST = "legacy_full_contrast"
DEFAULT_STIMULUS_PROFILE_ID = STIMULUS_PROFILE_COMFORT_FBCCA_V1

DEFAULT_COMFORT_FREQS = (8.0, 10.0, 12.0, 15.0)
DEFAULT_COMFORT_REFRESH_RATE_HZ = 240.0
DEFAULT_SCREEN_BRIGHTNESS_NOTE = "not_recorded"
DEFAULT_COMFORT_RATING = None
STIMULUS_MODE_ELAPSED_TIME_SINE = "elapsed_time_sine"
STIMULUS_MODE_FRAME_LOCKED_SINE = "frame_locked_sine"


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
        description="Comfort-first FBCCA SSVEP stimulus for 8/10/12/15Hz async control.",
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
    if profile.profile_id == STIMULUS_PROFILE_COMFORT_FBCCA_V1:
        if refresh_rate_is_stable_240hz(refresh_rate_hz):
            return profile.preferred_mode, "stable_240hz_frame_locked"
        return profile.fallback_mode, "fallback_refresh_not_confirmed_240hz"
    return profile.preferred_mode, "profile_default"


def stimulus_profile_metadata(
    profile_id: str | None,
    *,
    stimulus_mode: str,
    refresh_rate_hz: float,
    mode_selection_reason: str = "",
    comfort_rating: int | None = None,
    screen_brightness_note: str | None = None,
    frame_interval_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = get_stimulus_profile(profile_id)
    note = str(screen_brightness_note or profile.screen_brightness_note or DEFAULT_SCREEN_BRIGHTNESS_NOTE)
    rating = profile.comfort_rating if comfort_rating is None else comfort_rating
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
        "comfort_rating": rating,
        "screen_brightness_note": note,
    }


def profile_matches_freqs(profile_id: str | None, freqs: Sequence[float]) -> bool:
    profile_freqs = tuple(float(freq) for freq in get_stimulus_profile(profile_id).freqs)
    values = tuple(float(freq) for freq in freqs)
    return len(values) == len(profile_freqs) and all(abs(left - right) <= 1e-6 for left, right in zip(values, profile_freqs))

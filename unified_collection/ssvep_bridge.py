"""Compatibility imports for the SSVEP collection subsystem."""

from __future__ import annotations

from brain_workspace.paths import ensure_runtime_import_paths

ensure_runtime_import_paths()

from apps.async_fbcca_validation_ui import (  # noqa: E402
    PHASE_CAL_ACTIVE,
    PHASE_CAL_PREPARE,
    PHASE_CAL_REST,
    PHASE_ERROR,
    PHASE_STOPPED,
    STIMULUS_MODE_ELAPSED_TIME_SINE,
    STIMULUS_MODE_FRAME_LOCKED_SINE,
    validate_stimulus_mode,
)
from apps.data_collection_ui import (  # noqa: E402
    ACTIVE_START_CUE_SEC,
    MAX_TRIAL_RETRIES,
    MIN_TRIAL_QUALITY_RATIO,
    STIM_AMP,
    STIM_FRAME_FORMULA,
    STIM_MEAN,
    STIM_PHI,
    STIMULUS_BACKEND_PYQT_FULLSCREEN,
    STIMULUS_PHASE_APPLY_TIMEOUT_SEC,
    CollectionFullscreenStimWindow,
    build_collection_output_session_id,
    estimate_active_stimulus_arm_sec,
    play_collection_tone_event,
    play_collection_tone_event_sync,
    prompt_text_for_trial,
    resolve_collection_stim_refresh_rate_hz,
    stimulus_backend_metadata,
    stimulus_sample_window_alignment_metadata,
    validate_stimulus_frequency_set,
)
from ssvep_core.async_fbcca_idle_standalone import parse_freqs  # noqa: E402
from ssvep_core.dataset import (  # noqa: E402
    CollectionProtocol,
    build_collection_trials,
    save_collection_dataset_bundle,
)

__all__ = [
    "PHASE_CAL_ACTIVE",
    "PHASE_CAL_PREPARE",
    "PHASE_CAL_REST",
    "PHASE_ERROR",
    "PHASE_STOPPED",
    "STIMULUS_MODE_ELAPSED_TIME_SINE",
    "STIMULUS_MODE_FRAME_LOCKED_SINE",
    "validate_stimulus_mode",
    "ACTIVE_START_CUE_SEC",
    "MAX_TRIAL_RETRIES",
    "MIN_TRIAL_QUALITY_RATIO",
    "STIM_AMP",
    "STIM_FRAME_FORMULA",
    "STIM_MEAN",
    "STIM_PHI",
    "STIMULUS_BACKEND_PYQT_FULLSCREEN",
    "STIMULUS_PHASE_APPLY_TIMEOUT_SEC",
    "CollectionFullscreenStimWindow",
    "build_collection_output_session_id",
    "estimate_active_stimulus_arm_sec",
    "play_collection_tone_event",
    "play_collection_tone_event_sync",
    "prompt_text_for_trial",
    "resolve_collection_stim_refresh_rate_hz",
    "stimulus_backend_metadata",
    "stimulus_sample_window_alignment_metadata",
    "validate_stimulus_frequency_set",
    "parse_freqs",
    "CollectionProtocol",
    "build_collection_trials",
    "save_collection_dataset_bundle",
]

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ENHANCED_45M_PROTOCOL",
    "CollectionProtocol",
    "LoadedDataset",
    "build_collection_trials",
    "load_collection_dataset",
    "save_collection_dataset_bundle",
    "AsyncGateAdapter",
    "BoardShim",
    "describe_runtime_error",
    "ensure_stream_ready",
    "normalize_serial_port",
    "prepare_board_session",
    "read_recent_eeg_segment",
    "resolve_selected_eeg_channels",
    "SUPPORTED_MODEL_NAMES",
    "DecoderModelAdapter",
    "create_model_adapter",
    "ModelRegistry",
    "ArtifactStore",
    "ArtifactKey",
    "BenchmarkReport",
    "evaluate_models",
    "enrich_primary_metrics",
    "ProfileV2",
    "build_profile_v2",
    "PROFILE_V2_VERSION",
    "is_profile_v2_payload",
    "export_evaluation_figures",
    "OfflineTrainEvalConfig",
    "run_offline_train_eval",
]

_EXPORT_MAP = {
    "ENHANCED_45M_PROTOCOL": (".dataset", "ENHANCED_45M_PROTOCOL"),
    "CollectionProtocol": (".dataset", "CollectionProtocol"),
    "LoadedDataset": (".dataset", "LoadedDataset"),
    "build_collection_trials": (".dataset", "build_collection_trials"),
    "load_collection_dataset": (".dataset", "load_collection_dataset"),
    "save_collection_dataset_bundle": (".dataset", "save_collection_dataset_bundle"),
    "AsyncGateAdapter": (".gate", "AsyncGateAdapter"),
    "BoardShim": (".io_brainflow", "BoardShim"),
    "describe_runtime_error": (".io_brainflow", "describe_runtime_error"),
    "ensure_stream_ready": (".io_brainflow", "ensure_stream_ready"),
    "normalize_serial_port": (".io_brainflow", "normalize_serial_port"),
    "prepare_board_session": (".io_brainflow", "prepare_board_session"),
    "read_recent_eeg_segment": (".io_brainflow", "read_recent_eeg_segment"),
    "resolve_selected_eeg_channels": (".io_brainflow", "resolve_selected_eeg_channels"),
    "SUPPORTED_MODEL_NAMES": (".models", "SUPPORTED_MODEL_NAMES"),
    "DecoderModelAdapter": (".models", "DecoderModelAdapter"),
    "create_model_adapter": (".models", "create_model_adapter"),
    "ModelRegistry": (".registry", "ModelRegistry"),
    "ArtifactStore": (".artifact_store", "ArtifactStore"),
    "ArtifactKey": (".artifact_store", "ArtifactKey"),
    "BenchmarkReport": (".benchmark_suite", "BenchmarkReport"),
    "evaluate_models": (".benchmark_suite", "evaluate_models"),
    "enrich_primary_metrics": (".benchmark_suite", "enrich_primary_metrics"),
    "ProfileV2": (".profile_v2", "ProfileV2"),
    "build_profile_v2": (".profile_v2", "build_profile_v2"),
    "PROFILE_V2_VERSION": (".profile_v2", "PROFILE_V2_VERSION"),
    "is_profile_v2_payload": (".profile_v2", "is_profile_v2_payload"),
    "export_evaluation_figures": (".reporting", "export_evaluation_figures"),
    "OfflineTrainEvalConfig": (".train_eval", "OfflineTrainEvalConfig"),
    "run_offline_train_eval": (".train_eval", "run_offline_train_eval"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'ssvep_core' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

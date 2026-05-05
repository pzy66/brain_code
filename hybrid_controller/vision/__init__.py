from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "SlotState": ("hybrid_controller.vision.processing", "SlotState"),
    "VisionCalibration": ("hybrid_controller.vision.processing", "VisionCalibration"),
    "annotate_slots_with_cylindrical": ("hybrid_controller.vision.processing", "annotate_slots_with_cylindrical"),
    "build_vision_packet": ("hybrid_controller.vision.processing", "build_vision_packet"),
    "extract_candidates": ("hybrid_controller.vision.processing", "extract_candidates"),
    "packet_to_targets": ("hybrid_controller.vision.processing", "packet_to_targets"),
    "update_slots": ("hybrid_controller.vision.processing", "update_slots"),
    "VisionRuntime": ("hybrid_controller.vision.runtime", "VisionRuntime"),
    "VisionResolutionResult": ("hybrid_controller.vision.target_resolver", "VisionResolutionResult"),
    "resolve_vision_packet": ("hybrid_controller.vision.target_resolver", "resolve_vision_packet"),
    "VisionCalibrationProfile": ("hybrid_controller.vision.calibration_profile", "VisionCalibrationProfile"),
    "VisionMappingResult": ("hybrid_controller.vision.calibration_profile", "VisionMappingResult"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

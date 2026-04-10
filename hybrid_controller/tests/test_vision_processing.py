from __future__ import annotations

import numpy as np

from hybrid_controller.vision.processing import (
    SlotState,
    VisionCalibration,
    annotate_slots_with_cylindrical,
    build_vision_packet,
    extract_candidates,
    packet_to_targets,
    update_slots,
)


class _TensorLike:
    def __init__(self, values):
        self._values = np.array(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _TensorLike(xyxy)
        self.conf = _TensorLike(conf)


class _Result:
    def __init__(self, xyxy, conf):
        self.boxes = _Boxes(xyxy, conf)
        self.masks = None


def test_vision_calibration_identity_maps_pixel_to_world_plane():
    calibration = VisionCalibration.from_param_dict(
        {
            "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "R": [[0], [0], [0]],
            "T": [[0], [0], [1]],
        }
    )
    world_xyz = calibration.camera_to_world(12.0, 34.0)
    assert world_xyz == (12.0, 34.0, 0.0)


def test_extract_candidates_filters_to_roi_and_keeps_distance_order():
    result = _Result(
        xyxy=[
            [100, 100, 140, 140],
            [300, 300, 360, 360],
            [110, 110, 150, 150],
        ],
        conf=[0.9, 0.95, 0.8],
    )
    candidates, detected_count = extract_candidates(
        result,
        frame_shape=(480, 640),
        roi_center=(128, 128),
        roi_radius=60,
        max_det=4,
        confidence_threshold=0.25,
    )
    assert detected_count == 2
    assert [candidate.center for candidate in candidates] == [(130, 130), (120, 120)]


def test_slot_tracking_and_packet_output_emit_cylindrical_targets():
    slots = [SlotState(slot=index + 1, freq_hz=freq) for index, freq in enumerate((8.0, 10.0, 12.0, 15.0))]
    result = _Result(
        xyxy=[
            [100, 100, 140, 140],
            [180, 100, 220, 140],
        ],
        conf=[0.9, 0.88],
    )
    candidates, _ = extract_candidates(
        result,
        frame_shape=(480, 640),
        roi_center=(160, 120),
        roi_radius=120,
        max_det=4,
        confidence_threshold=0.25,
    )
    update_slots(slots, candidates, match_distance=120.0, lost_ttl=6)
    update_slots(slots, candidates, match_distance=120.0, lost_ttl=6)

    calibration = VisionCalibration.from_param_dict(
        {
            "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "R": [[0], [0], [0]],
            "T": [[0], [0], [1]],
        }
    )
    annotate_slots_with_cylindrical(slots, calibration=calibration)
    packet = build_vision_packet(
        frame_id=3,
        frame_size=(640, 480),
        roi_center=(160, 120),
        roi_radius=120,
        slots=slots,
        capture_fps=30.0,
        infer_ms=12.5,
        queue_age_ms=4.0,
        detected_count=2,
        calibration_ready=True,
    )
    targets = packet_to_targets(packet)

    assert len(targets) == 2
    assert targets[0].command_mode == "world"
    assert targets[0].slot_id == 1
    assert targets[0].freq_hz == 8.0
    assert targets[0].command_point is not None
    assert targets[0].cylindrical_center is not None
    assert targets[0].actionable is True
    assert packet["slots"][0]["slot_id"] == 1
    assert packet["slots"][0]["command_mode"] == "world"

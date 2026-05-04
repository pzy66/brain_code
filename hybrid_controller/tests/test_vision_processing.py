from __future__ import annotations

import numpy as np
import cv2

from hybrid_controller.vision.processing import (
    DetectionCandidate,
    SlotState,
    VisionCalibration,
    annotate_slots_with_cylindrical,
    build_vision_packet,
    extract_candidates,
    frame_to_block_candidates,
    mask_to_grasp_geometry,
    packet_to_targets,
    update_slots,
)
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile


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


def test_vision_calibration_undistorts_pixels_when_distortion_coefficients_exist():
    calibration = VisionCalibration.from_param_dict(
        {
            "K": [[300, 0, 320], [0, 300, 240], [0, 0, 1]],
            "R": [[0], [0], [0]],
            "T": [[0], [0], [1]],
            "D": [0.15, -0.05, 0.0, 0.0, 0.0],
        }
    )

    undistorted = calibration.undistort_pixel(520.0, 240.0)

    assert undistorted[0] != 520.0
    assert abs(undistorted[1] - 240.0) < 1e-6


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
    assert packet["slots"][0]["grasp_pixel"] is not None
    assert packet["slots"][0]["grasp_quality"] > 0.0


def test_grasp_history_requires_stable_recent_pixels_and_resets_on_jump():
    slots = [SlotState(slot=1, freq_hz=8.0)]

    def candidate(point: tuple[int, int]) -> DetectionCandidate:
        x, y = point
        return DetectionCandidate(
            center=point,
            grasp_pixel=point,
            bbox=(x - 10, y - 10, x + 10, y + 10),
            area_px=400,
            confidence=0.9,
            polygon=[(x - 10, y - 10), (x + 10, y - 10), (x + 10, y + 10), (x - 10, y + 10)],
            grasp_quality=1.0,
            oriented_bbox=[(x - 10, y - 10), (x + 10, y - 10), (x + 10, y + 10), (x - 10, y + 10)],
            distance_to_roi=0.0,
        )

    for point in ((319, 240), (321, 241), (320, 239)):
        update_slots(
            slots,
            [candidate(point)],
            match_distance=120.0,
            lost_ttl=6,
            grasp_history_len=5,
            grasp_stability_tolerance_px=3.0,
            grasp_history_reset_px=22.0,
        )

    assert slots[0].grasp_pixel == (320, 240)
    assert slots[0].grasp_stable_frames == 3
    assert slots[0].grasp_stability_px is not None
    assert slots[0].grasp_stability_px <= 3.0

    update_slots(
        slots,
        [candidate((360, 240))],
        match_distance=120.0,
        lost_ttl=6,
        grasp_history_len=5,
        grasp_stability_tolerance_px=3.0,
        grasp_history_reset_px=22.0,
    )

    assert slots[0].grasp_pixel == (360, 240)
    assert slots[0].grasp_stable_frames == 1


def test_mask_grasp_geometry_prefers_bright_top_face_over_full_silhouette_center():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:80, 20:80] = 1.0
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:80, 20:60] = (0, 140, 255)
    frame[20:80, 60:80] = (0, 70, 120)

    geometry = mask_to_grasp_geometry(mask, (100, 100), frame_bgr=frame)

    assert geometry is not None
    assert geometry.grasp_pixel[0] < 50
    assert 43 <= geometry.grasp_pixel[0] <= 48


def test_mask_grasp_geometry_reports_color_independent_rect_angle():
    mask = np.zeros((160, 160), dtype=np.float32)
    rect = ((80.0, 80.0), (80.0, 30.0), 30.0)
    points = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [points], 1.0)
    frame = np.full((160, 160, 3), 230, dtype=np.uint8)
    cv2.fillPoly(frame, [points], (180, 40, 210))

    geometry = mask_to_grasp_geometry(mask, (160, 160), frame_bgr=frame)

    assert geometry is not None
    assert geometry.grasp_angle_deg is not None
    assert 25.0 <= geometry.grasp_angle_deg <= 35.0
    assert geometry.grasp_angle_quality > 0.5


def test_frame_block_fallback_detects_colored_block_without_color_name():
    frame = np.full((240, 320, 3), 235, dtype=np.uint8)
    cv2.line(frame, (20, 20), (300, 40), (20, 20, 20), 4)
    for x in range(170, 250, 18):
        cv2.line(frame, (x, 150), (x + 10, 150), (160, 160, 160), 3)
    frame[80:150, 70:145] = (210, 40, 30)

    candidates = frame_to_block_candidates(
        frame,
        roi_center=(160, 120),
        roi_radius=220,
        max_det=4,
    )

    assert candidates
    assert candidates[0].bbox[0] <= 75
    assert candidates[0].bbox[2] >= 140
    assert candidates[0].grasp_quality > 0.6


def test_profile_mapping_marks_far_eye_in_hand_target_for_servo():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    result = _Result(xyxy=[[690, 350, 730, 390]], conf=[0.9])
    candidates, _ = extract_candidates(
        result,
        frame_shape=(480, 640),
        roi_center=(640, 360),
        roi_radius=400,
        max_det=1,
        confidence_threshold=0.25,
    )
    update_slots(slots, candidates, match_distance=120.0, lost_ttl=6)
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -640], [0, 1, -360]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"center_tolerance_px": 8.0, "gain": 0.8, "max_attempts": 3},
        }
    )

    annotate_slots_with_cylindrical(
        slots,
        calibration=None,
        calibration_profile=profile,
        frame_size=(640, 480),
        roi_center=(640, 360),
        mapping_mode="delta_servo",
        calibration_profile_required=True,
        action_error_threshold_mm=6.0,
    )

    packet_slot = slots[0].to_packet()
    assert packet_slot["calibration_profile_id"] == "unit-profile"
    assert packet_slot["camera_to_world_raw"] == [70.0, 10.0, 0.0]
    assert packet_slot["estimated_xy_error_mm"] == 2.0
    assert packet_slot["servo_required"] is True


def test_profile_mapping_subtracts_alignment_target_pixel():
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
        }
    )

    mapped = profile.map_pixel_to_delta((350.0, 250.0), frame_size=(640, 480), target_pixel=(330.0, 245.0))

    assert mapped.delta_xy_mm == (20.0, 5.0)
    assert mapped.estimated_error_mm == 2.0


def test_annotation_uses_alignment_target_for_servo_distance_and_delta():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (350, 245)
    slot.grasp_pixel = (350, 245)
    slot.grasp_quality = 1.0
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"center_tolerance_px": 8.0},
        }
    )

    annotate_slots_with_cylindrical(
        slots,
        calibration=None,
        calibration_profile=profile,
        frame_size=(640, 480),
        roi_center=(320, 240),
        alignment_target_pixel=(344.0, 245.0),
        mapping_mode="delta_servo",
        calibration_profile_required=True,
        action_error_threshold_mm=6.0,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["camera_to_world_raw"] == [6.0, 0.0, 0.0]
    assert packet_slot["alignment_target_pixel"] == [344.0, 245.0]
    assert packet_slot["servo_required"] is False


def test_delta_servo_near_center_waits_for_stable_grasp_frames():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (326, 244)
    slot.grasp_pixel = (326, 244)
    slot.grasp_quality = 1.0
    slot.grasp_stable_frames = 1
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"center_tolerance_px": 8.0},
        }
    )

    annotate_slots_with_cylindrical(
        slots,
        calibration=None,
        calibration_profile=profile,
        frame_size=(640, 480),
        roi_center=(320, 240),
        alignment_target_pixel=(320.0, 240.0),
        mapping_mode="delta_servo",
        calibration_profile_required=True,
        action_error_threshold_mm=6.0,
        action_center_tolerance_px=14.0,
        required_stable_frames=3,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] < 8.0
    assert packet_slot["servo_required"] is False
    assert packet_slot["invalid_reason"] == "grasp_unstable"

    slot.grasp_stable_frames = 3
    annotate_slots_with_cylindrical(
        slots,
        calibration=None,
        calibration_profile=profile,
        frame_size=(640, 480),
        roi_center=(320, 240),
        alignment_target_pixel=(320.0, 240.0),
        mapping_mode="delta_servo",
        calibration_profile_required=True,
        action_error_threshold_mm=6.0,
        action_center_tolerance_px=14.0,
        required_stable_frames=3,
    )

    assert slot.invalid_reason == "awaiting_robot_snapshot_delta_resolve"

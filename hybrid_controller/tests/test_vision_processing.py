from __future__ import annotations

import numpy as np
import cv2
import pytest

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


def test_extract_candidates_filters_to_roi_and_uses_unified_score():
    result = _Result(
        xyxy=[
            [95, 95, 145, 145],
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
    assert [candidate.center for candidate in candidates] == [(120, 120), (130, 130)]


def test_extract_candidates_prefers_large_graspable_yolo_candidate_over_center_artifact():
    result = _Result(
        xyxy=[
            [315, 235, 325, 245],
            [245, 180, 335, 270],
        ],
        conf=[0.99, 0.78],
    )
    candidates, detected_count = extract_candidates(
        result,
        frame_shape=(480, 640),
        roi_center=(320, 240),
        roi_radius=160,
        max_det=4,
        confidence_threshold=0.25,
    )

    assert detected_count == 2
    assert candidates[0].center == (290, 225)
    assert candidates[0].area_px > candidates[1].area_px


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


def test_grasp_angle_history_uses_stable_median_angle():
    slots = [SlotState(slot=1, freq_hz=8.0)]

    def candidate(angle: float) -> DetectionCandidate:
        return DetectionCandidate(
            center=(100, 100),
            grasp_pixel=(100, 100),
            bbox=(80, 90, 120, 110),
            area_px=800,
            confidence=0.9,
            polygon=[(80, 90), (120, 90), (120, 110), (80, 110)],
            grasp_quality=1.0,
            oriented_bbox=[(80, 90), (120, 90), (120, 110), (80, 110)],
            distance_to_roi=0.0,
            grasp_angle_deg=angle,
            grasp_angle_quality=0.9,
        )

    for angle in (10.0, 12.0, 11.0):
        update_slots(
            slots,
            [candidate(angle)],
            match_distance=120.0,
            lost_ttl=6,
            grasp_history_len=5,
            grasp_angle_stability_tolerance_deg=8.0,
        )

    assert slots[0].grasp_angle_deg == 11.0
    assert slots[0].grasp_angle_stability_deg is not None
    assert slots[0].grasp_angle_stability_deg <= 2.0


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


def test_frame_block_fallback_prefers_full_block_over_small_colored_artifact():
    frame = np.full((240, 320, 3), 235, dtype=np.uint8)
    frame[98:133, 148:174] = (210, 40, 30)
    frame[152:224, 86:166] = (40, 170, 70)

    candidates = frame_to_block_candidates(
        frame,
        roi_center=(160, 120),
        roi_radius=220,
        max_det=4,
        min_area_px=500,
    )

    assert len(candidates) >= 2
    assert candidates[0].bbox[0] <= 90
    assert candidates[0].bbox[2] >= 160
    assert candidates[0].area_px > candidates[1].area_px


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


def test_profile_mapping_applies_residual_grid_correction():
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "grid-profile",
            "image_size": [100, 100],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "residual_grid": {
                "model": "grid",
                "x_values": [0, 10],
                "y_values": [0, 10],
                "correction_dx_mm": [[2, 2], [2, 2]],
                "correction_dy_mm": [[-1, -1], [-1, -1]],
                "error_mm": [[1, 2], [3, 5]],
            },
            "valid_workspace": {"undistorted_pixel_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        }
    )

    mapped = profile.map_pixel_to_delta((5.0, 5.0), frame_size=(100, 100))

    assert mapped.delta_xy_mm == (7.0, 4.0)
    assert mapped.estimated_error_mm == 2.75


def test_profile_mapping_selects_stage_model_by_name():
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "stage-profile",
            "image_size": [100, 100],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "stage_models": {
                "search": {
                    "z_mm": 190.0,
                    "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
                },
                "confirm": {
                    "z_mm": 130.0,
                    "pixel_to_delta": {"model": "affine", "matrix": [[2, 0, 0], [0, 2, 0]]},
                },
            },
        }
    )

    mapped = profile.map_pixel_to_delta((5.0, 4.0), frame_size=(100, 100), stage="confirm")

    assert mapped.delta_xy_mm == (10.0, 8.0)
    assert profile.model_for_stage("confirm").z_mm == 130.0


def test_stage_target_pixel_overrides_parent_alignment_target():
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "stage-target-profile",
            "image_size": [100, 100],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "servo": {"target_pixel": [50.0, 50.0]},
            "stage_models": {
                "confirm": {
                    "z_mm": 130.0,
                    "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
                    "servo": {"target_pixel": [50.0, 44.0]},
                }
            },
        }
    )
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (50, 44)
    slot.center_stable_frames = 3
    slot.grasp_pixel = (50, 44)
    slot.grasp_stable_frames = 3
    slot.grasp_quality = 1.0

    annotate_slots_with_cylindrical(
        slots,
        calibration=None,
        calibration_profile=profile,
        frame_size=(100, 100),
        roi_center=(50, 50),
        mapping_mode="delta_servo",
        calibration_profile_required=True,
        alignment_target_pixel=None,
        alignment_target_required=True,
        calibration_stage="confirm",
        calibration_z_mm=130.0,
        center_tolerance_px=3.0,
        action_center_tolerance_px=3.0,
    )

    assert slot.alignment_target_pixel == (50.0, 44.0)
    assert slot.center_distance_px == 0.0
    assert slot.camera_to_world_raw == (0.0, 0.0, 0.0)


def test_stage_target_only_model_inherits_parent_pixel_to_delta():
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "stage-target-only-profile",
            "image_size": [100, 100],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "servo": {"target_pixel": [50.0, 50.0]},
            "stage_models": {
                "confirm": {
                    "z_mm": 130.0,
                    "servo": {"target_pixel": [50.0, 44.0]},
                }
            },
        }
    )

    stage_profile = profile.model_for_stage("confirm")
    mapped = profile.map_pixel_to_delta((50.0, 44.0), frame_size=(100, 100), stage="confirm")

    assert stage_profile.has_pixel_to_delta_model is True
    assert stage_profile.target_pixel == (50.0, 44.0)
    assert mapped.delta_xy_mm == (0.0, 0.0)


def test_profile_mapping_does_not_reuse_wrong_stage_when_named_stage_is_missing():
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "stage-profile",
            "image_size": [100, 100],
            "stage_models": {
                "search": {
                    "z_mm": 190.0,
                    "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
                }
            },
        }
    )
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (50, 50)
    slot.grasp_pixel = (50, 50)
    slot.grasp_quality = 1.0

    annotate_slots_with_cylindrical(
        slots,
        calibration=None,
        calibration_profile=profile,
        frame_size=(100, 100),
        roi_center=(50, 50),
        mapping_mode="delta_servo",
        calibration_profile_required=True,
        calibration_stage="confirm",
        calibration_z_mm=130.0,
    )

    assert slot.actionable is False
    assert slot.invalid_reason == "calibration_unavailable"


def test_mask_grasp_geometry_prefers_inner_safe_point_over_silhouette_center():
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[25:85, 25:85] = 255
    mask[25:45, 85:110] = 255

    geometry = mask_to_grasp_geometry(mask, (120, 120))

    assert geometry is not None
    assert geometry.center == (60, 52)
    assert geometry.grasp_pixel[0] < geometry.center[0]
    assert geometry.grasp_quality > 0.5


def test_profile_mapping_rejects_points_outside_valid_workspace():
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "workspace-profile",
            "image_size": [100, 100],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "valid_workspace": {"undistorted_pixel_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        }
    )

    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (50, 50)
    slot.grasp_pixel = (50, 50)
    slot.grasp_quality = 1.0

    annotate_slots_with_cylindrical(
        slots,
        calibration=None,
        calibration_profile=profile,
        frame_size=(100, 100),
        roi_center=(5, 5),
        mapping_mode="delta_servo",
        calibration_profile_required=True,
    )

    assert slot.actionable is False
    assert "calibration_profile_point_outside_valid_workspace" in slot.invalid_reason


def test_target_pixel_mode_requires_alignment_target():
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "no-target-profile",
            "image_size": [100, 100],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
        }
    )
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (50, 50)
    slot.grasp_pixel = (50, 50)
    slot.grasp_quality = 1.0

    annotate_slots_with_cylindrical(
        slots,
        calibration=None,
        calibration_profile=profile,
        frame_size=(100, 100),
        roi_center=(5, 5),
        mapping_mode="delta_servo",
        calibration_profile_required=True,
        alignment_target_required=True,
    )

    assert slot.invalid_reason == "alignment_target_unavailable"


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
    slot.center_stable_frames = 3
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


def test_delta_servo_low_confirm_centered_does_not_wait_for_grasp_pixel_stability():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (314, 240)
    slot.grasp_pixel = (304, 238)
    slot.grasp_quality = 1.0
    slot.center_stable_frames = 5
    slot.grasp_stable_frames = 1
    slot.area_stability_ratio = 0.20
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        calibration_stage="confirm",
        calibration_z_mm=140.0,
        action_error_threshold_mm=20.0,
        center_tolerance_px=6.0,
        action_center_tolerance_px=6.0,
        required_stable_frames=3,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] == 6.0
    assert packet_slot["servo_required"] is False
    assert packet_slot["invalid_reason"] == "awaiting_robot_snapshot_delta_resolve"


def test_delta_servo_near_center_waits_for_stable_center_frames():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (321, 239)
    slot.grasp_pixel = (321, 239)
    slot.grasp_quality = 1.0
    slot.center_stable_frames = 1
    slot.grasp_stable_frames = 3
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
        action_center_tolerance_px=8.0,
        required_stable_frames=3,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] < 8.0
    assert packet_slot["servo_required"] is False
    assert packet_slot["invalid_reason"] == "grasp_unstable"


def test_delta_servo_centering_uses_block_center_not_smoothed_grasp_pixel():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (330, 240)
    slot.grasp_pixel = (320, 240)
    slot.grasp_quality = 1.0
    slot.grasp_stable_frames = 3
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        action_center_tolerance_px=8.0,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] == 10.0
    assert packet_slot["servo_required"] is True
    assert packet_slot["camera_to_world_raw"] == [10.0, 0.0, 0.0]


def test_delta_servo_confirm_height_waits_before_unstable_off_center_move():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (328, 242)
    slot.grasp_pixel = (328, 242)
    slot.grasp_quality = 1.0
    slot.center_stable_frames = 1
    slot.grasp_stable_frames = 1
    slot.area_stability_ratio = 0.20
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        calibration_stage="confirm",
        calibration_z_mm=140.0,
        action_error_threshold_mm=20.0,
        center_tolerance_px=6.0,
        action_center_tolerance_px=6.0,
        required_stable_frames=3,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] > 6.0
    assert packet_slot["servo_required"] is False
    assert packet_slot["invalid_reason"] == "grasp_unstable"


def test_delta_servo_confirm_height_allows_stable_off_center_move():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (328, 242)
    slot.grasp_pixel = (328, 242)
    slot.grasp_quality = 1.0
    slot.center_stable_frames = 3
    slot.grasp_stable_frames = 3
    slot.area_stability_ratio = 0.02
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        calibration_stage="confirm",
        calibration_z_mm=140.0,
        action_error_threshold_mm=20.0,
        center_tolerance_px=6.0,
        action_center_tolerance_px=6.0,
        required_stable_frames=3,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] > 6.0
    assert packet_slot["servo_required"] is True
    assert packet_slot["invalid_reason"] == "awaiting_robot_snapshot_delta_resolve"


def test_delta_servo_confirm_height_allows_centered_large_area_change():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (321, 239)
    slot.grasp_pixel = (321, 239)
    slot.grasp_quality = 1.0
    slot.center_stable_frames = 3
    slot.grasp_stable_frames = 3
    slot.area_stability_ratio = 0.20
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        calibration_stage="confirm",
        calibration_z_mm=140.0,
        action_error_threshold_mm=20.0,
        center_tolerance_px=6.0,
        action_center_tolerance_px=6.0,
        required_stable_frames=3,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] < 6.0
    assert packet_slot["servo_required"] is False
    assert packet_slot["invalid_reason"] == "awaiting_robot_snapshot_delta_resolve"


def test_profile_center_tolerance_can_be_tightened_by_config():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (327, 240)
    slot.grasp_pixel = (327, 240)
    slot.grasp_quality = 1.0
    slot.grasp_stable_frames = 3
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        center_tolerance_px=6.0,
        action_center_tolerance_px=6.0,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] == 7.0
    assert packet_slot["center_tolerance_px"] == 6.0
    assert packet_slot["servo_required"] is True


def test_search_stage_can_use_wider_action_tolerance_without_relaxing_center_tolerance():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (335, 240)
    slot.grasp_pixel = (335, 240)
    slot.grasp_quality = 1.0
    slot.center_stable_frames = 3
    slot.grasp_stable_frames = 3
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        action_error_threshold_mm=20.0,
        center_tolerance_px=6.0,
        action_center_tolerance_px=16.0,
        required_stable_frames=3,
        calibration_stage="search",
        calibration_z_mm=190.0,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] == 15.0
    assert packet_slot["center_tolerance_px"] == 6.0
    assert packet_slot["action_tolerance_px"] == 16.0
    assert packet_slot["servo_required"] is False
    assert packet_slot["invalid_reason"] == "awaiting_robot_snapshot_delta_resolve"


def test_confirm_stage_keeps_strict_action_tolerance():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (335, 240)
    slot.grasp_pixel = (335, 240)
    slot.grasp_quality = 1.0
    slot.center_stable_frames = 3
    slot.grasp_stable_frames = 3
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        action_error_threshold_mm=20.0,
        center_tolerance_px=6.0,
        action_center_tolerance_px=6.0,
        required_stable_frames=3,
        calibration_stage="confirm",
        calibration_z_mm=185.0,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] == 15.0
    assert packet_slot["action_tolerance_px"] == 6.0
    assert packet_slot["servo_required"] is True


def test_confirm_stage_can_use_low_action_tolerance_for_near_center_noise():
    slots = [SlotState(slot=1, freq_hz=8.0)]
    slot = slots[0]
    slot.valid = True
    slot.observed = True
    slot.pixel_center = (314, 241)
    slot.grasp_pixel = (316, 232)
    slot.grasp_quality = 1.0
    slot.center_stable_frames = 1
    slot.grasp_stable_frames = 1
    slot.area_stability_ratio = 0.20
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "unit-profile",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            "residual": {"p95_error_mm": 2.0},
            "servo": {"target_pixel": [320, 240], "center_tolerance_px": 8.0},
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
        calibration_stage="confirm",
        calibration_z_mm=140.0,
        action_error_threshold_mm=20.0,
        center_tolerance_px=6.0,
        action_center_tolerance_px=8.0,
        required_stable_frames=3,
    )

    packet_slot = slot.to_packet()
    assert packet_slot["center_distance_px"] == pytest.approx(6.082762530298219)
    assert packet_slot["action_tolerance_px"] == 8.0
    assert packet_slot["servo_required"] is False
    assert packet_slot["invalid_reason"] == "awaiting_robot_snapshot_delta_resolve"

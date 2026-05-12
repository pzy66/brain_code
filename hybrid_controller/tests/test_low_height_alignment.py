from __future__ import annotations

import pytest

from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile
from hybrid_controller.vision.low_height_alignment import fit_low_height_response_model
from hybrid_controller.vision.low_height_alignment import merge_confirm_stage_model


def test_low_height_response_model_maps_pixel_error_to_correct_robot_delta() -> None:
    samples = []
    for x_mm, y_mm in (
        (-2.0, -2.0),
        (-2.0, 2.0),
        (0.0, 0.0),
        (2.0, -2.0),
        (2.0, 2.0),
    ):
        samples.append(
            {
                "pose_cyl": [0.0, 150.0, 120.0],
                "pose_xy": [x_mm, y_mm],
                "pixel": [320.0 + 10.0 * x_mm, 240.0 - 5.0 * y_mm],
            }
        )

    model = fit_low_height_response_model(samples, target_pixel=(320.0, 240.0), z_mm=120.0)
    stage_payload = model.to_stage_model_payload(profile_id="unit-confirm")
    profile = VisionCalibrationProfile.from_dict(
        {
            "profile_id": "parent",
            "image_size": [640, 480],
            "pixel_to_delta": {"model": "affine", "matrix": [[99, 0, 0], [0, 99, 0]]},
            "stage_models": {"confirm": stage_payload},
        }
    )

    mapped = profile.map_pixel_to_delta(
        (340.0, 230.0),
        frame_size=(640, 480),
        target_pixel=(320.0, 240.0),
        stage="confirm",
        z_mm=120.0,
    )

    assert mapped.delta_xy_mm == pytest.approx((-2.0, -2.0))
    assert model.rms_pixel_error_px < 1e-9
    assert model.used_sample_count == 5


def test_low_height_response_model_rejects_degenerate_motion() -> None:
    samples = [
        {
            "pose_cyl": [0.0, 150.0, 120.0],
            "pose_xy": [float(index), 0.0],
            "pixel": [320.0 + float(index), 240.0],
        }
        for index in range(5)
    ]

    with pytest.raises(ValueError, match="ill_conditioned"):
        fit_low_height_response_model(samples, target_pixel=(320.0, 240.0), z_mm=120.0)


def test_low_height_sample_prefers_subpixel_geometry_center() -> None:
    from hybrid_controller.vision.low_height_alignment import sample_from_mapping

    sample = sample_from_mapping(
        {
            "pose_cyl": [0.0, 150.0, 120.0],
            "pose_xy": [1.0, 2.0],
            "pixel_center": [316, 235],
            "geometry_center": [316, 235],
            "geometry_center_f": [316.47, 235.28],
        }
    )

    assert sample.pixel_x == pytest.approx(316.47)
    assert sample.pixel_y == pytest.approx(235.28)


def test_merge_confirm_stage_model_preserves_parent_payload() -> None:
    merged = merge_confirm_stage_model(
        {
            "profile_id": "parent",
            "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, 0], [0, 1, 0]]},
            "stage_models": {"search": {"z_mm": 190.0}},
        },
        {"profile_id": "confirm-local", "stage": "confirm", "z_mm": 120.0},
    )

    assert merged["profile_id"] == "parent"
    assert merged["stage_models"]["search"]["z_mm"] == 190.0
    assert merged["stage_models"]["confirm"]["profile_id"] == "confirm-local"


def test_low_height_response_model_exposes_large_target_delta_for_guard() -> None:
    samples = [
        {
            "pose_cyl": [0.0, 150.0, 120.0],
            "pose_xy": [0.0, 0.0],
            "pixel": [315.0, 235.0],
        },
        {
            "pose_cyl": [0.0, 150.0, 120.0],
            "pose_xy": [4.0, 0.0],
            "pixel": [316.0, 235.0],
        },
        {
            "pose_cyl": [0.0, 150.0, 120.0],
            "pose_xy": [0.0, 4.0],
            "pixel": [315.0, 236.0],
        },
        {
            "pose_cyl": [0.0, 150.0, 120.0],
            "pose_xy": [4.0, 4.0],
            "pixel": [316.0, 236.0],
        },
    ]

    model = fit_low_height_response_model(samples, target_pixel=(320.0, 240.0), z_mm=120.0, min_samples=4)

    assert model.target_robot_xy_mm == pytest.approx((20.0, 20.0))

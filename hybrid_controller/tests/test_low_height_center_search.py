from __future__ import annotations

import pytest

from hybrid_controller.tools.search_low_height_center import _candidate_offsets
from hybrid_controller.tools.search_low_height_center import _filter_candidate_offsets
from hybrid_controller.tools.search_low_height_center import _measurement_summary
from hybrid_controller.tools.search_low_height_center import _pose_close
from hybrid_controller.tools.search_low_height_center import _point_spread_px


def test_low_height_center_search_candidate_offsets_are_axis_first() -> None:
    assert _candidate_offsets(0.5, 2.0, include_diagonal=False) == [
        (0.0, 0.0),
        (-0.5, 0.0),
        (0.5, 0.0),
        (0.0, -2.0),
        (0.0, 2.0),
    ]


def test_low_height_center_search_can_restrict_backlash_direction() -> None:
    offsets = _candidate_offsets(0.5, 2.0, include_diagonal=True)

    assert _filter_candidate_offsets(offsets, theta_direction="none", radius_direction="positive") == [
        (0.0, 0.0),
        (0.0, 2.0),
    ]
    assert _filter_candidate_offsets(offsets, theta_direction="negative", radius_direction="none") == [
        (0.0, 0.0),
        (-0.5, 0.0),
    ]


def test_low_height_center_search_point_spread_uses_pairwise_max() -> None:
    assert _point_spread_px([(0.0, 0.0), (3.0, 4.0), (1.0, 1.0)]) == pytest.approx(5.0)


def test_low_height_center_search_rejects_unstable_repeats() -> None:
    summary = _measurement_summary(
        [
            {
                "center_distance_px": 4.0,
                "pixel": [320.0, 240.0],
                "measurement_point": "geometry_subpixel",
                "alignment_target_pixel": [320.0, 240.0],
                "point_distances_px": {"geometry_center_f": 4.0, "grasp_pixel_f": 7.0},
                "captured_frames": 10,
                "processed_frames": 3,
                "camera_transport": {"content_length_payloads": 10, "frames_rejected": 0},
            },
            {
                "center_distance_px": 6.0,
                "pixel": [330.0, 240.0],
                "measurement_point": "geometry_subpixel",
                "alignment_target_pixel": [320.0, 240.0],
                "point_distances_px": {"geometry_center_f": 6.0, "grasp_pixel_f": 9.0},
                "captured_frames": 10,
                "processed_frames": 3,
                "camera_transport": {"content_length_payloads": 20, "frames_rejected": 1},
            },
        ],
        max_repeat_spread_px=4.0,
    )

    assert summary["median_center_distance_px"] == pytest.approx(5.0)
    assert summary["repeat_spread_px"] == pytest.approx(10.0)
    assert summary["stable"] is False
    assert summary["sample_count"] == 2
    assert summary["captured_frames"] == 20
    assert summary["processed_frames"] == 6
    assert summary["camera_transport_last"]["frames_rejected"] == 1
    assert summary["measurement_point"] == "geometry_subpixel"
    assert summary["alignment_target_pixel"] == [320.0, 240.0]
    assert summary["median_point_distances_px"]["geometry_center_f"] == pytest.approx(5.0)
    assert summary["median_point_distances_px"]["grasp_pixel_f"] == pytest.approx(8.0)


def test_low_height_center_search_rejects_shape_jumps_even_when_center_repeat_is_small() -> None:
    summary = _measurement_summary(
        [
            {
                "center_distance_px": 3.0,
                "pixel": [321.0, 241.0],
                "area_px": 34000,
                "bbox": [210, 125, 434, 368],
            },
            {
                "center_distance_px": 3.1,
                "pixel": [321.2, 241.1],
                "area_px": 35400,
                "bbox": [210, 125, 434, 384],
            },
        ],
        max_repeat_spread_px=4.0,
    )

    assert summary["repeat_spread_px"] < 1.0
    assert summary["bottom_edge_span_px"] == pytest.approx(16.0)
    assert summary["shape_stable"] is False
    assert summary["stable"] is False


def test_low_height_center_search_accepts_small_shape_noise() -> None:
    summary = _measurement_summary(
        [
            {
                "center_distance_px": 2.5,
                "pixel": [320.5, 240.4],
                "area_px": 35000,
                "bbox": [210, 125, 434, 375],
            },
            {
                "center_distance_px": 2.7,
                "pixel": [320.7, 240.2],
                "area_px": 35100,
                "bbox": [210, 125, 434, 376],
            },
        ],
        max_repeat_spread_px=4.0,
    )

    assert summary["shape_stable"] is True
    assert summary["stable"] is True


def test_low_height_center_search_requires_pose_close_after_move() -> None:
    assert _pose_close((8.0, 170.0, 120.0), (8.03, 170.2, 120.1), theta_tol_deg=0.08, radius_tol_mm=0.35, z_tol_mm=0.35)
    assert not _pose_close((8.0, 170.0, 140.0), (8.0, 170.0, 120.0), theta_tol_deg=0.08, radius_tol_mm=0.35, z_tol_mm=0.35)


def test_low_height_center_search_move_limit_ignores_initial_height_delta() -> None:
    from hybrid_controller.tools.search_low_height_center import _cartesian_distance_between_cyl

    start_after_descent = (8.77, 176.34, 120.0)
    same_low_pose = (8.77, 176.34, 120.0)
    stale_high_start = (8.77, 176.34, 140.0)

    assert _cartesian_distance_between_cyl(start_after_descent, same_low_pose) == pytest.approx(0.0)
    assert _cartesian_distance_between_cyl(stale_high_start, same_low_pose) > 18.0

from __future__ import annotations

import pytest

from hybrid_controller.config import AppConfig
from hybrid_controller.tools.search_low_height_center import _candidate_offsets
from hybrid_controller.tools.search_low_height_center import _filter_candidate_offsets
from hybrid_controller.tools.search_low_height_center import _measurement_summary
from hybrid_controller.tools.search_low_height_center import _measure_repeated
from hybrid_controller.tools.search_low_height_center import _apply_report_alignment_from_summary
from hybrid_controller.tools.search_low_height_center import _pose_close
from hybrid_controller.tools.search_low_height_center import _point_spread_px
from hybrid_controller.vision.calibration_profile import VisionCalibrationProfile


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


def test_low_height_center_search_reports_shape_jumps_without_blocking_center_stability() -> None:
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
    assert summary["center_stable"] is True
    assert summary["shape_stable"] is False
    assert summary["shape_warning"] == "shape_changed_between_repeats"
    assert summary["stable"] is True


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
    assert summary["center_stable"] is True
    assert summary["stable"] is True


def test_low_height_center_search_report_alignment_uses_sample_provenance() -> None:
    report: dict[str, object] = {
        "alignment_target_pixel": None,
        "alignment_target_source": "pending_first_sample",
    }

    _apply_report_alignment_from_summary(report, {"alignment_target_pixel": [319.5, 241.25]})

    assert report["alignment_target_pixel"] == [319.5, 241.25]
    assert report["alignment_target_source"] == "sample_provenance"


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


def test_low_height_center_search_measure_retries_transient_lost_slot(monkeypatch, tmp_path) -> None:
    calls = {"count": 0}

    class _Reader:
        def __init__(self) -> None:
            self.reopen_count = 0

        def reopen(self) -> None:
            self.reopen_count += 1

    class _Cv2:
        def imwrite(self, *_args, **_kwargs) -> bool:
            return True

    def fake_measure_slot(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Slot 1 not detected in low-height calibration frame.")
        sample = {
            "center_distance_px": 3.0,
            "pixel": [321.0, 241.0],
            "measurement_point": "geometry_subpixel",
            "alignment_target_pixel": [320.0, 240.0],
            "point_distances_px": {"geometry_center_f": 3.0},
            "captured_frames": 3,
            "processed_frames": 1,
            "camera_transport": {"frames_rejected": 0},
        }
        packet = {"frame_id": calls["count"]}
        return sample, packet, object(), calls["count"], None

    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center._measure_slot", fake_measure_slot)
    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center._save_overlay", lambda **_kwargs: None)

    summary, frame_id, _debug_slots = _measure_repeated(
        reader=_Reader(),
        cv2_module=_Cv2(),
        model=None,
        config=None,
        calibration_profile=None,
        client=None,
        device=None,
        half=False,
        slot_id=1,
        repeats=1,
        frame_id=0,
        frames=3,
        drain_frames=0,
        settle_sec=0.0,
        timeout_sec=1.0,
        ros_timeout_sec=1.0,
        max_repeat_spread_px=4.0,
        output_dir=tmp_path,
        label="retry",
        debug_slots=None,
        fresh_reopen_before_measure=True,
        max_measure_attempts=3,
    )

    assert calls["count"] == 2
    assert frame_id == 2
    assert summary["sample_count"] == 1
    assert summary["median_center_distance_px"] == pytest.approx(3.0)
    assert summary["measurement_errors"] == ["Slot 1 not detected in low-height calibration frame."]


def test_low_height_center_search_cli_overrides_measurement_point_after_profile(monkeypatch, tmp_path) -> None:
    from hybrid_controller.tools.search_low_height_center import main

    captured: dict[str, object] = {}

    class FakeClient:
        def connect(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeReader:
        def __init__(self, **_kwargs) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_measure_repeated(**kwargs):
        captured["config"] = kwargs["config"]
        return (
            {
                "median_center_distance_px": 0.8,
                "repeat_spread_px": 0.2,
                "stable": True,
                "measurement_point": "top_face_subpixel",
            },
            1,
            None,
        )

    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center.RosBridgeClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center._PersistentCaptureReader", FakeReader)
    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center._freeze_sucker", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "hybrid_controller.tools.search_low_height_center._wait_for_idle",
        lambda *_a, **_k: {
            "state": "IDLE",
            "busy": False,
            "robot_cyl": {"theta_deg": 7.0, "radius_mm": 170.0, "z_mm": 120.0},
        },
    )
    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center._current_cyl_pose", lambda _snapshot: (7.0, 170.0, 120.0))
    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center._load_model", lambda *_a, **_k: None)
    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center._resolve_device", lambda _device: ("cpu", False))
    monkeypatch.setattr(
        "hybrid_controller.tools.search_low_height_center.load_vision_grasp_profile",
        lambda _config: type("Profile", (), {"ready": False, "error": ""})(),
    )
    monkeypatch.setattr(
        "hybrid_controller.tools.search_low_height_center.VisionCalibrationProfile.load",
        lambda _path: VisionCalibrationProfile.from_dict(
            {
                "profile_id": "unit-profile",
                "image_size": [640, 480],
                "pixel_to_delta": {"model": "affine", "matrix": [[1, 0, -320], [0, 1, -240]]},
            }
        ),
    )
    monkeypatch.setattr("hybrid_controller.tools.search_low_height_center._measure_repeated", fake_measure_repeated)

    exit_code = main(
        [
            "--slot-id",
            "1",
            "--z-mm",
            "120",
            "--dry-run",
            "--low-height-measurement-point",
            "top_face_subpixel",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert isinstance(captured["config"], AppConfig)
    assert captured["config"].vision_servo_low_height_measurement_point == "top_face_subpixel"

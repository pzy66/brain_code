from __future__ import annotations

import argparse
import types
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_controller.config import AppConfig


def _build_slots(*, center_distance_mm: float) -> list[dict[str, object]]:
    """Build four demo detection slots with deterministic geometry and stable quality fields."""
    base_distance = float(center_distance_mm)
    slot_defs = [
        (1, -110.0, -150.0, 12.0),
        (2, -60.0, -142.0, 28.0),
        (3, -5.0, -136.0, 34.0),
        (4, 45.0, -128.0, 40.0),
    ]
    slots: list[dict[str, object]] = []
    for slot_id, world_x, world_y, px_bias in slot_defs:
        if slot_id == 1:
            center_distance = base_distance
        elif slot_id == 2:
            center_distance = base_distance + 40.0
        elif slot_id == 3:
            center_distance = base_distance + 55.0
        else:
            center_distance = base_distance + 70.0
        pixel_x = 320.0 + px_bias
        pixel_y = 240.0
        command = f"PICK_CYL {world_x:.2f} {world_y:.2f}"
        slots.append(
            {
                "slot_id": slot_id,
                "valid": True,
                "actionable": True,
                "camera_to_world_raw": [world_x, world_y, 0.0],
                "world_xyz": [world_x, world_y, 0.0],
                "grasp_quality": 1.0,
                "confidence": 0.99,
                "command": command,
                "area_px": 3600.0,
                "bbox": [pixel_x - 28.0, pixel_y - 22.0, pixel_x + 28.0, pixel_y + 22.0],
                "pixel_center": [pixel_x, pixel_y],
                "pixel_center_f": [pixel_x, pixel_y],
                "geometry_center": [pixel_x, pixel_y],
                "geometry_center_f": [pixel_x, pixel_y],
        "servo_required": False,
                "center_distance_px": max(0.2, float(center_distance)),
                "measurement_point": "geometry_subpixel",
            }
        )
    return slots


def _install_offline_qt_stubs() -> None:
    import types
    import importlib.machinery

    class _Signal:
        def __init__(self) -> None:
            self._callbacks: list[object] = []

        def connect(self, callback: object) -> None:
            if callback is not None:
                self._callbacks.append(callback)

        def emit(self, *args: object, **kwargs: object) -> None:
            for callback in list(self._callbacks):
                if callable(callback):
                    callback(*args, **kwargs)

    class _QObject:
        pass

    class _QApplication:
        _instance = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            _QApplication._instance = self

        @classmethod
        def instance(cls) -> object | None:
            return cls._instance

        @staticmethod
        def processEvents() -> None:
            return None

        @staticmethod
        def quit() -> None:
            return None

    class _QTimer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.timeout = _Signal()
            self._is_active = False

        def setSingleShot(self, *_args: object, **_kwargs: object) -> None:
            return None

        def setTimerType(self, *_args: object, **_kwargs: object) -> None:
            return None

        def start(self, *_args: object, **_kwargs: object) -> None:
            self._is_active = True
            return None

        def stop(self) -> None:
            self._is_active = False
            return None

        def isActive(self) -> bool:
            return bool(self._is_active)

        def deleteLater(self) -> None:
            self.stop()
            return None

        @staticmethod
        def singleShot(*_args: object, **_kwargs: object) -> None:
            if _args and callable(_args[-1]):
                _args[-1]()

    class _Qt:
        PreciseTimer = 0

    class _QFileDialog:
        @staticmethod
        def getOpenFileName(*_args: object, **_kwargs: object) -> tuple[str, str]:
            return ("", "")

    qt_pkg = types.ModuleType("PyQt5")
    qt_core = types.ModuleType("PyQt5.QtCore")
    qt_core.QObject = _QObject
    qt_core.QTimer = _QTimer
    qt_core.Qt = _Qt
    qt_core.pyqtSignal = lambda *args, **kwargs: _Signal()
    qt_gui = types.ModuleType("PyQt5.QtGui")
    qt_widgets = types.ModuleType("PyQt5.QtWidgets")
    qt_widgets.QApplication = _QApplication
    qt_widgets.QFileDialog = _QFileDialog

    sys.modules["PyQt5"] = qt_pkg
    sys.modules["PyQt5.QtCore"] = qt_core
    sys.modules["PyQt5.QtGui"] = qt_gui
    sys.modules["PyQt5.QtWidgets"] = qt_widgets

    fake_window_pkg = types.ModuleType("hybrid_controller.ui.main_window")

    class MainWindow:
        key_pressed = _Signal()
        key_released = _Signal()
        robot_start_requested = _Signal()
        robot_connect_requested = _Signal()
        abort_requested = _Signal()
        reset_requested = _Signal()
        sucker_off_requested = _Signal()
        ssvep_connect_requested = _Signal()
        ssvep_config_apply_requested = _Signal()
        ssvep_pretrain_requested = _Signal()
        ssvep_load_profile_requested = _Signal()
        ssvep_open_profile_dir_requested = _Signal()
        ssvep_stim_toggled = _Signal()
        ssvep_start_requested = _Signal()
        ssvep_stop_requested = _Signal()
        manual_pick_slot_requested = _Signal()
        manual_place_requested = _Signal()
        pick_radius_bias_delta_requested = _Signal()
        pick_bias_reset_requested = _Signal()
        pick_tangent_bias_delta_requested = _Signal()
        pick_tangent_bias_reset_requested = _Signal()
        pick_theta_bias_delta_requested = _Signal()
        pick_theta_bias_reset_requested = _Signal()
        pick_tuning_delta_requested = _Signal()
        pick_release_mode_toggle_requested = _Signal()
        pick_tuning_apply_requested = _Signal()
        pick_tuning_reset_requested = _Signal()
        pick_tuning_save_requested = _Signal()

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def update_pick_bias_display(self, *_args: object, **_kwargs: object) -> None:
            pass

        def update_pick_tuning_display(self, *_args: object, **_kwargs: object) -> None:
            pass

        def append_log(self, *_args: object, **_kwargs: object) -> None:
            pass

        def update_panels(self, *_args: object, **_kwargs: object) -> None:
            pass

        def update_vision_payload(self, *args: object, **kwargs: object) -> None:
            pass

        def ssvep_runtime_config(self) -> dict[str, object]:
            return {}

        def ssvep_pretrain_config(self) -> dict[str, object]:
            return {}

        def is_ssvep_profile_auto_selected(self) -> bool:
            return False

        def selected_ssvep_profile_path(self) -> str:
            return ""

        def set_ssvep_runtime_config(self, *_args: object, **_kwargs: object) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def close(self) -> None:
            pass

    fake_window_pkg.MainWindow = MainWindow
    sys.modules["hybrid_controller.ui.main_window"] = fake_window_pkg

    if "cv2" not in sys.modules:
        cv2_stub = types.ModuleType("cv2")

        cv2_stub.COLOR_BGR2GRAY = 0
        cv2_stub.CC_STAT_AREA = 4
        cv2_stub.RETR_EXTERNAL = 0
        cv2_stub.CHAIN_APPROX_SIMPLE = 2

        def _noop(*_args: object, **_kwargs: object) -> object:
            return None

        def _connected_components_with_stats(*_args: object, **_kwargs: object) -> tuple[int, object, object, object]:
            return 0, None, None, None

        def _morphology_ex(*_args: object, **_kwargs: object) -> object:
            return _args[0] if _args else None

        def _min_area_rect(*_args: object, **_kwargs: object) -> tuple[float, tuple[float, float], float]:
            return ((0.0, 0.0), (0.0, 0.0), 0.0)

        def _find_contours(*_args: object, **_kwargs: object) -> tuple[list, tuple]:
            return [], (0, 0)

        cv2_stub.connectedComponentsWithStats = _connected_components_with_stats
        cv2_stub.morphologyEx = _morphology_ex
        cv2_stub.cvtColor = _noop
        cv2_stub.minAreaRect = _min_area_rect
        cv2_stub.findContours = _find_contours
        cv2_stub.__spec__ = importlib.machinery.ModuleSpec(name="cv2", loader=None)
        sys.modules["cv2"] = cv2_stub

    if "numpy" not in sys.modules:
        np_stub = types.ModuleType("numpy")

        def _asarray(value: object) -> object:
            return value

        def _mean(value: object, *_args: object, **_kwargs: object) -> float:
            if isinstance(value, (list, tuple)) and value:
                return sum(float(v) for v in value) / len(value)
            return 0.0

        def _percentile(value: object, *_args: object, **_kwargs: object) -> float:
            if not isinstance(value, (list, tuple)) or not value:
                return 0.0
            data = sorted(float(v) for v in value)
            return data[int((len(data) - 1) * 0.5)]

        def _std(value: object, *_args: object, **_kwargs: object) -> float:
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                return 0.0
            m = _mean(value)
            return float((sum((float(v) - m) ** 2 for v in value) / float(len(value))) ** 0.5)

        def _ptp(value: object, *_args: object, **_kwargs: object) -> float:
            if not isinstance(value, (list, tuple)) or not value:
                return 0.0
            data = [float(v) for v in value]
            return float(max(data) - min(data))

        def _amin(value: object, *_args: object, **_kwargs: object) -> float:
            return float(min(value)) if value else 0.0

        def _amax(value: object, *_args: object, **_kwargs: object) -> float:
            return float(max(value)) if value else 0.0

        def _median(value: object, *_args: object, **_kwargs: object) -> float:
            if not isinstance(value, (list, tuple)) or not value:
                return 0.0
            data = sorted(float(v) for v in value)
            return data[int((len(data) - 1) // 2)]

        class _DType(float):
            pass

        np_stub.float32 = _DType
        np_stub.float64 = _DType
        np_stub.float = float
        np_stub.int = int
        np_stub.int64 = int
        np_stub.ndarray = list
        np_stub.asarray = _asarray
        np_stub.mean = _mean
        np_stub.percentile = _percentile
        np_stub.std = _std
        np_stub.median = _median
        np_stub.ptp = _ptp
        np_stub.min = _amin
        np_stub.max = _amax
        np_stub.array = _asarray
        np_stub.__spec__ = importlib.machinery.ModuleSpec(name="numpy", loader=None)
        sys.modules["numpy"] = np_stub


def _resolve_qt_application():
    try:
        from PyQt5.QtWidgets import QApplication
        QApplication.instance()
        return QApplication
    except Exception:
        _install_offline_qt_stubs()
        from PyQt5.QtWidgets import QApplication
        return QApplication


def _build_frame(
    *,
    frame_id: int,
    center_distance_mm: float,
) -> dict[str, object]:
    return {
        "frame_id": int(frame_id),
        "frame_size": [640, 480],
        "image_size": [640, 480],
        "roi_center": [320, 240],
        "roi_radius": 240,
        "alignment_target_pixel": [320.0, 240.0],
        "capture_fps": 30.0,
        "infer_ms": 12.0,
        "queue_age_ms": 5.0,
        "stream_age_ms": None,
        "frame_quality": {"gray_mean": 94.0, "gray_p95": 128.0},
        "frame_block_reason": "",
        "detected_count": 4,
        "selected_slot": None,
        "slots": _build_slots(center_distance_mm=center_distance_mm),
        "calibration_ready": True,
        "mapping_mode": "absolute_base",
        "calibration_profile_required": False,
        "calibration_profile_id": "offline_sim_profile",
    }


def _pick_slots(packet: dict[str, object]) -> Iterable[dict[str, object]]:
    for slot in packet.get("slots", []):
        if isinstance(slot, dict):
            yield slot


def run_simulation(
    *,
    steps: int = 120,
    step_interval_sec: float = 0.05,
    start_distance: float = 52.0,
    end_distance: float = 1.0,
    start_z: float = 170.0,
    confirm_z: float = 130.0,
    z_rate_limit_mm_s: float | None = None,
    continuous_stop_at_confirm: bool = True,
    trace_decisions: bool = False,
) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication = _resolve_qt_application()
    app = QApplication.instance() or QApplication([])
    from hybrid_controller.app import HybridControllerApplication
    config = AppConfig(
        robot_mode="sim",
        vision_mode="robot_camera_detection",
        move_source="sim",
        decision_source="sim",
        control_sim_enabled=True,
        robot_connect_on_start=False,
        vision_auto_start=False,
        vision_mapping_mode="absolute_base",
        vision_continuous_servo_enabled=True,
        vision_continuous_servo_stop_at_confirm=continuous_stop_at_confirm,
        vision_eye_in_hand_pick_flow_enabled=True,
        vision_pick_confirm_z_mm=float(confirm_z),
        pick_tool_offset_source="target_pixel",
    ).resolved()
    if z_rate_limit_mm_s is not None and math.isfinite(float(z_rate_limit_mm_s)):
        config = replace(config, vision_continuous_servo_z_rate_limit_mm_s=float(z_rate_limit_mm_s)).resolved()
    runtime = HybridControllerApplication(config)
    runtime.controller.context.robot_cyl = (8.0, 130.0, float(start_z))
    runtime.controller.context.robot_xy = (0.0, -120.0)
    runtime._capture_world_snapshot(reason="offline_servo_align_sim", force=True)

    original_step = runtime._step_continuous_sim_servo_pose

    def _step_trace(
        *,
        theta_rate_deg_s: float,
        radius_rate_mm_s: float,
        z_rate_mm_s: float,
    ) -> bool:
        print(
            "  step_cmd:"
            f" dTheta={theta_rate_deg_s:>7.3f} deg/s"
            f"  dR={radius_rate_mm_s:>8.3f} mm/s"
            f"  dZ={z_rate_mm_s:>8.3f} mm/s"
        )
        return original_step(
            theta_rate_deg_s=theta_rate_deg_s,
            radius_rate_mm_s=radius_rate_mm_s,
            z_rate_mm_s=z_rate_mm_s,
        )

    runtime._step_continuous_sim_servo_pose = _step_trace

    if trace_decisions:
        servo_controller = runtime._continuous_vision_servo_controller_instance()

        original_decide = servo_controller.decide

        def _logged_decide(
            self,
            *,
            slot_id: int,
            slot_payload: Mapping[str, object] | None,
            packet: Mapping[str, object] | None,
            pending: Mapping[str, object] | None = None,
            current_cyl_pose: tuple[float, float, float] | None = None,
            frame_pose_age_ms: float | None = None,
        ):
            slot_payload_map = slot_payload if isinstance(slot_payload, Mapping) else {}
            payload = self.config if self.config is not None else runtime.config
            confirm_z = float(getattr(payload, "vision_pick_confirm_z_mm", 130.0))
            z_tolerance = float(getattr(payload, "vision_pick_z_tolerance_mm", 4.0))
            pick_ready_px = float(getattr(payload, "vision_continuous_servo_pick_ready_center_px", 2.0))
            required_stable = max(1, int(getattr(payload, "vision_continuous_servo_stable_frames", 2)))
            center_distance = float(slot_payload_map.get("center_distance_px", float("nan")))
            current_z = float(current_cyl_pose[2]) if isinstance(current_cyl_pose, tuple) and len(current_cyl_pose) >= 3 else float("nan")
            at_confirm = "True" if (abs(current_z - confirm_z) <= z_tolerance) else "False"
            pending_map = pending if isinstance(pending, Mapping) else {}
            pick_ready_frames = int(pending_map.get("pick_ready_frames", 0))
            print(
                "  gate:"
                f" at_confirm={at_confirm}"
                f" z={current_z:.3f}"
                f" confirm={confirm_z:.1f}"
                f" z_tol={z_tolerance:.1f}"
                f" pick_ready_px={pick_ready_px:.2f}"
                f" req_stable={required_stable}"
                f" pick_ready={pick_ready_frames}"
                f" action={bool(slot_payload_map.get('actionable', False))}"
                f" invalid={str(slot_payload_map.get('invalid_reason', ''))}"
                f" center={center_distance:.3f}"
            )
            decision = original_decide(
                slot_id=slot_id,
                slot_payload=slot_payload,
                packet=packet,
                pending=pending,
                current_cyl_pose=current_cyl_pose,
                frame_pose_age_ms=frame_pose_age_ms,
            )
            pending = getattr(decision, "pending_dict", None)
            action = getattr(decision, "action", "--")
            status = getattr(decision, "status", "")
            reason = getattr(decision, "reason", "")
            trace = getattr(decision, "trace", None)
            print(
                "  decision:"
                f" action={action}"
                f" reason={reason}"
                f" status={status}"
                f" trace_center={trace.get('center_distance_px') if isinstance(trace, dict) else '--'}"
                f" stable={getattr(pending, 'stable_frames', pending.get('stable_frames', '--') if isinstance(pending, dict) else '--') if pending is not None else '--'}"
                f" pick_ready={getattr(pending, 'pick_ready_frames', pending.get('pick_ready_frames', '--') if isinstance(pending, dict) else '--') if pending is not None else '--'}"
            )
            return decision

        servo_controller.decide = types.MethodType(_logged_decide, servo_controller)

    distance_start = float(start_distance)
    distance_end = float(end_distance)
    distance_delta = max(distance_start - distance_end, 0.2)
    last_status = ""
    had_pending = runtime._continuous_vision_servo_pick is not None
    frame_interval = max(0.0, float(step_interval_sec))
    if frame_interval > 0.0:
        runtime._continuous_servo_sim_last_step_ts = time.monotonic() - frame_interval

    initial_packet = _build_frame(frame_id=0, center_distance_mm=distance_start)
    runtime._on_vision_packet_received(initial_packet)
    target_slot = next((s for s in _pick_slots(initial_packet) if int(s.get("slot_id", -1)) == 1), None)
    print("initial packet:")
    print(
        f"  slot1 center_distance={target_slot['center_distance_px'] if target_slot else '--'}"
        f" invalid={target_slot.get('invalid_reason') if isinstance(target_slot, dict) else '--'}"
    )
    print(f"  config_confirm_z={float(confirm_z):.1f}  z_rate_limit={float(getattr(config, 'vision_continuous_servo_z_rate_limit_mm_s', 0.0)):.2f}")

    runtime._on_manual_pick_slot_requested(1)
    app.processEvents()
    start_status = str(runtime._rt_get("vision_servo_status", "--"))
    print(f"start status: {start_status}")

    for step in range(1, steps + 1):
        if frame_interval > 0.0:
            runtime._continuous_servo_sim_last_step_ts = time.monotonic() - frame_interval
        ratio = max(0.0, min(1.0, step / max(1, steps)))
        current_distance = max(
            distance_end,
            distance_start - distance_delta * ratio,
        )
        packet = _build_frame(frame_id=step, center_distance_mm=current_distance)
        runtime._on_vision_packet_received(packet)
        app.processEvents()
        resolved = runtime._latest_vision_packet if isinstance(runtime._latest_vision_packet, dict) else {}
        slot1 = next((s for s in _pick_slots(resolved) if int(s.get("slot_id", -1)) == 1), None)
        status = str(runtime._rt_get("vision_servo_status", "--"))
        pending = runtime._continuous_vision_servo_pick
        if status != last_status:
            print(f"status_update: {last_status or '--'} -> {status}")
            last_status = status

        pose = tuple(runtime.controller.context.robot_cyl)
        pick_ready = "--"
        stable = "--"
        if isinstance(pending, dict):
            pick_ready = str(pending.get("pick_ready_frames", "--"))
            stable = str(pending.get("stable_frames", "--"))
        print(
            f"frame={step:03d}"
            f" center={float(slot1.get('center_distance_px', float('nan'))) if isinstance(slot1, dict) else float('nan'):>6.2f}"
            f" invalid={slot1.get('invalid_reason') if isinstance(slot1, dict) else '--':>22}"
            f" status={status}"
            f" pending={type(pending).__name__}"
            f" stable={stable:>4s} pick={pick_ready:>4s}"
            f" pose={pose[0]:>7.2f},{pose[1]:>8.2f},{pose[2]:>7.2f}"
        )

        if pending is None and had_pending:
            print(f"servo loop finished: {status}")
            break
        if "continuous_stop_at_confirm" in status:
            print(f"servo stop confirm reached: {status}")
            break
        if "continuous_idle awaiting_pick" in status and had_pending:
            print(f"pending lost before completion: {status}")
            break
        if slot1 is None:
            print("slot1 disappeared from packet, stop simulation.")
            break
        had_pending = pending is not None
        if step_interval_sec > 0.0:
            time.sleep(step_interval_sec)

    app.processEvents()
    final_status = str(runtime._rt_get("vision_servo_status", "--"))
    confirm_blocked = runtime._rt_get("vision_servo_confirm_command_blocked", None)
    final_pose = runtime.controller.context.robot_cyl
    print(f"final status: {final_status}")
    print(f"confirm blocked: {confirm_blocked}")
    print(f"final pose: {final_pose[0]:.3f}, {final_pose[1]:.3f}, {final_pose[2]:.3f}")
    runtime.shutdown()
    return 0 if runtime._continuous_vision_servo_pick is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate vision continuous servo alignment in offline mode."
    )
    parser.add_argument("--steps", type=int, default=120, help="Frames to simulate.")
    parser.add_argument("--interval", type=float, default=0.05, help="Seconds between frames.")
    parser.add_argument("--start-z", type=float, default=170.0, help="Simulated robot start z (mm).")
    parser.add_argument("--confirm-z", type=float, default=130.0, help="Confirmation z for stop condition (mm).")
    parser.add_argument("--start-distance", type=float, default=52.0, help="Slot start center distance (px).")
    parser.add_argument("--end-distance", type=float, default=1.0, help="Slot center distance (px) at final frame.")
    parser.add_argument(
        "--z-rate-limit",
        type=float,
        default=60.0,
        help="Optional z-axis rate limit override for faster simulation (mm/s).",
    )
    parser.add_argument("--trace-decisions", action="store_true", help="Log raw vision servo decisions each frame.")
    parser.add_argument(
        "--disable-continuous-stop-at-confirm",
        action="store_true",
        help="Continue after PICK_READY instead of stopping at confirm position.",
    )
    args = parser.parse_args()
    return run_simulation(
        steps=args.steps,
        step_interval_sec=args.interval,
        start_distance=args.start_distance,
        end_distance=args.end_distance,
        start_z=args.start_z,
        confirm_z=args.confirm_z,
        z_rate_limit_mm_s=args.z_rate_limit,
        continuous_stop_at_confirm=not args.disable_continuous_stop_at_confirm,
        trace_decisions=args.trace_decisions,
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import threading
import time

import roslibpy


def _connect(host: str, port: int, timeout_sec: float) -> roslibpy.Ros:
    ros = roslibpy.Ros(host=str(host), port=int(port))
    ready = threading.Event()
    error_holder: dict[str, str] = {}

    def _on_ready(*_args: object) -> None:
        ready.set()

    def _on_error(error: object) -> None:
        error_holder["error"] = str(error)
        ready.set()

    ros.on_ready(_on_ready, run_in_thread=False)
    ros.on("error", _on_error)
    ros.run()
    if not ready.wait(timeout=float(timeout_sec)):
        ros.close()
        raise TimeoutError("Timed out waiting for rosbridge connection.")
    if error_holder.get("error"):
        ros.close()
        raise RuntimeError(error_holder["error"])
    if not ros.is_connected:
        ros.close()
        raise RuntimeError("rosbridge not connected.")
    return ros


def _call_service(
    ros: roslibpy.Ros,
    name: str,
    service_type: str,
    request: dict[str, object],
    *,
    timeout_sec: float,
) -> dict[str, object]:
    service = roslibpy.Service(ros, name, service_type)
    done = threading.Event()
    holder: dict[str, object] = {"response": None, "error": None}

    def _ok(response: dict[str, object]) -> None:
        holder["response"] = dict(response)
        done.set()

    def _err(error: object) -> None:
        holder["error"] = str(error)
        done.set()

    service.call(roslibpy.ServiceRequest(request), callback=_ok, errback=_err)
    if not done.wait(timeout=float(timeout_sec)):
        raise TimeoutError(f"Timed out waiting for service '{name}'.")
    if holder["error"] is not None:
        raise RuntimeError(str(holder["error"]))
    response = holder["response"]
    if not isinstance(response, dict):
        raise RuntimeError(f"Invalid service response for '{name}': {response!r}")
    return response


def _fetch_state(ros: roslibpy.Ros, *, timeout_sec: float) -> dict[str, object]:
    topic = roslibpy.Topic(ros, "/hybrid_controller/state", "hybrid_controller_ros/RobotState")
    done = threading.Event()
    holder: dict[str, object] = {"message": None}

    def _cb(message: dict[str, object]) -> None:
        holder["message"] = dict(message)
        done.set()

    topic.subscribe(_cb)
    try:
        if not done.wait(timeout=float(timeout_sec)):
            raise TimeoutError("Timed out waiting for /hybrid_controller/state.")
    finally:
        try:
            topic.unsubscribe()
        except Exception:
            pass
    message = holder["message"]
    if not isinstance(message, dict):
        raise RuntimeError("Invalid /hybrid_controller/state payload.")
    return message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe hybrid_controller ROS services.")
    parser.add_argument("--host", default="192.168.149.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--timeout-sec", type=float, default=12.0)
    parser.add_argument(
        "--action",
        choices=[
            "status",
            "move_cyl",
            "move_cyl_auto",
            "pick_world",
            "pick_cyl",
            "place",
            "abort",
            "reset",
        ],
        required=True,
    )
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=120.0)
    parser.add_argument("--z", type=float, default=160.0)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=-120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ros = _connect(args.host, args.port, timeout_sec=args.timeout_sec)
    try:
        action = str(args.action)
        if action == "status":
            payload = _fetch_state(ros, timeout_sec=args.timeout_sec)
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            return 0
        if action == "move_cyl":
            response = _call_service(
                ros,
                "/hybrid_controller/move_cyl",
                "hybrid_controller_ros/MoveCyl",
                {"theta_deg": float(args.theta), "radius_mm": float(args.radius), "z_mm": float(args.z)},
                timeout_sec=args.timeout_sec,
            )
        elif action == "move_cyl_auto":
            response = _call_service(
                ros,
                "/hybrid_controller/move_cyl_auto",
                "hybrid_controller_ros/MoveCylAuto",
                {"theta_deg": float(args.theta), "radius_mm": float(args.radius)},
                timeout_sec=args.timeout_sec,
            )
        elif action == "pick_world":
            response = _call_service(
                ros,
                "/hybrid_controller/pick_world",
                "hybrid_controller_ros/PickWorld",
                {"x_mm": float(args.x), "y_mm": float(args.y)},
                timeout_sec=max(float(args.timeout_sec), 25.0),
            )
        elif action == "pick_cyl":
            response = _call_service(
                ros,
                "/hybrid_controller/pick_cyl",
                "hybrid_controller_ros/PickCyl",
                {"theta_deg": float(args.theta), "radius_mm": float(args.radius)},
                timeout_sec=max(float(args.timeout_sec), 25.0),
            )
        elif action == "place":
            response = _call_service(
                ros,
                "/hybrid_controller/place",
                "std_srvs/Trigger",
                {},
                timeout_sec=max(float(args.timeout_sec), 25.0),
            )
        elif action == "abort":
            response = _call_service(
                ros,
                "/hybrid_controller/abort",
                "std_srvs/Trigger",
                {},
                timeout_sec=args.timeout_sec,
            )
        elif action == "reset":
            response = _call_service(
                ros,
                "/hybrid_controller/reset",
                "std_srvs/Trigger",
                {},
                timeout_sec=max(float(args.timeout_sec), 20.0),
            )
        else:
            raise RuntimeError(f"Unsupported action: {action}")
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
        time.sleep(0.1)
        return 0
    finally:
        try:
            ros.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

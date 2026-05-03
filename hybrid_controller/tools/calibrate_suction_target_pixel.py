from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from hybrid_controller.config import AppConfig


def _default_stream_url(host: str) -> str:
    return (
        f"http://{host}:8080/stream?"
        "topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
    )


def _capture_frame(stream_url: str, *, timeout_sec: float, drain_frames: int) -> np.ndarray:
    source = int(stream_url) if str(stream_url).isdigit() else str(stream_url)
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG if isinstance(source, str) else cv2.CAP_ANY)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera stream: {stream_url}")
    deadline = time.time() + max(0.5, float(timeout_sec))
    frame = None
    try:
        for _ in range(max(1, int(drain_frames))):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
            if time.time() > deadline:
                break
        while frame is None and time.time() <= deadline:
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
                break
            time.sleep(0.05)
    finally:
        capture.release()
    if frame is None:
        raise RuntimeError("Timed out waiting for camera frame.")
    return frame


def _find_orange_block_center(frame_bgr: np.ndarray, *, min_area: float) -> tuple[float, float, float]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([3, 45, 45]), np.array([35, 255, 255])),
        cv2.inRange(hsv, np.array([0, 35, 40]), np.array([45, 255, 255])),
    ]
    mask = masks[0]
    for extra in masks[1:]:
        mask = cv2.bitwise_or(mask, extra)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No orange block contour found.")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < float(min_area):
        raise RuntimeError(f"Detected contour is too small: area={area:.1f}px.")
    rect = cv2.minAreaRect(contour)
    center = rect[0]
    if not all(math.isfinite(float(value)) for value in center):
        raise RuntimeError("Detected block center is not finite.")
    return (float(center[0]), float(center[1]), area)


def _write_profile_target(profile_path: Path, target_pixel: tuple[float, float]) -> None:
    with profile_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError("Calibration profile must contain a JSON object.")
    servo = payload.get("servo")
    if not isinstance(servo, dict):
        servo = {}
        payload["servo"] = servo
    servo["target_pixel"] = [float(target_pixel[0]), float(target_pixel[1])]
    servo["target_pixel_method"] = "manual_suction_alignment"
    servo["target_pixel_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with profile_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _save_overlay(frame: np.ndarray, target_pixel: tuple[float, float], output_path: Path, area: float) -> None:
    overlay = frame.copy()
    point = (int(round(target_pixel[0])), int(round(target_pixel[1])))
    cv2.drawMarker(overlay, point, (255, 80, 220), cv2.MARKER_TILTED_CROSS, 34, 2)
    cv2.circle(overlay, point, 10, (255, 80, 220), 2)
    cv2.putText(
        overlay,
        f"target_pixel=({target_pixel[0]:.1f},{target_pixel[1]:.1f}) area={area:.0f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        f"target_pixel=({target_pixel[0]:.1f},{target_pixel[1]:.1f}) area={area:.0f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record the camera pixel that corresponds to the suction cup projection. "
            "Physically align the suction cup over the block first, with suction off."
        )
    )
    parser.add_argument("--host", default=AppConfig().robot_host)
    parser.add_argument("--stream-url", default="")
    parser.add_argument("--profile", type=Path, default=AppConfig().vision_calibration_profile_path)
    parser.add_argument("--output", type=Path, default=AppConfig().vision_debug_bundle_dir / "suction_target_overlay.jpg")
    parser.add_argument("--timeout-sec", type=float, default=4.0)
    parser.add_argument("--drain-frames", type=int, default=12)
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stream_url = str(args.stream_url).strip() or _default_stream_url(str(args.host))
    parsed = urlparse(stream_url)
    if parsed.scheme and not parsed.netloc:
        raise RuntimeError(f"Invalid stream URL: {stream_url}")
    frame = _capture_frame(stream_url, timeout_sec=float(args.timeout_sec), drain_frames=int(args.drain_frames))
    x, y, area = _find_orange_block_center(frame, min_area=float(args.min_area))
    target_pixel = (float(x), float(y))
    if not bool(args.no_write):
        _write_profile_target(Path(args.profile), target_pixel)
    _save_overlay(frame, target_pixel, Path(args.output), float(area))
    print(
        json.dumps(
            {
                "target_pixel": [float(target_pixel[0]), float(target_pixel[1])],
                "area_px": float(area),
                "profile": str(args.profile),
                "overlay": str(args.output),
                "wrote_profile": not bool(args.no_write),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

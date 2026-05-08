#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlparse, urlunparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


DEFAULT_BASE_URL = (
    "http://192.168.149.1:8080/stream?"
    "topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
)
DEFAULT_DURATION_SEC = 4.0
DEFAULT_PROFILES: Dict[str, Dict[str, str]] = {
    "mjpeg_640_q80": {
        "type": "mjpeg",
        "width": "640",
        "height": "480",
        "quality": "80",
    },
    "mjpeg_640_q80_compressed": {
        "type": "mjpeg",
        "width": "640",
        "height": "480",
        "quality": "80",
        "default_transport": "compressed",
    },
    "mjpeg_480_q70_compressed": {
        "type": "mjpeg",
        "width": "480",
        "height": "360",
        "quality": "70",
        "default_transport": "compressed",
    },
    "mjpeg_320_q60_compressed": {
        "type": "mjpeg",
        "width": "320",
        "height": "240",
        "quality": "60",
        "default_transport": "compressed",
    },
}


class HttpMjpegCapture:
    def __init__(self, url: str, *, timeout_sec: float) -> None:
        self._url = normalize_web_video_url(str(url))
        self._timeout_sec = max(0.2, float(timeout_sec))
        self._response = None
        self._buffer = bytearray()
        self._open()

    def _open(self) -> None:
        request = Request(self._url, headers={"User-Agent": "jetmax-stream-probe"})
        self._response = urlopen(request, timeout=self._timeout_sec)

    def isOpened(self) -> bool:
        return self._response is not None

    def read(self):
        if self._response is None:
            return False, None
        deadline = time.perf_counter() + self._timeout_sec
        while time.perf_counter() < deadline:
            start = self._buffer.find(b"\xff\xd8")
            end = self._buffer.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
            if start >= 0 and end >= 0:
                jpg = bytes(self._buffer[start : end + 2])
                del self._buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
                continue
            chunk = self._response.read(4096)
            if not chunk:
                return False, None
            self._buffer.extend(chunk)
            if len(self._buffer) > 4_000_000:
                del self._buffer[:-1_000_000]
        return False, None

    def release(self) -> None:
        response = self._response
        self._response = None
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def is_web_video_mjpeg_stream(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.path.endswith("/stream"):
        return False
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    requested_type = str(query.get("type", "")).strip().lower()
    return requested_type in {"", "mjpeg"}


def normalize_web_video_url(source: str) -> str:
    value = str(source).strip()
    try:
        parsed = urlparse(value)
    except Exception:
        return value
    if parsed.scheme not in {"http", "https"} or not parsed.path.endswith("/stream") or not parsed.query:
        return value
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if not query_items:
        return value
    normalized_query = "&".join(
        f"{quote(str(key), safe='')}={quote(str(val), safe='/')}" for key, val in query_items
    )
    return urlunparse(parsed._replace(query=normalized_query))


@dataclass
class ProbeResult:
    profile: str
    url: str
    opened: bool
    frame_count: int
    elapsed_sec: float
    fps: float
    first_frame_ms: Optional[float]
    frame_size: Optional[Tuple[int, int]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile": self.profile,
            "url": self.url,
            "opened": self.opened,
            "frame_count": self.frame_count,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "fps": round(self.fps, 3),
            "first_frame_ms": None if self.first_frame_ms is None else round(self.first_frame_ms, 3),
            "frame_size": None if self.frame_size is None else [self.frame_size[0], self.frame_size[1]],
        }


def build_url(base_url: str, extra_params: Dict[str, str]) -> str:
    if not extra_params:
        return base_url
    parsed = urlparse(base_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(extra_params)
    query_items = [f"{quote(str(key), safe='')}={quote(str(value), safe='/')}" for key, value in query.items()]
    return urlunparse(parsed._replace(query="&".join(query_items)))


def open_capture(url: str) -> cv2.VideoCapture:
    url = normalize_web_video_url(str(url))
    if is_web_video_mjpeg_stream(url):
        capture = None
        try:
            capture = HttpMjpegCapture(url, timeout_sec=3.0)
            return capture
        except Exception:
            pass
        if capture is not None:
            capture.release()
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    requested_type = str(query.get("type", "")).strip().lower()
    backend_candidates = [cv2.CAP_ANY]
    if hasattr(cv2, "CAP_FFMPEG"):
        if parsed.scheme in {"rtsp", "tcp"}:
            backend_candidates = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
        elif parsed.scheme in {"http", "https"} and requested_type not in {"h264", "vp8", "vp9"}:
            backend_candidates = [cv2.CAP_ANY, cv2.CAP_FFMPEG]
        else:
            backend_candidates = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
    deduped_backends: List[int] = []
    for backend in backend_candidates:
        if backend not in deduped_backends:
            deduped_backends.append(backend)
    for backend in deduped_backends:
        try:
            capture = cv2.VideoCapture(url, backend)
        except TypeError:
            capture = cv2.VideoCapture(url)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture()


def measure_stream(profile: str, url: str, duration_sec: float) -> ProbeResult:
    capture = open_capture(url)
    opened = bool(capture.isOpened())
    if not opened:
        capture.release()
        return ProbeResult(profile, url, False, 0, 0.0, 0.0, None, None)

    start = time.perf_counter()
    deadline = start + duration_sec
    first_frame_ms: Optional[float] = None
    frame_count = 0
    frame_size: Optional[Tuple[int, int]] = None

    while time.perf_counter() < deadline:
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        now = time.perf_counter()
        if first_frame_ms is None:
            first_frame_ms = (now - start) * 1000.0
        frame_count += 1
        if frame_size is None:
            frame_h, frame_w = frame.shape[:2]
            frame_size = (int(frame_w), int(frame_h))

    elapsed_sec = max(time.perf_counter() - start, 1e-6)
    capture.release()
    return ProbeResult(
        profile=profile,
        url=url,
        opened=True,
        frame_count=frame_count,
        elapsed_sec=elapsed_sec,
        fps=frame_count / elapsed_sec,
        first_frame_ms=first_frame_ms,
        frame_size=frame_size,
    )


def parse_profiles(raw: str) -> List[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or list(DEFAULT_PROFILES.keys())


def resolve_profiles(names: Iterable[str]) -> List[Tuple[str, Dict[str, str]]]:
    resolved: List[Tuple[str, Dict[str, str]]] = []
    for name in names:
        if name not in DEFAULT_PROFILES:
            raise ValueError(f"Unknown profile: {name}")
        resolved.append((name, DEFAULT_PROFILES[name]))
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe JetMax web_video_server stream profiles from Windows/PC.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base web_video_server URL with topic query")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC, help="Probe duration in seconds for each profile")
    parser.add_argument(
        "--profiles",
        type=str,
        default="mjpeg_640_q80",
        help=f"Comma-separated profile names. Available: {', '.join(DEFAULT_PROFILES.keys())}",
    )
    args = parser.parse_args()

    results: List[ProbeResult] = []
    for name, params in resolve_profiles(parse_profiles(args.profiles)):
        url = build_url(args.base_url, params)
        result = measure_stream(name, url, max(0.5, float(args.duration)))
        results.append(result)
        print(json.dumps(result.to_dict(), ensure_ascii=False), flush=True)

    best = max(results, key=lambda item: item.fps, default=None)
    summary = {
        "base_url": args.base_url,
        "profiles_tested": len(results),
        "best_profile": None if best is None else best.profile,
        "best_fps": None if best is None else round(best.fps, 3),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

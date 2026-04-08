from __future__ import annotations

import json
import socket
import threading
import time
from typing import Callable, Optional

from hybrid_controller.controller.events import Event


def parse_robot_line(line: str) -> Event:
    cleaned = line.strip()
    if not cleaned:
        raise ValueError("Robot message is empty.")
    if cleaned == "BUSY":
        return Event(source="robot", type="robot_busy", value="BUSY")
    if cleaned.startswith("ERR"):
        message = cleaned[3:].strip() or "Unknown robot error"
        return Event(source="robot", type="robot_error", value=message)
    if cleaned.startswith("ACK"):
        ack = cleaned[3:].strip()
        return Event(source="robot", type="robot_ack", value=ack)
    raise ValueError(f"Unsupported robot message: {cleaned}")


def fetch_robot_status(host: str, port: int, timeout_sec: float = 0.5) -> dict[str, object]:
    with socket.create_connection((host, int(port)), timeout=timeout_sec) as sock:
        sock.settimeout(timeout_sec)
        sock.sendall(b"STATUS\n")
        buffer = ""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="ignore")
            if "\n" not in buffer:
                continue
            line, _ = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            prefix = "ACK STATUS "
            if line.startswith(prefix):
                return json.loads(line[len(prefix) :].strip())
            if line == "ACK STATUS":
                return {}
            if line.startswith("ERR"):
                raise RuntimeError(line[3:].strip() or "Robot STATUS failed")
            raise RuntimeError(f"Unexpected STATUS response: {line}")
    raise RuntimeError("Robot STATUS timed out")


class RobotClient:
    def __init__(
        self,
        host: str,
        port: int,
        event_callback: Optional[Callable[[Event], None]] = None,
        timeout_sec: float = 0.2,
        reconnect_delay_sec: float = 1.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.event_callback = event_callback
        self.timeout_sec = float(timeout_sec)
        self.reconnect_delay_sec = float(reconnect_delay_sec)
        self._sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._send_lock = threading.Lock()
        self._pong_event = threading.Event()
        self._connected_lock = threading.Lock()
        self._connected = False

    def connect(self) -> None:
        if self._sock is not None:
            return
        self._stop_event.clear()
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        self._sock.settimeout(self.timeout_sec)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._connected_lock:
            self._connected = True
        self._recv_thread = threading.Thread(target=self._recv_loop, name="robot-client-recv", daemon=True)
        self._recv_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        with self._connected_lock:
            self._connected = False
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.0)
            self._recv_thread = None

    def is_connected(self) -> bool:
        with self._connected_lock:
            return self._connected

    def reconnect(self) -> None:
        self.close()
        time.sleep(self.reconnect_delay_sec)
        self.connect()

    def send_command(self, command: str) -> None:
        if self._sock is None:
            self.connect()
        if self._sock is None:
            raise RuntimeError("Robot client is not connected.")
        payload = (command.strip() + "\n").encode("utf-8")
        with self._send_lock:
            self._sock.sendall(payload)

    def send_move(self, x: float, y: float) -> None:
        self.send_command(f"MOVE {float(x):.2f} {float(y):.2f}")

    def send_move_cyl(self, theta_deg: float, radius_mm: float, z_mm: float) -> None:
        self.send_command(f"MOVE_CYL {float(theta_deg):.2f} {float(radius_mm):.2f} {float(z_mm):.2f}")

    def send_move_cyl_auto(self, theta_deg: float, radius_mm: float) -> None:
        self.send_command(f"MOVE_CYL_AUTO {float(theta_deg):.2f} {float(radius_mm):.2f}")

    def send_pick(self, x: float, y: float) -> None:
        self.send_command(f"PICK {float(x):.2f} {float(y):.2f}")

    def send_pick_cyl(self, theta_deg: float, radius_mm: float) -> None:
        self.send_command(f"PICK_CYL {float(theta_deg):.2f} {float(radius_mm):.2f}")

    def send_place(self) -> None:
        self.send_command("PLACE")

    def send_ping(self) -> None:
        self.send_command("PING")

    def healthcheck(self, timeout_sec: float = 1.0) -> bool:
        self._pong_event.clear()
        self.send_ping()
        return self._pong_event.wait(timeout_sec)

    def handle_line(self, line: str) -> Event:
        event = parse_robot_line(line)
        if event.type == "robot_ack" and str(event.value).strip().upper() == "PONG":
            self._pong_event.set()
        if self.event_callback is not None:
            self.event_callback(event)
        return event

    def _recv_loop(self) -> None:
        assert self._sock is not None
        buffer = ""
        while not self._stop_event.is_set():
            try:
                data = self._sock.recv(1024)
                if not data:
                    break
                buffer += data.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        self.handle_line(line)
            except socket.timeout:
                continue
            except OSError:
                break
        with self._connected_lock:
            self._connected = False
        self._sock = None
        if not self._stop_event.is_set() and self.event_callback is not None:
            self.event_callback(Event(source="robot", type="robot_error", value="Robot connection lost"))

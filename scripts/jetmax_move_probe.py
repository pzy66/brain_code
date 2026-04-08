from __future__ import annotations

import argparse
import json
import socket
import time


def read_line(sock: socket.socket, timeout: float = 5.0) -> str:
    sock.settimeout(timeout)
    buffer = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="ignore")
        if "\n" in buffer:
            line, _ = buffer.split("\n", 1)
            return line.strip()
    raise TimeoutError("Timed out waiting for robot response")


def send_command(sock: socket.socket, command: str, timeout: float = 5.0) -> str:
    sock.sendall((command.strip() + "\n").encode("utf-8"))
    return read_line(sock, timeout=timeout)


def fetch_status(sock: socket.socket, timeout: float = 5.0) -> dict[str, object]:
    response = send_command(sock, "STATUS", timeout=timeout)
    prefix = "ACK STATUS "
    if not response.startswith(prefix):
        raise RuntimeError(f"Unexpected STATUS response: {response}")
    return json.loads(response[len(prefix) :].strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe JetMax incremental MOVE behavior")
    parser.add_argument("--host", default="192.168.149.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--dx", type=float, default=8.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pause-ms", type=int, default=0)
    parser.add_argument("--reset-first", action="store_true")
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
        if args.reset_first:
            print(send_command(sock, "RESET"))
            time.sleep(1.8)
        status = fetch_status(sock)
        print("START", status)
        for index in range(max(0, int(args.steps))):
            robot_xy = status.get("robot_xy", [0.0, 0.0])
            next_x = float(robot_xy[0]) + float(args.dx)
            next_y = float(robot_xy[1]) + float(args.dy)
            print("MOVE", index + 1, next_x, next_y)
            print(send_command(sock, f"MOVE {next_x:.2f} {next_y:.2f}", timeout=10.0))
            status = fetch_status(sock)
            print("STATUS", status)
            if args.pause_ms > 0:
                time.sleep(float(args.pause_ms) / 1000.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

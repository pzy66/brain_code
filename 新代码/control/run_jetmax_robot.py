from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path


ROBOT_HOST = "192.168.149.1"
ROBOT_USER = "hiwonder"
ROBOT_PASSWORD = "hiwonder"

ROOT = Path(__file__).resolve().parent
START_RUNTIME = ROOT / "hybrid_controller" / "robot" / "tools" / "jetmax_start_ros_runtime.py"
RUN_REAL = ROOT / "hybrid_controller" / "run_real.py"


def log(message: str) -> None:
    print(message, flush=True)


def require_modules() -> None:
    missing: list[str] = []
    for name in ("paramiko",):
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing dependencies: {0}\n"
            "Activate the allowed environment and install there only:\n"
            "  conda activate brain_robot\n"
            "  pip install paramiko".format(", ".join(missing))
        )


def tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout)):
            return True
    except OSError:
        return False


def hybrid_runtime_ready(host: str, user: str, password: str) -> bool:
    try:
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=user, password=password, timeout=8)
        command = (
            "bash -lc 'source /opt/ros/melodic/setup.bash; "
            "source ~/ros/devel/setup.bash 2>/dev/null || true; "
            "rosservice list 2>/dev/null | grep -qx /hybrid_controller/move_cyl_auto'"
        )
        _, stdout, stderr = client.exec_command(command, timeout=15)
        status = stdout.channel.recv_exit_status()
        stdout.read()
        stderr.read()
        client.close()
        return status == 0
    except Exception:
        return False


def run_tool(args: list[str], *, timeout: int = 180) -> str:
    command = [sys.executable, str(START_RUNTIME), *args]
    result = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-60:])
        raise SystemExit(tail or f"Command failed: {' '.join(command)}")
    return result.stdout


def start_runtime(host: str, user: str, password: str, *, force_rebuild: bool) -> None:
    log("[1/3] Starting JetMax ROS runtime...")
    args = [
        "--host",
        host,
        "--user",
        user,
        "--password",
        password,
        "--remote-root",
        "/home/hiwonder/brain_code",
        "--keep-autostart-rosbridge",
        "--skip-camera-check",
        "--skip-tcp-check",
        "--ready-timeout-sec",
        "120",
    ]
    if force_rebuild:
        args.append("--force-catkin-rebuild")
    out = run_tool(args, timeout=240)
    for line in out.splitlines()[-12:]:
        log(line)


def repair_camera(host: str, user: str, password: str) -> None:
    log("[camera] Repairing JetMax usb_cam.service with auto camera-device detection...")
    out = run_tool(
        [
            "--host",
            host,
            "--user",
            user,
            "--password",
            password,
            "--camera-only",
            "--repair-camera-sender",
            "--allow-camera-sender-mutation",
            "--camera-video-device",
            "/dev/usb_cam0",
            "--camera-io-method",
            "mmap",
            "--camera-framerate",
            "20",
            "--check-camera-stream",
        ],
        timeout=180,
    )
    for line in out.splitlines()[-40:]:
        log(line)


def check_video(host: str) -> bool:
    url = (
        f"http://{host}:8080/stream?topic=/usb_cam/image_rect_color"
        "&type=mjpeg&width=640&height=480&quality=80"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read(256_000)
        return data.find(b"\xff\xd8") >= 0 and data.find(b"\xff\xd9") >= 0
    except Exception:
        return False


def launch_gui(host: str, *, confirm_only: bool) -> int:
    args = [sys.executable, str(RUN_REAL), "--robot-host", host]
    if confirm_only:
        args.append("--vision-continuous-servo-stop-at-confirm")
    return int(subprocess.run(args, cwd=str(ROOT)).returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JetMax robot launcher")
    parser.add_argument("--host", default=ROBOT_HOST)
    parser.add_argument("--user", default=ROBOT_USER)
    parser.add_argument("--password", default=os.environ.get("JETMAX_PASSWORD", ROBOT_PASSWORD))
    parser.add_argument("--skip-start", action="store_true")
    parser.add_argument("--force-start", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--repair-camera", "--reset-camera", action="store_true", dest="repair_camera")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Open the PyQt camera/robot GUI after checks.")
    parser.add_argument("--confirm-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_modules()
    if not START_RUNTIME.exists() or not RUN_REAL.exists():
        raise SystemExit(f"Missing extracted control files under {ROOT}")

    log(f"Python: {sys.executable}")
    log(f"JetMax: {args.user}@{args.host}")
    if not tcp_open(args.host, 22, timeout=4):
        raise SystemExit(f"Cannot reach JetMax SSH at {args.host}:22")

    if args.repair_camera:
        repair_camera(args.host, args.user, args.password)

    if not args.skip_start:
        if (
            args.force_start
            or args.force_rebuild
            or not tcp_open(args.host, 9091, timeout=2)
            or not hybrid_runtime_ready(args.host, args.user, args.password)
        ):
            start_runtime(args.host, args.user, args.password, force_rebuild=bool(args.force_rebuild))
        else:
            log("[1/3] hybrid_controller runtime is already ready.")

    log("[2/3] Checking camera MJPEG stream...")
    if check_video(args.host):
        log("OK: camera stream has JPEG frames.")
    else:
        raise SystemExit(
            "Camera stream is reachable but has no JPEG frames. Run:\n"
            "  python .\\run_jetmax_robot.py --repair-camera --check-only"
        )

    if args.check_only:
        log("[3/3] Check complete.")
        return 0

    log("[3/3] Launching GUI.")
    return launch_gui(args.host, confirm_only=bool(args.confirm_only))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import paramiko


DEFAULT_HOST = "192.168.149.1"
DEFAULT_USER = "hiwonder"
DEFAULT_PASSWORD = "hiwonder"
DEFAULT_REMOTE_ROOT = "/home/hiwonder/brain_code"
DEFAULT_ROSBRIDGE_PORT = 9091
DEFAULT_WEB_VIDEO_PORT = 8080


@dataclass(frozen=True, slots=True)
class PortCheck:
    name: str
    port: int
    required: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start JetMax ROS runtime over SSH and wait until ports are ready.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--ssh-timeout-sec", type=float, default=10.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=90.0)
    parser.add_argument("--rosbridge-port", type=int, default=DEFAULT_ROSBRIDGE_PORT)
    parser.add_argument("--web-video-port", type=int, default=DEFAULT_WEB_VIDEO_PORT)
    parser.add_argument("--disable-autostart-rosbridge", action="store_true", default=True)
    parser.add_argument("--keep-autostart-rosbridge", action="store_false", dest="disable_autostart_rosbridge")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--skip-camera-check", action="store_true")
    parser.add_argument(
        "--require-tcp-check",
        action="store_true",
        help="Require legacy TCP runtime port 8888 to be reachable before reporting success.",
    )
    parser.add_argument("--skip-tcp-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    local_robot_dir = Path(__file__).resolve().parents[1]
    remote_robot_dir = f"{args.remote_root}/hybrid_controller/robot"
    remote_log = f"{args.remote_root}/hybrid_ros_runtime.log"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        args.host,
        username=args.user,
        password=args.password,
        timeout=float(args.ssh_timeout_sec),
    )
    try:
        if not args.no_sync:
            sync_robot_bundle(ssh, local_robot_dir=local_robot_dir, remote_robot_dir=remote_robot_dir)
        run_remote_command(ssh, f"test -d {remote_robot_dir}")
        run_remote_command(ssh, f"test -f {remote_robot_dir}/run_hybrid_controller_ros_runtime.sh")
        if args.disable_autostart_rosbridge:
            run_remote_command(
                ssh,
                f"echo '{args.password}' | sudo -S systemctl stop rosbridge.service >/dev/null 2>&1 || true",
            )
            run_remote_command(
                ssh,
                f"echo '{args.password}' | sudo -S systemctl disable rosbridge.service >/dev/null 2>&1 || true",
            )
        run_remote_command(
            ssh,
            "pkill -f hybrid_controller_runtime_node.py >/dev/null 2>&1 || true; "
            "pkill -f run_hybrid_controller_ros_runtime.sh >/dev/null 2>&1 || true; "
            "pkill -f rosbridge_websocket >/dev/null 2>&1 || true; "
            "pkill -f web_video_server >/dev/null 2>&1 || true; "
            "pkill -f robot_runtime_py36.py >/dev/null 2>&1 || true",
        )
        run_remote_command(ssh, f"rm -f {remote_log}")
        run_remote_command(
            ssh,
            f"cd {remote_robot_dir}; "
            f"ROSBRIDGE_PORT={int(args.rosbridge_port)} WEB_VIDEO_PORT={int(args.web_video_port)} "
            f"nohup bash run_hybrid_controller_ros_runtime.sh > {remote_log} 2>&1 < /dev/null &",
        )
        time.sleep(2.0)
        process_info = run_remote_command(
            ssh,
            "pgrep -af 'run_hybrid_controller_ros_runtime.sh|hybrid_controller_runtime_node.py|rosbridge_websocket|web_video_server|robot_runtime_py36.py' || true",
            capture=True,
        )
        if process_info:
            print(process_info)
        log_tail = run_remote_command(ssh, f"tail -n 60 {remote_log} 2>/dev/null || true", capture=True)
        if log_tail:
            print(log_tail)
    finally:
        ssh.close()

    checks = [PortCheck(name="rosbridge", port=int(args.rosbridge_port), required=True)]
    checks.append(PortCheck(name="web_video_server", port=int(args.web_video_port), required=not args.skip_camera_check))
    require_tcp_check = bool(args.require_tcp_check) and not bool(args.skip_tcp_check)
    checks.append(PortCheck(name="tcp_runtime", port=8888, required=require_tcp_check))

    wait_for_ports(args.host, checks, timeout_sec=float(args.ready_timeout_sec))
    verify_runtime_services(
        host=args.host,
        user=args.user,
        password=args.password,
        timeout_sec=float(args.ssh_timeout_sec),
    )
    if require_tcp_check:
        status_payload = query_status(args.host, 8888, timeout_sec=5.0)
        print(status_payload)
    print("JetMax ROS runtime ready.")
    return 0


def sync_robot_bundle(ssh: paramiko.SSHClient, *, local_robot_dir: Path, remote_robot_dir: str) -> None:
    local_robot_dir = local_robot_dir.resolve()
    sync_paths = [
        local_robot_dir / "run_hybrid_controller_ros_runtime.sh",
        local_robot_dir / "requirements-jetmax-robot-python.txt",
        local_robot_dir / "runtime",
        local_robot_dir / "ros_pkg",
    ]
    sftp = ssh.open_sftp()
    try:
        ensure_remote_dir(sftp, remote_robot_dir)
        for source in sync_paths:
            if not source.exists():
                continue
            if source.is_file():
                rel = source.relative_to(local_robot_dir)
                remote_file = f"{remote_robot_dir}/{rel.as_posix()}"
                ensure_remote_dir(sftp, remote_file.rsplit("/", 1)[0])
                sftp.put(str(source), remote_file)
                continue
            for file_path in source.rglob("*"):
                if not file_path.is_file():
                    continue
                if "__pycache__" in file_path.parts:
                    continue
                if file_path.suffix in {".pyc", ".pyo"}:
                    continue
                rel = file_path.relative_to(local_robot_dir)
                remote_file = f"{remote_robot_dir}/{rel.as_posix()}"
                ensure_remote_dir(sftp, remote_file.rsplit("/", 1)[0])
                sftp.put(str(file_path), remote_file)
    finally:
        sftp.close()


def run_remote_command(ssh: paramiko.SSHClient, command: str, *, capture: bool = False) -> str:
    _, stdout, stderr = ssh.exec_command("bash -lc " + repr(command))
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="ignore").strip()
    err = stderr.read().decode("utf-8", errors="ignore").strip()
    if exit_code not in (0, -1):
        raise RuntimeError(err or out or f"Remote command failed ({exit_code}): {command}")
    if exit_code == -1 and err:
        raise RuntimeError(err or out or f"Remote command failed ({exit_code}): {command}")
    if capture:
        return out
    return ""


def wait_for_ports(host: str, checks: list[PortCheck], *, timeout_sec: float) -> None:
    deadline = time.time() + max(1.0, float(timeout_sec))
    waiting = {item.name: item for item in checks if item.required}
    while waiting and time.time() < deadline:
        done: list[str] = []
        for name, check in waiting.items():
            if can_connect(host, check.port, timeout_sec=1.5):
                done.append(name)
        for name in done:
            check = waiting.pop(name)
            print(f"{check.name} ready on {host}:{check.port}")
        if waiting:
            time.sleep(0.8)
    if waiting:
        detail = ", ".join(f"{item.name}:{item.port}" for item in waiting.values())
        raise RuntimeError(f"Timed out waiting for ports: {detail}")


def can_connect(host: str, port: int, *, timeout_sec: float) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.2, float(timeout_sec))):
            return True
    except OSError:
        return False


def query_status(host: str, port: int, *, timeout_sec: float) -> str:
    with socket.create_connection((host, int(port)), timeout=max(0.5, float(timeout_sec))) as sock:
        sock.sendall(b"STATUS\n")
        payload = sock.recv(4096).decode("utf-8", errors="ignore").strip()
    return payload


def verify_runtime_services(*, host: str, user: str, password: str, timeout_sec: float) -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password, timeout=float(timeout_sec))
    try:
        process_info = run_remote_command(
            ssh,
            "pgrep -af hybrid_controller_runtime_node.py || true",
            capture=True,
        )
        if not process_info:
            raise RuntimeError("hybrid_controller_runtime_node.py is not running on JetMax.")
        required = {
            "/hybrid_controller/move_cyl",
            "/hybrid_controller/move_cyl_auto",
            "/hybrid_controller/pick_world",
            "/hybrid_controller/place",
            "/hybrid_controller/reset",
            "/hybrid_controller/abort",
        }
        listed = run_remote_command(
            ssh,
            "source /opt/ros/melodic/setup.bash; "
            "source ~/catkin_ws/devel/setup.bash; "
            "rosservice list 2>/dev/null | grep -E \"^/hybrid_controller/\" || true",
            capture=True,
        )
        available = {line.strip() for line in str(listed).splitlines() if line.strip()}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError("Missing ROS services: {0}".format(", ".join(missing)))
    finally:
        ssh.close()


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts: list[str] = []
    path = remote_dir.strip()
    while path not in ("", "/"):
        parts.append(path.rsplit("/", 1)[-1])
        path = path.rsplit("/", 1)[0] if "/" in path else ""
    current = "/" if remote_dir.startswith("/") else ""
    for part in reversed(parts):
        current = f"{current.rstrip('/')}/{part}" if current else part
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import paramiko


DEFAULT_HOST = "192.168.149.1"
DEFAULT_USER = "hiwonder"
DEFAULT_PASSWORD = "hiwonder"
DEFAULT_REMOTE_ROOT = "/home/hiwonder/brain_code"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload and start JetMax runtime over SSH.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    local_runtime = repo_root / "robot" / "runtime" / "robot_runtime_py36.py"
    remote_runtime = f"{args.remote_root}/hybrid_controller/robot/runtime/robot_runtime_py36.py"
    remote_log = f"{args.remote_root}/robot_runtime.log"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.host, username=args.user, password=args.password, timeout=10)
    try:
        sftp = ssh.open_sftp()
        try:
            ensure_remote_dir(sftp, f"{args.remote_root}/hybrid_controller/robot/runtime")
            sftp.put(str(local_runtime), remote_runtime)
        finally:
            sftp.close()

        run_remote_command(ssh, "pkill -f robot_runtime_py36.py >/dev/null 2>&1 || true")
        run_remote_command(ssh, f"rm -f {remote_log}")
        run_remote_command(
            ssh,
            "source /opt/ros/melodic/setup.bash >/dev/null 2>&1 || true; "
            "source ~/ros/devel/setup.bash >/dev/null 2>&1 || true; "
            "source ~/catkin_ws/devel/setup.bash >/dev/null 2>&1 || true; "
            f"mkdir -p {args.remote_root}; "
            f"cd {args.remote_root}; "
            f"nohup python3 {remote_runtime} --host 0.0.0.0 --port 8888 > {remote_log} 2>&1 < /dev/null &",
        )
        time.sleep(2.0)
        print(run_remote_command(ssh, "pgrep -af robot_runtime_py36.py || true", capture=True))
        print(run_remote_command(ssh, f"tail -n 40 {remote_log} 2>/dev/null || true", capture=True))
    finally:
        ssh.close()

    deadline = time.time() + 15.0
    last_error: OSError | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((args.host, 8888), timeout=3.0) as sock:
                sock.sendall(b"STATUS\n")
                payload = sock.recv(4096).decode("utf-8", errors="ignore").strip()
                print(payload)
                return 0
        except OSError as error:
            last_error = error
            time.sleep(1.0)
    raise RuntimeError(f"JetMax runtime did not start listening on 8888: {last_error}")


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts: list[str] = []
    path = remote_dir
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


if __name__ == "__main__":
    raise SystemExit(main())

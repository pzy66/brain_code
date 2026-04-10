from __future__ import annotations

import argparse
import posixpath
import socket
import sys
import time
from pathlib import Path

import paramiko


DEFAULT_HOST = "192.168.149.1"
DEFAULT_USER = "hiwonder"
DEFAULT_PASSWORD = "hiwonder"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy brain_code JetMax runtime to the robot controller.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--remote-root", default="/home/hiwonder/brain_code")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    local_runtime = repo_root / "robot" / "runtime" / "robot_runtime_py36.py"
    remote_runtime = f"{args.remote_root}/hybrid_controller/robot/runtime/robot_runtime_py36.py"

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

        strict_commands = [
            f"python3 -m py_compile {remote_runtime}",
        ]
        for command in strict_commands:
            _, stdout, stderr = ssh.exec_command(f"bash -lc {command!r}")
            exit_code = stdout.channel.recv_exit_status()
            out = stdout.read().decode("utf-8", errors="ignore").strip()
            err = stderr.read().decode("utf-8", errors="ignore").strip()
            if out:
                print(out)
            if err:
                print(err, file=sys.stderr)
            if exit_code != 0:
                raise RuntimeError(f"Remote command failed ({exit_code}): {command}")
        restart_commands = [
            "pkill -f robot_runtime_py36.py >/dev/null 2>&1 || true",
            f"rm -f {args.remote_root}/robot_runtime.log",
            (
                "source /opt/ros/melodic/setup.bash >/dev/null 2>&1 || true; "
                "source ~/ros/devel/setup.bash >/dev/null 2>&1 || true; "
                "source ~/catkin_ws/devel/setup.bash >/dev/null 2>&1 || true; "
                f"mkdir -p {args.remote_root}; "
                f"cd {args.remote_root}; "
                f"nohup python3 {remote_runtime} --host 0.0.0.0 --port 8888 >{args.remote_root}/robot_runtime.log 2>&1 &"
            ),
        ]
        for command in restart_commands:
            ssh.exec_command(f"bash -lc {command!r}")
        _, stdout, _ = ssh.exec_command("bash -lc 'ps -ef | grep robot_runtime_py36.py | grep -v grep || true'")
        process_listing = stdout.read().decode("utf-8", errors="ignore").strip()
        if process_listing:
            print(process_listing)
    finally:
        ssh.close()

    deadline = time.time() + 15.0
    payload = ""
    last_error: OSError | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((args.host, 8888), timeout=3.0) as sock:
                sock.sendall(b"STATUS\n")
                payload = sock.recv(4096).decode("utf-8", errors="ignore").strip()
                last_error = None
                break
        except OSError as error:
            last_error = error
            time.sleep(1.0)
    if last_error is not None:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(args.host, username=args.user, password=args.password, timeout=10)
        try:
            _, stdout, stderr = ssh.exec_command(f"bash -lc 'tail -n 80 {args.remote_root}/robot_runtime.log 2>/dev/null || true'")
            out = stdout.read().decode("utf-8", errors="ignore").strip()
            err = stderr.read().decode("utf-8", errors="ignore").strip()
            _, diag_stdout, diag_stderr = ssh.exec_command(
                "bash -lc "
                + repr(
                    "source /opt/ros/melodic/setup.bash >/dev/null 2>&1 || true; "
                    "source ~/ros/devel/setup.bash >/dev/null 2>&1 || true; "
                    "source ~/catkin_ws/devel/setup.bash >/dev/null 2>&1 || true; "
                    "find /home/hiwonder -maxdepth 4 -path '*/devel/setup.bash' 2>/dev/null | head -n 10; "
                    f"python3 --version; "
                    "python3 -c \"import sys; print(sys.path); import rospy; print(rospy.__file__)\"; "
                    f"ls -l {remote_runtime}; "
                    f"timeout 5s python3 {remote_runtime} --host 0.0.0.0 --port 8888"
                )
            )
            diag_out = diag_stdout.read().decode("utf-8", errors="ignore").strip()
            diag_err = diag_stderr.read().decode("utf-8", errors="ignore").strip()
        finally:
            ssh.close()
        if out:
            print(out)
        if err:
            print(err, file=sys.stderr)
        if diag_out:
            print(diag_out)
        if diag_err:
            print(diag_err, file=sys.stderr)
        raise RuntimeError(f"JetMax runtime did not start listening on 8888: {last_error}")
    print(payload)
    return 0


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    path = posixpath.normpath(remote_dir)
    parts = []
    while path not in ("", "/"):
        parts.append(posixpath.basename(path))
        path = posixpath.dirname(path)
    current = "/" if remote_dir.startswith("/") else ""
    for part in reversed(parts):
        current = posixpath.join(current, part) if current else part
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


if __name__ == "__main__":
    raise SystemExit(main())

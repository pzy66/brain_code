from __future__ import annotations

import argparse
import sys

import paramiko


DEFAULT_HOST = "192.168.149.1"
DEFAULT_USER = "hiwonder"
DEFAULT_PASSWORD = "hiwonder"


def run_remote(ssh: paramiko.SSHClient, label: str, command: str) -> None:
    print(f"\n===== {label} =====")
    _, stdout, stderr = ssh.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="ignore").strip()
    err = stderr.read().decode("utf-8", errors="ignore").strip()
    print(f"[exit] {exit_code}")
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe JetMax ROS/Python environment.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--runtime-path", default="/home/hiwonder/brain_code/hybrid_controller/robot/runtime/robot_runtime_py36.py")
    args = parser.parse_args(argv)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.host, username=args.user, password=args.password, timeout=10)
    try:
        run_remote(ssh, "whoami pwd", "bash -lc 'whoami; pwd'")
        run_remote(ssh, "bashrc", "bash -lc 'sed -n \"1,220p\" ~/.bashrc'")
        run_remote(
            ssh,
            "devel setup files",
            "bash -lc 'find /home/hiwonder -maxdepth 4 -path \"*/devel/setup.bash\" -print | sort'",
        )
        run_remote(
            ssh,
            "plain python3",
            "bash -lc 'python3 --version; python3 -c \"import sys; print(sys.executable); print(sys.path)\"'",
        )
        run_remote(
            ssh,
            "source opt ros only",
            "bash -lc 'source /opt/ros/melodic/setup.bash >/dev/null 2>&1 || true; "
            "python3 -c \"import sys; print(sys.path); import rospy; print(rospy.__file__)\"'",
        )
        run_remote(
            ssh,
            "source catkin_ws only",
            "bash -lc 'source /opt/ros/melodic/setup.bash >/dev/null 2>&1 || true; "
            "source ~/catkin_ws/devel/setup.bash >/dev/null 2>&1 || true; "
            "python3 -c \"import sys; print(sys.path); import rospy; print(rospy.__file__)\"'",
        )
        run_remote(
            ssh,
            "source ros ws only",
            "bash -lc 'source /opt/ros/melodic/setup.bash >/dev/null 2>&1 || true; "
            "source ~/ros/devel/setup.bash >/dev/null 2>&1 || true; "
            "python3 -c \"import sys; print(sys.path); import rospy; print(rospy.__file__)\"'",
        )
        run_remote(
            ssh,
            "source catkin_ws and ros ws",
            "bash -lc 'source /opt/ros/melodic/setup.bash >/dev/null 2>&1 || true; "
            "source ~/catkin_ws/devel/setup.bash >/dev/null 2>&1 || true; "
            "source ~/ros/devel/setup.bash >/dev/null 2>&1 || true; "
            "python3 -c \"import sys; print(sys.path); import rospy; print(rospy.__file__)\"'",
        )
        run_remote(
            ssh,
            "runtime direct start test",
            "bash -lc 'source /opt/ros/melodic/setup.bash >/dev/null 2>&1 || true; "
            "source ~/ros/devel/setup.bash >/dev/null 2>&1 || true; "
            "source ~/catkin_ws/devel/setup.bash >/dev/null 2>&1 || true; "
            f"timeout 5s python3 {args.runtime_path} --host 0.0.0.0 --port 8888'",
        )
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

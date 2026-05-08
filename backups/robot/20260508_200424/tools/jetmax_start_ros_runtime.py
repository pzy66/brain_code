from __future__ import annotations

import argparse
import json
import shlex
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import paramiko


DEFAULT_HOST = "192.168.149.1"
DEFAULT_USER = "hiwonder"
DEFAULT_PASSWORD = "hiwonder"
DEFAULT_REMOTE_ROOT = "/home/hiwonder/brain_code"
DEFAULT_ROSBRIDGE_PORT = 9091
DEFAULT_WEB_VIDEO_PORT = 8080
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_QUALITY = 80
DEFAULT_CAMERA_STREAM_TYPE = "mjpeg"
DEFAULT_CAMERA_IO_METHOD = "mmap"
DEFAULT_CAMERA_FRAMERATE = 20
DEFAULT_UVCVIDEO_CONF_PATH = "/etc/modprobe.d/hiwonder-uvcvideo.conf"
DEFAULT_UVCVIDEO_QUIRKS = 128
DEFAULT_UVCVIDEO_NODROP = 1
DEFAULT_UVCVIDEO_TIMEOUT = 5000
HIWONDER_CAMERA_TOPIC = "/usb_cam/image_rect_color"
HIWONDER_CAMERA_WIDTH = 640
HIWONDER_CAMERA_HEIGHT = 480
HIWONDER_CAMERA_QUALITY = 80
HIWONDER_CAMERA_STREAM_TYPE = "mjpeg"
# Locked JetMax/Hiwonder camera sender contract:
#   /dev/usb_cam0 -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080
# This tool must leave that sender untouched by default. Only explicit repair
# flags together with --allow-camera-sender-mutation may rewrite/restart it.
DEFAULT_CAMERA_STREAM_PATH = (
    f"/stream?topic={HIWONDER_CAMERA_TOPIC}"
    f"&type={HIWONDER_CAMERA_STREAM_TYPE}"
    f"&width={HIWONDER_CAMERA_WIDTH}"
    f"&height={HIWONDER_CAMERA_HEIGHT}"
    f"&quality={HIWONDER_CAMERA_QUALITY}"
)


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
    parser.add_argument(
        "--disable-autostart-rosbridge",
        action="store_true",
        default=False,
        help="Explicitly stop/disable JetMax rosbridge.service before starting the hybrid runtime.",
    )
    parser.add_argument("--keep-autostart-rosbridge", action="store_false", dest="disable_autostart_rosbridge")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help=(
            "Only repair/restart the official usb_cam.service camera sender. Requires "
            "--allow-camera-sender-mutation."
        ),
    )
    parser.add_argument(
        "--skip-camera-check",
        action="store_true",
        default=True,
        help="Skip 8080 port and official camera stream health checks.",
    )
    parser.add_argument(
        "--check-camera-stream",
        action="store_false",
        dest="skip_camera_check",
        help=(
            "Explicitly verify the official camera stream by subscribing/reading frames. "
            "Off by default so robot startup never pulls video."
        ),
    )
    parser.add_argument(
        "--skip-camera-repair",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The default already leaves the official usb_cam.service sender untouched; "
            "use --repair-camera-sender when an explicit repair/restart is required."
        ),
    )
    parser.add_argument(
        "--repair-camera-sender",
        action="store_true",
        help=(
            "Explicitly rewrite and restart the official usb_cam.service sender. Off by default so JetMax keeps its "
            "factory camera startup path."
        ),
    )
    parser.add_argument(
        "--skip-camera-driver-repair",
        action="store_true",
        help="Deprecated compatibility flag; driver repair is off by default.",
    )
    parser.add_argument(
        "--repair-camera-driver",
        action="store_true",
        help=(
            "Persist/reload JetMax uvcvideo compatibility options. Off by default so the camera sender "
            "matches the Hiwonder official usb_cam.service path."
        ),
    )
    parser.add_argument(
        "--keep-camera-driver-override",
        action="store_false",
        dest="remove_camera_driver_override",
        default=False,
        help=(
            "Deprecated compatibility flag. Driver override files are kept by default."
        ),
    )
    parser.add_argument(
        "--remove-camera-driver-override",
        action="store_true",
        dest="remove_camera_driver_override",
        help="Explicitly remove this tool's uvcvideo override file during a camera repair.",
    )
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--camera-quality", type=int, default=DEFAULT_CAMERA_QUALITY)
    parser.add_argument("--camera-stream-type", default=DEFAULT_CAMERA_STREAM_TYPE, choices=("mjpeg", "h264"))
    parser.add_argument("--camera-io-method", default=DEFAULT_CAMERA_IO_METHOD, choices=("read", "mmap", "userptr"))
    parser.add_argument(
        "--camera-framerate",
        type=int,
        default=DEFAULT_CAMERA_FRAMERATE,
        help="Set usb_cam framerate. The locked JetMax camera sender uses 20 FPS; use 0 only for manual diagnosis.",
    )
    parser.add_argument("--camera-driver-quirks", type=int, default=DEFAULT_UVCVIDEO_QUIRKS)
    parser.add_argument("--camera-driver-nodrop", type=int, default=DEFAULT_UVCVIDEO_NODROP, choices=(0, 1))
    parser.add_argument("--camera-driver-timeout", type=int, default=DEFAULT_UVCVIDEO_TIMEOUT)
    parser.add_argument("--camera-driver-conf-path", default=DEFAULT_UVCVIDEO_CONF_PATH)
    parser.add_argument(
        "--manage-web-video",
        action="store_true",
        help=(
            "Deprecated and refused. The hybrid runtime must not manage web_video_server; "
            "JetMax usb_cam.service owns camera output."
        ),
    )
    parser.add_argument(
        "--restart-web-video",
        action="store_true",
        help=(
            "Deprecated and refused. The hybrid runtime must not restart web_video_server; "
            "JetMax usb_cam.service owns camera output."
        ),
    )
    parser.add_argument(
        "--allow-camera-sender-mutation",
        action="store_true",
        help=(
            "Required with options that modify JetMax camera output, restart usb_cam.service, "
            "or reload uvcvideo."
        ),
    )
    parser.add_argument(
        "--require-tcp-check",
        action="store_true",
        help="Require legacy TCP runtime port 8888 to be reachable before reporting success.",
    )
    parser.add_argument("--skip-tcp-check", action="store_true")
    parser.add_argument(
        "--force-catkin-rebuild",
        action="store_true",
        help="Force rebuilding the hybrid_controller_ros catkin package on the JetMax.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.manage_web_video) or bool(args.restart_web_video):
        raise RuntimeError(
            "Refusing to manage JetMax web_video_server. Camera output is fixed to the official usb_cam.service path."
        )
    if _camera_sender_mutation_requested(args) and not bool(args.allow_camera_sender_mutation):
        raise RuntimeError(
            "Refusing to modify JetMax camera output. The camera sender is fixed to the Hiwonder official "
            "default path; pass --allow-camera-sender-mutation only for an intentional camera repair."
        )
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
        if args.camera_only:
            if args.skip_camera_repair:
                raise RuntimeError("--camera-only requires camera repair; remove --skip-camera-repair.")
            repair_official_camera_sender(ssh, args=args)
            return 0
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
        sudo = f"printf '%s\\n' {shlex.quote(str(args.password))} | sudo -S"
        if bool(args.repair_camera_sender) and not bool(args.skip_camera_repair):
            repair_official_camera_sender(ssh, args=args)
        elif bool(args.repair_camera_driver) and not bool(args.skip_camera_driver_repair):
            repair_uvcvideo_driver(ssh, args=args, sudo=sudo)
        elif bool(args.remove_camera_driver_override):
            remove_uvcvideo_override_file(ssh, args=args, sudo=sudo)
        run_remote_command(
            ssh,
            "source /opt/ros/melodic/setup.bash >/dev/null 2>&1; "
            "source ~/catkin_ws/devel/setup.bash >/dev/null 2>&1 || true; "
            "rosnode kill /hybrid_controller_runtime_node >/dev/null 2>&1 || true",
        )
        time.sleep(1.0)
        run_remote_command(
            ssh,
            "pkill -f hybrid_controller_runtime_node.py >/dev/null 2>&1 || true; "
            "pkill -f run_hybrid_controller_ros_runtime.sh >/dev/null 2>&1 || true; "
            "pkill -f robot_runtime_py36.py >/dev/null 2>&1 || true",
        )
        run_remote_command(ssh, f"rm -f {remote_log}")
        run_remote_command(
            ssh,
            f"cd {remote_robot_dir}; "
            f"ROSBRIDGE_PORT={int(args.rosbridge_port)} "
            "HYBRID_FORCE_RESTART_ROSBRIDGE=0 "
            f"HYBRID_FORCE_CATKIN_REBUILD={1 if bool(args.force_catkin_rebuild) else 0} "
            f"nohup bash run_hybrid_controller_ros_runtime.sh > {remote_log} 2>&1 < /dev/null &",
        )
        time.sleep(2.0)
        process_info = run_remote_command(
            ssh,
            "pgrep -af 'run_hybrid_controller_ros_runtime.sh|hybrid_controller_runtime_node.py|rosbridge_websocket|robot_runtime_py36.py' || true",
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
    if not bool(args.skip_camera_check):
        checks.append(PortCheck(name="web_video_server", port=int(args.web_video_port), required=True))
    require_tcp_check = bool(args.require_tcp_check) and not bool(args.skip_tcp_check)
    checks.append(PortCheck(name="tcp_runtime", port=8888, required=require_tcp_check))

    wait_for_ports(args.host, checks, timeout_sec=float(args.ready_timeout_sec))
    verify_runtime_services(
        host=args.host,
        user=args.user,
        password=args.password,
        timeout_sec=float(args.ssh_timeout_sec),
    )
    if not args.skip_camera_check:
        verify_official_camera_sender(
            host=args.host,
            user=args.user,
            password=args.password,
            timeout_sec=float(args.ssh_timeout_sec),
        )
        verify_web_video_mjpeg_stream(
            host=args.host,
            port=int(args.web_video_port),
            timeout_sec=max(1.0, float(args.ssh_timeout_sec)),
        )
    if require_tcp_check:
        status_payload = query_status(args.host, 8888, timeout_sec=5.0)
        print(status_payload)
    print("JetMax ROS runtime ready.")
    return 0


def _camera_sender_mutation_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            bool(args.camera_only),
            bool(args.repair_camera_sender),
            bool(args.repair_camera_driver),
            bool(args.remove_camera_driver_override),
        )
    )


def repair_official_camera_sender(ssh: paramiko.SSHClient, *, args: argparse.Namespace) -> None:
    config = {
        "launch_path": "/home/hiwonder/ros/autostart/usb_cam.launch",
        "video_device": "/dev/usb_cam0",
        "width": max(1, int(args.camera_width)),
        "height": max(1, int(args.camera_height)),
        "quality": max(1, min(100, int(args.camera_quality))),
        "stream_type": str(args.camera_stream_type or DEFAULT_CAMERA_STREAM_TYPE).strip().lower(),
        "io_method": str(args.camera_io_method or DEFAULT_CAMERA_IO_METHOD).strip().lower(),
        "framerate": max(0, int(args.camera_framerate)),
        "port": int(args.web_video_port),
    }
    remote_script_path = "/tmp/hybrid_repair_usb_cam.py"
    upload_text(ssh, remote_script_path, _camera_repair_script(config))
    print(run_remote_command(ssh, f"python3 {remote_script_path}", capture=True))
    sudo = f"printf '%s\\n' {shlex.quote(str(args.password))} | sudo -S"
    run_remote_command(ssh, f"{sudo} systemctl stop usb_cam.service >/dev/null 2>&1 || true", capture=True)
    clear_usb_cam_rosparams(ssh)
    repair_driver = bool(args.repair_camera_driver) and not bool(args.skip_camera_driver_repair)
    if repair_driver:
        persist_and_reload_uvcvideo(ssh, args=args, sudo=sudo)
    elif bool(args.remove_camera_driver_override):
        remove_uvcvideo_override_file(ssh, args=args, sudo=sudo)
    run_remote_command(ssh, f"{sudo} systemctl daemon-reload", capture=True)
    run_remote_command(ssh, f"{sudo} systemctl enable usb_cam.service >/dev/null", capture=True)
    run_remote_command(ssh, f"{sudo} systemctl start usb_cam.service", capture=True)
    time.sleep(4.0)
    status = run_remote_command(
        ssh,
        "systemctl is-active usb_cam.service; "
        "pgrep -af 'usb_cam_node|web_video_server|usb_cam.launch' || true; "
        "source /opt/ros/melodic/setup.bash >/dev/null 2>&1; "
        "printf '\\n[topics]\\n'; "
        "rostopic list 2>/dev/null | grep -E '^/usb_cam/(image_rect_color|camera_info)$' || true; "
        "printf '\\n[image_rect_color hz]\\n'; "
        "timeout 6 rostopic hz /usb_cam/image_rect_color 2>/dev/null || true",
        capture=True,
    )
    if status:
        print(status)


def repair_uvcvideo_driver(
    ssh: paramiko.SSHClient,
    *,
    args: argparse.Namespace,
    sudo: str,
) -> None:
    run_remote_command(ssh, f"{sudo} systemctl stop usb_cam.service >/dev/null 2>&1 || true", capture=True)
    persist_and_reload_uvcvideo(ssh, args=args, sudo=sudo)
    run_remote_command(ssh, f"{sudo} systemctl daemon-reload", capture=True)
    run_remote_command(ssh, f"{sudo} systemctl enable usb_cam.service >/dev/null", capture=True)
    run_remote_command(ssh, f"{sudo} systemctl start usb_cam.service", capture=True)
    time.sleep(4.0)
    status = run_remote_command(
        ssh,
        "systemctl is-active usb_cam.service; "
        "pgrep -af 'usb_cam_node|web_video_server|usb_cam.launch' || true",
        capture=True,
    )
    if status:
        print(status)


def clear_usb_cam_rosparams(ssh: paramiko.SSHClient) -> None:
    status = run_remote_command(
        ssh,
        "if [ -f /opt/ros/melodic/setup.bash ]; then "
        "source /opt/ros/melodic/setup.bash >/dev/null 2>&1; "
        "rosparam delete /usb_cam >/dev/null 2>&1 || true; "
        "echo '[usb_cam rosparams cleared]'; "
        "fi",
        capture=True,
    )
    if status:
        print(status)


def persist_and_reload_uvcvideo(
    ssh: paramiko.SSHClient,
    *,
    args: argparse.Namespace,
    sudo: str,
) -> None:
    conf_path = str(args.camera_driver_conf_path or DEFAULT_UVCVIDEO_CONF_PATH).strip()
    if not conf_path.startswith("/etc/modprobe.d/") or not conf_path.endswith(".conf"):
        raise ValueError(f"Refusing to write unexpected modprobe config path: {conf_path}")
    quirks = max(0, int(args.camera_driver_quirks))
    nodrop = 1 if int(args.camera_driver_nodrop) else 0
    timeout = max(0, int(args.camera_driver_timeout))
    remote_tmp = "/tmp/hiwonder-uvcvideo.conf"
    content = (
        "# Hiwonder JetMax USB camera stability for official usb_cam.service chain.\n"
        "# Keep camera output on the official path:\n"
        "# /dev/usb_cam0 -> usb_cam_node -> /usb_cam/image_rect_color -> web_video_server:8080\n"
        "# Required for the 32e6:9005 icspring UVC camera on this Jetson image.\n"
        f"options uvcvideo quirks={quirks} nodrop={nodrop} timeout={timeout}\n"
    )
    upload_text(ssh, remote_tmp, content)
    run_remote_command(ssh, f"{sudo} cp {shlex.quote(remote_tmp)} {shlex.quote(conf_path)}", capture=True)
    run_remote_command(ssh, f"{sudo} chmod 0644 {shlex.quote(conf_path)}", capture=True)
    run_remote_command(ssh, f"{sudo} modprobe -r uvcvideo >/dev/null 2>&1 || true", capture=True)
    time.sleep(2.0)
    run_remote_command(
        ssh,
        f"{sudo} modprobe uvcvideo quirks={quirks} nodrop={nodrop} timeout={timeout}",
        capture=True,
    )
    time.sleep(4.0)
    params = run_remote_command(
        ssh,
        "printf '[uvcvideo params]\\n'; "
        "for name in quirks nodrop timeout; do "
        "path=/sys/module/uvcvideo/parameters/$name; "
        "if [ -r \"$path\" ]; then printf '%s=' \"$name\"; cat \"$path\"; fi; "
        "done; "
        f"printf '\\n[uvcvideo conf]\\n'; cat {shlex.quote(conf_path)} 2>/dev/null || true",
        capture=True,
    )
    if params:
        print(params)


def remove_uvcvideo_override_file(
    ssh: paramiko.SSHClient,
    *,
    args: argparse.Namespace,
    sudo: str,
) -> None:
    conf_path = str(args.camera_driver_conf_path or DEFAULT_UVCVIDEO_CONF_PATH).strip()
    if not conf_path.startswith("/etc/modprobe.d/") or not conf_path.endswith(".conf"):
        raise ValueError(f"Refusing to remove unexpected modprobe config path: {conf_path}")
    run_remote_command(ssh, f"{sudo} rm -f {shlex.quote(conf_path)}", capture=True)
    params = run_remote_command(
        ssh,
        "printf '[uvcvideo override removed]\\n'; "
        "for name in quirks nodrop timeout; do "
        "path=/sys/module/uvcvideo/parameters/$name; "
        "if [ -r \"$path\" ]; then printf '%s=' \"$name\"; cat \"$path\"; fi; "
        "done; "
        f"printf '\\n[removed override file]\\n'; test ! -e {shlex.quote(conf_path)} && echo {shlex.quote(conf_path)}",
        capture=True,
    )
    if params:
        print(params)


def _camera_repair_script(config: dict[str, object]) -> str:
    payload = json.dumps(config, ensure_ascii=False)
    return f"""
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

CONFIG = {payload}

def indent(elem, level=0):
    pad = "\\n" + level * "  "
    child_pad = "\\n" + (level + 1) * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_pad
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad

def find_node(root, name):
    for node in root.findall("node"):
        if node.get("name") == name:
            return node
    return None

def ensure_param(node, name, value, param_type=None):
    for param in node.findall("param"):
        if param.get("name") == name:
            param.set("value", str(value))
            if param_type is not None:
                param.set("type", str(param_type))
            return param
    attrib = {{"name": name, "value": str(value)}}
    if param_type is not None:
        attrib["type"] = str(param_type)
    return ET.SubElement(node, "param", attrib)

def remove_param(node, name):
    for param in list(node.findall("param")):
        if param.get("name") == name:
            node.remove(param)

launch_path = Path(str(CONFIG["launch_path"]))
launch_path.parent.mkdir(parents=True, exist_ok=True)
backup_path = None
if launch_path.exists():
    backup_path = launch_path.with_name(launch_path.name + time.strftime(".bak-%Y%m%d-%H%M%S"))
    shutil.copy2(str(launch_path), str(backup_path))

try:
    tree = ET.parse(str(launch_path))
    root = tree.getroot()
    if root.tag != "launch":
        raise ValueError("root element is not <launch>")
except Exception:
    root = ET.Element("launch")
    ET.SubElement(root, "arg", {{"name": "camera_info_topic_name", "default": "/usb_cam/camera_info"}})
    tree = ET.ElementTree(root)

usb_node = find_node(root, "usb_cam")
if usb_node is None:
    usb_node = ET.SubElement(
        root,
        "node",
        {{"name": "usb_cam", "pkg": "usb_cam", "type": "usb_cam_node", "output": "screen", "respawn": "true", "respawn_delay": "2"}},
    )
ensure_param(usb_node, "video_device", CONFIG["video_device"])
ensure_param(usb_node, "image_width", CONFIG["width"])
ensure_param(usb_node, "image_height", CONFIG["height"])
ensure_param(usb_node, "pixel_format", "yuyv")
if int(CONFIG.get("framerate", 0) or 0) > 0:
    ensure_param(usb_node, "framerate", CONFIG["framerate"])
else:
    remove_param(usb_node, "framerate")
ensure_param(usb_node, "camera_frame_id", "usb_cam")
ensure_param(usb_node, "io_method", CONFIG["io_method"])

image_proc = find_node(root, "image_proc")
if image_proc is None:
    ET.SubElement(root, "node", {{"name": "image_proc", "pkg": "image_proc", "type": "image_proc", "ns": "usb_cam"}})

web_node = find_node(root, "web_video_server")
if web_node is None:
    web_node = ET.SubElement(
        root,
        "node",
        {{"name": "web_video_server", "pkg": "web_video_server", "type": "web_video_server", "output": "screen"}},
    )
ensure_param(web_node, "port", CONFIG["port"], "int")
ensure_param(web_node, "address", "0.0.0.0", "string")
ensure_param(web_node, "server_threads", "2", "int")
ensure_param(web_node, "ros_threads", "2", "int")
ensure_param(web_node, "width", CONFIG["width"], "int")
ensure_param(web_node, "height", CONFIG["height"], "int")
ensure_param(web_node, "quality", CONFIG["quality"], "int")
ensure_param(web_node, "type", CONFIG["stream_type"], "string")

indent(root)
tree.write(str(launch_path), encoding="utf-8", xml_declaration=False)
print("camera launch repaired: {{0}}".format(launch_path))
if backup_path is not None:
    print("camera launch backup: {{0}}".format(backup_path))
print("camera stream: port={{0}} type={{1}} size={{2}}x{{3}} quality={{4}} io={{5}} fps={{6}}".format(
    CONFIG["port"], CONFIG["stream_type"], CONFIG["width"], CONFIG["height"], CONFIG["quality"], CONFIG["io_method"],
    CONFIG["framerate"] if int(CONFIG.get("framerate", 0) or 0) > 0 else "factory-default"
))
"""


def upload_text(ssh: paramiko.SSHClient, remote_path: str, content: str) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, newline="\n") as handle:
        local_path = Path(handle.name)
        handle.write(content)
    try:
        sftp = ssh.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()
    finally:
        try:
            local_path.unlink()
        except OSError:
            pass


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
    _, stdout, stderr = ssh.exec_command("bash -lc " + shlex.quote(command))
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


def verify_web_video_mjpeg_stream(host: str, port: int, *, timeout_sec: float) -> None:
    # Camera contract: usb_cam.service owns /dev/usb_cam0 and web_video_server publishes
    # Hiwonder's official rectified color topic. Health checks must verify that path
    # without restarting or replacing the camera sender.
    request = (
        f"GET {DEFAULT_CAMERA_STREAM_PATH} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    deadline = time.time() + max(1.0, float(timeout_sec))
    payload = bytearray()
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.5, float(timeout_sec))) as sock:
            sock.settimeout(max(0.5, min(2.0, float(timeout_sec))))
            sock.sendall(request)
            while time.time() < deadline and len(payload) < 256_000:
                try:
                    chunk = sock.recv(8192)
                except socket.timeout:
                    break
                if not chunk:
                    break
                payload.extend(chunk)
                header_end = payload.find(b"\r\n\r\n")
                body = payload[header_end + 4 :] if header_end >= 0 else payload
                start = body.find(b"\xff\xd8")
                end = body.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
                if start >= 0 and end >= 0:
                    print(f"camera mjpeg stream ready on {host}:{port}")
                    return
    except OSError as error:
        raise RuntimeError(f"Cannot connect to JetMax web_video_server on {host}:{port}: {error}") from error
    raise RuntimeError(
        "JetMax web_video_server is reachable but did not return a complete MJPEG frame from "
        f"{DEFAULT_CAMERA_STREAM_PATH}."
    )


def verify_official_camera_sender(*, host: str, user: str, password: str, timeout_sec: float) -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password, timeout=float(timeout_sec))
    try:
        command = (
            "set -o pipefail; "
            "systemctl is-active --quiet usb_cam.service || { "
            "echo 'usb_cam.service is not active'; exit 11; }; "
            "pgrep -af '/opt/ros/melodic/lib/usb_cam/usb_cam_node' >/dev/null || { "
            "echo 'usb_cam_node is not running'; exit 12; }; "
            "pgrep -af '[w]eb_video_server' >/dev/null || { "
            "echo 'web_video_server is not running'; exit 13; }; "
            "source /opt/ros/melodic/setup.bash >/dev/null 2>&1; "
            f"rect_meta=$(timeout 6 rostopic echo -n 1 --noarr {shlex.quote(HIWONDER_CAMERA_TOPIC)} 2>&1) || rect_rc=$?; "
            "if [ -n \"${rect_rc:-}\" ]; then "
            f"echo 'no frame from {HIWONDER_CAMERA_TOPIC}'; echo \"$rect_meta\"; exit 14; "
            "fi; "
            "echo \"$rect_meta\" | grep -q 'encoding: \"rgb8\"' || { "
            "echo 'unexpected image_rect_color metadata'; echo \"$rect_meta\"; exit 15; }; "
            f"echo \"$rect_meta\" | grep -q 'height: {HIWONDER_CAMERA_HEIGHT}' || {{ "
            "echo 'unexpected image_rect_color height'; echo \"$rect_meta\"; exit 16; }; "
            f"echo \"$rect_meta\" | grep -q 'width: {HIWONDER_CAMERA_WIDTH}' || {{ "
            "echo 'unexpected image_rect_color width'; echo \"$rect_meta\"; exit 17; }; "
            f"echo 'official camera sender ready: {HIWONDER_CAMERA_TOPIC} rgb8 {HIWONDER_CAMERA_WIDTH}x{HIWONDER_CAMERA_HEIGHT}'; "
            f"timeout 4 rostopic hz {shlex.quote(HIWONDER_CAMERA_TOPIC)} 2>/dev/null | sed -n '1,4p' || true"
        )
        status = run_remote_command(ssh, command, capture=True)
        if status:
            print(status)
    finally:
        ssh.close()


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
            "/hybrid_controller/sucker_off",
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
        state_payload = run_remote_command(
            ssh,
            "source /opt/ros/melodic/setup.bash; "
            "source ~/catkin_ws/devel/setup.bash; "
            "timeout 8 rostopic echo -n 1 --noarr /hybrid_controller/state 2>/dev/null || true",
            capture=True,
        )
        if "state:" not in state_payload or "busy:" not in state_payload or "robot_ts:" not in state_payload:
            raise RuntimeError("/hybrid_controller/state did not publish a complete runtime state sample.")
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

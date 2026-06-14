#!/usr/bin/env bash
set -euo pipefail

# Run this on the JetMax itself when SSH is not available.
# It restores the Hiwonder camera sender path without changing resolution,
# frame rate, stream type, or JPEG quality.

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  sudo -v
  SUDO="sudo"
fi

LAUNCH_PATH="/home/hiwonder/ros/autostart/usb_cam.launch"
UVC_CONF="/etc/modprobe.d/hiwonder-uvcvideo.conf"

echo "[camera factory restore] remove hybrid uvcvideo override"
$SUDO rm -f "$UVC_CONF"

echo "[camera factory restore] repair $LAUNCH_PATH"
$SUDO python3 - <<'PY'
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

CONFIG = {
    "launch_path": "/home/hiwonder/ros/autostart/usb_cam.launch",
    "video_device": "/dev/usb_cam0",
    "width": 640,
    "height": 480,
    "pixel_format": "yuyv",
    "framerate": 20,
    "io_method": "mmap",
    "port": 8080,
    "stream_type": "mjpeg",
    "quality": 80,
}


def indent(elem, level=0):
    pad = "\n" + level * "  "
    child_pad = "\n" + (level + 1) * "  "
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
    attrib = {"name": name, "value": str(value)}
    if param_type is not None:
        attrib["type"] = str(param_type)
    return ET.SubElement(node, "param", attrib)


launch_path = Path(CONFIG["launch_path"])
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
    ET.SubElement(root, "arg", {"name": "camera_info_topic_name", "default": "/usb_cam/camera_info"})
    tree = ET.ElementTree(root)

usb_node = find_node(root, "usb_cam")
if usb_node is None:
    usb_node = ET.SubElement(
        root,
        "node",
        {"name": "usb_cam", "pkg": "usb_cam", "type": "usb_cam_node", "output": "screen", "respawn": "true", "respawn_delay": "2"},
    )
ensure_param(usb_node, "video_device", CONFIG["video_device"])
ensure_param(usb_node, "image_width", CONFIG["width"])
ensure_param(usb_node, "image_height", CONFIG["height"])
ensure_param(usb_node, "pixel_format", CONFIG["pixel_format"])
ensure_param(usb_node, "framerate", CONFIG["framerate"])
ensure_param(usb_node, "camera_frame_id", "usb_cam")
ensure_param(usb_node, "io_method", CONFIG["io_method"])

image_proc = find_node(root, "image_proc")
if image_proc is None:
    ET.SubElement(root, "node", {"name": "image_proc", "pkg": "image_proc", "type": "image_proc", "ns": "usb_cam"})

web_node = find_node(root, "web_video_server")
if web_node is None:
    web_node = ET.SubElement(
        root,
        "node",
        {"name": "web_video_server", "pkg": "web_video_server", "type": "web_video_server", "output": "screen"},
    )
ensure_param(web_node, "port", CONFIG["port"], "int")
ensure_param(web_node, "address", "0.0.0.0", "string")
ensure_param(web_node, "server_threads", 2, "int")
ensure_param(web_node, "ros_threads", 2, "int")
ensure_param(web_node, "width", CONFIG["width"], "int")
ensure_param(web_node, "height", CONFIG["height"], "int")
ensure_param(web_node, "quality", CONFIG["quality"], "int")
ensure_param(web_node, "type", CONFIG["stream_type"], "string")

indent(root)
tree.write(str(launch_path), encoding="utf-8", xml_declaration=False)
print("camera launch repaired:", launch_path)
if backup_path is not None:
    print("camera launch backup:", backup_path)
PY

echo "[camera factory restore] clear stale ROS params and reload uvcvideo"
bash -lc 'source /opt/ros/melodic/setup.bash >/dev/null 2>&1; rosparam delete /usb_cam >/dev/null 2>&1 || true'
$SUDO systemctl daemon-reload
$SUDO systemctl enable usb_cam.service >/dev/null
$SUDO systemctl stop usb_cam.service >/dev/null 2>&1 || true
$SUDO modprobe -r uvcvideo >/dev/null 2>&1 || true
sleep 2
$SUDO modprobe uvcvideo
sleep 2

$SUDO systemctl restart usb_cam.service
sleep 4

echo "[camera factory restore] status"
systemctl is-active usb_cam.service || true
pgrep -af 'usb_cam_node|web_video_server|usb_cam.launch' || true
bash -lc 'source /opt/ros/melodic/setup.bash >/dev/null 2>&1; rostopic list 2>/dev/null | grep -E "^/usb_cam/(image_rect_color|camera_info)$" || true'

echo "[camera factory restore] if the stream is still corrupt after this clean reload, reboot/power-cycle JetMax and the camera USB path."

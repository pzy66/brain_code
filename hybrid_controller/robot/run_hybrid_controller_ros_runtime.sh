#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
ROS_PACKAGE_ROOT="${SCRIPT_DIR}/ros_pkg/hybrid_controller_ros"
RUNTIME_NODE="${ROS_PACKAGE_ROOT}/scripts/hybrid_controller_runtime_node.py"
CATKIN_WS="${CATKIN_WS:-${HOME}/catkin_ws}"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9091}"
WEB_VIDEO_PORT="${WEB_VIDEO_PORT:-8080}"
ROSBRIDGE_PING_INTERVAL="${ROSBRIDGE_PING_INTERVAL:-10}"
ROSBRIDGE_PING_TIMEOUT="${ROSBRIDGE_PING_TIMEOUT:-30}"
ROSBRIDGE_RETRY_STARTUP_DELAY="${ROSBRIDGE_RETRY_STARTUP_DELAY:-2.0}"
ROSBRIDGE_USE_COMPRESSION="${ROSBRIDGE_USE_COMPRESSION:-false}"
HYBRID_FORCE_RESTART_ROSBRIDGE="${HYBRID_FORCE_RESTART_ROSBRIDGE:-1}"
HYBRID_FORCE_RESTART_WEB_VIDEO="${HYBRID_FORCE_RESTART_WEB_VIDEO:-1}"

if command -v hostname >/dev/null 2>&1; then
    DEFAULT_ROS_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
ROS_IP="${ROS_IP:-${DEFAULT_ROS_IP:-192.168.149.1}}"
ROS_HOSTNAME="${ROS_HOSTNAME:-${ROS_IP}}"

if [ -f /opt/ros/melodic/setup.bash ]; then
    source /opt/ros/melodic/setup.bash
fi

export ROS_IP
export ROS_HOSTNAME

mkdir -p "${CATKIN_WS}/src"
rm -rf "${CATKIN_WS}/src/hybrid_controller_ros"
cp -R "${ROS_PACKAGE_ROOT}" "${CATKIN_WS}/src/hybrid_controller_ros"

# Clean stale generated artifacts for this package before rebuilding.
rm -rf "${CATKIN_WS}/build/hybrid_controller_ros"
rm -rf "${CATKIN_WS}/devel/share/hybrid_controller_ros"
rm -rf "${CATKIN_WS}/devel/include/hybrid_controller_ros"
rm -rf "${CATKIN_WS}/devel/lib/python2.7/dist-packages/hybrid_controller_ros"
rm -rf "${CATKIN_WS}/devel/lib/python3/dist-packages/hybrid_controller_ros"

pushd "${CATKIN_WS}" >/dev/null
catkin_make --force-cmake --pkg hybrid_controller_ros
popd >/dev/null

source "${CATKIN_WS}/devel/setup.bash"
export HYBRID_CONTROLLER_REPO_ROOT="${PROJECT_ROOT}"

set +e
python3 - <<'PY'
import os
import socket

sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", int(os.environ.get("ROSBRIDGE_PORT", "9091"))))
except Exception:
    raise SystemExit(1)
finally:
    sock.close()
PY
ROSBRIDGE_RUNNING=$?
set -e

if [ "${HYBRID_FORCE_RESTART_ROSBRIDGE}" = "1" ]; then
    pkill -f rosbridge_websocket >/dev/null 2>&1 || true
    ROSBRIDGE_RUNNING=1
fi

if [ "${ROSBRIDGE_RUNNING}" -ne 0 ]; then
    nohup /opt/ros/melodic/bin/roslaunch rosbridge_server rosbridge_websocket.launch \
        port:="${ROSBRIDGE_PORT}" \
        websocket_ping_interval:="${ROSBRIDGE_PING_INTERVAL}" \
        websocket_ping_timeout:="${ROSBRIDGE_PING_TIMEOUT}" \
        retry_startup_delay:="${ROSBRIDGE_RETRY_STARTUP_DELAY}" \
        use_compression:="${ROSBRIDGE_USE_COMPRESSION}" \
        >/tmp/hybrid_rosbridge.log 2>&1 &
    sleep 2
fi

set +e
python3 - <<'PY'
import os
import socket

sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", int(os.environ.get("WEB_VIDEO_PORT", "8080"))))
except Exception:
    raise SystemExit(1)
finally:
    sock.close()
PY
WEB_VIDEO_RUNNING=$?
set -e

if [ "${HYBRID_FORCE_RESTART_WEB_VIDEO}" = "1" ]; then
    pkill -f web_video_server >/dev/null 2>&1 || true
    WEB_VIDEO_RUNNING=1
fi

if [ "${WEB_VIDEO_RUNNING}" -ne 0 ]; then
    if command -v rosrun >/dev/null 2>&1; then
        nohup rosrun web_video_server web_video_server _port:="${WEB_VIDEO_PORT}" >/tmp/hybrid_web_video.log 2>&1 &
        sleep 2
    fi
fi

python3 "${RUNTIME_NODE}"

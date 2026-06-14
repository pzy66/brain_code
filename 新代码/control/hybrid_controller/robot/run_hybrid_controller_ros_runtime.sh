#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
ROS_PACKAGE_ROOT="${SCRIPT_DIR}/ros_pkg/hybrid_controller_ros"
RUNTIME_NODE="${ROS_PACKAGE_ROOT}/scripts/hybrid_controller_runtime_node.py"
CATKIN_WS="${CATKIN_WS:-${HOME}/catkin_ws}"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9091}"
ROSBRIDGE_PING_INTERVAL="${ROSBRIDGE_PING_INTERVAL:-10}"
ROSBRIDGE_PING_TIMEOUT="${ROSBRIDGE_PING_TIMEOUT:-30}"
ROSBRIDGE_RETRY_STARTUP_DELAY="${ROSBRIDGE_RETRY_STARTUP_DELAY:-2.0}"
ROSBRIDGE_USE_COMPRESSION="${ROSBRIDGE_USE_COMPRESSION:-false}"
HYBRID_FORCE_RESTART_ROSBRIDGE="${HYBRID_FORCE_RESTART_ROSBRIDGE:-0}"
HYBRID_FORCE_CATKIN_REBUILD="${HYBRID_FORCE_CATKIN_REBUILD:-0}"

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
if command -v rsync >/dev/null 2>&1; then
    mkdir -p "${CATKIN_WS}/src/hybrid_controller_ros"
    rsync -a --delete "${ROS_PACKAGE_ROOT}/" "${CATKIN_WS}/src/hybrid_controller_ros/"
else
    rm -rf "${CATKIN_WS}/src/hybrid_controller_ros"
    cp -R "${ROS_PACKAGE_ROOT}" "${CATKIN_WS}/src/hybrid_controller_ros"
fi

PACKAGE_BUILD_DIR="${CATKIN_WS}/build/hybrid_controller_ros"
PACKAGE_DEVEL_DIR="${CATKIN_WS}/devel/share/hybrid_controller_ros"
GENERATED_PY_DIR="${CATKIN_WS}/devel/lib/python2.7/dist-packages/hybrid_controller_ros"
BUILD_STAMP="${PACKAGE_BUILD_DIR}/.hybrid_interface.sha256"
CURRENT_INTERFACE_HASH=""
if command -v sha256sum >/dev/null 2>&1; then
    CURRENT_INTERFACE_HASH="$(
        cd "${CATKIN_WS}/src/hybrid_controller_ros"
        {
            find msg srv -type f -print 2>/dev/null
            [ -f CMakeLists.txt ] && printf '%s\n' CMakeLists.txt
            [ -f package.xml ] && printf '%s\n' package.xml
        } | sort | xargs sha256sum | sha256sum | awk '{print $1}'
    )"
fi
BUILT_INTERFACE_HASH="$(cat "${BUILD_STAMP}" 2>/dev/null || true)"
if [ "${HYBRID_FORCE_CATKIN_REBUILD}" = "1" ]; then
    rm -rf "${PACKAGE_BUILD_DIR}"
    rm -rf "${PACKAGE_DEVEL_DIR}"
    rm -rf "${CATKIN_WS}/devel/include/hybrid_controller_ros"
    rm -rf "${GENERATED_PY_DIR}"
    rm -rf "${CATKIN_WS}/devel/lib/python3/dist-packages/hybrid_controller_ros"
fi

pushd "${CATKIN_WS}" >/dev/null
if [ "${HYBRID_FORCE_CATKIN_REBUILD}" = "1" ] || [ ! -d "${PACKAGE_BUILD_DIR}" ] || [ ! -d "${PACKAGE_DEVEL_DIR}" ] || [ ! -d "${GENERATED_PY_DIR}" ] || [ -z "${CURRENT_INTERFACE_HASH}" ] || [ "${CURRENT_INTERFACE_HASH}" != "${BUILT_INTERFACE_HASH}" ]; then
    catkin_make --pkg hybrid_controller_ros
    if [ -n "${CURRENT_INTERFACE_HASH}" ]; then
        mkdir -p "$(dirname "${BUILD_STAMP}")"
        printf '%s\n' "${CURRENT_INTERFACE_HASH}" > "${BUILD_STAMP}"
    fi
else
    echo "hybrid_controller_ros catkin build is current; skipping rebuild."
fi
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

python3 "${RUNTIME_NODE}"

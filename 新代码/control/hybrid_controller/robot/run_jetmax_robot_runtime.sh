#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"

if [ -f /opt/ros/melodic/setup.bash ]; then
    source /opt/ros/melodic/setup.bash
fi

if [ -f "${HOME}/ros/devel/setup.bash" ]; then
    source "${HOME}/ros/devel/setup.bash"
elif [ -f "${HOME}/catkin_ws/devel/setup.bash" ]; then
    source "${HOME}/catkin_ws/devel/setup.bash"
fi

python3 "${PACKAGE_ROOT}/robot/runtime/robot_runtime_py36.py" --host "${HOST}" --port "${PORT}"

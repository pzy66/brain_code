#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"

cd "${REPO_ROOT}"

if [ -f /opt/ros/melodic/setup.bash ]; then
    # JetMax official image uses ROS Melodic on Ubuntu 18.04.
    source /opt/ros/melodic/setup.bash
fi

if [ -f "${HOME}/ros/devel/setup.bash" ]; then
    source "${HOME}/ros/devel/setup.bash"
elif [ -f "${HOME}/catkin_ws/devel/setup.bash" ]; then
    source "${HOME}/catkin_ws/devel/setup.bash"
fi

python3 "${REPO_ROOT}/hybrid_controller/integrations/robot_runtime_py36.py" --host "${HOST}" --port "${PORT}"

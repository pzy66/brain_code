#!/usr/bin/env bash
set -u

DURATION="${1:-10}"

echo "=== JetMax Camera Probe ==="
echo "duration_sec=${DURATION}"
echo

echo "=== ROS Nodes ==="
rosnode list || true
echo

TOPIC="/usb_cam/image_rect_color"
echo "=== rostopic info ${TOPIC} ==="
rostopic info "${TOPIC}" || true
echo

echo "=== rostopic hz ${TOPIC} (${DURATION}s) ==="
timeout "${DURATION}" rostopic hz "${TOPIC}" || true
echo

echo "=== rostopic bw ${TOPIC} (${DURATION}s) ==="
timeout "${DURATION}" rostopic bw "${TOPIC}" || true
echo

echo "=== tegrastats (${DURATION}s) ==="
timeout "${DURATION}" tegrastats || true
echo

echo "=== top snapshot ==="
top -b -n 1 | head -n 30 || true
echo

echo "=== Done ==="

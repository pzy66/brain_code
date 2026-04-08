#!/usr/bin/env bash
set -u

DURATION="${1:-10}"

echo "=== JetMax Camera Probe ==="
echo "duration_sec=${DURATION}"
echo

echo "=== V4L2 Formats ==="
v4l2-ctl --list-formats-ext || true
echo

echo "=== ROS Nodes ==="
rosnode list || true
echo

for topic in /usb_cam/image_raw /usb_cam/image_rect_color; do
  echo "=== rostopic info ${topic} ==="
  rostopic info "${topic}" || true
  echo

  echo "=== rostopic hz ${topic} (${DURATION}s) ==="
  timeout "${DURATION}" rostopic hz "${topic}" || true
  echo

  echo "=== rostopic bw ${topic} (${DURATION}s) ==="
  timeout "${DURATION}" rostopic bw "${topic}" || true
  echo
done

echo "=== tegrastats (${DURATION}s) ==="
timeout "${DURATION}" tegrastats || true
echo

echo "=== top snapshot ==="
top -b -n 1 | head -n 30 || true
echo

echo "=== Done ==="

"""Suction end-effector rotation helpers for the JetMax runtime.

The public command angle is a logical grasp angle around 0 degrees. HiWonder's
servo convention is centered at 90 degrees, so the runtime maps the logical
angle into the physical PWM servo range before sending it to the robot.
"""

import math

DEFAULT_SERVO_CENTER_DEG = 90.0
DEFAULT_SERVO_MIN_DEG = 45.0
DEFAULT_SERVO_MAX_DEG = 135.0


def clamp_float(value, lower, upper):
    low = float(lower)
    high = float(upper)
    if low > high:
        low, high = high, low
    return max(low, min(high, float(value)))


def normalize_logical_sucker_angle_deg(angle_deg):
    """Normalize a grasp/wrist angle into the symmetric [-45, 45] range."""
    value = float(angle_deg)
    if not math.isfinite(value):
        raise ValueError("Sucker rotation angle must be finite.")
    while value <= -45.0:
        value += 90.0
    while value > 45.0:
        value -= 90.0
    return float(value)


def map_logical_to_servo_angle_deg(
    logical_angle_deg,
    offset_deg=0.0,
    invert=False,
    center_deg=DEFAULT_SERVO_CENTER_DEG,
    min_deg=DEFAULT_SERVO_MIN_DEG,
    max_deg=DEFAULT_SERVO_MAX_DEG,
):
    logical = normalize_logical_sucker_angle_deg(float(logical_angle_deg))
    if bool(invert):
        logical = -logical
    servo_angle = float(center_deg) + logical + float(offset_deg)
    return clamp_float(servo_angle, float(min_deg), float(max_deg))

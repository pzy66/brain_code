from enum import Enum


class TaskState(str, Enum):
    IDLE = "idle"
    S1_MI_MOVE = "s1_mi_move"
    S1_DECISION = "s1_decision"
    S2_TARGET_SELECT = "s2_target_select"
    S2_GRAB_CONFIRM = "s2_grab_confirm"
    S2_PICKING = "s2_picking"
    S3_MI_CARRY = "s3_mi_carry"
    S3_DECISION = "s3_decision"
    S3_PLACING = "s3_placing"
    FINISHED = "finished"
    ERROR = "error"

from enum import Enum


class RobotExecutorState(str, Enum):
    IDLE = "IDLE"
    MOVING_XY = "MOVING_XY"
    PICK_APPROACH = "PICK_APPROACH"
    PICK_SUCTION_ON = "PICK_SUCTION_ON"
    PICK_DESCEND = "PICK_DESCEND"
    PICK_LIFT = "PICK_LIFT"
    CARRY_READY = "CARRY_READY"
    PLACE_DESCEND = "PLACE_DESCEND"
    PLACE_RELEASE = "PLACE_RELEASE"
    PLACE_LIFT = "PLACE_LIFT"
    RECOVERING = "RECOVERING"
    ERROR = "ERROR"


class RobotErrorCode(str, Enum):
    ABORTED = "aborted"
    CALIBRATION_UNAVAILABLE = "calibration_unavailable"
    CALIBRATION_INVALID = "calibration_invalid"
    TARGET_OUT_OF_WORKSPACE = "target_out_of_workspace"
    INVALID_STATE = "invalid_state"
    BUSY = "busy"
    HARDWARE_FAILURE = "hardware_failure"
    RECOVER_FAILED = "recover_failed"
    CONNECTION_LOST = "connection_lost"


class RobotCommandError(RuntimeError):
    def __init__(self, code, message):
        self.code = code.value if isinstance(code, Enum) else str(code)
        self.message = str(message)
        super().__init__(self.message)

    def to_wire_payload(self):
        return "{0}: {1}".format(self.code, self.message)


def format_error_line(code, message):
    code_text = code.value if isinstance(code, Enum) else str(code)
    return "ERR {0}: {1}".format(code_text, message)


def is_busy_state(state):
    return state not in {RobotExecutorState.IDLE, RobotExecutorState.CARRY_READY, RobotExecutorState.ERROR}

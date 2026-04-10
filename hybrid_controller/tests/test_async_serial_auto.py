from __future__ import annotations

from hybrid_controller.ssvep import async_fbcca_idle as module


def test_normalize_and_auto_detection() -> None:
    assert module.normalize_serial_port("") == "auto"
    assert module.normalize_serial_port(None) == "auto"
    assert module.normalize_serial_port("COM4") == "COM4"
    assert module.serial_port_is_auto("auto")
    assert module.serial_port_is_auto("")
    assert not module.serial_port_is_auto("COM4")


def test_prepare_board_session_auto_retries_until_success() -> None:
    class FakeParams:
        def __init__(self) -> None:
            self.serial_port = ""

    class FakePortInfo:
        def __init__(self, device: str) -> None:
            self.device = device
            self.description = "USB Serial"
            self.hwid = "USB VID:PID=1234:5678"

    class FakeListPorts:
        @staticmethod
        def comports():
            return [FakePortInfo("COM4"), FakePortInfo("COM3")]

    class FakeBoard:
        attempts: list[str] = []

        def __init__(self, _board_id: int, params: FakeParams) -> None:
            self.port = params.serial_port

        def prepare_session(self) -> None:
            FakeBoard.attempts.append(self.port)
            if self.port != "COM3":
                raise RuntimeError("unable_to_open_port_error")

        def release_session(self) -> None:
            return None

    original_board = module.BoardShim
    original_params = module.BrainFlowInputParams
    original_list_ports = module.serial_list_ports
    try:
        module.BoardShim = FakeBoard
        module.BrainFlowInputParams = FakeParams
        module.serial_list_ports = FakeListPorts

        board, resolved, attempted = module.prepare_board_session(0, "auto")

        assert isinstance(board, FakeBoard)
        assert resolved == "COM3"
        assert attempted == ["COM4", "COM3"]
        assert FakeBoard.attempts == ["COM4", "COM3"]
    finally:
        module.BoardShim = original_board
        module.BrainFlowInputParams = original_params
        module.serial_list_ports = original_list_ports

from __future__ import annotations

import argparse
import json
import socketserver
import threading
import time
from pathlib import Path

from hybrid_controller.config import AppConfig
from hybrid_controller.observability.event_logger import EventLogger
from hybrid_controller.robot.runtime.robot_protocol import RobotErrorCode, format_error_line

from .simulation_world import SimulationWorld


class FakeRobotTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        world: SimulationWorld,
        *,
        logger: EventLogger | None = None,
    ) -> None:
        super().__init__(server_address, FakeRobotRequestHandler)
        self.world = world
        self.logger = logger
        self._connection_lost_used = False

    def log_world(self, reason: str) -> None:
        if self.logger is not None:
            self.logger.log_world_snapshot(self.world.snapshot(), reason=reason)

    def log_runtime(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log_runtime_status("fake_robot_server", message)


class FakeRobotRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.server.log_runtime(f"Client connected: {self.client_address}")
        try:
            while True:
                try:
                    raw = self.rfile.readline()
                except OSError:
                    return
                if not raw:
                    return
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self._dispatch(line)
        finally:
            self.server.log_runtime(f"Client disconnected: {self.client_address}")

    def _dispatch(self, line: str) -> None:
        self.server.log_runtime(f"RX <= {line}")
        parts = line.split()
        command = parts[0].upper()
        if command == "PING":
            self._send_line("ACK PONG")
            self.server.log_world("ping")
            return
        if command == "STATUS":
            if len(parts) != 1:
                self._send_line("ERR STATUS takes no arguments")
                return
            self.server.log_world("status")
            self._send_line(f"ACK STATUS {json.dumps(self.server.world.snapshot(), ensure_ascii=True)}")
            return
        if command == "ABORT":
            if len(parts) != 1:
                self._send_line("ERR ABORT takes no arguments")
                return
            self.server.world.abort_current_action(
                "Abort requested by operator.",
                code=RobotErrorCode.ABORTED,
            )
            self.server.log_world("abort")
            self._send_line("ACK ABORT")
            return
        if command == "RESET":
            if len(parts) != 1:
                self._send_line("ERR RESET takes no arguments")
                return
            self.server.world.reset_error()
            self.server.log_world("reset")
            self._send_line("ACK RESET")
            return
        if command == "MOVE":
            if len(parts) != 3:
                self._send_line("ERR MOVE requires x y")
                return
            result = self.server.world.handle_move(float(parts[1]), float(parts[2]))
            self.server.log_world("move")
            if result["status"] == "busy":
                self._send_line("BUSY")
                return
            if result["status"] == "error":
                self._send_line(format_error_line(result["code"], result["message"]))
                return
            threading.Thread(target=self._run_move, args=(float(result["duration_sec"]),), daemon=True).start()
            return
        if command == "MOVE_CYL":
            if len(parts) != 4:
                self._send_line("ERR MOVE_CYL requires theta r z")
                return
            result = self.server.world.handle_move_cyl(float(parts[1]), float(parts[2]), float(parts[3]))
            self.server.log_world("move_cyl")
            if result["status"] == "busy":
                self._send_line("BUSY")
                return
            if result["status"] == "error":
                self._send_line(format_error_line(result["code"], result["message"]))
                return
            threading.Thread(target=self._run_move, args=(float(result["duration_sec"]),), daemon=True).start()
            return
        if command == "MOVE_CYL_AUTO":
            if len(parts) != 3:
                self._send_line("ERR MOVE_CYL_AUTO requires theta r")
                return
            result = self.server.world.handle_move_cyl_auto(float(parts[1]), float(parts[2]))
            self.server.log_world("move_cyl_auto")
            if result["status"] == "busy":
                self._send_line("BUSY")
                return
            if result["status"] == "error":
                self._send_line(format_error_line(result["code"], result["message"]))
                return
            threading.Thread(target=self._run_move, args=(float(result["duration_sec"]),), daemon=True).start()
            return
        if command == "PICK":
            if len(parts) != 3:
                self._send_line("ERR PICK requires pixel_x pixel_y")
                return
            result = self.server.world.begin_pick(float(parts[1]), float(parts[2]))
            self.server.log_world("pick_started")
            if result["status"] == "busy":
                self._send_line("BUSY")
                return
            if result["status"] == "error":
                self._send_line(format_error_line(result["code"], result["message"]))
                return
            self._send_line("ACK PICK_STARTED")
            if self._maybe_drop_connection():
                self.server.world.abort_current_action("Simulated connection lost")
                self.server.log_world("connection_lost")
                return
            threading.Thread(target=self._run_pick, args=(float(result["delay_sec"]),), daemon=True).start()
            return
        if command == "PICK_WORLD":
            if len(parts) != 3:
                self._send_line("ERR PICK_WORLD requires x y")
                return
            result = self.server.world.begin_pick_world(float(parts[1]), float(parts[2]))
            self.server.log_world("pick_world_started")
            if result["status"] == "busy":
                self._send_line("BUSY")
                return
            if result["status"] == "error":
                self._send_line(format_error_line(result["code"], result["message"]))
                return
            self._send_line("ACK PICK_STARTED")
            if self._maybe_drop_connection():
                self.server.world.abort_current_action("Simulated connection lost")
                self.server.log_world("connection_lost")
                return
            threading.Thread(target=self._run_pick, args=(float(result["delay_sec"]),), daemon=True).start()
            return
        if command == "PICK_CYL":
            if len(parts) != 3:
                self._send_line("ERR PICK_CYL requires theta r")
                return
            result = self.server.world.begin_pick_cyl(float(parts[1]), float(parts[2]))
            self.server.log_world("pick_cyl_started")
            if result["status"] == "busy":
                self._send_line("BUSY")
                return
            if result["status"] == "error":
                self._send_line(format_error_line(result["code"], result["message"]))
                return
            self._send_line("ACK PICK_STARTED")
            if self._maybe_drop_connection():
                self.server.world.abort_current_action("Simulated connection lost")
                self.server.log_world("connection_lost")
                return
            threading.Thread(target=self._run_pick, args=(float(result["delay_sec"]),), daemon=True).start()
            return
        if command == "PLACE":
            if len(parts) != 1:
                self._send_line("ERR PLACE takes no arguments")
                return
            result = self.server.world.begin_place()
            self.server.log_world("place_started")
            if result["status"] == "busy":
                self._send_line("BUSY")
                return
            if result["status"] == "error":
                self._send_line(format_error_line(result["code"], result["message"]))
                return
            self._send_line("ACK PLACE_STARTED")
            if self._maybe_drop_connection():
                self.server.world.abort_current_action("Simulated connection lost")
                self.server.log_world("connection_lost")
                return
            threading.Thread(target=self._run_place, args=(float(result["delay_sec"]),), daemon=True).start()
            return
        self._send_line(f"ERR Unsupported command: {line}")

    def _run_pick(self, delay_sec: float) -> None:
        next_delay = delay_sec
        while True:
            time.sleep(next_delay)
            snapshot = self.server.world.snapshot()
            if snapshot.get("state") == "ERROR" and snapshot.get("last_error_code") == "aborted":
                return
            result = self.server.world.step_pick()
            self.server.log_world("pick_step")
            if result["status"] == "progress":
                next_delay = float(result["delay_sec"])
                continue
            if result["status"] == "done":
                self._send_line("ACK PICK_DONE")
                return
            self._send_line(format_error_line(result["code"], result["message"]))
            return

    def _run_move(self, delay_sec: float) -> None:
        time.sleep(delay_sec)
        while True:
            snapshot = self.server.world.snapshot()
            if snapshot.get("state") == "ERROR":
                if snapshot.get("last_error_code") == "aborted":
                    return
                self._send_line(
                    format_error_line(
                        snapshot.get("last_error_code") or "hardware_failure",
                        snapshot.get("last_error") or "Move failed.",
                    )
                )
                return
            if snapshot.get("state") != "MOVING_XY":
                break
            time.sleep(0.01)
        self._send_line("ACK MOVE")
        self._maybe_drop_connection()

    def _run_place(self, delay_sec: float) -> None:
        next_delay = delay_sec
        while True:
            time.sleep(next_delay)
            snapshot = self.server.world.snapshot()
            if snapshot.get("state") == "ERROR" and snapshot.get("last_error_code") == "aborted":
                return
            result = self.server.world.step_place()
            self.server.log_world("place_step")
            if result["status"] == "progress":
                next_delay = float(result["delay_sec"])
                continue
            if result["status"] == "done":
                self._send_line("ACK PLACE_DONE")
                return
            self._send_line(format_error_line(result["code"], result["message"]))
            return

    def _maybe_drop_connection(self) -> bool:
        if self.server.world.scenario_name != "connection_lost" or self.server._connection_lost_used:
            return False
        self.server._connection_lost_used = True
        try:
            self.request.shutdown(2)
        except OSError:
            pass
        try:
            self.request.close()
        except OSError:
            pass
        return True

    def _send_line(self, line: str) -> None:
        try:
            self.wfile.write((line + "\n").encode("utf-8"))
            self.wfile.flush()
            self.server.log_runtime(f"TX => {line}")
        except (OSError, ValueError):
            pass


class FakeRobotServer:
    def __init__(
        self,
        host: str,
        port: int,
        world: SimulationWorld,
        *,
        log_path: Path | str | None = None,
    ) -> None:
        self.logger = EventLogger(log_path) if log_path else None
        self.server = FakeRobotTCPServer((host, int(port)), world, logger=self.logger)
        self.host, self.port = self.server.server_address
        self._thread = threading.Thread(target=self.server.serve_forever, name="fake-robot-server", daemon=True)

    def start(self) -> None:
        self.server.log_runtime(f"Listening on {self.host}:{self.port}")
        self.server.log_world("server_start")
        self._thread.start()

    def stop(self) -> None:
        self.server.log_runtime("Stopping fake robot server")
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=1.0)

    def snapshot(self) -> dict[str, object]:
        return self.server.world.snapshot()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone fake robot server for control simulation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--timing-profile", choices=("formal", "fast"), default="formal")
    parser.add_argument(
        "--scenario-name",
        choices=(
            "basic",
            "pick_busy",
            "pick_error",
            "place_error",
            "invalid_pick_slot",
            "invalid_world_slot",
            "out_of_bounds_move",
            "connection_lost",
            "sparse_targets",
            "empty_roi",
        ),
        default="basic",
    )
    parser.add_argument("--vision-mode", choices=("slots", "fixed_world_slots", "fixed_cyl_slots"), default="slots")
    parser.add_argument("--slot-profile", default="default")
    parser.add_argument("--log-path", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = AppConfig(
        control_sim_enabled=True,
        sim_process_mode="dual",
        slot_profile=args.slot_profile,
        timing_profile=args.timing_profile,
        scenario_name=args.scenario_name,
        robot_mode="fake-remote",
        vision_mode=args.vision_mode,
    ).resolved()
    world = SimulationWorld(config, scenario_name=args.scenario_name)
    server = FakeRobotServer(args.host, args.port, world, log_path=args.log_path or None)
    server.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

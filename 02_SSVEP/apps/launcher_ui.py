from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.data_collection_ui import DatasetCollectionWindow
from apps.realtime_online_ui import RealtimeOnlineWindow
from apps.training_evaluation_ui import TrainingEvaluationWindow
from brain_workspace.paths import SSVEP_DATASET_DIR, SSVEP_PROFILE_DIR
from ssvep_core.async_fbcca_idle_standalone import DEFAULT_BOARD_ID, parse_freqs


class LauncherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP Launcher")
        self.resize(960, 520)
        self._child_windows: list[QWidget] = []

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        title = QLabel("02_SSVEP Main Launcher")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        subtitle = QLabel("Open the active collection, session-calibration, realtime, and evaluation tools.")
        subtitle.setWordWrap(True)
        info = QLabel(
            "\n".join(
                [
                    f"project_root={PROJECT_DIR}",
                    f"dataset_dir={SSVEP_DATASET_DIR}",
                    f"run_artifacts={PROJECT_DIR / 'artifacts' / 'runs'}",
                    f"profile_dir={SSVEP_PROFILE_DIR}",
                ]
            )
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #4b5563;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(info)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        layout.addLayout(grid)

        buttons = [
            ("Data Collection", "Collect SSVEP sessions into the configured local datasets root.", self._open_data_collection),
            ("Realtime Decode", "Load a profile and inspect realtime decisions and shadow output.", self._open_realtime),
            ("Training Eval", "Run and inspect training, reports, logs, and generated profiles.", self._open_training_eval),
            ("FBCCA Pretrain", "Calibrate FBCCA realtime thresholds from local collection data.", self._open_fbcca_threshold_pretrain),
            ("FBCCA Local Opt", "Research-only: optimize FBCCA profiles from copied local datasets.", self._open_fbcca_local_opt),
            ("TDCA Local Opt", "Research-only comparison task; not the current realtime mainline.", self._open_tdca_local_opt),
        ]

        for index, (label, desc, handler) in enumerate(buttons):
            card = QWidget(self)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 16, 16, 16)
            card_layout.setSpacing(10)
            card.setStyleSheet("border: 1px solid #d1d5db; border-radius: 8px;")

            button = QPushButton(label)
            button.setMinimumHeight(44)
            button.clicked.connect(handler)
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #4b5563;")
            card_layout.addWidget(button)
            card_layout.addWidget(desc_label)
            grid.addWidget(card, index // 2, index % 2)

        footer = QLabel("Realtime startup rejects research-only gate variants unless they are promoted by validation.")
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #6b7280;")
        layout.addWidget(footer)

    def _track_window(self, window: QWidget, *, fullscreen: bool = False) -> QWidget:
        self._child_windows.append(window)

        def _release(*_args) -> None:
            self._child_windows[:] = [item for item in self._child_windows if item is not window]

        window.destroyed.connect(_release)
        if bool(fullscreen):
            window.showFullScreen()
        else:
            window.show()
        window.raise_()
        window.activateWindow()
        return window

    def _open_data_collection(self) -> None:
        window = DatasetCollectionWindow(serial_port="auto", board_id=DEFAULT_BOARD_ID, freqs=parse_freqs("8,10,12,15"))
        window.dataset_dir_edit.setText(str(SSVEP_DATASET_DIR))
        self._track_window(window)

    def _open_realtime(self) -> None:
        window = RealtimeOnlineWindow(serial_port="auto", board_id=DEFAULT_BOARD_ID, freqs=parse_freqs("8,10,12,15"))
        self._track_window(window, fullscreen=True)

    def _open_training_eval(self) -> None:
        self._track_window(TrainingEvaluationWindow())

    def _open_tdca_local_opt(self) -> None:
        window = TrainingEvaluationWindow()
        window.configure_tdca_local_opt_mode(auto_start=False)
        self._track_window(window)

    def _open_fbcca_local_opt(self) -> None:
        window = TrainingEvaluationWindow()
        window.configure_fbcca_local_opt_mode(auto_start=False)
        self._track_window(window)

    def _open_fbcca_threshold_pretrain(self) -> None:
        window = TrainingEvaluationWindow()
        window.configure_fbcca_threshold_pretrain_mode(auto_start=False)
        self._track_window(window)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = argv
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = LauncherWindow()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from apps.data_collection_ui import DatasetCollectionWindow
from apps.realtime_online_ui import RealtimeOnlineWindow
from apps.training_evaluation_ui import TrainingEvaluationWindow
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

        title = QLabel("02_SSVEP 主线入口")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        subtitle = QLabel("从这里直接进入采集、实时解码、训练评测和 TDCA 本地异步优化。")
        subtitle.setWordWrap(True)
        info = QLabel(
            "\n".join(
                [
                    f"项目根目录：{PROJECT_DIR}",
                    f"数据集目录：{PROJECT_DIR / 'artifacts' / 'datasets'}",
                    f"运行产物目录：{PROJECT_DIR / 'artifacts' / 'runs'}",
                    f"部署 profile：{PROJECT_DIR / 'artifacts' / 'deployed_profiles'}",
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
            ("数据采集", "按协议采集 session，并直接落到 artifacts/datasets。", self._open_data_collection),
            ("实时在线解码", "加载 deployed profile，在线查看实时判定与 shadow 输出。", self._open_realtime),
            ("训练评测", "统一查看 run 目录、日志、进度、报告和 profile 产物。", self._open_training_eval),
            ("TDCA 本地异步优化", "直接进入 TDCA 本地优化模式，保留实时进度与 run 归档。", self._open_tdca_local_opt),
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

        footer = QLabel("所有新运行都按 run 目录归档；旧代码与旧产物统一进 _archive 和 legacy_imported。")
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #6b7280;")
        layout.addWidget(footer)

    def _track_window(self, window: QWidget) -> QWidget:
        self._child_windows.append(window)

        def _release(*_args) -> None:
            self._child_windows[:] = [item for item in self._child_windows if item is not window]

        window.destroyed.connect(_release)
        window.show()
        window.raise_()
        window.activateWindow()
        return window

    def _open_data_collection(self) -> None:
        window = DatasetCollectionWindow(serial_port="auto", board_id=DEFAULT_BOARD_ID, freqs=parse_freqs("8,10,12,15"))
        window.dataset_dir_edit.setText(str(PROJECT_DIR / "artifacts" / "datasets"))
        self._track_window(window)

    def _open_realtime(self) -> None:
        window = RealtimeOnlineWindow(serial_port="auto", board_id=DEFAULT_BOARD_ID, freqs=parse_freqs("8,10,12,15"))
        self._track_window(window)

    def _open_training_eval(self) -> None:
        self._track_window(TrainingEvaluationWindow())

    def _open_tdca_local_opt(self) -> None:
        window = TrainingEvaluationWindow()
        window.configure_tdca_local_opt_mode(auto_start=False)
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

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ExternalReplayViewer(QMainWindow):
    def __init__(self, report_path: Path) -> None:
        super().__init__()
        self.report_path = Path(report_path).expanduser().resolve()
        self.payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.timeline = list(self.payload.get("replay_timeline_board", []) or [])
        self.setWindowTitle(f"External Replay Viewer - {self.report_path.name}")
        self.resize(1000, 760)

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        summary_form = QFormLayout()
        self.subject_label = QLabel(str(self.payload.get("subject", "")))
        self.status_label = QLabel(str(self.payload.get("status", "")))
        self.variant_label = QLabel(
            f"{self.payload.get('fbcca_variant', '')} | conf={self.payload.get('confidence_variant', '')}"
        )
        self.session_label = QLabel("")
        self.time_label = QLabel("")
        self.trial_type_label = QLabel("")
        self.pred_label = QLabel("")
        self.p_correct_label = QLabel("")
        self.selected_label = QLabel("")
        self.gate_label = QLabel("")
        self.commit_label = QLabel("")
        summary_form.addRow("Subject", self.subject_label)
        summary_form.addRow("Status", self.status_label)
        summary_form.addRow("Variant", self.variant_label)
        summary_form.addRow("Session", self.session_label)
        summary_form.addRow("Time", self.time_label)
        summary_form.addRow("Trial", self.trial_type_label)
        summary_form.addRow("Top1", self.pred_label)
        summary_form.addRow("p_correct", self.p_correct_label)
        summary_form.addRow("Selected", self.selected_label)
        summary_form.addRow("Gate", self.gate_label)
        summary_form.addRow("Commit", self.commit_label)
        layout.addLayout(summary_form)

        slider_row = QHBoxLayout()
        self.index_label = QLabel("0 / 0")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(len(self.timeline) - 1, 0))
        self.slider.valueChanged.connect(self._update_row)
        slider_row.addWidget(QLabel("Timeline"))
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.index_label)
        layout.addLayout(slider_row)

        self.row_text = QPlainTextEdit()
        self.row_text.setReadOnly(True)
        layout.addWidget(self.row_text, 1)
        self._update_row(0)

    def _update_row(self, index: int) -> None:
        if not self.timeline:
            self.index_label.setText("0 / 0")
            self.session_label.setText("-")
            self.time_label.setText("-")
            self.trial_type_label.setText("-")
            self.pred_label.setText("-")
            self.p_correct_label.setText("-")
            self.selected_label.setText("-")
            self.gate_label.setText("-")
            self.commit_label.setText("-")
            self.row_text.setPlainText("replay_timeline_board is empty")
            return
        safe_index = max(0, min(int(index), len(self.timeline) - 1))
        row = dict(self.timeline[safe_index])
        self.index_label.setText(f"{safe_index + 1} / {len(self.timeline)}")
        self.session_label.setText(f"{row.get('session_id', '')} (#{row.get('session_index', '')})")
        self.time_label.setText(f"{float(row.get('time_sec', 0.0)):.3f}s")
        self.trial_type_label.setText(str(row.get("current_trial_type", "")))
        self.pred_label.setText("" if row.get("pred_freq") is None else f"{float(row.get('pred_freq')):g} Hz")
        self.p_correct_label.setText(f"{float(row.get('p_correct', 0.0)):.4f}")
        self.selected_label.setText(
            "-"
            if row.get("selected_freq") is None
            else f"{float(row.get('selected_freq')):g} Hz"
        )
        gate_text = f"{row.get('gate_event', '')} | open={bool(row.get('gate_is_open', False))}"
        if row.get("gate_open_freq") is not None:
            gate_text += f" | freq={float(row.get('gate_open_freq')):g} Hz"
        self.gate_label.setText(gate_text)
        self.commit_label.setText(
            f"commit={bool(row.get('commit', False))} | "
            f"commit_freq={row.get('commit_freq', None)} | tracked={row.get('tracked_freq', None)}"
        )
        self.row_text.setPlainText(json.dumps(row, ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="View external replay timeline from a report.json")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication([])
        created = True
    window = ExternalReplayViewer(args.report)
    window.show()
    if created:
        return int(app.exec_())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

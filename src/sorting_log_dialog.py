# -*- coding: utf-8 -*-
"""sorting_log_dialog.py —— 分拣日志查看弹窗。"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox,
)


class SortingLogDialog(QDialog):
    """只读日志弹窗，用于查看主界面累计的分拣日志。"""

    DIALOG_QSS = """
        QDialog { background: #f5f7fa; }
        QTextEdit {
            background: #ffffff;
            border: 1px solid #d9e0e8;
            border-radius: 8px;
            color: #1f2937;
            font-family: "Menlo", "Monaco", "Microsoft YaHei", monospace;
            font-size: 12px;
            padding: 10px;
        }
        QPushButton {
            background: #eef3f8;
            border: 1px solid #cfd8e3;
            border-radius: 6px;
            padding: 7px 14px;
            min-height: 30px;
        }
        QPushButton:hover { background: #e0ebf6; }
    """

    def __init__(self, parent, log_text):
        super().__init__(parent)
        self.setWindowTitle("分拣日志")
        self.setModal(False)
        self.resize(640, 460)
        self.setMinimumSize(420, 300)
        self.setStyleSheet(self.DIALOG_QSS)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QTextEdit.NoWrap)
        self.set_log_text(log_text)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, Qt.Horizontal)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout()
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)
        root.addWidget(self.txt_log, 1)
        root.addWidget(buttons)
        self.setLayout(root)

    def set_log_text(self, log_text):
        text = log_text or "暂无分拣日志"
        self.txt_log.setPlainText(text)
        cursor = self.txt_log.textCursor()
        cursor.movePosition(cursor.End)
        self.txt_log.setTextCursor(cursor)

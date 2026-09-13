"""透明答案悬浮窗。

悬浮窗必须保持鼠标穿透，并尽量从 Windows 屏幕捕获中排除，避免自身文字
进入下一轮 OCR。
"""
import ctypes
import logging
import sys
from typing import Dict

from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


logger = logging.getLogger(__name__)


class AnswerOverlay(QWidget):
    """显示答案、解析和运行状态的置顶悬浮窗。"""

    _show_answer_signal = pyqtSignal(dict)
    _show_message_signal = pyqtSignal(str)
    _update_status_signal = pyqtSignal(str)

    def __init__(self) -> None:
        """创建无边框、透明且不可交互的窗口。"""
        super().__init__()
        flags = (
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowTransparentForInput
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._border_radius = 10
        self._width = 380

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(7)

        self.answer_label = QLabel("准备就绪")
        self.answer_label.setFont(QFont("Microsoft YaHei", 19, QFont.Bold))
        self.answer_label.setStyleSheet(
            "color: #66F2A3; background: transparent;"
        )
        self.answer_label.setAlignment(Qt.AlignCenter)
        self.answer_label.setWordWrap(True)

        self.detail_label = QLabel("")
        self.detail_label.setFont(QFont("Microsoft YaHei", 10))
        self.detail_label.setStyleSheet(
            "color: #F5F5F5; background: transparent;"
        )
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("状态：等待扫描")
        self.status_label.setFont(QFont("Microsoft YaHei", 8))
        self.status_label.setStyleSheet(
            "color: #B8C1CC; background: transparent;"
        )
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.answer_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self.setFixedWidth(self._width)

        self._detail_clear_timer = QTimer(self)
        self._detail_clear_timer.setSingleShot(True)
        self._detail_clear_timer.timeout.connect(self._clear_detail)
        self._persistent_status = "状态：等待扫描"

        self._show_answer_signal.connect(self._do_show_answer)
        self._show_message_signal.connect(self._do_show_message)
        self._update_status_signal.connect(self._do_update_status)

        self._reposition()
        self.show()
        self._exclude_from_capture()

    def _reposition(self) -> None:
        """把窗口固定在主屏左下角。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        x = available.left() + 24
        y = available.bottom() - self.height() - 60
        self.move(x, y)

    def _exclude_from_capture(self) -> None:
        """Windows 10/11 下尽量从屏幕捕获中排除自身。"""
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            # WDA_EXCLUDEFROMCAPTURE = 0x11，失败时保留普通透明窗口行为。
            if not ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11):
                logger.debug("当前系统不支持从屏幕捕获中排除悬浮窗")
        except Exception as exc:
            logger.debug("设置悬浮窗捕获排除失败: %s", exc)

    def paintEvent(self, _event) -> None:
        """绘制半透明深色圆角背景。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(12, 18, 28, 220)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(
            self.rect(),
            self._border_radius,
            self._border_radius,
        )

    def show_answer(self, answer_data: Dict) -> None:
        """线程安全地提交答案更新。"""
        self._show_answer_signal.emit(answer_data)

    def show_message(self, message: str) -> None:
        """线程安全地显示临时消息。"""
        self._show_message_signal.emit(message)

    def update_status(self, status: str) -> None:
        """线程安全地更新持久状态。"""
        self._update_status_signal.emit(status)

    def _do_show_answer(self, answer_data: Dict) -> None:
        """刷新答案、类型、选项文字和来源信息。"""
        answer = str(answer_data.get("answer", "?"))
        valid = bool(answer_data.get("is_valid", answer != "?"))
        question_type = str(answer_data.get("question_type_label", ""))
        texts = answer_data.get("texts") or []
        source = str(answer_data.get("source", ""))
        detail = str(answer_data.get("detail", ""))
        confidence = answer_data.get("confidence")

        if valid:
            self.answer_label.setText("答案：%s" % answer)
            self.answer_label.setStyleSheet(
                "color: #66F2A3; background: transparent;"
            )
        else:
            self.answer_label.setText("需人工确认")
            self.answer_label.setStyleSheet(
                "color: #FFB86B; background: transparent;"
            )

        detail_parts = []
        if question_type:
            detail_parts.append(question_type)
        if texts:
            detail_parts.append("选项：" + "；".join(str(text) for text in texts))
        if source:
            detail_parts.append("来源：" + source)
        if confidence is not None:
            detail_parts.append("OCR %.0f%%" % (float(confidence) * 100))
        if detail:
            detail_parts.append(detail)
        self.detail_label.setText("\n".join(detail_parts))

        self._detail_clear_timer.stop()
        self._detail_clear_timer.start(9000)
        self._resize_to_content()

    def _do_show_message(self, message: str) -> None:
        """显示短时状态消息。"""
        self.detail_label.setText(message)
        self._detail_clear_timer.stop()
        self._detail_clear_timer.start(2500)
        self._resize_to_content()

    def _do_update_status(self, status: str) -> None:
        """保存并显示持续状态。"""
        self._persistent_status = status
        self.status_label.setText(status)
        self._resize_to_content()

    def _resize_to_content(self) -> None:
        """限制最大高度并重新定位。"""
        self.setFixedWidth(self._width)
        self.adjustSize()
        if self.height() > 360:
            self.setFixedHeight(360)
        else:
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
        self._reposition()

    def _clear_detail(self) -> None:
        """自动清除临时解析，保留当前答案。"""
        self.detail_label.setText("")
        self._resize_to_content()


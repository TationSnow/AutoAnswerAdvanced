"""悬浮窗 - 显示答案/状态，定位左下角，修复定时器重叠。"""
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush


class AnswerOverlay(QWidget):
    _show_answer_signal = pyqtSignal(dict)
    _show_message_signal = pyqtSignal(str)
    _update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._border_radius = 12

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)

        self.answer_label = QLabel("准备就绪")
        self.answer_label.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        self.answer_label.setStyleSheet("color: #00FF88; background: transparent;")
        self.answer_label.setAlignment(Qt.AlignCenter)

        self.detail_label = QLabel("")
        self.detail_label.setFont(QFont("Microsoft YaHei", 10))
        self.detail_label.setStyleSheet("color: #FFFFFF; background: transparent;")
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setMinimumHeight(0)

        self.status_label = QLabel("状态：等待扫描")
        self.status_label.setFont(QFont("Microsoft YaHei", 8))
        self.status_label.setStyleSheet("color: #AAAAAA; background: transparent;")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.answer_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        # 固定宽度，高度自适应
        self.setFixedWidth(320)
        self.resize(320, 90)

        # 单例定时器：清空 detail
        self._detail_clear_timer = QTimer(self)
        self._detail_clear_timer.setSingleShot(True)
        self._detail_clear_timer.timeout.connect(self._clear_detail)

        # 单例定时器：恢复 status
        self._status_restore_timer = QTimer(self)
        self._status_restore_timer.setSingleShot(True)
        self._status_restore_timer.timeout.connect(self._restore_status)
        self._persistent_status = "状态：等待扫描"

        # 信号
        self._show_answer_signal.connect(self._do_show_answer)
        self._show_message_signal.connect(self._do_show_message)
        self._update_status_signal.connect(self._do_update_status)

        self._reposition()
        self.show()

    # ------------------------------------------------------------------
    def _reposition(self):
        """定位到屏幕左下角。"""
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.left() + 30
        y = screen.bottom() - self.height() - 80
        self.move(x, y)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(QColor(0, 0, 0, 190)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), self._border_radius, self._border_radius)

    # ------------------------------------------------------------------
    def show_answer(self, answer_data: dict):
        self._show_answer_signal.emit(answer_data)

    def show_message(self, message: str):
        self._show_message_signal.emit(message)

    def update_status(self, status: str):
        self._update_status_signal.emit(status)

    # ------------------------------------------------------------------
    def _do_show_answer(self, answer_data: dict):
        answer = answer_data.get("answer", "?")
        detail = answer_data.get("detail", "")
        source = answer_data.get("source", "")
        self.answer_label.setText(f"答案：{answer}")
        if source:
            detail = f"[{source}] {detail}"
        self.detail_label.setText(detail)

        # 重启清理定时器（避免多个定时器互相清空文本）
        self._detail_clear_timer.stop()
        self._detail_clear_timer.start(6000)

        self.adjustSize()
        self.setFixedWidth(320)
        self._reposition()

    def _do_show_message(self, message: str):
        """临时消息显示在 detail 上，并自动清除。"""
        self.detail_label.setText(message)
        self._detail_clear_timer.stop()
        self._detail_clear_timer.start(2500)
        self.adjustSize()
        self.setFixedWidth(320)
        self._reposition()

    def _do_update_status(self, status: str):
        self._persistent_status = status
        self.status_label.setText(status)

    def _clear_detail(self):
        self.detail_label.setText("")
        self.adjustSize()
        self.setFixedWidth(320)
        self._reposition()

    def _restore_status(self):
        self.status_label.setText(self._persistent_status)
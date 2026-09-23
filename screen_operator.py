"""屏幕自动化操作。

除点击外，本模块还提供**滑动手势**原语：部分答题平台没有“上一题/下一题”
按钮，只能靠水平拖动来翻页。手势相关的安全约束：

1. 拖动起止点始终落在捕获区域内（并留出边缘余量），避免拖到答题选项、
   悬浮按钮或窗口边缘上造成误操作；
2. 手势期间临时关闭 ``pyautogui.FAILSAFE``，避免用户鼠标恰好停在屏幕角落时
   中途抛异常、导致左键被一直按住；
3. 无论中途是否异常，都必须松开鼠标按键；
4. 关闭自动点击时保持干跑（DRY-RUN），只打日志不产生真实输入。
"""
import logging
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from models import SwipeDirection, SwipeProfile

logger = logging.getLogger(__name__)

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    _HAS_PYAUTOGUI = True
except Exception as e:  # 依赖缺失或环境限制
    logger.error(f"pyautogui 加载失败: {e}")
    _HAS_PYAUTOGUI = False


@contextmanager
def _failsafe_suspended() -> Iterator[None]:
    """在受控的手势区间内临时关闭 pyautogui 的角落保护。

    ``FAILSAFE`` 会在鼠标位于屏幕角落时让**每一次** pyautogui 调用抛异常。
    如果用户把鼠标停在角落，拖拽会在中途被中断，紧接着的 ``mouseUp`` 也会失败，
    鼠标左键就会一直被按住。这里在自动手势期间临时关闭该保护，结束后无条件恢复。
    """
    if not _HAS_PYAUTOGUI:
        yield
        return
    previous = bool(getattr(pyautogui, "FAILSAFE", False))
    pyautogui.FAILSAFE = False
    try:
        yield
    finally:
        pyautogui.FAILSAFE = previous


class ScreenOperator:
    """在捕获区域内执行点击与滑动手势。"""

    #: 手势纵向锚点默认值（占捕获区域高度比例），提取为常量便于统一调整。
    DEFAULT_ANCHOR_RATIO = 0.5
    #: 拖动起止点距捕获区域左右边缘的留白比例，避免贴边触发窗口级手势。
    EDGE_MARGIN_RATIO = 0.05
    #: 滑动起点的横向位置比例：左滑从右侧起手，右滑从左侧起手。
    START_RATIO = 0.85
    #: 有效拖动的像素下限，低于该值视为配置异常，不产生手势。
    MIN_SWIPE_PIXELS = 8

    def __init__(self, capture_region: Dict, enabled: bool = True):
        self.region = dict(capture_region or {})
        self.enabled = enabled and _HAS_PYAUTOGUI

    def update_region(self, region: Dict):
        self.region = dict(region or {})

    # ------------------------------------------------------------------
    def _to_screen(self, local_x: int, local_y: int) -> Tuple[int, int]:
        left = int(self.region.get('left', 0))
        top = int(self.region.get('top', 0))
        return left + int(local_x), top + int(local_y)

    # ------------------------------------------------------------------
    def click_at(self, local_center: Tuple[int, int]) -> bool:
        if not self.enabled:
            logger.info(f"[DRY-RUN] 本应点击 local={local_center}")
            return False
        try:
            x, y = self._to_screen(*local_center)
            pyautogui.moveTo(x, y, duration=random.uniform(0.12, 0.28))
            time.sleep(random.uniform(0.05, 0.15))
            pyautogui.click()
            logger.info(f"已点击屏幕坐标 ({x}, {y}) [local {local_center}]")
            return True
        except Exception as e:
            logger.error(f"点击失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 滑动手势
    # ------------------------------------------------------------------
    def swipe(
        self,
        direction: Any,
        profile: Optional[SwipeProfile] = None,
    ) -> bool:
        """在捕获区域内执行一次水平拖拽手势。

        :param direction: :class:`SwipeDirection` 或 ``"left"`` / ``"right"``。
        :param profile: 手势档位（距离/锚点/时长/步数）；``None`` 时使用默认档位。
        :return: 是否真的执行了手势；干跑模式、区域非法或异常时返回 ``False``。
        """
        shape = profile or SwipeProfile(name="默认", anchor_ratio=self.DEFAULT_ANCHOR_RATIO)
        try:
            gesture_direction = (
                direction
                if isinstance(direction, SwipeDirection)
                else SwipeDirection(str(direction).strip().lower())
            )
        except ValueError:
            logger.error("无法识别的滑动方向: %r", direction)
            return False

        if not self.enabled:
            logger.info(
                "[DRY-RUN] 本应%s（档位 %s，锚点 %.2f，时长 %.2fs）",
                gesture_direction.label,
                shape.name,
                shape.anchor_ratio,
                shape.duration,
            )
            return False

        geometry = self._gesture_geometry(gesture_direction, shape)
        if geometry is None:
            logger.warning(
                "无法计算%s几何参数（捕获区域 %s），已跳过本次手势",
                gesture_direction.label,
                self.region,
            )
            return False
        start_x, end_x, anchor_y = geometry
        step_duration = max(0.01, float(shape.duration) / max(1, int(shape.steps)))

        with _failsafe_suspended():
            try:
                self._drag_horizontally(
                    start_x, end_x, anchor_y, shape.steps, step_duration
                )
                logger.info(
                    "已%s: (%d, %d) → (%d, %d)，档位 %s",
                    gesture_direction.label,
                    start_x,
                    anchor_y,
                    end_x,
                    anchor_y,
                    shape.name,
                )
                return True
            except Exception as e:
                logger.error("%s失败: %s", gesture_direction.label, e)
                self._release_mouse()
                return False

    def _gesture_geometry(
        self,
        direction: SwipeDirection,
        shape: SwipeProfile,
    ) -> Optional[Tuple[int, int, int]]:
        """计算手势的 ``(起点 x, 终点 x, 纵坐标 y)``（屏幕坐标）。

        返回 ``None`` 表示捕获区域不可用或拖动距离过短，此时不应产生手势。
        """
        left = int(self.region.get("left", 0))
        top = int(self.region.get("top", 0))
        width = int(self.region.get("width", 0))
        height = int(self.region.get("height", 0))
        if width <= 0 or height <= 0:
            return None

        margin = max(2, int(width * self.EDGE_MARGIN_RATIO))
        low = left + margin
        high = left + width - margin
        if high <= low:
            return None

        travel = int(width * max(0.0, float(shape.distance_ratio)))
        if direction is SwipeDirection.LEFT:
            start_x = min(high, left + int(width * self.START_RATIO))
            end_x = max(low, start_x - travel)
        else:
            start_x = max(low, left + width - int(width * self.START_RATIO))
            end_x = min(high, start_x + travel)

        if abs(end_x - start_x) < self.MIN_SWIPE_PIXELS:
            logger.warning(
                "滑动距离过短（%dpx），可能不被页面手势识别；"
                "请检查 config.SWIPE_PROFILES 中的距离比例与捕获区域宽度",
                abs(end_x - start_x),
            )
            return None

        anchor_y = top + int(height * max(0.0, min(1.0, float(shape.anchor_ratio))))
        return start_x, end_x, anchor_y

    def _drag_horizontally(
        self,
        start_x: int,
        end_x: int,
        y: int,
        steps: int,
        step_duration: float,
    ) -> None:
        """按住左键沿水平方向分步拖动，最后释放。

        分步移动是必要的：页面手势识别器依赖一段时间内的多次移动事件，
        一步到位的瞬移通常不会被识别为滑动。
        """
        steps = max(1, int(steps))
        pyautogui.moveTo(start_x, y, duration=min(0.1, step_duration))
        pyautogui.mouseDown()
        try:
            for step in range(1, steps + 1):
                ratio = step / float(steps)
                current_x = int(round(start_x + (end_x - start_x) * ratio))
                pyautogui.moveTo(current_x, y, duration=step_duration)
        finally:
            # 无论拖动是否被中断，都必须松开按键。
            self._release_mouse()

    def _release_mouse(self) -> None:
        """尽力释放鼠标左键，失败只记录日志。"""
        try:
            pyautogui.mouseUp()
        except Exception as e:
            logger.error("释放鼠标按键失败: %s", e)

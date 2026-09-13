"""屏幕自动化操作。"""
import time
import random
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    _HAS_PYAUTOGUI = True
except Exception as e:  # 依赖缺失或环境限制
    logger.error(f"pyautogui 加载失败: {e}")
    _HAS_PYAUTOGUI = False


class ScreenOperator:
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
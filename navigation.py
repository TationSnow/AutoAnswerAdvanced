"""题目导航策略：按钮点击优先，滑动手势兜底。

## 为什么需要这个模块

部分答题平台**完全不提供“上一题 / 下一题”按钮**，题目切换只能靠手势拖动。
在这类平台上，旧实现会一直找不到 “下一题” 按钮，于是：

- 答完一题后无法推进，同一题被反复识别并刷出相同的日志；
- 触发“无进展退避”后扫描频率逐步放慢，看起来像程序卡死。

本模块把“翻到下一题 / 上一题”抽象为**导航指令**，
由多条策略按优先级竞争产出：

1. :class:`ButtonNavigationStrategy` —— 页面存在明确按钮时点击按钮（与旧行为等价）。
2. :class:`SwipeNavigationStrategy` —— 没有按钮时，用水平鼠标拖拽模拟手势翻页。

## 扩展方式

新增导航方式（例如键盘方向键、滚轮、无障碍接口）只需：

1. 继承 :class:`NavigationStrategy`，实现 ``resolve``；
2. 在 :func:`build_default_planner` 中按优先级插入。

调用方（``main`` / ``automation``）不需要任何改动。
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple

from config import (
    SWIPE_FALLBACK_ENABLED,
    SWIPE_NEXT_DIRECTION,
    SWIPE_PREV_DIRECTION,
    SWIPE_PREV_ENABLED,
    SWIPE_PROFILES,
)
from models import NavigationDirection, QuestionSnapshot, SwipeDirection, SwipeProfile

logger = logging.getLogger(__name__)


class NavigationMethod(str, Enum):
    """一次导航采用的落地方式。"""

    BUTTON = "button"
    SWIPE = "swipe"


@dataclass(frozen=True)
class NavigationCommand:
    """一条导航指令：告诉执行层“去哪一题、怎么去”。

    ``target`` 与 ``profile`` 互斥地对应两种方式：

    - 按钮方式：携带 ``target`` 屏幕（区域内）坐标；
    - 滑动方式：携带 ``profile`` 手势参数，方向由 ``swipe_direction`` 决定。
    """

    method: NavigationMethod
    direction: NavigationDirection
    target: Optional[Tuple[int, int]] = None
    profile: Optional[SwipeProfile] = None
    swipe_direction: Optional[SwipeDirection] = None
    label: str = ""

    @property
    def is_gesture(self) -> bool:
        """是否为滑动手势指令。"""
        return self.method is NavigationMethod.SWIPE

    def describe(self) -> str:
        """生成用于日志的一句话描述。"""
        if self.is_gesture:
            gesture = self.swipe_direction.label if self.swipe_direction else "滑动"
            shape = self.profile.name if self.profile else "-"
            return "%s（%s，档位 %s）" % (self.direction.label, gesture, shape)
        return "%s（点击按钮 %s）" % (self.direction.label, self.label or "-")


class NavigationStrategy(ABC):
    """导航方式基类。

    子类只负责回答一个问题：“在当前页面，用这种方式能否走到某个方向？”
    能则返回 :class:`NavigationCommand`，不能则返回 ``None``（交由下一条策略）。
    """

    #: 策略名称，用于启动日志与失败排查。
    name: str = "导航策略"

    def describe(self) -> str:
        """返回策略描述，默认即策略名；子类可补充关键参数。"""
        return self.name

    @abstractmethod
    def resolve(
        self,
        question: Optional[QuestionSnapshot],
        direction: NavigationDirection,
        attempt: int = 0,
    ) -> Optional[NavigationCommand]:
        """尝试产出一条导航指令。

        :param question: 当前题目快照；为 ``None`` 表示本帧没有可用页面信息。
        :param direction: 目标方向。
        :param attempt: 本方向已尝试过的次数，供策略做梯度升级（如加长滑动距离）。
        """


class ButtonNavigationStrategy(NavigationStrategy):
    """点击页面上的“下一题 / 上一题”按钮。

    安全约束：只有按钮文本被明确识别、且坐标完整时才产出指令，
    避免在坐标缺失时退化成点击 (0, 0) 这种危险行为。
    """

    name = "按钮导航"

    def resolve(
        self,
        question: Optional[QuestionSnapshot],
        direction: NavigationDirection,
        attempt: int = 0,
    ) -> Optional[NavigationCommand]:
        """查找对应角色的按钮并生成点击指令。"""
        if question is None:
            return None
        button = question.button_boxes.get(direction.button_role)
        if button is None or button.center is None:
            return None
        return NavigationCommand(
            method=NavigationMethod.BUTTON,
            direction=direction,
            target=button.center,
            label=button.text,
        )


class SwipeNavigationStrategy(NavigationStrategy):
    """用手势滑动模拟翻题。

    方向映射、手势档位全部来自配置，便于按平台微调；
    同一方向连续滑动无效时，会按档位梯度自动加长拖动距离或上移锚点，
    而不是原样重复同一个手势。
    """

    name = "滑动导航"

    def __init__(
        self,
        enabled: bool = SWIPE_FALLBACK_ENABLED,
        prev_enabled: bool = SWIPE_PREV_ENABLED,
        profiles: Optional[Sequence[Any]] = None,
        next_direction: Any = SWIPE_NEXT_DIRECTION,
        prev_direction: Any = SWIPE_PREV_DIRECTION,
    ) -> None:
        """初始化滑动策略。

        :param enabled: 是否允许用滑动替代“下一题”。
        :param prev_enabled: 是否允许用滑动回退“上一题”。
        :param profiles: 手势档位，支持配置里的元组形式或 :class:`SwipeProfile`。
        :param next_direction: “下一题”对应的滑动手势方向。
        :param prev_direction: “上一题”对应的滑动手势方向。
        """
        self.enabled = enabled
        self.prev_enabled = prev_enabled
        self.profiles: List[SwipeProfile] = build_profile_ladder(profiles)
        self.next_swipe = coerce_swipe_direction(next_direction, SwipeDirection.LEFT)
        self.prev_swipe = coerce_swipe_direction(prev_direction, SwipeDirection.RIGHT)
        if self.next_swipe is self.prev_swipe:
            # 两个方向滑法相同一定是配置笔误：翻不过去比翻错方向更容易排查，
            # 这里只告警不抛异常，避免因为配置问题导致整个程序起不来。
            logger.warning(
                "滑动翻题的“下一题/上一题”方向配置相同（%s），"
                "请检查 config.SWIPE_NEXT_DIRECTION / SWIPE_PREV_DIRECTION",
                self.next_swipe.value,
            )

    def resolve(
        self,
        question: Optional[QuestionSnapshot],
        direction: NavigationDirection,
        attempt: int = 0,
    ) -> Optional[NavigationCommand]:
        """按方向开关与档位梯度生成滑动指令。"""
        if direction is NavigationDirection.NEXT:
            if not self.enabled:
                return None
            swipe_direction = self.next_swipe
        else:
            if not self.prev_enabled:
                return None
            swipe_direction = self.prev_swipe

        return NavigationCommand(
            method=NavigationMethod.SWIPE,
            direction=direction,
            profile=self.profile_for(attempt),
            swipe_direction=swipe_direction,
            label=swipe_direction.label,
        )

    def profile_for(self, attempt: int) -> SwipeProfile:
        """按尝试次数取手势档位；超出档位范围时固定使用最后一档。"""
        if not self.profiles:
            return SwipeProfile(name="默认")
        index = max(0, min(int(attempt), len(self.profiles) - 1))
        return self.profiles[index]

    def describe(self) -> str:
        """补充方向映射与档位数，便于用户确认滑动配置是否符合平台操作习惯。"""
        if not self.enabled and not self.prev_enabled:
            return "%s（已关闭）" % self.name
        parts = []
        if self.enabled:
            parts.append("%s→下一题" % self.next_swipe.label)
        if self.prev_enabled:
            parts.append("%s→上一题" % self.prev_swipe.label)
        return "%s（%s，共 %d 档）" % (self.name, " / ".join(parts), len(self.profiles))


# ----------------------------------------------------------------------
# 配置归一化辅助
# ----------------------------------------------------------------------


def coerce_swipe_direction(value: Any, default: SwipeDirection) -> SwipeDirection:
    """把配置中的字符串/枚举统一成 :class:`SwipeDirection`。"""
    if isinstance(value, SwipeDirection):
        return value
    try:
        return SwipeDirection(str(value).strip().lower())
    except ValueError:
        logger.warning("无法识别的滑动方向配置 %r，回退为 %s", value, default.value)
        return default


def build_profile_ladder(profiles: Optional[Sequence[Any]] = None) -> List[SwipeProfile]:
    """把配置里的手势档位转换为 :class:`SwipeProfile` 列表。

    兼容两种写法：

    - 元组 ``(名称, 距离比例, 锚点比例, 时长, 步数)``；
    - 已经构造好的 :class:`SwipeProfile` 实例（便于测试注入与外部扩展）。
    """
    raw_items = SWIPE_PROFILES if profiles is None else profiles
    ladder: List[SwipeProfile] = []
    for item in raw_items or ():
        if isinstance(item, SwipeProfile):
            ladder.append(item)
            continue
        try:
            name, distance, anchor, duration, steps = item
        except (TypeError, ValueError):
            logger.warning("跳过无法解析的滑动档位配置: %r", item)
            continue
        ladder.append(
            SwipeProfile(
                name=str(name),
                distance_ratio=float(distance),
                anchor_ratio=float(anchor),
                duration=float(duration),
                steps=max(1, int(steps)),
            )
        )
    if not ladder:
        # 配置被写坏时仍然给出一个可用档位，保证功能不至于整体失效。
        logger.warning("滑动档位配置为空，已回退为内置默认档位")
        ladder.append(SwipeProfile(name="默认"))
    return ladder


@dataclass
class NavigationPlanner:
    """按优先级选择导航方式，实现“按钮优先、滑动兜底”。

    调用方只需一次 :meth:`resolve`，不必关心页面上到底有没有按钮：
    有按钮时行为与旧版本完全一致，没有按钮时自动退化为手势滑动。
    """

    strategies: List[NavigationStrategy] = field(default_factory=list)

    def resolve(
        self,
        question: Optional[QuestionSnapshot],
        direction: NavigationDirection,
        attempt: int = 0,
    ) -> Optional[NavigationCommand]:
        """依次询问各策略，返回第一条可用指令。"""
        for strategy in self.strategies:
            try:
                command = strategy.resolve(question, direction, attempt)
            except Exception as exc:  # 单条策略异常不能影响其它导航方式
                logger.error("导航策略[%s]执行失败: %s", strategy.name, exc)
                continue
            if command is not None:
                return command
        return None

    def can_resolve(self, question: Optional[QuestionSnapshot], direction: NavigationDirection) -> bool:
        """判断当前是否存在任何可用的导航方式。"""
        return self.resolve(question, direction) is not None

    def describe(self) -> str:
        """生成策略列表描述，便于启动时确认导航能力。"""
        return "导航策略: %s" % " → ".join(
            strategy.describe() for strategy in self.strategies
        )


def build_default_planner() -> NavigationPlanner:
    """构造默认规划器：按钮优先，滑动兜底。"""
    return NavigationPlanner(
        strategies=[
            ButtonNavigationStrategy(),
            SwipeNavigationStrategy(),
        ]
    )

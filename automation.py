"""自动答题动作规划与同一题状态机。"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models import (
    AnswerCandidate,
    AutomationAction,
    AutomationActionType,
    AutomationPlan,
    NavigationDirection,
    QuestionSnapshot,
)
from navigation import NavigationPlanner, build_default_planner


class SessionState(str, Enum):
    """同一道题在当前页面上的处理状态。"""

    IDLE = "idle"
    SOLVING = "solving"
    WAITING_FEEDBACK = "waiting_feedback"
    WAITING_NEXT = "waiting_next"
    DONE = "done"


# 同一题最多重试“推进到下一题”的次数（点击与滑动共用同一份预算）。
# 点击后页面没有推进（网络慢、按钮未响应、滑动未被识别）时允许有限次重试，
# 达到上限即停止，避免在同一题上无限点击或无限滑动。
MAX_NEXT_CLICKS = 3

# 同一题最多回退到“上一题”的次数。
# 回退主要用于滑动过头或需要重新识别的场景；次数必须受限，
# 否则一旦前进/回退判定抖动，程序会在两道题之间来回滑动。
MAX_PREV_SWIPES = 2


class QuestionSession:
    """保证同一题只执行一次作答，并支持下一题恢复。"""

    def __init__(self) -> None:
        self.current_key: Optional[str] = None
        self.state = SessionState.IDLE
        self.last_action_at = 0.0
        self.next_click_count = 0
        self.prev_swipe_count = 0
        #: 最近一次推进使用的方式（"click" / "swipe"），仅用于日志与排查。
        self.last_navigation_method = ""

    def should_process(self, question: QuestionSnapshot) -> bool:
        """新题返回 True，同一题已经处理则返回 False。"""
        key = question.identity_key
        if key != self.current_key:
            self.current_key = key
            self.state = SessionState.SOLVING
            self.last_action_at = time.time()
            self.next_click_count = 0
            self.prev_swipe_count = 0
            self.last_navigation_method = ""
            return True
        return False

    def mark_waiting_feedback(self) -> None:
        """记录已提交答案，正在等待页面反馈。"""
        self.state = SessionState.WAITING_FEEDBACK
        self.last_action_at = time.time()

    def mark_waiting_next(self) -> None:
        """记录仍未找到明确下一题按钮。"""
        self.state = SessionState.WAITING_NEXT
        self.last_action_at = time.time()

    def record_next_click(self, method: str = "click") -> None:
        """记录一次“推进到下一题”的尝试，并允许页面未推进时重试。

        与直接标记完成不同，这里保留 WAITING_NEXT 状态，
        使页面在等待时间内没有变化时可以再次尝试推进。

        :param method: 本次尝试的方式（``"click"`` 点击按钮 / ``"swipe"`` 滑动手势）。
            无按钮平台上滑动是唯一手段，因此与点击共用同一份尝试预算，
            避免两个计数器各自放行导致实际尝试次数翻倍。
        """
        self.next_click_count += 1
        self.state = SessionState.WAITING_NEXT
        self.last_action_at = time.time()
        self.last_navigation_method = method

    def can_retry_next(self, retry_after: float, now: Optional[float] = None) -> bool:
        """判断是否已经到安全重试下一题的时间。"""
        current = time.time() if now is None else now
        return (
            self.state in {SessionState.WAITING_FEEDBACK, SessionState.WAITING_NEXT}
            and self.next_click_count < MAX_NEXT_CLICKS
            and current - self.last_action_at >= retry_after
        )

    def record_prev_swipe(self) -> None:
        """记录一次“回退到上一题”的滑动尝试。"""
        self.prev_swipe_count += 1
        self.last_action_at = time.time()
        self.last_navigation_method = "swipe"

    def can_swipe_prev(
        self,
        retry_after: float = 0.0,
        now: Optional[float] = None,
    ) -> bool:
        """判断是否还允许回退“上一题”。

        与 :meth:`can_retry_next` 不同，这里不限定会话状态：
        回退既可能发生在等待反馈阶段（滑过头），也可能发生在题目完全无法解析时。
        """
        current = time.time() if now is None else now
        return (
            self.prev_swipe_count < MAX_PREV_SWIPES
            and current - self.last_action_at >= retry_after
        )

    def mark_finished(self) -> None:
        """当前题已处理完成。"""
        self.state = SessionState.DONE
        self.last_action_at = time.time()

    def reset(self) -> None:
        """清空会话状态。"""
        self.current_key = None
        self.state = SessionState.IDLE
        self.last_action_at = 0.0
        self.next_click_count = 0
        self.prev_swipe_count = 0
        self.last_navigation_method = ""


@dataclass
class AutomationPlanner:
    """根据题目和答案生成安全、可测试的动作列表。

    ``navigator`` 负责回答“怎么翻到下一题”：
    页面有明确按钮时点击按钮，没有按钮时（部分平台只有手势翻页）
    自动退化为滑动手势，具体由 :mod:`navigation` 的策略链决定。
    """

    auto_next: bool = True
    feedback_timeout: float = 4.0
    navigator: NavigationPlanner = field(default_factory=build_default_planner)

    def build(
        self,
        question: QuestionSnapshot,
        answer: AnswerCandidate,
    ) -> AutomationPlan:
        """创建动作计划，任何不确定条件都停止而不是猜测点击。"""
        if not question.is_valid:
            return self._stop("题目无效，禁止自动点击")
        if not answer.is_valid:
            return self._stop(answer.reason or "答案无效，禁止自动点击")
        if not question.question_type.clickable:
            return self._stop("当前题型仅展示答案，不执行自动点击")

        actions = []
        for label in answer.labels:
            option = question.option_by_label(label)
            if option is None or option.center is None:
                return self._stop("答案选项 %s 缺少可点击坐标" % label)
            actions.append(
                AutomationAction(
                    kind=AutomationActionType.CLICK_OPTION,
                    target=option.center,
                    label=option.label,
                )
            )

        confirm = question.button_boxes.get("confirm")
        if confirm is not None:
            if confirm.center is None:
                return self._stop("确认答案按钮缺少可点击坐标")
            actions.append(
                AutomationAction(
                    kind=AutomationActionType.CLICK_CONFIRM,
                    target=confirm.center,
                    label=confirm.text,
                )
            )

        actions.append(
            AutomationAction(
                kind=AutomationActionType.WAIT_FEEDBACK,
                timeout=self.feedback_timeout,
            )
        )

        # 纯提交按钮可能结束整场考试，无论配置如何都不进入自动计划。
        if self.auto_next:
            navigation = self.navigator.resolve(question, NavigationDirection.NEXT)
            if navigation is not None:
                if navigation.is_gesture:
                    # 平台没有“下一题”按钮：退化为滑动手势翻页。
                    actions.append(
                        AutomationAction(
                            kind=AutomationActionType.SWIPE,
                            direction=navigation.direction,
                            profile=navigation.profile,
                            label=navigation.label,
                        )
                    )
                elif navigation.target is None:
                    return self._stop("下一题按钮缺少可点击坐标")
                else:
                    actions.append(
                        AutomationAction(
                            kind=AutomationActionType.CLICK_NEXT,
                            target=navigation.target,
                            label=navigation.label,
                        )
                    )
        return AutomationPlan(actions=actions)

    @staticmethod
    def _stop(reason: str) -> AutomationPlan:
        """创建立即停止的计划。"""
        return AutomationPlan(
            actions=[
                AutomationAction(
                    kind=AutomationActionType.STOP,
                    reason=reason,
                )
            ]
        )



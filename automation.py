"""自动答题动作规划与同一题状态机。"""
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from models import (
    AnswerCandidate,
    AutomationAction,
    AutomationActionType,
    AutomationPlan,
    QuestionSnapshot,
)


class SessionState(str, Enum):
    """同一道题在当前页面上的处理状态。"""

    IDLE = "idle"
    SOLVING = "solving"
    WAITING_FEEDBACK = "waiting_feedback"
    WAITING_NEXT = "waiting_next"
    DONE = "done"


# 同一题最多重试点击“下一题”的次数。
# 点击后页面没有推进（网络慢、按钮未响应）时允许有限次重试，
# 达到上限即停止，避免在同一题上无限点击。
MAX_NEXT_CLICKS = 3


class QuestionSession:
    """保证同一题只执行一次作答，并支持下一题恢复。"""

    def __init__(self) -> None:
        self.current_key: Optional[str] = None
        self.state = SessionState.IDLE
        self.last_action_at = 0.0
        self.next_click_count = 0

    def should_process(self, question: QuestionSnapshot) -> bool:
        """新题返回 True，同一题已经处理则返回 False。"""
        key = question.identity_key
        if key != self.current_key:
            self.current_key = key
            self.state = SessionState.SOLVING
            self.last_action_at = time.time()
            self.next_click_count = 0
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

    def record_next_click(self) -> None:
        """记录一次“下一题”点击，并允许页面未推进时重试。

        与直接标记完成不同，这里保留 WAITING_NEXT 状态，
        使页面在等待时间内没有变化时可以再次尝试推进。
        """
        self.next_click_count += 1
        self.state = SessionState.WAITING_NEXT
        self.last_action_at = time.time()

    def can_retry_next(self, retry_after: float, now: Optional[float] = None) -> bool:
        """判断是否已经到安全重试下一题的时间。"""
        current = time.time() if now is None else now
        return (
            self.state in {SessionState.WAITING_FEEDBACK, SessionState.WAITING_NEXT}
            and self.next_click_count < MAX_NEXT_CLICKS
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


@dataclass
class AutomationPlanner:
    """根据题目和答案生成安全、可测试的动作列表。"""

    auto_next: bool = True
    feedback_timeout: float = 4.0

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
            next_button = question.button_boxes.get("next")
            if next_button is not None:
                if next_button.center is None:
                    return self._stop("下一题按钮缺少可点击坐标")
                actions.append(
                    AutomationAction(
                        kind=AutomationActionType.CLICK_NEXT,
                        target=next_button.center,
                        label=next_button.text,
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


"""答题流程统一领域模型。

各模块只在这些结构之间传递数据，避免继续依赖含义模糊的字典字段。
"""
import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


_MATCH_NOISE_RE = re.compile(
    r"[\s，。？！、；：\"'“”‘’（）【】《》〈〉,.!?;:()\[\]{}<>·—\-_/\\]+"
)


def normalize_for_match(text: str) -> str:
    """生成用于哈希和文字比较的稳定文本。"""
    return _MATCH_NOISE_RE.sub("", text or "").lower()


def text_similarity(left: str, right: str) -> float:
    """计算轻量文字相似度，优先处理 OCR 标点和空格差异。"""
    from difflib import SequenceMatcher

    norm_left = normalize_for_match(left)
    norm_right = normalize_for_match(right)
    if not norm_left or not norm_right:
        return 0.0
    if norm_left == norm_right:
        return 1.0
    if norm_left in norm_right or norm_right in norm_left:
        short = min(len(norm_left), len(norm_right))
        long = max(len(norm_left), len(norm_right))
        return max(0.85, short / long)
    return SequenceMatcher(None, norm_left, norm_right).ratio()


class QuestionType(str, Enum):
    """当前支持识别的题型。"""

    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    OPEN_ENDED = "open_ended"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: Any) -> "QuestionType":
        """兼容旧值和常见中英文别名。"""
        if isinstance(value, cls):
            return value
        aliases = {
            "choice": cls.SINGLE_CHOICE,
            "single": cls.SINGLE_CHOICE,
            "single_choice": cls.SINGLE_CHOICE,
            "单选题": cls.SINGLE_CHOICE,
            "multiple": cls.MULTIPLE_CHOICE,
            "multi_choice": cls.MULTIPLE_CHOICE,
            "multiple_choice": cls.MULTIPLE_CHOICE,
            "多选题": cls.MULTIPLE_CHOICE,
            "true_false": cls.TRUE_FALSE,
            "judge": cls.TRUE_FALSE,
            "判断题": cls.TRUE_FALSE,
            "fill_blank": cls.FILL_BLANK,
            "填空题": cls.FILL_BLANK,
            "open_ended": cls.OPEN_ENDED,
            "开放题": cls.OPEN_ENDED,
        }
        return aliases.get(str(value or "").strip().lower(), cls.UNKNOWN)

    @property
    def label(self) -> str:
        """返回界面展示名称。"""
        return {
            QuestionType.SINGLE_CHOICE: "单选题",
            QuestionType.MULTIPLE_CHOICE: "多选题",
            QuestionType.TRUE_FALSE: "判断题",
            QuestionType.FILL_BLANK: "填空题",
            QuestionType.OPEN_ENDED: "开放题",
            QuestionType.UNKNOWN: "未知题型",
        }[self]

    @property
    def clickable(self) -> bool:
        """判断该题型是否支持自动点击选项。"""
        return self in {
            QuestionType.SINGLE_CHOICE,
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.TRUE_FALSE,
        }


@dataclass
class OptionItem:
    """题目选项的运行时结构。"""

    label: str
    text: str
    box: Optional[List[Any]] = None
    center: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为便于 UI 和调试工具读取的字典。"""
        return {
            "label": self.label,
            "text": self.text,
            "box": self.box,
            "center": self.center,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptionItem":
        """从数据库或旧字典恢复选项。"""
        center = data.get("center")
        if isinstance(center, list):
            center = tuple(center)
        return cls(
            label=str(data.get("label", "")).upper(),
            text=str(data.get("text", "")),
            box=data.get("box"),
            center=center,
        )


@dataclass
class ButtonItem:
    """按钮文本和坐标。"""

    role: str
    text: str
    box: Optional[List[Any]] = None
    center: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "role": self.role,
            "text": self.text,
            "box": self.box,
            "center": self.center,
        }


@dataclass
class QuestionSnapshot:
    """一次 OCR 识别得到的完整题目快照。"""

    external_id: Optional[str]
    question: str
    question_type: QuestionType
    options: List[OptionItem] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0
    is_valid: bool = False
    error: str = ""
    button_boxes: Dict[str, ButtonItem] = field(default_factory=dict)
    #: 本帧截图的画面签名（由识别层填充）。
    #: 上层用它判断“画面是否真的变了”，从而区分
    #: “页面已翻到新题但新题解析不出来”与“画面根本没动”——
    #: 后者不该再对旧题目重复点击或滑动。
    frame_signature: str = ""

    @property
    def identity_key(self) -> str:
        """生成用于去重的稳定题目身份。"""
        if self.external_id:
            source = "%s|id:%s" % (self.question_type.value, self.external_id)
        else:
            source = "%s|question:%s" % (
                self.question_type.value,
                normalize_for_match(self.question),
            )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]

    def option_by_label(self, label: str) -> Optional[OptionItem]:
        """按选项编号查找选项。"""
        normalized = (label or "").strip().upper()
        for option in self.options:
            if option.label.upper() == normalized:
                return option
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为旧接口和调试工具使用的字典。"""
        options = {option.label: option.text for option in self.options}
        option_boxes = {
            option.label: option.to_dict()
            for option in self.options
            if option.center is not None
        }
        return {
            "external_id": self.external_id,
            "question": self.question,
            "options": options,
            "ordered_options": [option.to_dict() for option in self.options],
            "raw_text": self.raw_text,
            "question_type": self.question_type.value,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "error": self.error,
            "option_boxes": option_boxes,
            "button_boxes": {
                role: button.to_dict() for role, button in self.button_boxes.items()
            },
        }


@dataclass
class AnswerCandidate:
    """经过题型策略规范化后的答案。"""

    labels: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    display_answer: str = "?"
    detail: str = ""
    source: str = ""
    verified: bool = False
    match_id: Optional[int] = None
    similarity: Optional[float] = None
    is_valid: bool = False
    reason: str = ""

    @property
    def answer(self) -> str:
        """兼容原悬浮窗使用的 answer 字段。"""
        return self.display_answer

    def to_dict(self) -> Dict[str, Any]:
        """转换为 Qt 信号和旧调用方可消费的字典。"""
        return {
            "answer": self.display_answer,
            "display_answer": self.display_answer,
            "labels": list(self.labels),
            "texts": list(self.texts),
            "detail": self.detail,
            "source": self.source,
            "verified": self.verified,
            "match_id": self.match_id,
            "similarity": self.similarity,
            "is_valid": self.is_valid,
            "reason": self.reason,
        }

    @classmethod
    def invalid(cls, reason: str, source: str = "") -> "AnswerCandidate":
        """创建禁止自动点击的无效答案。"""
        return cls(
            display_answer="?",
            detail=reason,
            source=source,
            is_valid=False,
            reason=reason,
        )


class NavigationDirection(str, Enum):
    """题目之间的翻页方向（语义层）。

    只描述“要去哪一题”，不描述用什么手段实现：
    按钮点击与手势滑动都复用同一个方向语义，
    由 navigation 模块决定具体用哪种方式落地。
    """

    NEXT = "next"
    PREV = "prev"

    @property
    def button_role(self) -> str:
        """返回该方向在 ``QuestionSnapshot.button_boxes`` 中的按钮角色名。"""
        return "next" if self is NavigationDirection.NEXT else "prev"

    @property
    def label(self) -> str:
        """返回界面与日志使用的中文名称。"""
        return "下一题" if self is NavigationDirection.NEXT else "上一题"


class SwipeDirection(str, Enum):
    """滑动手势的物理方向（实现层）。"""

    LEFT = "left"
    RIGHT = "right"

    @property
    def label(self) -> str:
        """返回界面与日志使用的中文名称。"""
        return "左滑" if self is SwipeDirection.LEFT else "右滑"


@dataclass(frozen=True)
class SwipeProfile:
    """一次水平滑动手势的形状参数（不含方向）。

    单独抽出该结构的目的：同一档手势参数既可以被“下一题”使用，
    也可以被“上一题”使用；方向与形状解耦后，两侧的复用手势策略
    只需要各配置一次，不必为每个方向复制一整套参数。

    :param name: 档位名称，仅用于日志与调试。
    :param distance_ratio: 拖动距离占捕获区域宽度的比例。
    :param anchor_ratio: 拖动起点的纵向位置（占捕获区域高度比例）。
    :param duration: 整段拖动耗时（秒）。
    :param steps: 中间移动次数；过少可能不被页面手势识别器采信。
    """

    name: str
    distance_ratio: float = 0.55
    anchor_ratio: float = 0.5
    duration: float = 0.32
    steps: int = 8

    def to_dict(self) -> Dict[str, Any]:
        """转换为便于日志输出的字典。"""
        return {
            "name": self.name,
            "distance_ratio": self.distance_ratio,
            "anchor_ratio": self.anchor_ratio,
            "duration": self.duration,
            "steps": self.steps,
        }


class AutomationActionType(str, Enum):
    """自动操作动作类型。"""

    CLICK_OPTION = "click_option"
    CLICK_CONFIRM = "click_confirm"
    WAIT_FEEDBACK = "wait_feedback"
    CLICK_NEXT = "click_next"
    # 滑动翻题：页面上不存在“上一题/下一题”按钮时，用手势模拟翻页。
    SWIPE = "swipe"
    CLICK_SUBMIT = "click_submit"
    STOP = "stop"


@dataclass
class AutomationAction:
    """一次自动化动作。

    ``target`` 与 ``direction``/``profile`` 分别对应两种导航方式：
    点击类动作使用 ``target`` 坐标；滑动类动作使用方向与手势参数。
    """

    kind: AutomationActionType
    target: Optional[Tuple[int, int]] = None
    label: str = ""
    reason: str = ""
    timeout: float = 0.0
    direction: Optional[NavigationDirection] = None
    profile: Optional[SwipeProfile] = None



@dataclass
class AutomationPlan:
    """按顺序执行的自动化计划。"""

    actions: List[AutomationAction] = field(default_factory=list)

    def kinds(self) -> List[AutomationActionType]:
        """返回动作序列，便于日志和测试。"""
        return [action.kind for action in self.actions]


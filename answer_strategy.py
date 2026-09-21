"""题型答案策略。

所有模型输出、题库答案和屏幕反馈都必须经过这里，才能进入自动点击流程。
"""
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from models import (
    AnswerCandidate,
    OptionItem,
    QuestionSnapshot,
    QuestionType,
    normalize_for_match,
    text_similarity,
)


TRUE_TEXTS = {"对", "正确", "是", "√", "true", "t", "yes"}
FALSE_TEXTS = {"错", "错误", "否", "×", "false", "f", "no"}


def _as_list(value: Any) -> List[str]:
    """把模型字段安全转换为字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (TypeError, ValueError):
                pass
        return [stripped]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _canonical_bool(text: str) -> Optional[bool]:
    """把常见中英文正误表达转换为布尔值。"""
    normalized = normalize_for_match(text)
    if not normalized:
        return None
    if normalized in TRUE_TEXTS:
        return True
    if normalized in FALSE_TEXTS:
        return False
    if re.search(r"(?:说法|答案|判断)?正确$", normalized):
        return True
    if re.search(r"(?:说法|答案|判断)?错误$", normalized):
        return False
    return None


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中容错提取 JSON 对象。

    线上接口不支持 JSON Schema 时会退化为纯文本协议，模型常常把 JSON 包在
    Markdown 代码块里、或在 JSON 前后附带思考文字。这里依次尝试：

    1. 直接解析整段文本；
    2. 去掉 ``` 代码块围栏后再解析；
    3. 扫描文本中所有成对花括号片段，**从后往前**解析（最终答案通常在末尾）。

    字符串中的花括号会被忽略，避免把选项文字里的括号当成 JSON 边界。
    """
    candidate = (text or "").strip()
    if not candidate:
        return None

    direct = _try_load_json(candidate)
    if direct is not None:
        return direct

    for stripped in _strip_code_fences(candidate):
        parsed = _try_load_json(stripped)
        if parsed is not None:
            return parsed

    for fragment in reversed(_iter_brace_fragments(candidate)):
        parsed = _try_load_json(fragment)
        if parsed is not None:
            return parsed
    return None


def _try_load_json(text: str) -> Optional[Dict[str, Any]]:
    """尝试把文本解析为 JSON 对象，失败返回 None。"""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _strip_code_fences(text: str) -> List[str]:
    """去掉 Markdown 代码块围栏，返回候选文本。"""
    if "```" not in text:
        return []
    results = []
    for block in re.findall(r"```[a-zA-Z]*\s*(.+?)```", text, re.DOTALL):
        stripped = block.strip()
        if stripped:
            results.append(stripped)
    return results


def _iter_brace_fragments(text: str) -> List[str]:
    """扫描出所有顶层花括号片段（忽略字符串内部的花括号）。"""
    fragments: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    fragments.append(text[start:index + 1])
                    start = -1
    return fragments


class BaseAnswerStrategy(ABC):
    """题型策略基类。"""

    question_type: QuestionType

    @abstractmethod
    def build(
        self,
        question: QuestionSnapshot,
        labels: Sequence[str],
        texts: Sequence[str],
        detail: str,
        source: str,
        verified: bool,
        match_id: Optional[int],
        similarity: Optional[float],
    ) -> AnswerCandidate:
        """把候选答案规范化。"""


class SingleChoiceStrategy(BaseAnswerStrategy):
    """单选题仅允许一个有效选项。"""

    question_type = QuestionType.SINGLE_CHOICE

    def build(
        self,
        question: QuestionSnapshot,
        labels: Sequence[str],
        texts: Sequence[str],
        detail: str,
        source: str,
        verified: bool,
        match_id: Optional[int],
        similarity: Optional[float],
    ) -> AnswerCandidate:
        labels, texts, reason = _normalize_choice_answers(question, labels, texts)
        if len(labels) != 1:
            return AnswerCandidate.invalid(
                reason or "单选题答案必须且只能包含一个选项", source
            )
        option = question.option_by_label(labels[0])
        if option is None:
            return AnswerCandidate.invalid("单选题答案不在当前选项中", source)
        return _candidate(
            [option.label],
            [option.text],
            option.label,
            detail,
            source,
            verified,
            match_id,
            similarity,
        )


class MultipleChoiceStrategy(BaseAnswerStrategy):
    """多选题保留全部有效选项，并按屏幕顺序排列。"""

    question_type = QuestionType.MULTIPLE_CHOICE

    def build(
        self,
        question: QuestionSnapshot,
        labels: Sequence[str],
        texts: Sequence[str],
        detail: str,
        source: str,
        verified: bool,
        match_id: Optional[int],
        similarity: Optional[float],
    ) -> AnswerCandidate:
        labels, texts, reason = _normalize_choice_answers(question, labels, texts)
        if not labels:
            return AnswerCandidate.invalid(reason or "多选题答案为空", source)
        options = [question.option_by_label(label) for label in labels]
        if any(option is None for option in options):
            return AnswerCandidate.invalid("多选题包含当前页面不存在的选项", source)
        valid_options = [option for option in options if option is not None]
        return _candidate(
            [option.label for option in valid_options],
            [option.text for option in valid_options],
            ", ".join(option.label for option in valid_options),
            detail,
            source,
            verified,
            match_id,
            similarity,
        )


class TrueFalseStrategy(BaseAnswerStrategy):
    """判断题统一映射为当前页面的“对/错”选项。"""

    question_type = QuestionType.TRUE_FALSE

    def build(
        self,
        question: QuestionSnapshot,
        labels: Sequence[str],
        texts: Sequence[str],
        detail: str,
        source: str,
        verified: bool,
        match_id: Optional[int],
        similarity: Optional[float],
    ) -> AnswerCandidate:
        expected = _find_true_false_value(question, labels, texts)
        if expected is None:
            return AnswerCandidate.invalid("无法判断正确答案是“对”还是“错”", source)

        for option in question.options:
            if _canonical_bool(option.text) is expected:
                return _candidate(
                    [option.label],
                    [option.text],
                    option.text,
                    detail,
                    source,
                    verified,
                    match_id,
                    similarity,
                )
        return AnswerCandidate.invalid("当前页面缺少对应的判断题选项", source)


class TextAnswerStrategy(BaseAnswerStrategy):
    """填空题和开放题只展示答案，不参与自动点击。"""

    def __init__(self, question_type: QuestionType) -> None:
        self.question_type = question_type

    def build(
        self,
        question: QuestionSnapshot,
        labels: Sequence[str],
        texts: Sequence[str],
        detail: str,
        source: str,
        verified: bool,
        match_id: Optional[int],
        similarity: Optional[float],
    ) -> AnswerCandidate:
        values = list(texts) or list(labels)
        if not values:
            return AnswerCandidate.invalid("模型未返回有效答案", source)
        display = "；".join(values)
        return _candidate(
            [],
            values,
            display,
            detail,
            source,
            verified,
            match_id,
            similarity,
        )


class AnswerStrategyRegistry:
    """按题型选择答案策略。"""

    def __init__(self) -> None:
        self._strategies = {
            QuestionType.SINGLE_CHOICE: SingleChoiceStrategy(),
            QuestionType.MULTIPLE_CHOICE: MultipleChoiceStrategy(),
            QuestionType.TRUE_FALSE: TrueFalseStrategy(),
            QuestionType.FILL_BLANK: TextAnswerStrategy(QuestionType.FILL_BLANK),
            QuestionType.OPEN_ENDED: TextAnswerStrategy(QuestionType.OPEN_ENDED),
            QuestionType.UNKNOWN: TextAnswerStrategy(QuestionType.UNKNOWN),
        }

    def get(self, question_type: QuestionType) -> BaseAnswerStrategy:
        """取得对应策略。"""
        return self._strategies.get(question_type, self._strategies[QuestionType.UNKNOWN])


class AnswerParser:
    """统一解析模型 JSON、文本答案和屏幕反馈。"""

    def __init__(self, text_match_threshold: float = 0.82) -> None:
        self.text_match_threshold = text_match_threshold
        self.registry = AnswerStrategyRegistry()

    def from_payload(
        self,
        question: QuestionSnapshot,
        payload: Dict[str, Any],
        source: str,
        detail: str = "",
        verified: bool = False,
        match_id: Optional[int] = None,
        similarity: Optional[float] = None,
    ) -> AnswerCandidate:
        """解析结构化模型响应。"""
        labels = _as_list(payload.get("answer_labels"))
        texts = _as_list(payload.get("answer_texts"))
        explanation = str(payload.get("explanation", detail) or "").strip()
        return self.from_values(
            question,
            labels,
            texts,
            explanation,
            source,
            verified,
            match_id,
            similarity,
        )

    def from_values(
        self,
        question: QuestionSnapshot,
        labels: Sequence[str],
        texts: Sequence[str],
        detail: str = "",
        source: str = "",
        verified: bool = False,
        match_id: Optional[int] = None,
        similarity: Optional[float] = None,
    ) -> AnswerCandidate:
        """按题型策略规范化答案。"""
        strategy = self.registry.get(question.question_type)
        return strategy.build(
            question,
            labels,
            texts,
            detail,
            source,
            verified,
            match_id,
            similarity,
        )

    def bind_texts(
        self,
        question: QuestionSnapshot,
        texts: Sequence[str],
        detail: str = "",
        source: str = "",
        verified: bool = False,
        match_id: Optional[int] = None,
        similarity: Optional[float] = None,
    ) -> AnswerCandidate:
        """把题库保存的文字答案绑定到当前页面的选项编号。"""
        labels, matched_texts, reason = _normalize_choice_answers(
            question, [], texts, self.text_match_threshold
        )
        if not labels:
            return AnswerCandidate.invalid(reason or "答案文字未匹配到当前选项", source)
        return self.from_values(
            question,
            labels,
            matched_texts,
            detail,
            source,
            verified,
            match_id,
            similarity,
        )

    def from_explicit_text(
        self,
        question: QuestionSnapshot,
        text: str,
        source: str,
        verified: bool = False,
        match_id: Optional[int] = None,
    ) -> AnswerCandidate:
        """解析“答案/正确答案/解析”格式的屏幕文字。"""
        detail = _extract_detail(text)
        if question.question_type == QuestionType.TRUE_FALSE:
            expected = _extract_true_false_answer(text)
            if expected is None:
                return AnswerCandidate.invalid("未识别到判断题答案", source)
            return self.from_values(
                question,
                [],
                ["对" if expected else "错"],
                detail,
                source,
                verified,
                match_id,
            )

        match = re.search(
            r"(?:正确答案|答案)\s*[：:]\s*([^\n\r]+)",
            text,
            re.IGNORECASE,
        )
        candidate = match.group(1).strip() if match else ""
        if not candidate:
            return AnswerCandidate.invalid("未识别到明确答案", source)

        labels = _extract_labels(candidate, question)
        if labels:
            return self.from_values(
                question,
                labels,
                [],
                detail,
                source,
                verified,
                match_id,
            )
        return self.bind_texts(
            question,
            [candidate],
            detail,
            source,
            verified,
            match_id,
        )

    def parse_json_text(
        self,
        question: QuestionSnapshot,
        content: str,
        source: str,
    ) -> AnswerCandidate:
        """尝试把模型文本内容解析为 JSON，失败时返回无效答案。"""
        payload = extract_json_payload(content)
        if payload is None:
            return AnswerCandidate.invalid("模型响应不是合法 JSON", source)
        return self.from_payload(question, payload, source)


def _candidate(
    labels: Sequence[str],
    texts: Sequence[str],
    display: str,
    detail: str,
    source: str,
    verified: bool,
    match_id: Optional[int],
    similarity: Optional[float],
) -> AnswerCandidate:
    """创建有效答案。"""
    return AnswerCandidate(
        labels=list(labels),
        texts=list(texts),
        display_answer=display,
        detail=detail,
        source=source,
        verified=verified,
        match_id=match_id,
        similarity=similarity,
        is_valid=True,
    )


def _normalize_choice_answers(
    question: QuestionSnapshot,
    labels: Sequence[str],
    texts: Sequence[str],
    text_match_threshold: float = 0.82,
) -> Tuple[List[str], List[str], str]:
    """规范选择题答案；优先校验编号，其次按选项文字绑定。"""
    normalized_labels = []
    for item in labels:
        for label in re.findall(r"(?<![A-Za-z0-9])[A-Z](?![A-Za-z0-9])", str(item).upper()):
            if label not in normalized_labels:
                normalized_labels.append(label)

    valid_labels = []
    for label in normalized_labels:
        if question.option_by_label(label) is not None:
            valid_labels.append(label)
        else:
            return [], [], "答案包含当前页面不存在的选项 %s" % label

    if valid_labels:
        order = {option.label.upper(): index for index, option in enumerate(question.options)}
        valid_labels.sort(key=lambda label: order.get(label, 999))
        valid_texts = [
            question.option_by_label(label).text
            for label in valid_labels
            if question.option_by_label(label) is not None
        ]
        return valid_labels, valid_texts, ""

    if not texts:
        return [], [], "答案未包含有效选项"

    resolved_labels: List[str] = []
    resolved_texts: List[str] = []
    for text in texts:
        option = _match_option_text(question.options, text, text_match_threshold)
        if option is None:
            return [], [], "答案文字未匹配到唯一选项"
        if option.label not in resolved_labels:
            resolved_labels.append(option.label)
            resolved_texts.append(option.text)

    order = {option.label.upper(): index for index, option in enumerate(question.options)}
    pairs = sorted(
        zip(resolved_labels, resolved_texts),
        key=lambda pair: order.get(pair[0], 999),
    )
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs], ""


def _match_option_text(
    options: Sequence[OptionItem],
    answer_text: str,
    threshold: float,
) -> Optional[OptionItem]:
    """在选项文字中寻找唯一且足够相似的匹配。"""
    normalized = normalize_for_match(answer_text)
    if not normalized:
        return None

    scored = []
    for option in options:
        score = text_similarity(answer_text, option.text)
        scored.append((score, option))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < threshold:
        return None
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.01:
        return None
    return scored[0][1]


def _find_true_false_value(
    question: QuestionSnapshot,
    labels: Sequence[str],
    texts: Sequence[str],
) -> Optional[bool]:
    """从文本、字母或 T/F 标记中确定判断题答案。"""
    for text in texts:
        value = _canonical_bool(text)
        if value is not None:
            return value

    for item in labels:
        raw = str(item).strip()
        value = _canonical_bool(raw)
        if value is not None:
            return value
        option = question.option_by_label(raw)
        if option is not None:
            value = _canonical_bool(option.text)
            if value is not None:
                return value
    return None


def _extract_detail(text: str) -> str:
    """提取解析文本。"""
    match = re.search(
        r"(?:题目解析|解析|原因)\s*[：:]\s*(.+?)(?=\n(?:题目依据|依据)\s*[：:]|\Z)",
        text,
        re.DOTALL,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip()[:300] if match else ""


def _extract_true_false_answer(text: str) -> Optional[bool]:
    """从反馈文本中提取正误值。"""
    match = re.search(r"(?:正确答案|答案)\s*[：:]\s*([^\s\n\r，,。]+)", text)
    if not match:
        return None
    return _canonical_bool(match.group(1))


def _extract_labels(text: str, question: QuestionSnapshot) -> List[str]:
    """从短答案文本中提取当前页面存在的选项编号。"""
    available = {option.label.upper() for option in question.options}
    labels = []
    for label in re.findall(r"(?<![A-Za-z0-9])[A-Z](?![A-Za-z0-9])", text.upper()):
        if label in available and label not in labels:
            labels.append(label)
    order = {option.label.upper(): index for index, option in enumerate(question.options)}
    labels.sort(key=lambda label: order.get(label, 999))
    return labels

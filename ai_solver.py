"""OpenAI 兼容接口答题器。

优先查询本地题库，未命中时请求结构化 JSON，并通过题型策略进行二次校验。

线上 / 本地接口兼容性设计（对应“线上 API 模型返回空响应、答案永远拿不到”问题）：

1. **输出协议梯度**：依次尝试 json_schema → json_object → 纯文本。
   只有确认“输出格式约束不被支持”的 400 才会降级；其它 400 原样暴露，
   避免把“模型名不存在 / 参数非法 / 鉴权失败”误报成“不支持 JSON Schema”。
2. **协议记忆**：探测结果缓存在实例上，同一次运行内不再重复试探，
   省掉每道题都多打一次注定失败请求的开销。
3. **推理型模型兼容**：max_tokens 过小会让推理型模型把额度全部用于思考，
   返回空正文（finish_reason=length）。这里默认给足额度、检测到截断时自动加倍重试，
   并在正文为空但 reasoning_content 有内容时从推理文本中提取答案。
4. **可诊断**：空响应 / 结构异常时把 finish_reason、usage 与原始响应片段
   同时写进日志和失败原因，不必靠猜。
"""
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from config import (
    DEEPSEEK_JSON_SCHEMA_ENABLED,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MAX_TOKENS_LIMIT,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_TIMEOUT,
)
from answer_strategy import AnswerParser
from models import AnswerCandidate, QuestionSnapshot, QuestionType


logger = logging.getLogger(__name__)


# ---------------- 输出协议 ----------------
PROTOCOL_JSON_SCHEMA = "json_schema"
PROTOCOL_JSON_OBJECT = "json_object"
PROTOCOL_TEXT = "text"
PROTOCOL_LABELS: Dict[str, str] = {
    PROTOCOL_JSON_SCHEMA: "JSON Schema 强约束",
    PROTOCOL_JSON_OBJECT: "JSON 对象模式",
    PROTOCOL_TEXT: "纯文本",
}

# ---------------- 失败来源分类 ----------------
SOURCE_API_ERROR = "api_error"        # 可重试：网络抖动、5xx、空响应等
SOURCE_API_FATAL = "api_fatal"        # 不可重试：模型名、鉴权、参数等配置问题
SOURCE_TRUNCATED = "api_truncated"    # 正文被 max_tokens 截断，可提高额度重试
RETRYABLE_SOURCES = (SOURCE_API_ERROR, SOURCE_TRUNCATED)

# 400 响应体中命中这些字样，才认为是“输出格式约束不被支持”。
FORMAT_REJECTION_TOKENS: Tuple[str, ...] = (
    "response_format",
    "response format",
    "json_schema",
    "json schema",
    "grammar",
    "structured output",
    "结构化输出",
)
# HTTP 错误体里命中这些字样视为配置类错误，重试无意义。
FATAL_ERROR_TOKENS: Tuple[str, ...] = (
    "invalid api key",
    "authentication",
    "unauthorized",
    "permission",
    "model not found",
    "not exist",
    "does not exist",
    "invalid_request_error",
    "insufficient",
    "balance",
)
# 这些状态码属于可重试的瞬时故障。
RETRYABLE_STATUS_CODES = (408, 409, 425, 429)


class DeepSeekSolver:
    """OpenAI 兼容接口答题器，类名保留以兼容现有配置和调用。"""

    MAX_RETRIES = 3
    RETRY_DELAY = 1

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        enable_search: bool = False,
        knowledge_base: Any = None,
        max_tokens: Optional[int] = None,
        max_tokens_limit: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        json_schema_enabled: Optional[bool] = None,
    ) -> None:
        """初始化会话和题库依赖。"""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enable_search = enable_search
        self.kb = knowledge_base
        self.max_tokens = int(max_tokens or DEEPSEEK_MAX_TOKENS)
        self.max_tokens_limit = int(max_tokens_limit or DEEPSEEK_MAX_TOKENS_LIMIT)
        self.temperature = (
            DEEPSEEK_TEMPERATURE if temperature is None else float(temperature)
        )
        self.timeout = float(timeout or DEEPSEEK_TIMEOUT)
        # 输出协议梯度：优先强约束，逐步放宽，最后用纯文本兜底。
        self.protocol_order: List[str] = [
            PROTOCOL_JSON_SCHEMA,
            PROTOCOL_JSON_OBJECT,
            PROTOCOL_TEXT,
        ]
        if json_schema_enabled is None:
            json_schema_enabled = DEEPSEEK_JSON_SCHEMA_ENABLED
        self._unsupported_protocols = set()
        if not json_schema_enabled:
            self._unsupported_protocols.add(PROTOCOL_JSON_SCHEMA)
        self._current_protocol = PROTOCOL_TEXT

        self.answer_parser = AnswerParser()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % api_key,
            }
        )
        logger.info(
            "AI 解题器初始化完成（%s），联网: %s",
            self.describe(),
            enable_search,
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def solve(self, question: QuestionSnapshot) -> AnswerCandidate:
        """执行题库优先、模型兜底的完整解题流程。"""
        if not question or not question.is_valid:
            return AnswerCandidate.invalid("未识别题目", "error")
        if not question.question:
            return AnswerCandidate.invalid("题目为空", "error")

        if self.kb is not None:
            started = time.time()
            hit = self.kb.search(question)
            if hit is not None and hit.is_valid:
                logger.info(
                    "题库命中 (%.3fs): %s",
                    time.time() - started,
                    hit.source,
                )
                return hit

        return self._call_api_with_retry(question)

    def describe(self) -> str:
        """返回接口配置摘要，便于启动日志与排查配置错配。

        典型误配：base_url 指向本机服务、model 却写的是线上模型名（或反之），
        此时接口仍可能返回 200 但内容异常，把配置打印出来能立刻发现问题。
        """
        return "model=%s | endpoint=%s | max_tokens=%d | 输出协议梯度=%s" % (
            self.model,
            self.base_url,
            self.max_tokens,
            "/".join(self._active_protocols()),
        )

    # ------------------------------------------------------------------
    # 请求与重试
    # ------------------------------------------------------------------
    def _call_api_with_retry(self, question: QuestionSnapshot) -> AnswerCandidate:
        """调用模型并执行有限次数重试，必要时自动提高输出额度。"""
        last_error = ""
        for attempt in range(self.MAX_RETRIES):
            started = time.time()
            try:
                result = self._call_api(question)
            except Exception as exc:  # 网络异常等
                last_error = str(exc)
                logger.warning(
                    "模型请求失败 (%d/%d): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    exc,
                )
            else:
                logger.info(
                    "模型请求耗时 %.2fs (尝试 %d/%d, %s, max_tokens=%d)",
                    time.time() - started,
                    attempt + 1,
                    self.MAX_RETRIES,
                    PROTOCOL_LABELS.get(self._current_protocol, "-"),
                    self.max_tokens,
                )
                if result.is_valid or result.source not in RETRYABLE_SOURCES:
                    return result
                last_error = result.reason
                if result.source == SOURCE_TRUNCATED:
                    self._escalate_max_tokens()
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY * (attempt + 1))
        return AnswerCandidate.invalid(
            "API 失败: %s" % last_error[:200],
            "error",
        )

    def _call_api(self, question: QuestionSnapshot) -> AnswerCandidate:
        """按输出协议梯度请求模型，返回首个可用结果。"""
        last_detail = ""
        for protocol in self._active_protocols():
            payload = self._build_payload(question, protocol)
            try:
                response = self.session.post(
                    "%s/chat/completions" % self.base_url,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.Timeout:
                # 推理型模型的长思考会显著拉长单次耗时，这里给出可操作的排查建议。
                logger.warning(
                    "模型请求超时（%.0fs，%s）",
                    self.timeout,
                    PROTOCOL_LABELS.get(protocol, protocol),
                )
                return AnswerCandidate.invalid(
                    "请求超时（当前 DEEPSEEK_TIMEOUT=%.0fs）：推理型模型耗时较长，"
                    "可提高该配置或换用非推理模型" % self.timeout,
                    SOURCE_API_ERROR,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "请求 %s 失败: %s", PROTOCOL_LABELS.get(protocol, protocol), exc
                )
                return AnswerCandidate.invalid("请求异常: %s" % exc, SOURCE_API_ERROR)

            if response.status_code == 400 and protocol != PROTOCOL_TEXT:
                body = self._safe_text(response)
                if self._is_format_rejection(body):
                    # 该协议不被支持：记住并降级，后续请求不再重复试探。
                    self._unsupported_protocols.add(protocol)
                    # 部分接口会在拒绝正文里写明支持的格式，可据此直接收敛协议梯度，
                    # 例如 LM Studio 返回 “'response_format.type' must be 'json_schema' or 'text'”。
                    self._apply_supported_protocols(body)
                    logger.warning(
                        "接口不接受%s（HTTP 400: %s），降级为下一档输出协议",
                        PROTOCOL_LABELS.get(protocol, protocol),
                        body[:120],
                    )
                    last_detail = body[:200]
                    continue
                logger.error("模型请求被拒绝（HTTP 400，与输出格式无关）: %s", body[:300])
                return AnswerCandidate.invalid(
                    "HTTP 400: %s" % body[:200], self._classify_error(body, default_fatal=True)
                )

            if response.status_code != 200:
                body = self._safe_text(response)
                logger.error(
                    "模型请求失败 HTTP %s: %s", response.status_code, body[:300]
                )
                source = (
                    SOURCE_API_ERROR
                    if response.status_code in RETRYABLE_STATUS_CODES
                    or response.status_code >= 500
                    else self._classify_error(body, default_fatal=True)
                )
                return AnswerCandidate.invalid(
                    "HTTP %s: %s" % (response.status_code, body[:200]), source
                )

            self._current_protocol = protocol
            return self._parse_api_response(question, response)

        return AnswerCandidate.invalid(
            "接口未接受任何输出协议: %s" % last_detail[:120], SOURCE_API_FATAL
        )

    def _active_protocols(self) -> List[str]:
        """返回当前可用的输出协议顺序（已探测不支持的协议会被剔除）。"""
        return [
            protocol
            for protocol in self.protocol_order
            if protocol not in self._unsupported_protocols
        ]

    def _apply_supported_protocols(self, body: str) -> None:
        """从拒绝正文中解析接口支持的输出格式，收紧协议梯度。

        只在能从正文里明确认出多个已知协议时才生效，并且永远不会剔除纯文本兜底，
        避免误判导致没有协议可用。
        """
        match = re.search(r"must be\s+(.+)", body or "", re.IGNORECASE)
        if not match:
            return
        allowed = {
            token
            for token in re.findall(r"[a-z_]+", match.group(1).lower())
            if token in self.protocol_order
        }
        if len(allowed) < 2:
            return
        for protocol in self.protocol_order:
            if protocol not in allowed and protocol != PROTOCOL_TEXT:
                self._unsupported_protocols.add(protocol)
        logger.info(
            "接口声明支持的输出格式为 %s，已调整协议梯度为 %s",
            "/".join(sorted(allowed)),
            "/".join(self._active_protocols()),
        )

    def _escalate_max_tokens(self) -> bool:
        """输出被截断时把额度加倍，直到硬上限。"""
        if self.max_tokens >= self.max_tokens_limit:
            return False
        previous = self.max_tokens
        self.max_tokens = min(previous * 2, self.max_tokens_limit)
        logger.warning(
            "模型输出被 max_tokens 截断，额度 %d → %d 后重试", previous, self.max_tokens
        )
        return True

    @staticmethod
    def _safe_text(response: Any) -> str:
        """安全读取响应正文，流式或异常响应也不能抛错。"""
        try:
            return str(getattr(response, "text", "") or "")
        except Exception:
            return ""

    @staticmethod
    def _is_format_rejection(body: str) -> bool:
        """判断 400 是否确实由“输出格式约束不被支持”引起。"""
        lowered = (body or "").lower()
        return any(token in lowered for token in FORMAT_REJECTION_TOKENS)

    @staticmethod
    def _classify_error(body: str, default_fatal: bool = False) -> str:
        """根据错误正文判断是否为不可重试的配置类错误。"""
        lowered = (body or "").lower()
        if any(token in lowered for token in FATAL_ERROR_TOKENS):
            return SOURCE_API_FATAL
        return SOURCE_API_FATAL if default_fatal else SOURCE_API_ERROR

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------
    def _parse_api_response(
        self, question: QuestionSnapshot, response: Any
    ) -> AnswerCandidate:
        """解析响应，空响应给出可诊断原因，推理型模型从推理文本兜底。"""
        raw_text = self._safe_text(response)
        try:
            data = response.json()
        except ValueError:
            # 常见原因：网关忽略了 stream=False 返回 SSE、或返回了 HTML 错误页。
            logger.error("模型响应不是合法 JSON: %s", raw_text[:300])
            return AnswerCandidate.invalid(
                "模型响应不是合法 JSON: %s" % raw_text[:150], SOURCE_API_ERROR
            )
        if not isinstance(data, dict):
            logger.error("模型响应结构异常: %s", raw_text[:300])
            return AnswerCandidate.invalid(
                "模型响应结构异常: %s" % raw_text[:150], SOURCE_API_ERROR
            )

        if data.get("error"):
            message = self._error_message(data["error"])
            logger.error("接口返回错误对象: %s", message)
            return AnswerCandidate.invalid(
                "接口错误: %s" % message, self._classify_error(message)
            )

        message, finish_reason, usage = self._extract_message(data)
        if message is None:
            logger.error("模型响应缺少 choices[0].message: %s", raw_text[:300])
            return AnswerCandidate.invalid(
                "模型响应结构异常（缺少 choices[0].message）: %s" % raw_text[:150],
                SOURCE_API_ERROR,
            )

        content = self._message_text(message, ("content",))
        used_reasoning = False
        if not content:
            # 推理型模型会把正文写进 reasoning_content，实测这是“空响应”的常见原因。
            reasoning = self._message_text(message, ("reasoning_content", "reasoning"))
            if reasoning:
                logger.warning(
                    "模型 content 为空，改用 reasoning_content 解析答案（推理型模型特征）"
                )
                content = reasoning
                used_reasoning = True
        if not content:
            detail = self._describe_empty_response(finish_reason, usage, raw_text)
            logger.error("模型返回空响应：%s", detail)
            source = SOURCE_TRUNCATED if finish_reason == "length" else SOURCE_API_ERROR
            return AnswerCandidate.invalid("模型返回空响应（%s）" % detail, source)

        structured = self.answer_parser.parse_json_text(question, content, "LLM")
        if structured.is_valid:
            return structured

        # 兼容模型没有遵循 JSON 格式或只返回“答案：...”的情况。
        fallback = self.answer_parser.from_explicit_text(
            question,
            content,
            source="LLM",
        )
        if fallback.is_valid:
            return fallback

        logger.warning("模型输出无法解析为答案: %s", content[:200])
        reason = structured.reason or fallback.reason or "模型答案无法校验"
        # 思考内容被截断时正文必然不完整：标记为可重试并提高额度，而不是直接放弃。
        if finish_reason == "length" or used_reasoning:
            detail = self._describe_empty_response(finish_reason, usage, raw_text)
            logger.warning("模型输出不完整（%s），将提高额度后重试", detail)
            return AnswerCandidate.invalid(
                "模型输出被截断，未包含可解析的答案（%s）" % detail,
                SOURCE_TRUNCATED if finish_reason == "length" else SOURCE_API_ERROR,
            )
        return AnswerCandidate.invalid(reason, "LLM")

    @staticmethod
    def _extract_message(data: Dict[str, Any]):
        """取出首个候选消息，返回 (message, finish_reason, usage)。"""
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            return None, "", None
        if not isinstance(choice, dict):
            return None, "", None
        message = choice.get("message")
        if not isinstance(message, dict):
            return None, "", None
        return message, str(choice.get("finish_reason") or ""), data.get("usage")

    @staticmethod
    def _message_text(message: Dict[str, Any], keys: Sequence[str]) -> str:
        """读取消息正文，兼容字符串与分片数组两种返回形态。"""
        for key in keys:
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                # 部分网关把正文返回成 [{type: text, text: ...}] 分片数组。
                merged = "".join(
                    str(part.get("text", ""))
                    for part in value
                    if isinstance(part, dict)
                )
                if merged.strip():
                    return merged.strip()
        return ""

    @staticmethod
    def _error_message(error: Any) -> str:
        """把错误对象统一转换成可读文本。"""
        if isinstance(error, dict):
            return str(
                error.get("message")
                or error.get("detail")
                or json.dumps(error, ensure_ascii=False)
            )
        return str(error)

    @staticmethod
    def _describe_empty_response(
        finish_reason: str, usage: Any, raw_text: str
    ) -> str:
        """拼装空响应 / 不完整输出的诊断信息。

        可操作的处置建议放在最前面，保证即使失败原因被截断展示也不会丢失关键提示。
        """
        parts = []
        if finish_reason == "length":
            parts.append(
                "输出额度被思考内容耗尽，请提高 DEEPSEEK_MAX_TOKENS 或换用非推理模型"
            )
        if finish_reason:
            parts.append("finish_reason=%s" % finish_reason)
        if usage:
            parts.append("usage=%s" % json.dumps(usage, ensure_ascii=False))
        parts.append("原始响应=%s" % raw_text[:200])
        return "，".join(parts)

    # ------------------------------------------------------------------
    # 请求体与提示词
    # ------------------------------------------------------------------
    def _build_payload(self, question: QuestionSnapshot, protocol: str) -> Dict[str, Any]:
        """按指定输出协议构造请求体。"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严谨的答题助手。必须根据题型和完整选项判断答案。"
                        "多选题要返回全部正确选项；答案文字必须原样对应选项文字；"
                        "解析使用简体中文。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(question, protocol),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        response_format = self._response_format(protocol)
        if response_format is not None:
            payload["response_format"] = response_format
        if self.enable_search:
            payload["enable_search"] = True
        return payload

    def _build_prompt(
        self, question: QuestionSnapshot, protocol: str = PROTOCOL_JSON_SCHEMA
    ) -> str:
        """根据题型与输出协议生成明确、可校验的提示词。"""
        rules = {
            QuestionType.SINGLE_CHOICE: (
                "这是单选题，answer_labels 必须且只能有一个选项编号，"
                "answer_texts 必须包含对应选项的完整文字。"
            ),
            QuestionType.MULTIPLE_CHOICE: (
                "这是多选题，answer_labels 必须列出全部正确选项，"
                "answer_texts 必须逐项复制对应选项的完整文字，不能只写编号。"
            ),
            QuestionType.TRUE_FALSE: (
                "这是判断题，answer_texts 只能填写“对”或“错”，"
                "answer_labels 可以为空。"
            ),
            QuestionType.FILL_BLANK: "这是填空题，请直接填写应填内容。",
            QuestionType.OPEN_ENDED: "这是开放题，请给出最精简的直接答案。",
        }
        rule = rules.get(question.question_type, "请给出最精简的直接答案。")
        options = self._format_options(question)
        if protocol == PROTOCOL_TEXT:
            # 纯文本协议下不再要求 JSON，改为固定单行格式，保证答案可解析。
            requirements = (
                "1. %s\n"
                "2. 只输出一行，固定格式：答案：<内容>；"
                "多选题用英文逗号分隔选项编号，例如 答案：A,C。\n"
                "3. 不要输出 Markdown，不要输出任何解释文字。" % rule
            )
            example = "答案：B"
        else:
            requirements = (
                "1. %s\n"
                "2. answer_labels、answer_texts 和 explanation 字段必须存在。\n"
                "3. 不要输出 Markdown，不要输出 JSON 之外的任何文字。\n"
                "4. 直接给出结论，不要输出思考过程。" % rule
            )
            example = (
                '{"answer_labels":["B"],"answer_texts":["选项文字"],'
                '"explanation":"简要理由"}'
            )
        return """题型：{type_label}
题目：{question}

选项：
{options}

要求：
{requirements}

输出示例：
{example}""".format(
            type_label=question.question_type.label,
            question=question.question,
            options=options,
            requirements=requirements,
            example=example,
        )

    @staticmethod
    def _format_options(question: QuestionSnapshot) -> str:
        """格式化有序选项，避免选题时丢失选项文字。"""
        if not question.options:
            return "无选项"
        return "\n".join(
            "%s. %s" % (option.label, option.text)
            for option in question.options
        )

    @staticmethod
    def _response_format(protocol: str) -> Optional[Dict[str, Any]]:
        """返回指定协议对应的 response_format，纯文本协议返回 None。"""
        if protocol == PROTOCOL_JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "quiz_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer_labels": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "answer_texts": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "explanation": {"type": "string"},
                        },
                        "required": [
                            "answer_labels",
                            "answer_texts",
                            "explanation",
                        ],
                        "additionalProperties": False,
                    },
                },
            }
        if protocol == PROTOCOL_JSON_OBJECT:
            # DeepSeek 等接口只支持 json_object，需要在提示词中出现 JSON 字样。
            return {"type": "json_object"}
        return None

    def _parse_response(
        self,
        text: str,
        question: QuestionSnapshot,
    ) -> AnswerCandidate:
        """兼容旧测试或外部调用方的响应解析入口。"""
        parsed = self.answer_parser.parse_json_text(question, text, "LLM")
        if parsed.is_valid:
            return parsed
        return self.answer_parser.from_explicit_text(question, text, "LLM")

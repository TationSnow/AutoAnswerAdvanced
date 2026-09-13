"""LM Studio 兼容接口答题器。

优先查询本地题库，未命中时请求结构化 JSON，并通过题型策略进行二次校验。
"""
import logging
import time
from typing import Any, Dict, Optional

import requests

from answer_strategy import AnswerParser
from models import AnswerCandidate, QuestionSnapshot, QuestionType


logger = logging.getLogger(__name__)


class DeepSeekSolver:
    """OpenAI 兼容接口答题器，类名保留以兼容现有配置和调用。"""

    MAX_RETRIES = 3
    RETRY_DELAY = 1
    TIMEOUT = 30

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        enable_search: bool = False,
        knowledge_base: Any = None,
    ) -> None:
        """初始化会话和题库依赖。"""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enable_search = enable_search
        self.kb = knowledge_base
        self.answer_parser = AnswerParser()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % api_key,
            }
        )
        logger.info(
            "AI 解题器初始化完成，模型: %s，联网: %s",
            model,
            enable_search,
        )

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

    def _call_api_with_retry(self, question: QuestionSnapshot) -> AnswerCandidate:
        """调用模型并执行有限次数重试。"""
        last_error = ""
        for attempt in range(self.MAX_RETRIES):
            started = time.time()
            try:
                result = self._call_api(question)
                logger.info(
                    "模型请求耗时 %.2fs (尝试 %d/%d)",
                    time.time() - started,
                    attempt + 1,
                    self.MAX_RETRIES,
                )
                if result.is_valid or result.source != "api_error":
                    return result
                last_error = result.reason
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "模型请求失败 (%d/%d): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    exc,
                )
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY * (attempt + 1))
        return AnswerCandidate.invalid(
            "API 失败: %s" % last_error[:80],
            "error",
        )

    def _call_api(self, question: QuestionSnapshot) -> AnswerCandidate:
        """发送结构化输出请求并解析模型答案。"""
        payload = {
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
                    "content": self._build_prompt(question),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "stream": False,
            "response_format": self._response_format(),
        }
        if self.enable_search:
            payload["enable_search"] = True

        response = self.session.post(
            "%s/chat/completions" % self.base_url,
            json=payload,
            timeout=self.TIMEOUT,
        )

        # 部分 OpenAI 兼容服务不支持 json_schema，此时降级为普通文本协议。
        if response.status_code == 400 and "response_format" in payload:
            logger.warning("模型不支持 JSON Schema，降级为文本答案协议")
            payload.pop("response_format", None)
            response = self.session.post(
                "%s/chat/completions" % self.base_url,
                json=payload,
                timeout=self.TIMEOUT,
            )

        if response.status_code != 200:
            detail = "HTTP %s: %s" % (
                response.status_code,
                response.text[:200],
            )
            logger.error(detail)
            return AnswerCandidate.invalid(detail, "api_error")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            return AnswerCandidate.invalid("模型响应结构异常: %s" % exc, "api_error")
        if not content:
            return AnswerCandidate.invalid("模型返回空响应", "api_error")

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
        return AnswerCandidate.invalid(
            structured.reason or fallback.reason or "模型答案无法校验",
            "LLM",
        )

    def _build_prompt(self, question: QuestionSnapshot) -> str:
        """根据题型生成明确、可校验的提示词。"""
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
        return """题型：{type_label}
题目：{question}

选项：
{options}

要求：
1. {rule}
2. answer_labels、answer_texts 和 explanation 字段必须存在。
3. 不要输出 Markdown，不要输出 JSON 之外的任何文字。""".format(
            type_label=question.question_type.label,
            question=question.question,
            options=options,
            rule=rule,
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
    def _response_format() -> Dict[str, Any]:
        """返回 LM Studio 可识别的 JSON Schema。"""
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


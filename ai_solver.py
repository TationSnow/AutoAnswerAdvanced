"""AI解题模块 - 直接用 requests 调用 LMStudio/DeepSeek 兼容接口。
避开 OpenAI SDK 与 LMStudio 的 payload 兼容性问题（502）。
"""
import re
import time
import logging
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class DeepSeekSolver:
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    TIMEOUT = 30

    def __init__(self, api_key: str, base_url: str, model: str,
                 enable_search: bool = False, knowledge_base=None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enable_search = enable_search
        self.kb = knowledge_base

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        logger.info(f"AI解题器初始化完成，模型: {model}，联网: {enable_search}")

    # ------------------------------------------------------------------
    def solve(self, question_data: Dict) -> Dict:
        if not question_data or not question_data.get('is_valid'):
            return {"answer": "?", "detail": "未识别题目", "source": "error"}

        question = question_data['question']
        options = question_data.get('options', {})
        question_type = question_data.get('question_type', 'choice')
        raw_text = question_data.get('raw_text', '')

        if not question:
            return {"answer": "?", "detail": "题目为空", "source": "error"}

        # 1) 题库查询
        if self.kb is not None:
            t0 = time.time()
            hit = self.kb.search(question, options)
            if hit:
                logger.info(f"⏱ 题库命中 ({time.time()-t0:.3f}s): {hit['source']}")
                return hit

        # 2) LLM 调用
        return self._call_api_with_retry(question, options, question_type, raw_text)

    # ------------------------------------------------------------------
    def _call_api_with_retry(self, question, options, qtype, raw_text) -> Dict:
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.time()
                result = self._call_api(question, options, qtype, raw_text)
                t1 = time.time()
                logger.info(f"⏱ API单次: {t1-t0:.2f}s (尝试{attempt+1}/{self.MAX_RETRIES})")
                if result.get('source') != 'api_error':
                    return result
                last_error = result.get('detail', '未知错误')
            except Exception as e:
                last_error = str(e)
                logger.warning(f"API失败 ({attempt+1}/{self.MAX_RETRIES}): {e}")
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY * (attempt + 1))
        logger.error(f"API全部失败: {last_error}")
        return {"answer": "?", "detail": f"API失败: {str(last_error)[:60]}",
                "source": "error"}

    # ------------------------------------------------------------------
    def _call_api(self, question, options, qtype, raw_text) -> Dict:
        options_text = self._format_options(options)
        prompt = self._build_prompt(question, options_text, qtype, raw_text)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是答题助手。请根据题型直接输出标准答案，并补一行简短解析。"
                        "选择题输出选项标记（如 A 或 A,C）；判断题输出\"正确\"或\"错误\"；"
                        "填空题直接输出应填内容；如果题干不完整，也尽量给出最可能答案。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 200,
            "stream": False,
        }
        if self.enable_search:
            payload["enable_search"] = True

        try:
            logger.debug(f"POST {self.base_url}/chat/completions "
                         f"len(question)={len(question)}")
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.TIMEOUT,
            )
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.error(err)
                return {"answer": "?", "detail": err, "source": "api_error"}

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            logger.debug(f"API响应: {content[:120]}")
            return self._parse_response(content, qtype, options)

        except requests.exceptions.Timeout:
            raise RuntimeError("请求超时")
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            raise

    # ------------------------------------------------------------------
    def _format_options(self, options: Dict) -> str:
        if not options:
            return "无选项"
        lines = []
        for k in sorted(options.keys()):
            v = options[k]
            if len(v) > 80:
                v = v[:77] + "..."
            lines.append(f"{k}. {v}")
        return '\n'.join(lines)

    def _build_prompt(self, question, options_text, qtype, raw_text) -> str:
        rule = {
            "choice": "这是选择题，请输出最可能的选项标记，例如 A 或 A,C。",
            "true_false": "这是判断题，请只输出\"正确\"或\"错误\"。",
            "fill_blank": "这是填空题，请直接输出应填内容；多个空用分号分隔。",
            "open_ended": "这不是标准选择题，请输出最精简的直接答案。",
        }.get(qtype, "请输出最精简的直接答案。")
        return f"""题型：{qtype}
题目：{question}

选项：
{options_text}

OCR原文：
{raw_text[:800]}

要求：
1. {rule}
2. 严格按以下格式输出两行：
答案：...
解析：..."""

    # ------------------------------------------------------------------
    def _parse_response(self, text: str, qtype: str, options: Dict) -> Dict:
        if not text:
            return {"answer": "?", "detail": "空响应", "source": "api_error"}

        answer = self._extract_answer(text, qtype, options)
        detail = ""
        for pat in [r'解析[：:]\s*(.+?)(?:\n|$)', r'原因[：:]\s*(.+?)(?:\n|$)',
                    r'因为[：:]\s*(.+?)(?:\n|$)']:
            m = re.search(pat, text, re.DOTALL)
            if m:
                detail = m.group(1).strip()
                break
        if not detail:
            m = re.search(r'[：:](.+?)(?:\n|$)', text)
            if m:
                cand = m.group(1).strip()
                if cand and cand.upper() not in ['A', 'B', 'C', 'D']:
                    detail = cand
        detail = detail[:200] if detail else ""
        return {"answer": answer, "detail": detail, "source": "LLM"}

    def _extract_answer(self, text: str, qtype: str, options: Dict) -> str:
        m = re.search(r'答案[：:]\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        answer_line = m.group(1).strip() if m else ""
        first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
        candidate = answer_line or first_line

        if qtype == "true_false":
            norm = candidate.replace(" ", "")
            if re.search(r'(正确|对|√|是)', norm):
                return "正确"
            if re.search(r'(错误|错|×|否)', norm):
                return "错误"

        if qtype == "choice" and options:
            labels = sorted(options.keys(), key=len, reverse=True)
            matched = []
            for label in labels:
                if re.search(
                    r'(?<![A-Za-z0-9])%s(?![A-Za-z0-9])' % re.escape(label),
                    candidate, re.IGNORECASE
                ):
                    if label.upper() not in matched:
                        matched.append(label.upper())
            if matched:
                return ",".join(matched)

        if candidate:
            candidate = re.split(r'解析[：:]', candidate)[0].strip()
            candidate = candidate.strip('，,。.;； ')
            if candidate:
                return candidate[:80]
        return "?"
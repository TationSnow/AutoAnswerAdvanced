"""
OCR engine module - RapidOCR (ONNX Runtime).
在原有基础上：保留文本框坐标、关联选项坐标、识别下一题/提交按钮。
"""
import os
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import BUTTON_KEYWORDS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCREngine:
    """RapidOCR engine: ONNX Runtime backend."""

    CIRCLED_NUM_MAP: Dict[str, str] = {
        '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
        '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
        '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
        '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20',
    }
    CN_NUM_MAP: Dict[str, str] = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
    }
    OPTION_LINE_RE = re.compile(
        r'^\s*(?P<label>[A-Za-z]|\d{1,2}|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|'
        r'[一二三四五六七八九十])\s*[\.\、．:：\)）]\s*(?P<content>.+?)\s*$'
    )
    INLINE_LETTER_OPTION_RE = re.compile(
        r'(?<![A-Za-z0-9])(?P<label>[A-Z])\s*[\.\、．:：\)）]\s*'
    )
    FILL_BLANK_PATTERNS: List[str] = [
        r'_{2,}', r'（\s*）', r'\(\s*\)', r'【\s*】', r'\[\s*\]', r'____+',
    ]
    TRUE_FALSE_HINTS: Tuple[str, ...] = (
        '判断题', '判断下列', '对错', '是否正确', '是否错误', '是非题'
    )
    OPTION_PATTERNS: Dict[str, List[str]] = {
        'A': [r'A[．、.、:：]\s*(.+?)(?=(?:[B-D][．、.、:：]|\Z))',
              r'A[）)]\s*(.+?)(?=(?:[B-D][）)]|\Z))',
              r'A\s{2,}(.+?)(?=(?:[B-D]\s{2,}|\Z))',
              r'①\s*[Aa]\.?\s*(.+?)(?=(?:②\s*[Bb]|\Z))'],
        'B': [r'B[．、.、:：]\s*(.+?)(?=(?:[C-D][．、.、:：]|\Z))',
              r'B[）)]\s*(.+?)(?=(?:[C-D][）)]|\Z))',
              r'B\s{2,}(.+?)(?=(?:[C-D]\s{2,}|\Z))',
              r'②\s*[Bb]\.?\s*(.+?)(?=(?:③\s*[Cc]|\Z))'],
        'C': [r'C[．、.、:：]\s*(.+?)(?=(?:D[．、.、:：]|\Z))',
              r'C[）)]\s*(.+?)(?=(?:D[）)]|\Z))',
              r'C\s{2,}(.+?)(?=(?:D\s{2,}|\Z))',
              r'③\s*[Cc]\.?\s*(.+?)(?=(?:④\s*[Dd]|\Z))'],
        'D': [r'D[．、.、:：]\s*(.+?)(?=\Z)',
              r'D[）)]\s*(.+?)(?=\Z)',
              r'D\s{2,}(.+?)(?=\Z)',
              r'④\s*[Dd]\.?\s*(.+?)(?=\Z)'],
    }

    QUESTION_CLEAN_PATTERNS: List[Tuple[str, str]] = [
        (r'^\d+[\.\u3001\s]*', ''),
        (r'^[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+[\.\u3001\s]*', ''),
        (r'^(\u9898\u76ee|\u95ee\u9898|\u8bf7\u56de\u7b54)[\uff1a:]?\s*', ''),
        (r'^\s*', ''),
        (r'\s*$', ''),
    ]

    def __init__(self, use_gpu: bool = False, lang: str = 'ch',
                 show_log: bool = True) -> None:
        self.ocr: Any = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._show_log = show_log
        if not show_log:
            logging.getLogger('rapidocr_onnxruntime').setLevel(logging.WARNING)
        self._init_model()

    # ------------------------------------------------------------------
    def _init_model(self) -> None:
        try:
            logger.info("Loading RapidOCR [ONNX Runtime]...")
            from rapidocr_onnxruntime import RapidOCR

            self.ocr = RapidOCR()

            warm = np.zeros((60, 120, 3), dtype=np.uint8)
            warm[:] = 255
            _ = self.ocr(warm)
            logger.info("RapidOCR loaded & warmed up")
        except ImportError as exc:
            logger.error("RapidOCR import failed: %s", exc)
            raise ImportError("pip install rapidocr_onnxruntime") from exc
        except Exception as exc:
            logger.error("RapidOCR init failed: %s", exc)
            raise RuntimeError("OCR engine init failed: %s" % exc) from exc

    # ------------------------------------------------------------------
    def recognize(self, image: Any) -> Dict[str, Any]:
        try:
            img_array = self._prepare_image(image)
            if img_array is None:
                return self._empty_result("unsupported image type")

            t0 = time.time()
            result, _ = self.ocr(img_array)
            t1 = time.time()

            if not result:
                return self._empty_result("no text detected")

            items = self._extract_items(result)
            if not items:
                return self._empty_result("text confidence too low")

            texts = [it['text'] for it in items]
            confidences = [it['score'] for it in items]
            full_text = '\n'.join(texts)
            avg_conf = sum(confidences) / len(confidences)

            parsed = self._parse_question(full_text)
            self._attach_boxes(parsed, items)

            parsed["confidence"] = avg_conf
            logger.info(
                "OCR: infer=%.2fs | %d lines conf=%.2f",
                t1 - t0, len(texts), avg_conf
            )
            return parsed
        except Exception as exc:
            logger.error("OCR recognition error: %s", exc)
            return self._empty_result(str(exc))

    # ------------------------------------------------------------------
    def _prepare_image(self, image: Any) -> Optional[np.ndarray]:
        try:
            if isinstance(image, Image.Image):
                arr = np.array(image)
            elif isinstance(image, np.ndarray):
                arr = image
            else:
                return None
            if len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            return arr
        except Exception as exc:
            logger.error("Image conversion failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    def _extract_items(self, ocr_result: Any) -> List[Dict[str, Any]]:
        """RapidOCR 返回 [[box], text, score]；这里保留 box。"""
        items: List[Dict[str, Any]] = []
        if not ocr_result:
            return items
        for item in ocr_result:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                box = item[0]
                text = str(item[1])
                score = float(item[2])
                if score > 0.2:
                    items.append({"text": text, "score": score, "box": box})
        return items

    @staticmethod
    def _box_center(box) -> Tuple[int, int]:
        try:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        except Exception:
            return 0, 0

    # ------------------------------------------------------------------
    def _attach_boxes(self, parsed: Dict, items: List[Dict]):
        """把 OCR 文本块坐标关联到选项标签与按钮。"""
        option_boxes: Dict[str, Dict] = {}
        button_boxes: Dict[str, Dict] = {}
        options = parsed.get('options', {})

        for it in items:
            text = it['text'].strip()
            if not text:
                continue
            box = it['box']
            center = self._box_center(box)
            info = {"box": box, "center": center, "text": text}

            # 选项标签
            m = self.OPTION_LINE_RE.match(text)
            if m:
                label = self._normalize_option_label(m.group('label'))
                if label in options and label not in option_boxes:
                    option_boxes[label] = info

            # 按钮（限制长度避免正文误命中）
            if len(text) <= 12:
                for btn_key, kws in BUTTON_KEYWORDS.items():
                    if btn_key in button_boxes:
                        continue
                    if any(kw in text for kw in kws):
                        button_boxes[btn_key] = info
                        break

        parsed['option_boxes'] = option_boxes
        parsed['button_boxes'] = button_boxes

    # ------------------------------------------------------------------
    def _parse_question(self, text: str) -> Dict[str, Any]:
        if not text:
            return self._empty_result("text is empty")
        lines = self._normalize_lines(text)
        text = '\n'.join(lines)

        options, option_start_index = self._extract_options_from_lines(lines)
        if options:
            question = self._extract_question_from_lines(lines, option_start_index)
        else:
            options = self._extract_inline_options(text)
            question = self._extract_question(text, options)

        question = self._clean_question(question)
        qtype = self._detect_question_type(question, options, text)
        return {
            "question": question[:300],
            "options": options,
            "raw_text": text,
            "question_type": qtype,
            "is_valid": self._is_valid_question(question, options, qtype),
            "confidence": 0.0,
            "option_boxes": {},
            "button_boxes": {},
        }

    # -------------------- 以下保持原有逻辑 --------------------
    def _normalize_lines(self, text: str) -> List[str]:
        out = []
        for line in text.split('\n'):
            cleaned = re.sub(r'[^\S\n]+', ' ', line).strip()
            if cleaned:
                out.append(cleaned)
        return out

    def _normalize_option_label(self, label: str) -> str:
        label = label.strip()
        if label in self.CIRCLED_NUM_MAP:
            return self.CIRCLED_NUM_MAP[label]
        if label in self.CN_NUM_MAP:
            return self.CN_NUM_MAP[label]
        if label.isalpha():
            return label.upper()
        return label

    def _extract_options_from_lines(self, lines):
        options: Dict[str, str] = {}
        current_label = None
        option_start_index = len(lines)
        for idx, line in enumerate(lines):
            m = self.OPTION_LINE_RE.match(line)
            if m:
                label = self._normalize_option_label(m.group('label'))
                content = m.group('content').strip()
                if option_start_index == len(lines):
                    option_start_index = idx
                if content:
                    options[label] = content[:200]
                    current_label = label
                continue
            if current_label and option_start_index < len(lines):
                options[current_label] = (
                    options[current_label] + ' ' + line
                )[:200]
        return options, option_start_index

    def _extract_inline_options(self, text: str) -> Dict[str, str]:
        options: Dict[str, str] = {}
        matches = list(self.INLINE_LETTER_OPTION_RE.finditer(text))
        if len(matches) < 2:
            for letter in ['A', 'B', 'C', 'D']:
                content = self._extract_option(text, letter)
                if content:
                    options[letter] = content[:200].strip()
            return options
        for idx, m in enumerate(matches):
            label = m.group('label').upper()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            content = text[start:end].strip(" \n\t,，;；")
            if content:
                options[label] = content[:200]
        return options

    def _extract_option(self, text: str, letter: str) -> Optional[str]:
        for pattern in self.OPTION_PATTERNS.get(letter, []):
            try:
                m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if m and m.group(1).strip():
                    return m.group(1).strip()
            except re.error:
                continue
        return None

    def _extract_question(self, text: str, options: Dict[str, str]) -> str:
        if not options:
            return text
        first_pos = len(text)
        for label in options:
            pat = re.compile(
                r'(?<![A-Za-z0-9])%s\s*[\.\、．:：\)）]' % re.escape(label),
                re.IGNORECASE
            )
            m = pat.search(text)
            if m and m.start() < first_pos:
                first_pos = m.start()
        question = text[:first_pos].strip() if first_pos < len(text) else text
        return self._clean_question(question)

    def _extract_question_from_lines(self, lines, option_start_index):
        if option_start_index <= 0:
            return '\n'.join(lines)
        return '\n'.join(lines[:option_start_index]).strip()

    def _clean_question(self, question: str) -> str:
        for pat, rep in self.QUESTION_CLEAN_PATTERNS:
            question = re.sub(pat, rep, question, flags=re.IGNORECASE)
        return question.strip()

    def _detect_question_type(self, question, options, raw_text) -> str:
        combined = f"{question}\n{raw_text}".strip()
        if self._is_true_false_question(question, options, combined):
            return "true_false"
        if self._is_fill_blank_question(combined, options):
            return "fill_blank"
        if options:
            return "choice"
        return "open_ended"

    def _is_true_false_question(self, question, options, combined) -> bool:
        vals = ''.join(options.values()).replace(' ', '')
        if len(options) > 2:
            return False
        if any(h in combined for h in self.TRUE_FALSE_HINTS):
            return True
        if '正确' in vals and '错误' in vals:
            return True
        if ('对' in vals and '错' in vals) or ('√' in vals and '×' in vals):
            return True
        if not options and re.search(r'(正确|错误|对|错|是否)', question):
            return True
        return False

    def _is_fill_blank_question(self, combined, options) -> bool:
        if options:
            return False
        if '填空题' in combined or '填空' in combined:
            return True
        return any(re.search(p, combined) for p in self.FILL_BLANK_PATTERNS)

    def _is_valid_question(self, question, options, qtype) -> bool:
        qlen = len(question.replace('\n', '').strip())
        if qtype == "choice":
            return qlen >= 4 and len(options) >= 2
        if qtype == "true_false":
            return qlen >= 4
        if qtype in ("fill_blank", "open_ended"):
            return qlen >= 6
        return qlen >= 6 or len(options) >= 2

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "question": "", "options": {}, "raw_text": "",
            "question_type": "unknown", "is_valid": False,
            "confidence": 0.0, "error": reason,
            "option_boxes": {}, "button_boxes": {},
        }

    def get_version_info(self) -> Dict[str, Any]:
        try:
            import rapidocr_onnxruntime
            return {
                "rapidocr_version": getattr(rapidocr_onnxruntime, '__version__', 'unknown'),
                "ocr_type": "RapidOCR (ONNX Runtime)",
                "language": self.lang,
                "gpu_enabled": self.use_gpu,
            }
        except Exception as exc:
            return {"error": str(exc)}
"""RapidOCR 识别与题目结构化解析。

本模块负责把屏幕文字转换为稳定的 QuestionSnapshot，并过滤题库页面上的
标题、分页和操作按钮，避免页面装饰进入题目语义。
"""
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import BUTTON_KEYWORDS
from models import ButtonItem, OptionItem, QuestionSnapshot, QuestionType


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OCREngine:
    """RapidOCR 封装和页面结构解析器。"""

    CIRCLED_NUM_MAP: Dict[str, str] = {
        "①": "A",
        "②": "B",
        "③": "C",
        "④": "D",
        "⑤": "E",
        "⑥": "F",
        "⑦": "G",
        "⑧": "H",
        "⑨": "I",
        "⑩": "J",
        "⑪": "K",
        "⑫": "L",
        "⑬": "M",
        "⑭": "N",
        "⑮": "O",
        "⑯": "P",
        "⑰": "Q",
        "⑱": "R",
        "⑲": "S",
        "⑳": "T",
    }
    CN_NUM_MAP: Dict[str, str] = {
        "一": "A",
        "二": "B",
        "三": "C",
        "四": "D",
        "五": "E",
        "六": "F",
        "七": "G",
        "八": "H",
        "九": "I",
        "十": "J",
    }
    OPTION_LINE_RE = re.compile(
        r"^\s*(?P<label>[A-Za-z]|\d{1,2}|"
        r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|"
        r"[一二三四五六七八九十])\s*[\.\、．:：\)）]\s*(?P<content>.+?)\s*$"
    )
    QUESTION_NUMBER_RE = re.compile(
        r"^\s*[\[【]?\s*A?\s*(?P<id>\d{7})\s*[\]】]?\s*"
        r"(?:\d{1,3}\s*[、.．:：]?)?\s*",
        re.IGNORECASE,
    )
    EXTERNAL_ID_RE = re.compile(r"A\s*(\d{7})", re.IGNORECASE)
    PAGE_NUMBER_RE = re.compile(r"^\s*[一二三四五六七八九十\d]+\s*/\s*\d+\s*$")
    FILL_BLANK_PATTERNS: Tuple[str, ...] = (
        r"_{2,}",
        r"（\s*）",
        r"\(\s*\)",
        r"【\s*】",
        r"\[\s*\]",
    )
    TYPE_LABELS: Dict[QuestionType, Tuple[str, ...]] = {
        QuestionType.SINGLE_CHOICE: ("单选题", "单選題", "单选"),
        QuestionType.MULTIPLE_CHOICE: ("多选题", "多選題", "多选"),
        QuestionType.TRUE_FALSE: ("判断题", "判斷題", "是非题"),
    }
    TRUE_FALSE_TEXTS: Tuple[str, ...] = (
        "对",
        "错",
        "正确",
        "错误",
        "√",
        "×",
        "是",
        "否",
    )

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "ch",
        show_log: bool = True,
    ) -> None:
        """初始化 OCR 模型。"""
        self.ocr: Any = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._show_log = show_log
        if not show_log:
            logging.getLogger("rapidocr_onnxruntime").setLevel(logging.WARNING)
        self._init_model()

    def _init_model(self) -> None:
        """加载并预热 RapidOCR。"""
        try:
            logger.info("正在加载 RapidOCR [ONNX Runtime]...")
            from rapidocr_onnxruntime import RapidOCR

            self.ocr = RapidOCR()
            warm = np.zeros((60, 120, 3), dtype=np.uint8)
            warm[:] = 255
            _ = self.ocr(warm)
            logger.info("RapidOCR 已加载并完成预热")
        except ImportError as exc:
            logger.error("RapidOCR 导入失败: %s", exc)
            raise ImportError("请安装 rapidocr_onnxruntime") from exc
        except Exception as exc:
            logger.error("RapidOCR 初始化失败: %s", exc)
            raise RuntimeError("OCR 引擎初始化失败: %s" % exc) from exc

    def recognize(self, image: Any) -> QuestionSnapshot:
        """识别图像并返回结构化题目。"""
        try:
            img_array = self._prepare_image(image)
            if img_array is None:
                return self._empty_result("不支持的图像类型")

            started = time.time()
            result, _ = self.ocr(img_array)
            elapsed = time.time() - started
            if not result:
                return self._empty_result("未检测到文字")

            items = self._extract_items(result)
            if not items:
                return self._empty_result("文字置信度过低")

            snapshot = self.parse_items(items)
            snapshot.confidence = sum(item["score"] for item in items) / len(items)
            logger.info(
                "OCR: 推理=%.2fs | %d 行 | 平均置信度=%.2f | 题型=%s",
                elapsed,
                len(items),
                snapshot.confidence,
                snapshot.question_type.value,
            )
            return snapshot
        except Exception as exc:
            logger.error("OCR 识别异常: %s", exc)
            return self._empty_result(str(exc))

    def parse_items(self, items: List[Dict[str, Any]]) -> QuestionSnapshot:
        """解析 RapidOCR 文本块，供正式流程和单元测试共用。"""
        if not items:
            return self._empty_result("没有可解析的文字块")

        ordered_items = self._sort_items(items)
        full_text = "\n".join(item["text"].strip() for item in ordered_items)
        external_id = self._extract_external_id(full_text)
        badge_type = self._detect_badge_type(ordered_items)
        options, option_start_index = self._extract_options(
            ordered_items, badge_type
        )
        question = self._extract_question(
            ordered_items, option_start_index, external_id, badge_type
        )
        buttons = self._extract_buttons(ordered_items, options)
        question_type = self._detect_question_type(
            question, options, full_text, badge_type, buttons
        )
        is_valid = self._is_valid_question(question, options, question_type)

        return QuestionSnapshot(
            external_id=external_id,
            question=question,
            question_type=question_type,
            options=options,
            raw_text=full_text,
            confidence=0.0,
            is_valid=is_valid,
            button_boxes=buttons,
        )

    def _prepare_image(self, image: Any) -> Optional[np.ndarray]:
        """将 PIL 图像或 NumPy 数组转换为 OCR 输入。"""
        try:
            if isinstance(image, Image.Image):
                array = np.array(image)
            elif isinstance(image, np.ndarray):
                array = image
            else:
                return None
            if len(array.shape) == 2:
                array = np.stack([array] * 3, axis=-1)
            elif array.shape[-1] == 4:
                array = array[:, :, :3]
            return array
        except Exception as exc:
            logger.error("图像转换失败: %s", exc)
            return None

    def _extract_items(self, ocr_result: Any) -> List[Dict[str, Any]]:
        """保留 RapidOCR 文本、置信度和坐标。"""
        items: List[Dict[str, Any]] = []
        for item in ocr_result or []:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            try:
                score = float(item[2])
            except (TypeError, ValueError):
                continue
            if score <= 0.2:
                continue
            text = str(item[1]).strip()
            if text:
                items.append({"text": text, "score": score, "box": item[0]})
        return items

    @staticmethod
    def _box_center(box: Any) -> Tuple[int, int]:
        """计算文本框中心点。"""
        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        except Exception:
            return 0, 0

    def _sort_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按从上到下、从左到右恢复阅读顺序。"""
        normalized = []
        for item in items:
            if isinstance(item, dict):
                copied = dict(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                copied = {
                    "box": item[0],
                    "text": str(item[1]).strip(),
                    "score": float(item[2]),
                }
            else:
                continue
            copied["center"] = self._box_center(copied.get("box"))
            normalized.append(copied)

        # 将纵向距离接近的文本框视为同一阅读行，再按横坐标排序。
        normalized.sort(key=lambda value: value["center"][0])
        lines: List[List[Dict[str, Any]]] = []
        for item in normalized:
            y = item["center"][1]
            for line in lines:
                if abs(line[0]["center"][1] - y) <= 10:
                    line.append(item)
                    break
            else:
                lines.append([item])
        lines.sort(key=lambda line: min(item["center"][1] for item in line))
        ordered = []
        for line in lines:
            line.sort(key=lambda item: item["center"][0])
            ordered.extend(line)
        return ordered

    def _detect_badge_type(
        self, items: List[Dict[str, Any]]
    ) -> Optional[QuestionType]:
        """从页面题型标签识别题型。"""
        for item in items:
            compact = re.sub(r"\s+", "", item["text"])
            for question_type, labels in self.TYPE_LABELS.items():
                if any(label in compact for label in labels):
                    return question_type
        return None

    def _extract_external_id(self, text: str) -> Optional[str]:
        """提取页面题号的七位数字部分。"""
        match = self.EXTERNAL_ID_RE.search(text or "")
        return "A%s" % match.group(1) if match else None

    def _extract_options(
        self,
        items: List[Dict[str, Any]],
        badge_type: Optional[QuestionType],
    ) -> Tuple[List[OptionItem], int]:
        """解析有序选项，并忽略按钮和分页文字。"""
        options: List[OptionItem] = []
        current: Optional[OptionItem] = None
        option_start_index = len(items)
        semantic_index = 0

        for index, item in enumerate(items):
            raw_text = item["text"].strip()
            text = self._strip_page_controls(raw_text)
            if not text or self._is_page_chrome(text):
                continue
            if self._button_role(raw_text) is not None:
                continue

            match = self.OPTION_LINE_RE.match(text)
            if match:
                label = self._normalize_option_label(match.group("label"))
                content = match.group("content").strip()
                if option_start_index == len(items):
                    option_start_index = index
                current = OptionItem(
                    label=label,
                    text=content,
                    box=item.get("box"),
                    center=item.get("center"),
                )
                options.append(current)
                continue

            semantic = self._normalize_semantic_bool(text)
            if (
                badge_type == QuestionType.TRUE_FALSE
                and semantic is not None
                and semantic_index < 2
            ):
                if option_start_index == len(items):
                    option_start_index = index
                current = OptionItem(
                    label="T" if semantic else "F",
                    text=text,
                    box=item.get("box"),
                    center=item.get("center"),
                )
                options.append(current)
                semantic_index += 1
                continue

            if current is not None:
                current.text = (current.text + " " + text).strip()[:500]

        return options, option_start_index

    def _extract_question(
        self,
        items: List[Dict[str, Any]],
        option_start_index: int,
        external_id: Optional[str],
        badge_type: Optional[QuestionType],
    ) -> str:
        """提取干净题干，移除页头、题型和题号。"""
        lines: List[str] = []
        using_id_anchor = False
        for item in items[:option_start_index]:
            text = self._strip_page_controls(item["text"])
            if not text or self._is_page_chrome(text):
                continue
            if self._button_role(text) is not None:
                continue
            if external_id and external_id[1:] in text.replace(" ", ""):
                using_id_anchor = True
            if not using_id_anchor and lines:
                continue
            cleaned = self.QUESTION_NUMBER_RE.sub("", text, count=1).strip()
            if cleaned:
                lines.append(cleaned)

        if not lines:
            fallback = []
            for item in items[:option_start_index]:
                text = self._strip_page_controls(item["text"])
                if text and not self._is_page_chrome(text):
                    fallback.append(
                        self.QUESTION_NUMBER_RE.sub("", text, count=1).strip()
                    )
            lines = [line for line in fallback if line]
        question = "\n".join(lines).strip()
        for label in self.TYPE_LABELS.get(badge_type or QuestionType.UNKNOWN, ()):
            question = question.replace(label, "")
        return question.strip()[:500]

    def _extract_buttons(
        self,
        items: List[Dict[str, Any]],
        options: List[OptionItem],
    ) -> Dict[str, ButtonItem]:
        """识别选项区下方的明确按钮角色。"""
        last_option_y = max(
            (option.center[1] for option in options if option.center is not None),
            default=0,
        )
        buttons: Dict[str, ButtonItem] = {}
        for item in items:
            raw_text = item["text"].strip()
            role = self._button_role(raw_text)
            if role is None or role in buttons:
                continue
            center = item.get("center") or (0, 0)
            # 只接受选项下方的按钮，避免把题干或导航文本误当成操作按钮。
            if center[1] < last_option_y:
                continue
            buttons[role] = ButtonItem(
                role=role,
                text=raw_text,
                box=item.get("box"),
                center=center,
            )
        return buttons

    def _button_role(self, text: str) -> Optional[str]:
        """使用配置中的明确关键词判定按钮角色。"""
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return None
        for role, keywords in BUTTON_KEYWORDS.items():
            for keyword in keywords:
                normalized = re.sub(r"\s+", "", keyword)
                if compact == normalized:
                    return role
        return None

    def _strip_page_controls(self, text: str) -> str:
        """从与选项粘连的 OCR 文本中截掉翻页按钮和页码。"""
        cleaned = str(text or "").strip()
        markers = [
            keyword
            for keywords in BUTTON_KEYWORDS.values()
            for keyword in keywords
        ]
        markers.extend(["上一题", "下一题"])
        positions = [
            cleaned.find(marker)
            for marker in markers
            if cleaned.find(marker) > 0
        ]
        if positions:
            cleaned = cleaned[: min(positions)].strip()
        cleaned = re.sub(r"\s*[一二三四五六七八九十\d]+\s*/\s*\d+\s*$", "", cleaned)
        return cleaned.strip()

    def _is_page_chrome(self, text: str) -> bool:
        """判断文本是否属于页面标题、题型标签或分页。"""
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return True
        if compact in {"题库练习", "上一题", "下一题"}:
            return True
        if self.PAGE_NUMBER_RE.match(compact):
            return True
        for labels in self.TYPE_LABELS.values():
            if compact in labels:
                return True
        return False

    def _normalize_option_label(self, label: str) -> str:
        """统一选项标签。"""
        value = (label or "").strip()
        if value in self.CIRCLED_NUM_MAP:
            return self.CIRCLED_NUM_MAP[value]
        if value in self.CN_NUM_MAP:
            return self.CN_NUM_MAP[value]
        if value.isdigit():
            number = int(value)
            return chr(ord("A") + number - 1) if 1 <= number <= 26 else value
        return value.upper()

    def _normalize_semantic_bool(self, text: str) -> Optional[bool]:
        """识别无字母判断题选项。"""
        compact = re.sub(r"[\s。.．]", "", text or "")
        if compact in {"对", "正确", "是", "√"}:
            return True
        if compact in {"错", "错误", "否", "×"}:
            return False
        return None

    def _detect_question_type(
        self,
        question: str,
        options: List[OptionItem],
        raw_text: str,
        badge_type: Optional[QuestionType],
        buttons: Dict[str, ButtonItem],
    ) -> QuestionType:
        """优先采用页面标签，再使用选项和按钮特征回退。"""
        if badge_type is not None:
            return badge_type
        if options and all(
            self._normalize_semantic_bool(option.text) is not None
            for option in options[:2]
        ):
            return QuestionType.TRUE_FALSE
        if options:
            if "confirm" in buttons:
                return QuestionType.MULTIPLE_CHOICE
            return QuestionType.SINGLE_CHOICE
        if any(re.search(pattern, raw_text) for pattern in self.FILL_BLANK_PATTERNS):
            return QuestionType.FILL_BLANK
        if re.search(r"是否正确|是否错误|判断题|对错|说法", question):
            return QuestionType.TRUE_FALSE
        if len(question) >= 6:
            return QuestionType.OPEN_ENDED
        return QuestionType.UNKNOWN

    def _is_valid_question(
        self,
        question: str,
        options: List[OptionItem],
        question_type: QuestionType,
    ) -> bool:
        """执行最小题目质量检查。"""
        question_length = len(re.sub(r"\s+", "", question or ""))
        if question_type in {
            QuestionType.SINGLE_CHOICE,
            QuestionType.MULTIPLE_CHOICE,
        }:
            return question_length >= 4 and len(options) >= 2
        if question_type == QuestionType.TRUE_FALSE:
            return question_length >= 4 and len(options) >= 2
        if question_type in {QuestionType.FILL_BLANK, QuestionType.OPEN_ENDED}:
            return question_length >= 6
        return question_length >= 6 or len(options) >= 2

    def _empty_result(self, reason: str) -> QuestionSnapshot:
        """创建统一空结果。"""
        return QuestionSnapshot(
            external_id=None,
            question="",
            question_type=QuestionType.UNKNOWN,
            options=[],
            raw_text="",
            confidence=0.0,
            is_valid=False,
            error=reason,
        )

    def get_version_info(self) -> Dict[str, Any]:
        """返回 OCR 版本信息。"""
        try:
            import rapidocr_onnxruntime

            return {
                "rapidocr_version": getattr(
                    rapidocr_onnxruntime, "__version__", "unknown"
                ),
                "ocr_type": "RapidOCR (ONNX Runtime)",
                "language": self.lang,
                "gpu_enabled": self.use_gpu,
            }
        except Exception as exc:
            return {"error": str(exc)}

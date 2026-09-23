"""RapidOCR 识别与题目结构化解析。

本模块负责把屏幕文字转换为稳定的 QuestionSnapshot，并过滤题库页面上的
标题、分页和操作按钮，避免页面装饰进入题目语义。

识别鲁棒性设计（对应“判断题长时间卡在同一处、连续识别却始终不推进”问题）：

1. **主通道严格阈值**：先按 RapidOCR 的默认阈值识别一次，结果干净、误检少。
2. **补救通道**：主通道解析失败时，自动用放宽的检测框/识别置信度阈值再识别一遍。
   实测题库页面里的孤立单字选项（判断题“对/错”）识别得分只有 0.499 左右，
   恰好压在默认阈值 0.5 之下被整块丢弃，这是判断题永远解析不出选项的根因。
3. **帧签名冷却**：同一画面在冷却时间内只补救一次，避免无法解析的画面反复付出识别开销。
4. **尺寸守卫**：超大截图先缩放、极端长宽比先补边，避免检测模型内部放大后耗时爆炸
   （这是偶发“单帧推理上百秒”的典型原因），并把坐标换算回原始截图空间。
5. **耗时告警**：单帧推理超过阈值时输出可操作的告警，便于定位偶发卡顿。
"""
import hashlib
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from config import (
    BUTTON_KEYWORDS,
    OCR_DET_BOX_THRESH,
    OCR_DET_UNCLIP_RATIO,
    OCR_MAX_ASPECT_RATIO,
    OCR_MAX_PIXELS,
    OCR_MIN_ITEM_SCORE,
    OCR_RESCUE_COOLDOWN,
    OCR_RESCUE_DET_BOX_THRESH,
    OCR_RESCUE_DET_UNCLIP_RATIO,
    OCR_RESCUE_MIN_ITEM_SCORE,
    OCR_RESCUE_TEXT_SCORE,
    OCR_SLOW_ERROR_SECONDS,
    OCR_SLOW_WARN_SECONDS,
    OCR_TEXT_SCORE,
    OCR_WARMUP_SIZE,
    PAGE_CHROME_KEYWORDS,
)
from models import ButtonItem, OptionItem, QuestionSnapshot, QuestionType


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OcrParams(NamedTuple):
    """一次 OCR 调用使用的通道参数。

    RapidOCR 的 ``__call__`` 会把传入的关键字参数**永久写回引擎实例**，
    因此每次调用都必须显式传齐全部参数，否则上一通道的宽松阈值会污染后续识别。
    """

    name: str
    box_thresh: float
    unclip_ratio: float
    text_score: float
    min_item_score: float


def _build_params(
    name: str,
    box_thresh: float,
    unclip_ratio: float,
    text_score: float,
    min_item_score: float,
) -> OcrParams:
    """构造通道参数。"""
    return OcrParams(name, box_thresh, unclip_ratio, text_score, min_item_score)


class OcrRescueStrategy(ABC):
    """补救识别策略基类。

    主通道解析失败时，注册表会按顺序尝试各策略；任一策略使题目变得可用即停止。
    新增补救手段（例如局部放大、二值化、多尺度融合）只需新增一个子类并注册。
    """

    name: str = "rescue"

    @property
    def params(self) -> OcrParams:
        """返回该策略使用的识别参数。"""
        return _build_params(
            self.name,
            OCR_RESCUE_DET_BOX_THRESH,
            OCR_RESCUE_DET_UNCLIP_RATIO,
            OCR_RESCUE_TEXT_SCORE,
            OCR_RESCUE_MIN_ITEM_SCORE,
        )

    @abstractmethod
    def prepare(self, image: np.ndarray, items: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
        """返回该策略要识别的图像；返回 None 表示当前画面不适用。"""


class RelaxedThresholdRescue(OcrRescueStrategy):
    """放宽阈值整图补救。

    实测有效：把 ``text_score`` 从 0.5 降到 0.1、``box_thresh`` 从 0.5 降到 0.2 后，
    判断题的“对/错”两个单字选项即可被稳定识别出来。
    """

    name = "放宽阈值"

    def prepare(self, image: np.ndarray, items: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
        """整图直接识别，不需要额外预处理。"""
        return image


class OcrRescueRegistry:
    """按顺序保存可用的补救识别策略。"""

    def __init__(self, strategies: Optional[Sequence[OcrRescueStrategy]] = None) -> None:
        self.strategies: List[OcrRescueStrategy] = list(
            strategies
            if strategies is not None
            else (RelaxedThresholdRescue(),)
        )

    def __iter__(self):
        """便于直接遍历。"""
        return iter(self.strategies)

    def register(self, strategy: OcrRescueStrategy, front: bool = False) -> None:
        """注册新策略，``front=True`` 时插到最前面优先尝试。"""
        if front:
            self.strategies.insert(0, strategy)
        else:
            self.strategies.append(strategy)


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
    # 单独的选项编号框（如 “A.”、“B、”），用于与同行内容框合并。
    OPTION_LABEL_ONLY_RE = re.compile(
        r"^\s*(?P<label>[A-Za-z]|\d{1,2}|"
        r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|"
        r"[一二三四五六七八九十])\s*[\.\、．:：\)）]?\s*$"
    )
    QUESTION_NUMBER_RE = re.compile(
        r"^\s*[\[【]?\s*A?\s*(?P<id>\d{7})\s*[\]】]?\s*"
        r"(?:\d{1,3}\s*[、.．:：]?)?\s*",
        re.IGNORECASE,
    )
    EXTERNAL_ID_RE = re.compile(r"A\s*(\d{7})", re.IGNORECASE)
    # 页码形如 “1 / 303”“三 1 / 303”“第3 / 20”，页面上还常带一个汉字序号前缀。
    PAGE_NUMBER_RE = re.compile(
        r"^\s*(?:第\s*)?(?:[一二三四五六七八九十百]+\s*)?\d{1,4}\s*/\s*\d{1,4}\s*$"
    )
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
    # 选项文字换行延续的最大行距倍数，以及同框判重的容差（像素）。
    CONTINUATION_GAP_RATIO = 1.8
    DUPLICATE_TOLERANCE = 14
    MIN_QUESTION_CHARS = 4
    # 选项标签与正文之间缺失标点时补上的标准分隔符，以及合法分隔符集合。
    # 真机实测：部分平台选项标签是灰底圆形徽章里的单个字母，与右侧文字之间
    # 没有任何标点，OCR 输出形如 “A 工作证”。
    DEFAULT_OPTION_SEPARATOR = "."
    OPTION_SEPARATOR_CHARS = ".\u3001\uff0e:\uff1a)\uff09"
    # 判定“这确实是一张题目画面”的题干最短长度，用于决定是否值得付出补救识别开销。
    RESCUE_MIN_QUESTION_CHARS = 10
    # 题干中出现这些字样时，即使题型标签识别失败也按判断题解析（对/错）选项。
    TRUE_FALSE_SIGNAL_RE = re.compile(r"是否正确|是否错误|对错|判断|说法正确|说法错误")

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "ch",
        show_log: bool = True,
        warmup_size: Optional[Tuple[int, int]] = None,
        rescue_registry: Optional[OcrRescueRegistry] = None,
    ) -> None:
        """初始化 OCR 模型。

        :param warmup_size: 预热图尺寸 (高, 宽)，建议传入真实捕获区域尺寸，
            把首次推理的一次性开销（线程池/内存分配）提前到启动阶段。
        """
        self.ocr: Any = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._show_log = show_log
        self.warmup_size = warmup_size or OCR_WARMUP_SIZE or (720, 1280)
        self.rescue_registry = rescue_registry or OcrRescueRegistry()
        # 底层引擎是否支持关键字参数（便于测试替身与不同版本兼容）。
        self._supports_kwargs: Optional[bool] = None
        # 补救通道冷却状态，避免同一画面反复补救。
        self._last_fallback_signature = ""
        self._last_fallback_at = 0.0
        # 上一次补救识别的结果与所用策略，供冷却期内复用（保证同一画面解析一致）。
        self._cached_rescue_items: List[Dict[str, Any]] = []
        self._cached_rescue_name = ""
        # 最近一次识别使用的补救策略名称，供日志与调试工具查看。
        self.last_rescue_name = ""
        if not show_log:
            logging.getLogger("rapidocr_onnxruntime").setLevel(logging.WARNING)
        self._init_model()

    # ------------------------------------------------------------------
    # 模型初始化
    # ------------------------------------------------------------------
    @classmethod
    def create_for_testing(
        cls,
        ocr_callable: Any = None,
        warmup_size: Tuple[int, int] = (64, 64),
    ) -> "OCREngine":
        """创建不加载模型的实例，供单元测试注入识别替身。

        与直接 ``__new__`` 相比，这里统一初始化全部运行时状态，
        新增字段时测试不会因为缺少属性而失败。
        """
        engine = cls.__new__(cls)
        engine.ocr = ocr_callable
        engine.use_gpu = False
        engine.lang = "ch"
        engine._show_log = False
        engine.warmup_size = warmup_size
        engine.rescue_registry = OcrRescueRegistry()
        engine._supports_kwargs = None
        engine._last_fallback_signature = ""
        engine._last_fallback_at = 0.0
        engine._cached_rescue_items = []
        engine._cached_rescue_name = ""
        engine.last_rescue_name = ""
        return engine

    def _init_model(self) -> None:
        """加载并预热 RapidOCR。"""
        try:
            logger.info("正在加载 RapidOCR [ONNX Runtime]...")
            from rapidocr_onnxruntime import RapidOCR

            self.ocr = RapidOCR()
            self._warmup()
            logger.info(
                "RapidOCR 已加载并完成预热（预热尺寸 %dx%d）",
                self.warmup_size[1],
                self.warmup_size[0],
            )
        except ImportError as exc:
            logger.error("RapidOCR 导入失败: %s", exc)
            raise ImportError("请安装 rapidocr_onnxruntime") from exc
        except Exception as exc:
            logger.error("RapidOCR 初始化失败: %s", exc)
            raise RuntimeError("OCR 引擎初始化失败: %s" % exc) from exc

    def _warmup(self) -> None:
        """用接近真实截图尺寸的合成图预热，避免运行中出现首次推理长卡顿。"""
        height, width = self.warmup_size
        height = max(int(height), 64)
        width = max(int(width), 64)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        # 画几条深色横条，强制检测与识别两个模型都跑一次真实推理。
        for index in range(3):
            top = int(height * (0.2 + 0.2 * index))
            canvas[top:top + max(8, height // 40), int(width * 0.1):int(width * 0.8)] = 0
        started = time.time()
        try:
            self.ocr(canvas, box_thresh=OCR_DET_BOX_THRESH,
                     unclip_ratio=OCR_DET_UNCLIP_RATIO,
                     text_score=OCR_TEXT_SCORE)
            self._supports_kwargs = True
        except TypeError:
            self._supports_kwargs = False
            self.ocr(canvas)
        logger.info("OCR 预热耗时 %.2fs", time.time() - started)

    # ------------------------------------------------------------------
    # 识别主流程
    # ------------------------------------------------------------------
    def recognize(self, image: Any) -> QuestionSnapshot:
        """识别图像并返回结构化题目。"""
        try:
            img_array = self._prepare_image(image)
            if img_array is None:
                return self._empty_result("不支持的图像类型")

            # 尺寸守卫：先补边、必要时缩放，并把坐标换算比例记录下来。
            img_array, scale = self._limit_image_size(img_array)
            # 画面签名在缩放/补边之后计算，保证同一帧输入始终得到同一签名。
            signature = self._frame_signature(img_array)

            primary = self._primary_params()
            result, elapsed = self._invoke_ocr(img_array, primary)
            items = self._extract_items(result, primary.min_item_score)
            snapshot = self.parse_items(items)
            snapshot = self._apply_confidence(snapshot, items)

            rescue_name = ""
            if self._should_rescue(snapshot, items, signature):
                rescuer, rescue_items, rescue_elapsed = self._run_rescue(
                    img_array, items, signature
                )
                if rescuer is not None:
                    elapsed += rescue_elapsed
                    rescue_name = rescuer.name
                    merged = self._merge_if_richer(items, rescue_items)
                    if merged is not None:
                        items = merged
                        snapshot = self.parse_items(items)
                        snapshot = self._apply_confidence(snapshot, items)
            else:
                # 冷却期内不再重复推理，但必须**复用**上次的补救结果：
                # 否则同一页面会产出两种不同的解析输入（含/不含补救文字块），
                # 上层靠文本判断“画面是否变化”就会每帧漂移，
                # 无进展退避与告警将永远无法触发。
                merged = self._merge_if_richer(
                    items, self._reusable_rescue_items(signature)
                )
                if merged is not None:
                    items = merged
                    rescue_name = "%s(复用)" % self._cached_rescue_name
                    snapshot = self.parse_items(items)
                    snapshot = self._apply_confidence(snapshot, items)

            if scale != 1.0:
                # 坐标换算回原始截图空间，保证后续点击位置正确。
                self._rescale_items(items, 1.0 / scale)
                snapshot = self.parse_items(items)
                snapshot = self._apply_confidence(snapshot, items)

            # 把帧签名交给上层：用于区分“画面真的变了”与“同一画面反复失败”。
            snapshot.frame_signature = signature
            logger.info(
                "OCR: 推理=%.2fs | %d 行 | 平均置信度=%.2f | 题型=%s%s",
                elapsed,
                len(items),
                snapshot.confidence,
                snapshot.question_type.value,
                " | 补救=%s" % rescue_name if rescue_name else "",
            )
            if not snapshot.is_valid:
                logger.info("OCR 解析未通过: %s", snapshot.error or "未知原因")
            return snapshot
        except Exception as exc:
            logger.error("OCR 识别异常: %s", exc)
            return self._empty_result(str(exc))

    def recognize_raw(self, image: Any) -> Tuple[List[Dict[str, Any]], float]:
        """只做主通道识别，返回文字块与耗时。

        供调试工具逐块查看识别结果使用，保证调试路径与实际运行使用
        完全相同的识别参数。
        """
        img_array = self._prepare_image(image)
        if img_array is None:
            return [], 0.0
        img_array, scale = self._limit_image_size(img_array)
        params = self._primary_params()
        result, elapsed = self._invoke_ocr(img_array, params)
        items = self._extract_items(result, params.min_item_score)
        self._rescale_items(items, 1.0 / scale)
        return self._sort_items(items), elapsed

    def _primary_params(self) -> OcrParams:
        """主通道参数。"""
        return _build_params(
            "主通道",
            OCR_DET_BOX_THRESH,
            OCR_DET_UNCLIP_RATIO,
            OCR_TEXT_SCORE,
            OCR_MIN_ITEM_SCORE,
        )

    def _invoke_ocr(
        self, image: np.ndarray, params: OcrParams
    ) -> Tuple[List[Any], float]:
        """调用底层引擎并记录耗时，必要时输出慢推理告警。"""
        started = time.time()
        result: Any = None
        try:
            result, _ = self._call_ocr(image, params)
        except Exception as exc:  # 单帧失败不能打断连续扫描
            logger.error("OCR 推理失败（%s）: %s", params.name, exc)
        elapsed = time.time() - started
        self._report_slow_inference(params, elapsed, len(result or []))
        return result or [], elapsed

    def _call_ocr(self, image: np.ndarray, params: OcrParams) -> Any:
        """调用 RapidOCR。

        参数必须每次都显式传齐：RapidOCR 会把关键字参数写回引擎实例，
        只传部分参数会让宽松阈值残留到后续识别。
        """
        if self._supports_kwargs is not False:
            try:
                result = self.ocr(
                    image,
                    box_thresh=params.box_thresh,
                    unclip_ratio=params.unclip_ratio,
                    text_score=params.text_score,
                )
                self._supports_kwargs = True
                return result
            except TypeError:
                # 测试替身或旧版本不支持关键字参数，退化为位置参数调用。
                self._supports_kwargs = False
        return self.ocr(image)

    def _report_slow_inference(self, params: OcrParams, elapsed: float, blocks: int) -> None:
        """对异常缓慢的单帧推理给出可操作提示。"""
        if elapsed >= OCR_SLOW_ERROR_SECONDS:
            logger.error(
                "OCR 单帧推理异常缓慢 %.1fs（%s，%d 个文本框）：请缩小捕获区域"
                "（Ctrl+F1 重新框选），并确认本机没有其他程序占满 CPU/内存。",
                elapsed,
                params.name,
                blocks,
            )
        elif elapsed >= OCR_SLOW_WARN_SECONDS:
            logger.warning(
                "OCR 单帧推理较慢 %.1fs（%s，%d 个文本框），若持续出现请缩小捕获区域。",
                elapsed,
                params.name,
                blocks,
            )

    # ------------------------------------------------------------------
    # 补救识别
    # ------------------------------------------------------------------
    def _should_rescue(
        self,
        snapshot: QuestionSnapshot,
        items: Sequence[Dict[str, Any]],
        signature: str,
    ) -> bool:
        """判断是否需要启动补救识别。

        :param signature: 当前帧的画面签名（由 :meth:`recognize` 统一计算）。
        """
        if snapshot.is_valid:
            return False
        # 只识别到零散文字、看不出题目结构时，不值得再付出一次识别开销。
        if items and not self._looks_like_question(snapshot):
            return False
        if (
            signature == self._last_fallback_signature
            and time.time() - self._last_fallback_at < OCR_RESCUE_COOLDOWN
        ):
            return False
        return True

    def _reusable_rescue_items(self, signature: str) -> List[Dict[str, Any]]:
        """取回上一次补救识别的结果（仅在画面签名完全相同时复用）。

        冷却期内重复补救同一画面既浪费算力，又会让同一页面产出两种不同的解析输入
        （补救前 / 补救后），使上层“画面是否变化”的判断每帧漂移。
        因此这里把上次结果按帧号缓存起来复用。
        """
        if not signature or signature != self._last_fallback_signature:
            return []
        return [dict(item) for item in self._cached_rescue_items]

    def _merge_if_richer(
        self,
        items: Sequence[Dict[str, Any]],
        extra: Sequence[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """仅在补救结果带来更多文字块时才接受合并。

        主通道结果更干净，补救结果更全但噪声更多；只有确实多出文字块时才替换，
        避免用噪声覆盖主通道的高质量结果。
        """
        if not extra:
            return None
        merged = self._merge_items(items, extra)
        return merged if len(merged) > len(items) else None

    def _looks_like_question(self, snapshot: QuestionSnapshot) -> bool:
        """判断一帧画面是否具备题目页面的结构特征。"""
        if snapshot.external_id:
            return True
        if snapshot.question_type != QuestionType.UNKNOWN:
            return True
        if snapshot.options:
            return True
        question_length = len(re.sub(r"\s+", "", snapshot.question or ""))
        return question_length >= self.RESCUE_MIN_QUESTION_CHARS

    def _run_rescue(
        self,
        image: np.ndarray,
        items: Sequence[Dict[str, Any]],
        signature: str,
    ) -> Tuple[Optional[OcrRescueStrategy], List[Dict[str, Any]], float]:
        """按注册顺序尝试补救策略，返回第一个产出文字块的策略。"""
        self._last_fallback_signature = signature
        self._last_fallback_at = time.time()
        self.last_rescue_name = ""
        self._cached_rescue_items = []
        self._cached_rescue_name = ""
        elapsed_total = 0.0
        for strategy in self.rescue_registry:
            prepared = strategy.prepare(image, items)
            if prepared is None:
                continue
            result, elapsed = self._invoke_ocr(prepared, strategy.params)
            elapsed_total += elapsed
            rescued = self._extract_items(result, strategy.params.min_item_score)
            if rescued:
                self.last_rescue_name = strategy.name
                self._cached_rescue_name = strategy.name
                self._cached_rescue_items = [dict(item) for item in rescued]
                logger.info(
                    "已启用补救识别（%s）: %d 个文字块，耗时 %.2fs",
                    strategy.name,
                    len(rescued),
                    elapsed,
                )
                return strategy, rescued, elapsed_total
        logger.info("补救识别未获得任何文字块，累计耗时 %.2fs", elapsed_total)
        return None, [], elapsed_total

    def _merge_items(
        self,
        base: Sequence[Dict[str, Any]],
        extra: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """合并两轮识别结果：同一位置的文字块只保留置信度更高的一份。"""
        merged = [dict(item) for item in base]
        for item in extra:
            center = self._box_center(item.get("box"))
            replaced = False
            for index, exist in enumerate(merged):
                exist_center = self._box_center(exist.get("box"))
                if (
                    abs(exist_center[0] - center[0]) <= self.DUPLICATE_TOLERANCE
                    and abs(exist_center[1] - center[1]) <= self.DUPLICATE_TOLERANCE
                ):
                    replaced = True
                    if item.get("score", 0.0) > exist.get("score", 0.0):
                        merged[index] = dict(item)
                    break
            if not replaced:
                merged.append(dict(item))
        return merged

    def _frame_signature(self, image: np.ndarray) -> str:
        """生成画面签名，用于判断两帧内容是否实质相同。"""
        try:
            sample = np.ascontiguousarray(image[::4, ::4])
            return hashlib.md5(sample.tobytes()).hexdigest()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 图像守卫
    # ------------------------------------------------------------------
    def _limit_image_size(self, array: np.ndarray) -> Tuple[np.ndarray, float]:
        """限制图像规模，返回 (处理后的图像, 缩放比例)。

        检测模型内部会把图像的最短边统一放大到固定长度，因此
        “极端长宽比 + 小尺寸”的截图会被放大到几千万像素，导致单帧推理耗时爆炸。
        这里先补边把长宽比压到安全范围，再对超限像素数做等比缩放。
        """
        array = self._pad_to_safe_aspect(array)
        height, width = array.shape[:2]
        pixels = height * width
        if pixels <= OCR_MAX_PIXELS:
            return array, 1.0
        ratio = math.sqrt(float(OCR_MAX_PIXELS) / float(pixels))
        new_width = max(1, int(round(width * ratio)))
        new_height = max(1, int(round(height * ratio)))
        resized = np.array(
            Image.fromarray(array).resize((new_width, new_height), Image.LANCZOS)
        )
        logger.warning(
            "截图过大（%dx%d=%d 像素），已缩放至 %dx%d 后再识别",
            width,
            height,
            pixels,
            new_width,
            new_height,
        )
        return resized, new_width / float(width)

    def _pad_to_safe_aspect(self, array: np.ndarray) -> np.ndarray:
        """用白色补边把极端长宽比压到安全范围，避免检测模型内部极端放大。"""
        height, width = array.shape[:2]
        if height <= 0 or width <= 0:
            return array
        ratio = max(height, width) / float(min(height, width))
        if ratio <= OCR_MAX_ASPECT_RATIO:
            return array
        if width >= height:
            # 横向过长时补高，保证内容不被裁掉。
            target_height = max(height, int(math.ceil(width / OCR_MAX_ASPECT_RATIO)))
            canvas = np.full((target_height, width, 3), 255, dtype=np.uint8)
            canvas[:height, :] = array
        else:
            target_width = max(width, int(math.ceil(height / OCR_MAX_ASPECT_RATIO)))
            canvas = np.full((height, target_width, 3), 255, dtype=np.uint8)
            canvas[:, :width] = array
        logger.warning(
            "截图长宽比异常（%dx%d），已补白边至 %dx%d 后再识别",
            width,
            height,
            canvas.shape[1],
            canvas.shape[0],
        )
        return canvas

    def _rescale_items(self, items: Sequence[Dict[str, Any]], factor: float) -> None:
        """就地缩放文字块坐标（用于把识别结果换算回原始截图空间）。"""
        if factor == 1.0:
            return
        for item in items:
            box = np.asarray(item.get("box"), dtype=float)
            if box.ndim != 2 or box.shape[1] != 2:
                continue
            item["box"] = (box * factor).tolist()

    # ------------------------------------------------------------------
    # 结果解析
    # ------------------------------------------------------------------
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
            error="" if is_valid else self._invalid_reason(question, options, question_type),
            button_boxes=buttons,
        )

    def _apply_confidence(
        self, snapshot: QuestionSnapshot, items: Sequence[Dict[str, Any]]
    ) -> QuestionSnapshot:
        """按文字块平均置信度回填快照。"""
        if items:
            snapshot.confidence = sum(item.get("score", 0.0) for item in items) / len(items)
        return snapshot

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
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(array)
        except Exception as exc:
            logger.error("图像转换失败: %s", exc)
            return None

    def _extract_items(
        self, ocr_result: Any, min_score: float = OCR_MIN_ITEM_SCORE
    ) -> List[Dict[str, Any]]:
        """保留 RapidOCR 文本、置信度和坐标。"""
        items: List[Dict[str, Any]] = []
        for item in ocr_result or []:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            try:
                score = float(item[2])
            except (TypeError, ValueError):
                continue
            if score < min_score:
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

    @staticmethod
    def _box_span(box: Any) -> Tuple[int, int, int, int]:
        """计算文本框的 (左, 上, 右, 下) 边界。"""
        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        except Exception:
            return 0, 0, 0, 0

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

    def _estimate_line_height(self, items: Sequence[Dict[str, Any]]) -> float:
        """估算页面行高，用于判断两个文字块是否属于相邻行。"""
        heights = []
        for item in items:
            _, top, _, bottom = self._box_span(item.get("box"))
            if bottom > top:
                heights.append(bottom - top)
        if not heights:
            return 24.0
        heights.sort()
        return float(heights[len(heights) // 2])

    def _footer_top(self, items: Sequence[Dict[str, Any]]) -> Optional[int]:
        """找出页脚（翻页按钮、页码）顶边，用于截断题目区域。"""
        tops = []
        for item in items:
            raw = item.get("text", "").strip()
            compact = re.sub(r"\s+", "", raw)
            if self._button_role(raw) is not None or self.PAGE_NUMBER_RE.match(compact):
                _, top, _, _ = self._box_span(item.get("box"))
                tops.append(top)
        return min(tops) if tops else None

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
        """解析有序选项，并忽略按钮、页脚和悬浮控件文字。"""
        options: List[OptionItem] = []
        current: Optional[OptionItem] = None
        option_start_index = len(items)
        semantic_index = 0
        line_height = self._estimate_line_height(items)
        footer_top = self._footer_top(items)
        # 题型标签识别失败时，用题干文字特征兜底识别“对/错”这类无字母选项。
        semantic_allowed = badge_type == QuestionType.TRUE_FALSE or bool(
            self.TRUE_FALSE_SIGNAL_RE.search(
                "".join(item["text"] for item in items)
            )
        )
        # 选项换行延续时用的参考块（记录最后一行，支持多行选项）。
        last_center: Optional[Tuple[int, int]] = None
        last_box: Any = None
        # OCR 可能把“A.”和“选项甲”拆成同一行的两个框，这里先合并再解析。
        merged_items = self._merge_split_option_boxes(items, line_height)

        for index, item in merged_items:
            raw_text = item["text"].strip()
            text = self._strip_page_controls(raw_text)
            if not text or self._is_page_chrome(text):
                continue
            if self._button_role(raw_text) is not None:
                continue
            center = item.get("center") or (0, 0)
            if footer_top is not None and center[1] >= footer_top:
                # 页脚区域（翻页按钮、页码）不参与选项识别与换行延续。
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
                last_center, last_box = current.center, current.box
                continue

            semantic = self._normalize_semantic_bool(text)
            if (
                semantic_allowed
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
                last_center, last_box = current.center, current.box
                continue

            if current is not None and self._is_adjacent_line(
                last_center, last_box, item, line_height
            ):
                current.text = (current.text + " " + text).strip()[:500]
                last_center, last_box = item.get("center"), item.get("box")

        return options, option_start_index

    def _merge_split_option_boxes(
        self,
        items: List[Dict[str, Any]],
        line_height: float,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """把被 OCR 拆成两个框的选项行合并回一个逻辑文字块。

        OCR 常把“A.”与选项正文识别成同一行的两个独立框，若不合并会导致
        选项整体丢失（进而整题判定为无效题目）。返回 (原始下标, 文字块) 列表，
        原始下标用于把选项起点映射回未合并的列表。
        """
        tolerance = max(6.0, line_height * 0.6)
        result: List[Tuple[int, Dict[str, Any]]] = []
        index = 0
        while index < len(items):
            item = items[index]
            text = item["text"].strip()
            if (
                index + 1 < len(items)
                and self.OPTION_LINE_RE.match(text) is None
                and self.OPTION_LABEL_ONLY_RE.match(text) is not None
            ):
                following = items[index + 1]
                same_line = abs(item["center"][1] - following["center"][1]) <= tolerance
                right_side = following["center"][0] > item["center"][0]
                if same_line and right_side:
                    result.append(
                        (
                            index,
                            self._combine_items(
                                item,
                                following,
                                self._join_option_label(text, following["text"]),
                            ),
                        )
                    )
                    index += 2
                    continue
            result.append((index, item))
            index += 1
        return result

    @classmethod
    def _join_option_label(cls, label_text: str, content: str) -> str:
        """把“徽章标签框 + 同行正文框”拼成标准选项行。

        部分平台的选项标签是圆形（或圆角）徽章里的单个字母，与右侧文字之间
        没有任何标点，OCR 会把它们识别成两个独立文本框。如果按原样用空格拼接，
        要求“标签后必须跟标点”的 :data:`OPTION_LINE_RE` 会判定该行不是选项，
        于是**整题选项全部丢失**（真机表现为“仅识别到 0 个选项”）。

        这里把这类版式差异在解析层收敛掉：标签缺少标点时补一个标准分隔符，
        使下游选项解析、点击坐标计算完全复用同一条路径。
        """
        label = (label_text or "").strip()
        content = (content or "").strip()
        if not label:
            return content
        if label[-1] in cls.OPTION_SEPARATOR_CHARS:
            # 已经带标点（如 “A.”“B、”）时保持原样，不改变既有语义。
            return "%s %s" % (label, content)
        return "%s%s%s" % (label, cls.DEFAULT_OPTION_SEPARATOR, content)

    def _combine_items(
        self, first: Dict[str, Any], second: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """合并同一行相邻的两个文字块，坐标取并集、置信度取较大值。"""
        left_a, top_a, right_a, bottom_a = self._box_span(first.get("box"))
        left_b, top_b, right_b, bottom_b = self._box_span(second.get("box"))
        left, top = min(left_a, left_b), min(top_a, top_b)
        right, bottom = max(right_a, right_b), max(bottom_a, bottom_b)
        box = [[left, top], [right, top], [right, bottom], [left, bottom]]
        return {
            "text": text,
            "score": max(first.get("score", 0.0), second.get("score", 0.0)),
            "box": box,
            "center": ((left + right) // 2, (top + bottom) // 2),
        }

    def _is_adjacent_line(
        self,
        reference_center: Optional[Tuple[int, int]],
        reference_box: Any,
        item: Dict[str, Any],
        line_height: float,
    ) -> bool:
        """判断文字块是否为选项文字的换行延续。

        只接受“紧贴参考行下方且横向有重叠”的文字块，避免把页面底部的悬浮按钮、
        页码或页脚文字粘到最后一个选项上（实测会把“错”污染成“错 纠 0 三”）。
        """
        if reference_center is None or item.get("center") is None:
            return False
        gap = item["center"][1] - reference_center[1]
        if gap <= 0 or gap > line_height * self.CONTINUATION_GAP_RATIO:
            return False
        left, _, right, _ = self._box_span(reference_box)
        item_left, _, item_right, _ = self._box_span(item.get("box"))
        if right <= left or item_right <= item_left:
            return False
        return not (item_right < left or item_left > right)

    def _extract_question(
        self,
        items: List[Dict[str, Any]],
        option_start_index: int,
        external_id: Optional[str],
        badge_type: Optional[QuestionType],
    ) -> str:
        """提取干净题干，移除页头、题型和题号。"""
        candidates: List[Tuple[Dict[str, Any], str]] = []
        for item in items[:option_start_index]:
            text = self._strip_page_controls(item["text"])
            if not text or self._is_page_chrome(text):
                continue
            if self._button_role(text) is not None:
                continue
            candidates.append((item, text))

        start = self._question_start_index(candidates, external_id)
        lines: List[str] = []
        for _, text in candidates[start:]:
            cleaned = self.QUESTION_NUMBER_RE.sub("", text, count=1).strip()
            if cleaned:
                lines.append(cleaned)

        if not lines:
            fallback = [
                self.QUESTION_NUMBER_RE.sub("", text, count=1).strip()
                for _, text in candidates
            ]
            lines = [line for line in fallback if line]
        question = "\n".join(lines).strip()
        for label in self.TYPE_LABELS.get(badge_type or QuestionType.UNKNOWN, ()):
            question = question.replace(label, "")
        return question.strip()[:500]

    def _question_start_index(
        self,
        candidates: Sequence[Tuple[Dict[str, Any], str]],
        external_id: Optional[str],
    ) -> int:
        """确定题干起始文字块。

        有题号时以题号块为锚点，避免把返回箭头、页面标题等噪声当成题干；
        没有题号时跳过顶部零散短文本（如被识别成“<”“人”的返回箭头）。
        """
        if not candidates:
            return 0
        if external_id:
            digits = external_id[1:]
            for index, (_, text) in enumerate(candidates):
                if digits in text.replace(" ", ""):
                    return index
        for index, (_, text) in enumerate(candidates):
            if len(re.sub(r"\s+", "", text)) >= self.MIN_QUESTION_CHARS:
                return index
        return 0

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
        # 页码可能带汉字序号前缀，如“三 1 / 303”，同样需要截掉。
        cleaned = re.sub(
            r"\s*(?:第\s*)?(?:[一二三四五六七八九十百]+\s*)?\d{1,4}\s*/\s*\d{1,4}\s*$",
            "",
            cleaned,
        )
        return cleaned.strip()

    def _is_page_chrome(self, text: str) -> bool:
        """判断文本是否属于页面标题、题型标签、悬浮按钮或分页。"""
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return True
        if compact in set(PAGE_CHROME_KEYWORDS) | {"上一题", "下一题"}:
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
            QuestionType.TRUE_FALSE,
        }:
            return question_length >= 4 and len(options) >= 2
        if question_type in {QuestionType.FILL_BLANK, QuestionType.OPEN_ENDED}:
            return question_length >= 6
        return question_length >= 6 or len(options) >= 2

    def _invalid_reason(
        self,
        question: str,
        options: List[OptionItem],
        question_type: QuestionType,
    ) -> str:
        """为无法作答的题目生成可操作的失败原因。

        没有原因说明的“静默失败”会让使用者在日志里看不出问题所在，
        这里统一给出题干长度、选项数量等可直接排查的信息。
        """
        length = len(re.sub(r"\s+", "", question or ""))
        if question_type in {
            QuestionType.SINGLE_CHOICE,
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.TRUE_FALSE,
        }:
            if len(options) < 2 and length < self.MIN_QUESTION_CHARS:
                return "题干过短且选项不足（题干 %d 字，选项 %d 个），无法自动作答" % (
                    length,
                    len(options),
                )
            if len(options) < 2:
                return "选项不足（仅识别到 %d 个选项），无法自动作答；请检查捕获区域是否完整包含所有选项" % len(
                    options
                )
            return "题干过短（%d 字），无法确认题目内容" % length
        if question_type in {QuestionType.FILL_BLANK, QuestionType.OPEN_ENDED}:
            return "题干过短（%d 字），无法确认题目内容" % length
        return "未能识别出可用题干或选项（题干 %d 字，选项 %d 个）" % (
            length,
            len(options),
        )

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
                "rescue_strategies": [strategy.name for strategy in self.rescue_registry],
            }
        except Exception as exc:
            return {"error": str(exc)}

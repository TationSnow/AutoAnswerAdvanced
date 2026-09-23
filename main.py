"""AutoAnswer 主程序。

程序负责捕获屏幕、调用 OCR/题库/模型，并按题型执行安全的自动答题状态机。
"""
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGION_JSON = os.path.join(BASE_DIR, "capture_region.json")
REGION_SELECTOR_SCRIPT = os.path.join(BASE_DIR, "region_selector.py")
KB_DB_FULL = os.path.join(BASE_DIR, "question_bank.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

#: 推进到下一题的两种方式标识，用于会话计数与等待时长选择。
ADVANCE_METHOD_CLICK = "click"
ADVANCE_METHOD_SWIPE = "swipe"


def check_dependencies() -> None:
    """检查运行依赖并给出中文安装提示。"""
    missing = []
    core_deps = {
        "PyQt5": "PyQt5",
        "mss": "mss",
        "PIL": "Pillow",
        "keyboard": "keyboard",
        "rapidocr_onnxruntime": "rapidocr_onnxruntime",
        "pyautogui": "pyautogui",
        "requests": "requests",
        "numpy": "numpy",
    }
    for module, package in core_deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
            logger.warning("缺少依赖: %s", package)

    if missing:
        print("\n" + "=" * 56)
        print("缺少以下依赖，请先安装：")
        print("pip install %s" % " ".join(missing))
        print("=" * 56)
        input("\n按回车退出...")
        sys.exit(1)
    logger.info("依赖检查通过")


# 保持第三方依赖检查完成后再导入业务模块，避免启动时输出难懂的 traceback。
from config import (  # noqa: E402
    AUTO_CLICK_ENABLED,
    AUTO_NEXT_ENABLED,
    CAPTURE_REGION,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_ENABLE_SEARCH,
    DEEPSEEK_JSON_SCHEMA_ENABLED,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEFAULT_MODE,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    FAIL_BACKOFF_LOG_EVERY,
    FAIL_BACKOFF_MAX_INTERVAL,
    FAIL_BACKOFF_THRESHOLD,
    FEEDBACK_POLL_INTERVAL,
    FEEDBACK_TIMEOUT,
    HOTKEY_CAPTURE,
    HOTKEY_EXIT,
    HOTKEY_NEXT,
    HOTKEY_PAUSE,
    HOTKEY_PREV,
    HOTKEY_SELECT,
    HOTKEY_TOGGLE_MODE,
    KB_DB_PATH,
    MIN_AUTO_ANSWER_CONFIDENCE,
    MODE_QUIZ,
    MODE_RECORD,
    OPTION_CLICK_INTERVAL,
    POST_CLICK_WAIT,
    POST_NEXT_WAIT,
    SCAN_INTERVAL,
    SIMILARITY_THRESHOLD,
    SWIPE_SETTLE_WAIT,
    SWIPE_SKIP_MAX_ESCAPES,
    SWIPE_SKIP_ROLLBACK_ENABLED,
    SWIPE_SKIP_STUCK_ENABLED,
    SWIPE_SKIP_STUCK_FRAMES,
)
from ai_solver import DeepSeekSolver  # noqa: E402
from answer_strategy import AnswerParser  # noqa: E402
from automation import (  # noqa: E402
    AutomationPlanner,
    QuestionSession,
    SessionState,
)
from embedding_client import EmbeddingClient  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402
from models import (  # noqa: E402
    AnswerCandidate,
    AutomationAction,
    AutomationActionType,
    NavigationDirection,
    QuestionSnapshot,
    SwipeDirection,
    SwipeProfile,
)
from navigation import (  # noqa: E402
    NavigationCommand,
    build_default_planner,
)
from ocr_engine import OCREngine  # noqa: E402
from overlay import AnswerOverlay  # noqa: E402
from screen_operator import ScreenOperator  # noqa: E402


@dataclass
class NavigationOutcome:
    """一次翻题（推进/回退）尝试的结果。

    ``acted`` 表示是否真的产生了输入动作；
    ``method`` 区分是点击按钮还是滑动手势，供调用方记录与选择等待时长。
    实现 ``__bool__`` 是为了让调用点可以直接写 ``if outcome:``。
    """

    acted: bool = False
    method: str = ""
    direction: NavigationDirection = NavigationDirection.NEXT
    detail: str = ""

    def __bool__(self) -> bool:
        """以是否产生动作为真值。"""
        return self.acted


class HotkeyHandler(QObject):
    """把全局热键转换为 Qt 主线程信号。"""

    select_region_signal = pyqtSignal()
    toggle_pause_signal = pyqtSignal()
    capture_signal = pyqtSignal()
    toggle_mode_signal = pyqtSignal()
    navigate_prev_signal = pyqtSignal()
    navigate_next_signal = pyqtSignal()
    exit_signal = pyqtSignal()

    def setup_hotkeys(self) -> None:
        """注册配置中的全局热键。"""
        try:
            import keyboard

            keyboard.add_hotkey(
                HOTKEY_SELECT,
                lambda: self.select_region_signal.emit(),
            )
            keyboard.add_hotkey(
                HOTKEY_PAUSE,
                lambda: self.toggle_pause_signal.emit(),
            )
            keyboard.add_hotkey(
                HOTKEY_CAPTURE,
                lambda: self.capture_signal.emit(),
            )
            keyboard.add_hotkey(
                HOTKEY_TOGGLE_MODE,
                lambda: self.toggle_mode_signal.emit(),
            )
            keyboard.add_hotkey(
                HOTKEY_PREV,
                lambda: self.navigate_prev_signal.emit(),
            )
            keyboard.add_hotkey(
                HOTKEY_NEXT,
                lambda: self.navigate_next_signal.emit(),
            )
            keyboard.add_hotkey(
                HOTKEY_EXIT,
                lambda: self.exit_signal.emit(),
            )
            logger.info(
                "热键已注册: %s / %s / %s / %s / %s / %s / %s",
                HOTKEY_SELECT,
                HOTKEY_PAUSE,
                HOTKEY_CAPTURE,
                HOTKEY_TOGGLE_MODE,
                HOTKEY_PREV,
                HOTKEY_NEXT,
                HOTKEY_EXIT,
            )
        except Exception as exc:
            logger.error("热键注册失败: %s", exc)


class CaptureThread(QThread):
    """OCR、解题和屏幕自动化后台线程。"""

    result_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)
    mode_signal = pyqtSignal(str)
    message_signal = pyqtSignal(str)

    def __init__(self, mode: str = DEFAULT_MODE) -> None:
        """初始化线程状态，所有外部资源延迟到 run 中创建。"""
        super().__init__()
        self.running = True
        self.paused = True
        self.single_shot = False
        self.mode = mode

        self.sct = None
        self.ocr: Optional[OCREngine] = None
        self.ai: Optional[DeepSeekSolver] = None
        self.kb: Optional[KnowledgeBase] = None
        self.operator: Optional[ScreenOperator] = None

        self.region = self._load_region()
        self.cooldown_until = 0.0
        self.start_time = time.time()
        self.ai_count = 0
        self.kb_hit_count = 0
        self.recorded_count = 0

        # 无进展退避状态：画面内容不变且始终无法推进时放慢扫描频率。
        self.scan_interval = SCAN_INTERVAL
        self.no_progress_key: Optional[str] = None
        self.no_progress_count = 0

        self.answer_parser = AnswerParser()
        self.session = QuestionSession()
        # 导航规划器：页面有“上一题/下一题”按钮时点击按钮，
        # 没有这类按钮时（部分平台只有手势翻页）退化为滑动模拟翻题。
        self.navigator = build_default_planner()
        self.planner = AutomationPlanner(
            auto_next=AUTO_NEXT_ENABLED,
            feedback_timeout=FEEDBACK_TIMEOUT,
            navigator=self.navigator,
        )
        self.last_question: Optional[QuestionSnapshot] = None
        self.last_answer: Optional[AnswerCandidate] = None
        # 手动翻题请求（Ctrl+F5 / Ctrl+F6）：由 UI 线程写入，捕获线程消费。
        self.manual_navigation: Optional[NavigationDirection] = None
        # 卡死画面的跳走计数，用于限制“越滑越远”。
        self.stuck_escape_count = 0

    def _load_region(self) -> Dict:
        """从磁盘加载用户选择的捕获区域。"""
        try:
            with open(REGION_JSON, "r", encoding="utf-8") as file:
                region = json.load(file)
                logger.info("已加载捕获区域: %s", region)
                return region
        except FileNotFoundError:
            return CAPTURE_REGION
        except Exception as exc:
            logger.error("加载区域失败: %s", exc)
            return CAPTURE_REGION

    def run(self) -> None:
        """启动截图、OCR、题库和解题服务。"""
        try:
            import mss

            self.sct = mss.mss()
            self.ocr = self._init_ocr()
            if self.ocr is None:
                return
            self.kb = self._init_kb()
            self.ai = self._init_ai()
            if self.ai is None:
                return
            self.operator = ScreenOperator(
                self.region,
                enabled=AUTO_CLICK_ENABLED,
            )

            self._print_banner()
            self.mode_signal.emit(self.mode)
            while self.running:
                if self.manual_navigation is not None:
                    # 手动翻题优先执行，保证用户按下热键后立刻响应。
                    direction = self.manual_navigation
                    self.manual_navigation = None
                    self._safe_manual_navigate(direction)
                elif self.single_shot:
                    self.single_shot = False
                    self._safe_process()
                elif not self.paused:
                    self._safe_process()
                time.sleep(self.scan_interval)
        except Exception as exc:
            logger.error("捕获线程异常: %s", exc)
        finally:
            if self.sct:
                try:
                    self.sct.close()
                except Exception:
                    pass

    def _safe_process(self) -> None:
        """隔离单帧异常，保证连续扫描不中断。"""
        try:
            self._process_frame()
        except Exception as exc:
            logger.error("处理帧失败: %s", exc)

    def _init_ocr(self) -> Optional[OCREngine]:
        """初始化 OCR。"""
        try:
            logger.info("正在初始化 OCR 引擎...")
            engine = OCREngine(
                use_gpu=False,
                lang="ch",
                show_log=False,
                warmup_size=self._warmup_size_from_region(),
            )
            logger.info("OCR 版本: %s", engine.get_version_info())
            return engine
        except Exception as exc:
            logger.error("OCR 初始化失败: %s", exc)
            return None

    def _warmup_size_from_region(self) -> Tuple[int, int]:
        """用真实捕获区域尺寸预热 OCR。

        首次推理会包含线程池与内存分配等一次性开销，实测可能长达上百秒。
        用接近真实截图尺寸的合成图在启动阶段预热，可把这段开销挪到启动时，
        避免运行中突然出现单帧超长卡顿。
        """
        try:
            height = int(self.region.get("height", 0))
            width = int(self.region.get("width", 0))
        except (TypeError, ValueError):
            height = width = 0
        if height <= 0 or width <= 0:
            return (720, 1280)
        # 上限 1600，避免预热本身耗时过长。
        return (min(height, 1600), min(width, 1600))

    def _init_kb(self) -> Optional[KnowledgeBase]:
        """初始化 Embedding 和题库。"""
        try:
            logger.info("正在初始化 Embedding 客户端...")
            embedding = EmbeddingClient(
                base_url=EMBEDDING_BASE_URL,
                model=EMBEDDING_MODEL,
                api_key=EMBEDDING_API_KEY,
            )
            vector = embedding.embed_one("测试")
            if vector is not None:
                logger.info("Embedding 维度: %s", vector.shape[0])
            else:
                logger.warning("Embedding 服务未响应，将退化为精确匹配")

            db_path = os.path.join(BASE_DIR, KB_DB_PATH)
            logger.info("正在加载题库: %s", db_path)
            knowledge_base = KnowledgeBase(
                db_path=db_path,
                embedding_client=embedding,
                similarity_threshold=SIMILARITY_THRESHOLD,
            )
            logger.info("题库统计: %s", knowledge_base.stats())
            return knowledge_base
        except Exception as exc:
            logger.error("题库初始化失败: %s", exc)
            return None

    def _init_ai(self) -> Optional[DeepSeekSolver]:
        """初始化本地/线上模型客户端。"""
        try:
            return DeepSeekSolver(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                model=DEEPSEEK_MODEL,
                enable_search=DEEPSEEK_ENABLE_SEARCH,
                knowledge_base=self.kb,
                max_tokens=DEEPSEEK_MAX_TOKENS,
                json_schema_enabled=DEEPSEEK_JSON_SCHEMA_ENABLED,
            )
        except Exception as exc:
            logger.error("AI 初始化失败: %s", exc)
            return None

    def _print_banner(self) -> None:
        """输出版本、模式和热键提示。"""
        mode_label = "做题模式" if self.mode == MODE_QUIZ else "题库纪录模式"
        ai_label = self.ai.describe() if self.ai is not None else "未初始化"
        logger.info(
            "\n%s\n自动答题助手已启动\n%s\n"
            "当前模式: %s\n捕获区域: %s\n"
            "生成模型: %s\n"
            "自动点击: %s | 自动下一题: %s\n"
            "%s\n"
            "快捷键: Ctrl+F1 区域 | Ctrl+F2 暂停 | Ctrl+F3 识别 | "
            "Ctrl+F4 模式 | Ctrl+F5 上一题 | Ctrl+F6 下一题 | Ctrl+Q 退出\n%s",
            "=" * 56,
            "=" * 56,
            mode_label,
            self.region,
            ai_label,
            "开" if AUTO_CLICK_ENABLED else "关",
            "开" if AUTO_NEXT_ENABLED else "关",
            self.navigator.describe(),
            "=" * 56,
        )

    def _process_frame(self) -> None:
        """处理当前屏幕帧，并按题目身份驱动状态机。"""
        if not self.ocr or not self.ai:
            return
        if time.time() < self.cooldown_until:
            return

        question = self._capture_question()
        if question is None or not question.is_valid:
            # 识别不到可用题目时记录无进展原因，便于定位“卡在同一处”的问题。
            self._note_no_progress(question)
            # 无按钮平台上“识别不出题目 + 没有按钮可点”会永久卡死，
            # 按配置尝试用滑动跳走这一帧（默认关闭）。
            self._try_escape_stuck_frame(question)
            return

        if self.session.should_process(question):
            self._reset_progress()
            self.stuck_escape_count = 0
            self._handle_new_question(question)
        elif self.session.state in {
            SessionState.WAITING_FEEDBACK,
            SessionState.WAITING_NEXT,
        }:
            if self._resume_question(question):
                self._reset_progress()
            else:
                self._note_no_progress(question)
        else:
            self._note_no_progress(question)

    def _no_progress_identity(self, question: Optional[QuestionSnapshot]) -> str:
        """生成“无进展画面”的身份标识，用于判断画面是否真的没有变化。

        优先使用识别层给出的**画面签名**：OCR 文本会因为补救通道的冷却而在
        “含补救文字块”与“仅主通道”之间抖动，按文本算身份会让同一画面每帧都
        被当成新画面，退避与告警因此永远不触发（真机日志中出现过该现象）。
        签名缺失时（例如手工构造的快照、调试路径）退回按文本计算。
        """
        if question is None:
            return "no-frame"
        if question.frame_signature:
            return "frame:%s" % question.frame_signature
        raw = question.raw_text or ""
        raw_signature = (
            hashlib.md5(raw.encode("utf-8")).hexdigest()[:12] if raw else ""
        )
        return "%s|%s" % (question.error, raw_signature or question.identity_key)

    def _note_no_progress(self, question: Optional[QuestionSnapshot]) -> None:
        """记录一次“本帧没有产生任何可执行动作”，并做退避。

        画面内容不变、题目又始终无法解析时，若继续按固定间隔重复识别，
        只会不断刷出同一批日志并持续占用 CPU。这里统计连续无进展次数，
        达到阈值后逐步放慢扫描频率，并只做少量说明性提示。
        """
        key = self._no_progress_identity(question)
        if key == self.no_progress_key:
            self.no_progress_count += 1
        else:
            self.no_progress_key = key
            self.no_progress_count = 1
            self.scan_interval = SCAN_INTERVAL

        if self.no_progress_count >= FAIL_BACKOFF_THRESHOLD:
            self.scan_interval = min(
                SCAN_INTERVAL * self.no_progress_count,
                FAIL_BACKOFF_MAX_INTERVAL,
            )

        should_log = self.no_progress_count == FAIL_BACKOFF_THRESHOLD or (
            self.no_progress_count > FAIL_BACKOFF_THRESHOLD
            and self.no_progress_count % FAIL_BACKOFF_LOG_EVERY == 0
        )
        if should_log:
            reason = question.error if question is not None else "未识别到题目画面"
            raw = (
                (question.raw_text or "").replace("\n", " / ")[:200]
                if question is not None
                else ""
            )
            logger.warning(
                "同一画面已连续 %d 次无法推进（%s），扫描间隔调整为 %.1fs。"
                " 识别到的文字: %s",
                self.no_progress_count,
                reason,
                self.scan_interval,
                raw or "（空）",
            )
            # 仅靠日志刷屏时用户完全看不到异常，这里同步给悬浮窗一个可见提示。
            self.message_signal.emit(
                "画面连续 %d 次无法识别题目，请检查捕获区域或用 Ctrl+F5/F6 手动翻题"
                % self.no_progress_count
            )

    def _reset_progress(self) -> None:
        """恢复正常扫描频率。"""
        self.no_progress_key = None
        self.no_progress_count = 0
        self.scan_interval = SCAN_INTERVAL

    def _capture_question(self) -> Optional[QuestionSnapshot]:
        """截取并识别当前题面。"""
        if not self.sct or not self.ocr:
            return None
        from PIL import Image

        screenshot = self.sct.grab(self.region)
        image = Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.bgra,
            "raw",
            "BGRX",
        )
        return self.ocr.recognize(image)

    def _handle_new_question(self, question: QuestionSnapshot) -> None:
        """解答新题、写入未验证记录并执行自动化计划。"""
        started = time.time()
        logger.info(
            "题目[%s] %s",
            question.question_type.label,
            question.question[:80],
        )

        answer = self.ai.solve(question)
        self.last_question = question
        self.last_answer = answer
        if answer.source == "LLM":
            self.ai_count += 1
        elif "题库" in answer.source:
            self.kb_hit_count += 1

        self._emit_answer(question, answer)
        logger.info("答案: 【%s】 | 来源: %s", answer.answer, answer.source)

        if not answer.is_valid:
            logger.warning("答案未通过校验，不执行自动点击: %s", answer.reason)
            self._emit_answer(
                question,
                AnswerCandidate.invalid(answer.reason, answer.source),
            )
            self.session.mark_finished()
            return

        if question.confidence < MIN_AUTO_ANSWER_CONFIDENCE:
            reason = "OCR 置信度过低，需人工确认"
            logger.warning("%s (%.2f)", reason, question.confidence)
            self._emit_answer(question, AnswerCandidate.invalid(reason, answer.source))
            self.session.mark_finished()
            return

        if self.mode == MODE_RECORD and self.kb is not None:
            if self.kb.add(question, answer):
                self.recorded_count += 1
                logger.info("已记录新题，共 %d 条", self.recorded_count)

        plan = self.planner.build(question, answer)
        if plan.actions and plan.actions[0].kind == AutomationActionType.STOP:
            logger.warning(plan.actions[0].reason)
            self.session.mark_finished()
            return
        self._execute_plan(question, answer, plan)

        runtime = int(time.time() - self.start_time)
        self.status_signal.emit(
            "[%s] 运行:%ss | 题库:%s | AI:%s | 新录:%s"
            % (
                "做题" if self.mode == MODE_QUIZ else "纪录",
                runtime,
                self.kb_hit_count,
                self.ai_count,
                self.recorded_count,
            )
        )
        logger.info("单题总耗时 %.2fs", time.time() - started)

    def _emit_answer(
        self,
        question: QuestionSnapshot,
        answer: AnswerCandidate,
    ) -> None:
        """把题目上下文和答案一起发送给 UI。"""
        data = answer.to_dict()
        data.update(
            {
                "question_type": question.question_type.value,
                "question_type_label": question.question_type.label,
                "confidence": question.confidence,
            }
        )
        self.result_signal.emit(data)

    def _execute_plan(
        self,
        question: QuestionSnapshot,
        answer: AnswerCandidate,
        plan,
    ) -> None:
        """顺序执行动作计划，并在等待阶段轮询页面反馈。"""
        if self.operator is None:
            return
        for action in plan.actions:
            if action.kind == AutomationActionType.CLICK_OPTION:
                if action.target is not None:
                    self.operator.click_at(action.target)
                    time.sleep(OPTION_CLICK_INTERVAL)
            elif action.kind == AutomationActionType.CLICK_CONFIRM:
                if action.target is not None:
                    self.operator.click_at(action.target)
                    time.sleep(POST_CLICK_WAIT)
            elif action.kind == AutomationActionType.WAIT_FEEDBACK:
                self.session.mark_waiting_feedback()
                latest, feedback, advanced = self._wait_for_feedback(
                    question,
                    answer,
                    action.timeout or FEEDBACK_TIMEOUT,
                )
                if advanced:
                    self.session.mark_finished()
                    return
                if feedback is not None:
                    self._emit_answer(question, feedback)
                outcome = self._try_advance(latest)
                if outcome.acted:
                    # 记录本次推进而不是直接判定完成：页面若未推进仍可有限次重试。
                    self._record_advance(outcome.method)
                    return
                self.session.mark_waiting_next()
            elif action.kind in {
                AutomationActionType.CLICK_NEXT,
                AutomationActionType.SWIPE,
            }:
                # 计划中声明的兜底动作：等待阶段没有推进时按声明执行一次。
                if self._execute_navigation_action(action):
                    self._record_advance(
                        ADVANCE_METHOD_SWIPE
                        if action.kind == AutomationActionType.SWIPE
                        else ADVANCE_METHOD_CLICK
                    )
                    return

    # ------------------------------------------------------------------
    # 翻题（上一题 / 下一题）—— 按钮优先，滑动兜底
    # ------------------------------------------------------------------
    def _try_advance(
        self,
        question: Optional[QuestionSnapshot],
        direction: NavigationDirection = NavigationDirection.NEXT,
    ) -> NavigationOutcome:
        """尝试翻到目标题目：优先点击页面按钮，按钮缺失时退化为滑动。

        这是自动流程与手动热键共用的唯一入口，
        “页面上到底有没有按钮”只在这里判断一次，避免逻辑分散。

        :param question: 最新画面快照；为 ``None`` 时只能依赖滑动。
        :param direction: 目标方向（下一题 / 上一题）。
        """
        if self.operator is None:
            return NavigationOutcome(direction=direction)

        attempt = (
            self.session.next_click_count
            if direction is NavigationDirection.NEXT
            else self.session.prev_swipe_count
        )
        command = self.navigator.resolve(question, direction, attempt)
        if command is None:
            logger.info("未找到%s按钮，且滑动回退未启用，保持等待", direction.label)
            return NavigationOutcome(direction=direction)

        if not self._execute_navigation(command):
            return NavigationOutcome(direction=direction, detail="执行失败")

        return NavigationOutcome(
            acted=True,
            method=(
                ADVANCE_METHOD_SWIPE
                if command.is_gesture
                else ADVANCE_METHOD_CLICK
            ),
            direction=direction,
            detail=command.describe(),
        )

    def _execute_navigation(self, command: NavigationCommand) -> bool:
        """按导航指令落地一次输入动作（按钮点击或滑动手势）。"""
        if command.is_gesture:
            return self._perform_swipe(command.swipe_direction, command.profile)
        if command.target is None or self.operator is None:
            return False
        if self.operator.click_at(command.target):
            logger.info(
                "已点击%s（%s）",
                command.direction.label,
                command.label or "-",
            )
            return True
        return False

    def _execute_navigation_action(self, action: AutomationAction) -> bool:
        """执行动作计划中声明的导航动作。"""
        if action.kind == AutomationActionType.SWIPE:
            return self._perform_swipe(
                action.direction or NavigationDirection.NEXT,
                action.profile,
            )
        if action.target is None or self.operator is None:
            return False
        if self.operator.click_at(action.target):
            logger.info("已点击%s", action.label or "下一题")
            return True
        return False

    def _perform_swipe(
        self,
        swipe_direction: Optional[SwipeDirection],
        profile: Optional[SwipeProfile] = None,
    ) -> bool:
        """在捕获区域内执行一次滑动手势，用于模拟“上一题 / 下一题”。"""
        if self.operator is None or swipe_direction is None:
            return False
        if self.operator.swipe(swipe_direction, profile):
            logger.info(
                "已用%s模拟翻题（档位 %s）",
                swipe_direction.label,
                profile.name if profile else "-",
            )
            return True
        return False

    def _record_advance(self, method: str) -> None:
        """记录一次推进尝试，并按方式选择等待时长后进入冷却。

        滑动手势后页面需要更长时间重绘，等待时长比点击更长，
        否则下一帧会截到动画中间态，导致同一题被重复识别。
        """
        method = method or ADVANCE_METHOD_CLICK
        self.session.record_next_click(method=method)
        self.cooldown_until = time.time() + (
            SWIPE_SETTLE_WAIT if method == ADVANCE_METHOD_SWIPE else POST_NEXT_WAIT
        )

    def _try_escape_stuck_frame(self, question: Optional[QuestionSnapshot]) -> bool:
        """画面长时间无法解析时，按配置尝试滑动跳走，避免永久卡死。

        仅在**页面上没有可点击按钮**（导航解析为手势）且连续无进展达到阈值时触发：
        有按钮的页面交给正常流程处理，绝不在无法解析的画面上下发点击。
        该能力默认关闭，需要时再打开 ``config.SWIPE_SKIP_STUCK_ENABLED``。

        :return: 是否执行了跳走动作。
        """
        if not SWIPE_SKIP_STUCK_ENABLED or self.operator is None:
            return False
        if self.no_progress_count < SWIPE_SKIP_STUCK_FRAMES:
            return False
        if self.stuck_escape_count >= SWIPE_SKIP_MAX_ESCAPES:
            return False

        command = self.navigator.resolve(
            question,
            NavigationDirection.NEXT,
            self.session.next_click_count,
        )
        if command is None or not command.is_gesture:
            return False
        if not self._execute_navigation(command):
            return False

        self.stuck_escape_count += 1
        self.session.record_next_click(method=ADVANCE_METHOD_SWIPE)
        self.cooldown_until = time.time() + SWIPE_SETTLE_WAIT
        logger.warning(
            "画面连续 %d 帧无法解析且没有可点击按钮，已%s跳过该画面（第 %d/%d 次）",
            self.no_progress_count,
            command.describe(),
            self.stuck_escape_count,
            SWIPE_SKIP_MAX_ESCAPES,
        )
        if (
            self.stuck_escape_count >= SWIPE_SKIP_MAX_ESCAPES
            and SWIPE_SKIP_ROLLBACK_ENABLED
        ):
            self._rollback_after_failed_escape(question)
        # 重新计时，避免对同一画面立刻重复跳走。
        self._reset_progress()
        return True

    def _rollback_after_failed_escape(
        self,
        question: Optional[QuestionSnapshot],
    ) -> None:
        """连续跳走仍解析不出题目时，反向滑动回退一帧。

        跳走用于越过“识别不出且没有按钮”的异常画面；但若连着几帧都解析不出内容，
        说明可能已经滑过头（例如越过了整页内容），此时回退一帧把页面恢复到
        上一次的状态，再由人工确认，而不是继续盲目向前滑。
        """
        if not self.session.can_swipe_prev():
            logger.warning("连续跳走仍无法解析题目，但回退次数已用尽，停止自动跳走")
            return
        command = self.navigator.resolve(
            question,
            NavigationDirection.PREV,
            self.session.prev_swipe_count,
        )
        if command is None or not command.is_gesture:
            logger.warning("连续跳走仍无法解析题目，且当前没有可用的回退方式，停止自动跳走")
            return
        if self._execute_navigation(command):
            self.session.record_prev_swipe()
            self.cooldown_until = time.time() + SWIPE_SETTLE_WAIT
            logger.warning(
                "已回退一帧（%s），请检查页面是否需要人工处理",
                command.describe(),
            )

    def _safe_manual_navigate(self, direction: NavigationDirection) -> None:
        """隔离手动翻题的异常，保证后台线程不中断。"""
        try:
            self.manual_navigate(direction)
        except Exception as exc:
            logger.error("手动%s失败: %s", direction.label, exc)

    def manual_navigate(self, direction: NavigationDirection) -> bool:
        """执行一次手动翻题（由 Ctrl+F5 / Ctrl+F6 触发）。

        与自动流程共用 :meth:`_try_advance`，因此：
        页面有按钮就点按钮，没有按钮就用滑动模拟，行为完全一致。

        :return: 是否成功产生了翻题动作。
        """
        if self.operator is None:
            self.message_signal.emit("尚未初始化完成，请稍后再试")
            return False

        snapshot = self.last_question if self.last_question is not None else None
        outcome = self._try_advance(snapshot, direction)
        if not outcome.acted:
            self.message_signal.emit(
                "未找到%s按钮，且滑动回退不可用" % direction.label
            )
            logger.warning("手动%s未生效: %s", direction.label, outcome.detail or "无可用导航方式")
            return False

        if direction is NavigationDirection.PREV:
            self.session.record_prev_swipe()
        else:
            self.session.record_next_click(method=outcome.method)
        self.cooldown_until = time.time() + SWIPE_SETTLE_WAIT
        logger.info("手动翻题: %s", outcome.detail)
        self.message_signal.emit("已%s" % outcome.detail)
        return True

    def _resume_question(self, question: QuestionSnapshot) -> bool:
        """继续等待反馈或下一题按钮，不重复选择答案。

        :return: 本帧是否产生了实际动作（用于无进展退避判断）。
        """
        acted = False
        answer = self.last_answer
        if answer is not None and self._has_feedback_text(question.raw_text):
            feedback = self._parse_feedback(question, answer)
            if feedback is not None:
                self._emit_answer(question, feedback)
                acted = True

        if not self.session.can_retry_next(POST_NEXT_WAIT):
            return acted
        outcome = self._try_advance(question)
        if outcome.acted:
            self._record_advance(outcome.method)
            return True
        self.session.mark_waiting_next()
        return acted

    def _wait_for_feedback(
        self,
        question: QuestionSnapshot,
        answer: AnswerCandidate,
        timeout: float,
    ) -> Tuple[QuestionSnapshot, Optional[AnswerCandidate], bool]:
        """轮询屏幕，等待答案解析或新题出现。

        判定顺序很关键：

        1. 解析成功且题目身份变了 —— 新题已出现，视为已推进；
        2. 解析成功且出现“正确答案/解析”字样 —— 捕获页面反馈；
        3. 解析失败但**画面签名变了** —— 说明页面确实翻走了，只是新题版式
           暂时解析不出来，同样按“已推进”处理。否则会对旧题目补发点击或滑动
           （真机日志里出现过这种无效动作），并让新题被反复重试。
        """
        deadline = time.time() + timeout
        latest = question
        while time.time() < deadline:
            time.sleep(FEEDBACK_POLL_INTERVAL)
            snapshot = self._capture_question()
            if snapshot is None:
                continue
            if not snapshot.is_valid:
                if self._frame_changed(question, snapshot):
                    return snapshot, None, True
                continue
            latest = snapshot
            if snapshot.identity_key != question.identity_key:
                return snapshot, None, True
            if self._has_feedback_text(snapshot.raw_text):
                feedback = self._parse_feedback(snapshot, answer)
                if feedback is not None:
                    return snapshot, feedback, False
        return latest, None, False

    @staticmethod
    def _frame_changed(
        previous: QuestionSnapshot,
        current: QuestionSnapshot,
    ) -> bool:
        """判断两帧画面是否确实不同（缺任一方签名时保守返回 False）。

        只用于“解析失败”的快照：能解析出题目的情况已经由题目身份判断覆盖。
        """
        if not previous.frame_signature or not current.frame_signature:
            return False
        return previous.frame_signature != current.frame_signature

    def _has_feedback_text(self, raw_text: str) -> bool:
        """判断页面是否真的出现了答案反馈。"""
        return bool(
            raw_text
            and (
                "正确答案" in raw_text
                or "题目解析" in raw_text
                or "答案：" in raw_text
                or "答案:" in raw_text
            )
        )

    def _parse_feedback(
        self,
        question: QuestionSnapshot,
        previous_answer: AnswerCandidate,
    ) -> Optional[AnswerCandidate]:
        """解析反馈并更新当前题库记录。"""
        feedback = self.answer_parser.from_explicit_text(
            question,
            question.raw_text,
            source="屏幕捕获",
            verified=True,
            match_id=previous_answer.match_id,
        )
        if not feedback.is_valid:
            return None
        if self.mode == MODE_RECORD and self.kb is not None:
            if feedback.match_id:
                self.kb.update_feedback(feedback.match_id, question, feedback)
            else:
                self.kb.add(question, feedback)
        logger.info("已捕获屏幕正确答案: %s", feedback.display_answer)
        return feedback

    def update_region(self, new_region: Dict) -> None:
        """更新捕获区域并重置当前题状态。"""
        self.region = dict(new_region)
        if self.operator:
            self.operator.update_region(new_region)
        self.session.reset()
        logger.info("捕获区域已更新: %s", new_region)

    def pause(self) -> bool:
        """切换暂停状态。"""
        self.paused = not self.paused
        return self.paused

    def stop(self) -> None:
        """请求线程停止。"""
        self.running = False

    def trigger_capture(self) -> None:
        """安排一次手动识别，并允许重新处理当前题。"""
        self.session.reset()
        self.single_shot = True

    def request_navigation(self, direction: NavigationDirection) -> None:
        """登记一次手动翻题请求，由捕获线程在下一轮循环中执行。

        不在 UI 线程直接操作鼠标：鼠标动作必须与 OCR、点击共用同一个线程，
        否则会出现两次输入互相打断的情况。
        """
        self.manual_navigation = direction

    def toggle_mode(self) -> str:
        """切换做题和题库记录模式。"""
        self.mode = MODE_RECORD if self.mode == MODE_QUIZ else MODE_QUIZ
        self.session.reset()
        return self.mode


class AutoAnswerApp:
    """Qt 应用编排器。"""

    def __init__(self) -> None:
        """创建悬浮窗、线程和热键。"""
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("AutoAnswer")
        self.overlay = AnswerOverlay()

        self.thread = CaptureThread(mode=DEFAULT_MODE)
        self.thread.result_signal.connect(self.overlay.show_answer)
        self.thread.status_signal.connect(self.overlay.update_status)
        self.thread.mode_signal.connect(self._on_mode_changed)
        self.thread.message_signal.connect(self.overlay.show_message)

        self.hotkey_handler = HotkeyHandler()
        self.hotkey_handler.select_region_signal.connect(self._on_select_region)
        self.hotkey_handler.toggle_pause_signal.connect(self._on_toggle_pause)
        self.hotkey_handler.capture_signal.connect(self._on_capture)
        self.hotkey_handler.toggle_mode_signal.connect(self._on_toggle_mode)
        self.hotkey_handler.navigate_prev_signal.connect(self._on_navigate_prev)
        self.hotkey_handler.navigate_next_signal.connect(self._on_navigate_next)
        self.hotkey_handler.exit_signal.connect(self._on_exit)
        self.hotkey_handler.setup_hotkeys()

        self.thread.start()
        self.overlay.update_status(
            "Ctrl+F3 手动识别 | Ctrl+F4 切换模式 | Ctrl+F5/F6 翻题"
        )

    def _on_navigate_prev(self) -> None:
        """手动翻到上一题（无按钮平台自动用右滑模拟）。"""
        self._request_navigation(NavigationDirection.PREV)

    def _on_navigate_next(self) -> None:
        """手动翻到下一题（无按钮平台自动用左滑模拟）。"""
        self._request_navigation(NavigationDirection.NEXT)

    def _request_navigation(self, direction: NavigationDirection) -> None:
        """把翻题请求交给后台线程执行，避免在 UI 线程里操作鼠标。"""
        self.overlay.show_message("正在请求%s…" % direction.label)
        self.thread.request_navigation(direction)

    def _on_mode_changed(self, mode: str) -> None:
        """显示模式切换结果。"""
        label = "做题模式" if mode == MODE_QUIZ else "题库纪录模式"
        self.overlay.show_message("已切换到: %s" % label)

    def _on_capture(self) -> None:
        """触发一次手动识别。"""
        self.overlay.show_message("正在识别...")
        self.thread.trigger_capture()

    def _on_select_region(self) -> None:
        """启动区域选择器并加载新坐标。"""
        logger.info("触发区域选择...")
        was_paused = self.thread.paused
        self.thread.paused = True
        try:
            import subprocess

            subprocess.run(
                [sys.executable, REGION_SELECTOR_SCRIPT],
                timeout=60,
                check=False,
            )
            with open(REGION_JSON, "r", encoding="utf-8") as file:
                new_region = json.load(file)
            self.thread.update_region(new_region)
            self.overlay.show_message("区域已更新")
        except Exception as exc:
            logger.error("区域选择失败: %s", exc)
            self.overlay.show_message("区域选择失败")
        finally:
            if not was_paused:
                self.thread.paused = False

    def _on_toggle_pause(self) -> None:
        """暂停或继续扫描。"""
        is_paused = self.thread.pause()
        self.overlay.show_message(
            "已暂停 | 按 Ctrl+F2 继续" if is_paused else "连续扫描中"
        )

    def _on_toggle_mode(self) -> None:
        """切换运行模式。"""
        new_mode = self.thread.toggle_mode()
        label = "做题模式" if new_mode == MODE_QUIZ else "题库纪录模式"
        self.overlay.show_message("模式切换: %s" % label)
        logger.info("模式切换: %s", label)

    def _on_exit(self) -> None:
        """输出统计并退出应用。"""
        runtime = time.time() - self.thread.start_time
        logger.info(
            "运行 %d 秒 | 题库命中 %d | AI %d | 新录 %d",
            int(runtime),
            self.thread.kb_hit_count,
            self.thread.ai_count,
            self.thread.recorded_count,
        )
        self.thread.stop()
        self.thread.wait(3000)
        self.app.quit()

    def run(self) -> None:
        """进入 Qt 事件循环。"""
        sys.exit(self.app.exec_())


def main() -> None:
    """程序入口。"""
    print("\n" + "=" * 56)
    print("AutoAnswer v4.0 - 三类题型状态机版")
    print("=" * 56 + "\n")
    check_dependencies()
    if not DEEPSEEK_API_KEY:
        print("未配置 DEEPSEEK_API_KEY")
        input("按回车退出...")
        sys.exit(1)
    try:
        AutoAnswerApp().run()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        logger.error("应用异常退出: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

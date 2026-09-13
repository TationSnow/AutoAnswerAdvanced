"""
主程序 - RapidOCR + 题库 + LLM + 自动点击
支持两种模式：
  - quiz   做题模式：查题库 → 未命中调 LLM → 点击选项/下一题
  - record 题库纪录模式：在 quiz 基础上，把新题的 AI 答案写入题库，
                        并尝试从屏幕捕获“正确答案/解析”做二次校验
"""
import sys
import os
import json
import time
import re
import logging
from typing import Optional, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGION_JSON = os.path.join(BASE_DIR, "capture_region.json")
REGION_SELECTOR_SCRIPT = os.path.join(BASE_DIR, "region_selector.py")
KB_DB_FULL = os.path.join(BASE_DIR, "question_bank.db")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    missing = []
    core_deps = {
        'PyQt5': 'PyQt5',
        'mss': 'mss',
        'PIL': 'Pillow',
        'keyboard': 'keyboard',
        'rapidocr_onnxruntime': 'rapidocr_onnxruntime',
        'pyautogui': 'pyautogui',
        'openai': 'openai',
    }
    for module, package in core_deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
            logger.warning(f"✗ {package} 未安装")

    if missing:
        print("\n" + "=" * 50)
        print("⚠️  缺少依赖包")
        print("=" * 50)
        print(f"\n  pip install {' '.join(missing)}")
        print("=" * 50)
        input("\n按回车退出...")
        sys.exit(1)
    logger.info("✓ 依赖检查通过")


from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from config import (
    CAPTURE_REGION, HOTKEY_SELECT, HOTKEY_PAUSE,
    HOTKEY_CAPTURE, HOTKEY_TOGGLE_MODE, HOTKEY_EXIT,
    SCAN_INTERVAL, POST_CLICK_WAIT, POST_NEXT_WAIT,
    AUTO_CLICK_ENABLED, AUTO_NEXT_ENABLED,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEEPSEEK_ENABLE_SEARCH,
    MODE_QUIZ, MODE_RECORD, DEFAULT_MODE,
    KB_DB_PATH, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    EMBEDDING_API_KEY, SIMILARITY_THRESHOLD,
)
from overlay import AnswerOverlay
from ocr_engine import OCREngine
from ai_solver import DeepSeekSolver
from embedding_client import EmbeddingClient
from knowledge_base import KnowledgeBase
from screen_operator import ScreenOperator


# ====================== 热键 ======================

class HotkeyHandler(QObject):
    select_region_signal = pyqtSignal()
    toggle_pause_signal = pyqtSignal()
    capture_signal = pyqtSignal()
    toggle_mode_signal = pyqtSignal()
    exit_signal = pyqtSignal()

    def setup_hotkeys(self):
        try:
            import keyboard
            keyboard.add_hotkey(HOTKEY_SELECT, lambda: self.select_region_signal.emit())
            keyboard.add_hotkey(HOTKEY_PAUSE, lambda: self.toggle_pause_signal.emit())
            keyboard.add_hotkey(HOTKEY_CAPTURE, lambda: self.capture_signal.emit())
            keyboard.add_hotkey(HOTKEY_TOGGLE_MODE, lambda: self.toggle_mode_signal.emit())
            keyboard.add_hotkey(HOTKEY_EXIT, lambda: self.exit_signal.emit())
            logger.info(
                f"热键已注册: {HOTKEY_SELECT} / {HOTKEY_PAUSE} / "
                f"{HOTKEY_CAPTURE} / {HOTKEY_TOGGLE_MODE} / {HOTKEY_EXIT}"
            )
        except Exception as e:
            logger.error(f"热键注册失败: {e}")


# ====================== 捕获线程 ======================

class CaptureThread(QThread):
    result_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)
    mode_signal = pyqtSignal(str)

    def __init__(self, mode: str = DEFAULT_MODE):
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
        self.last_text = ""
        self.cooldown_until = 0.0

        self.start_time = time.time()
        self.ai_count = 0
        self.kb_hit_count = 0
        self.recorded_count = 0

    # ------------------------------------------------------------------
    def _load_region(self) -> Dict:
        try:
            with open(REGION_JSON, "r", encoding="utf-8") as f:
                region = json.load(f)
                logger.info(f"已加载捕获区域: {region}")
                return region
        except FileNotFoundError:
            return CAPTURE_REGION
        except Exception as e:
            logger.error(f"加载区域失败: {e}")
            return CAPTURE_REGION

    # ------------------------------------------------------------------
    def run(self):
        try:
            import mss
            self.sct = mss.mss()

            # 初始化各引擎
            self.ocr = self._init_ocr()
            if self.ocr is None:
                return

            self.kb = self._init_kb()
            self.ai = self._init_ai()
            if self.ai is None:
                return

            self.operator = ScreenOperator(self.region, enabled=AUTO_CLICK_ENABLED)

            self._print_banner()
            self.mode_signal.emit(self.mode)

            while self.running:
                if self.single_shot:
                    self.single_shot = False
                    self._safe_process()
                elif not self.paused:
                    self._safe_process()
                time.sleep(SCAN_INTERVAL)
        except Exception as e:
            logger.error(f"捕获线程异常: {e}")
        finally:
            if self.sct:
                try:
                    self.sct.close()
                except Exception:
                    pass

    def _safe_process(self):
        try:
            self._process_frame()
        except Exception as e:
            logger.error(f"处理帧失败: {e}")

    # ------------------------------------------------------------------
    def _init_ocr(self) -> Optional[OCREngine]:
        try:
            logger.info("正在初始化 OCR 引擎...")
            engine = OCREngine(use_gpu=False, lang='ch', show_log=False)
            logger.info(f"OCR 版本: {engine.get_version_info()}")
            return engine
        except Exception as e:
            logger.error(f"OCR 初始化失败: {e}")
            return None

    def _init_kb(self) -> Optional[KnowledgeBase]:
        try:
            logger.info("正在初始化 Embedding 客户端...")
            emb = EmbeddingClient(
                base_url=EMBEDDING_BASE_URL,
                model=EMBEDDING_MODEL,
                api_key=EMBEDDING_API_KEY,
            )
            # 探测维度（同时验证服务可用性）
            v = emb.embed_one("test")
            if v is not None:
                logger.info(f"Embedding 维度: {v.shape[0]}")
            else:
                logger.warning("Embedding 服务未响应，将退化为精确匹配")

            logger.info(f"正在加载题库: {KB_DB_FULL}")
            kb = KnowledgeBase(
                db_path=KB_DB_FULL,
                embedding_client=emb,
                similarity_threshold=SIMILARITY_THRESHOLD,
            )
            s = kb.stats()
            logger.info(f"题库统计: {s}")
            return kb
        except Exception as e:
            logger.error(f"题库初始化失败: {e}")
            return None

    def _init_ai(self) -> Optional[DeepSeekSolver]:
        try:
            logger.info("正在初始化 AI 解题器...")
            solver = DeepSeekSolver(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                model=DEEPSEEK_MODEL,
                enable_search=DEEPSEEK_ENABLE_SEARCH,
                knowledge_base=self.kb,
            )
            return solver
        except Exception as e:
            logger.error(f"AI 初始化失败: {e}")
            return None

    # ------------------------------------------------------------------
    def _print_banner(self):
        mode_label = "做题模式" if self.mode == MODE_QUIZ else "题库纪录模式"
        logger.info(f"""
{'=' * 56}
🚀 自动答题助手已启动
{'=' * 56}
当前模式: {mode_label}
捕获区域: {self.region}
自动点击: {'开' if AUTO_CLICK_ENABLED else '关'} | 自动下一题: {'开' if AUTO_NEXT_ENABLED else '关'}
快捷键:
  • Ctrl+F1 = 重新框选区域
  • Ctrl+F2 = 暂停/继续
  • Ctrl+F3 = 手动识别一次
  • Ctrl+F4 = 切换 做题/纪录 模式
  • Ctrl+Q  = 退出
{'=' * 56}
""")

    # ------------------------------------------------------------------
    def _process_frame(self):
        if not self.ocr or not self.ai:
            return
        if time.time() < self.cooldown_until:
            return

        t_total = time.time()

        # 1) 截图
        screenshot = self.sct.grab(self.region)
        from PIL import Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        # 2) OCR
        result = self.ocr.recognize(img)
        if not result or not result.get('is_valid'):
            return

        # 3) 去重
        current_key = result['raw_text'][:120]
        if current_key == self.last_text:
            return
        self.last_text = current_key

        question = result['question']
        options = result.get('options', {})
        qtype = result.get('question_type', 'choice')

        logger.info(f"📝 [{self.mode}] 题目: {question[:60]}...")

        # 4) 解题（AI 内部已先查题库）
        answer_data = self.ai.solve(result)
        answer = answer_data.get('answer', '?')
        source = answer_data.get('source', '')

        if 'LLM' in source or source == 'DeepSeek':
            self.ai_count += 1
        elif '题库' in source:
            self.kb_hit_count += 1

        logger.info(f"✅ 答案: 【{answer}】 | 来源: {source}")

        # 5) 记录模式：写入题库
        if self.mode == MODE_RECORD and answer not in ('?', '', None):
            is_new = self.kb.add(
                question=question,
                options=options,
                question_type=qtype,
                answer=answer,
                detail=answer_data.get('detail', ''),
                source='AI',
                verified=False,
            )
            if is_new:
                self.recorded_count += 1
                logger.info(f"📥 已记录新题 (共 {self.recorded_count} 条)")

        # 6) 更新 UI
        self.result_signal.emit(answer_data)

        # 7) 自动点击
        if AUTO_CLICK_ENABLED and answer not in ('?', '', None):
            self._auto_click(result, answer_data)

        # 状态栏
        runtime = int(time.time() - self.start_time)
        self.status_signal.emit(
            f"[{'做题' if self.mode == MODE_QUIZ else '纪录'}] "
            f"运行:{runtime}s | 题库:{self.kb_hit_count} | AI:{self.ai_count} | 新录:{self.recorded_count}"
        )

        logger.info(f"⏱ 总计 {time.time() - t_total:.2f}s")

    # ------------------------------------------------------------------
    def _resolve_click_target(self, ocr_result: Dict, answer: str):
        """把答案映射到 OCR 选项框中心点（局部坐标）。"""
        option_boxes = ocr_result.get('option_boxes', {}) or {}
        qtype = ocr_result.get('question_type', 'choice')

        if qtype == 'choice':
            letters = [x.strip().upper()
                       for x in re.split(r'[,\s、，/]+', answer) if x.strip()]
            for letter in letters:
                if letter in option_boxes:
                    return option_boxes[letter]['center']

        if qtype == 'true_false':
            for letter, info in option_boxes.items():
                text = info.get('text', '')
                if '正确' in answer and re.search(r'正确|对|√|是', text):
                    return info['center']
                if '错误' in answer and re.search(r'错误|错|×|否', text):
                    return info['center']

        return None

    # ------------------------------------------------------------------
    def _auto_click(self, ocr_result: Dict, answer_data: Dict):
        answer = answer_data.get('answer', '')
        target = self._resolve_click_target(ocr_result, answer)
        if target is None:
            logger.warning(f"未找到答案 {answer} 对应的可点击位置，跳过自动点击")
            return

        # 点击选项
        self.operator.click_at(target)

        # 冷却
        self.cooldown_until = time.time() + POST_CLICK_WAIT
        time.sleep(POST_CLICK_WAIT)

        # 记录模式：尝试捕获屏幕上的正确答案/解析
        if self.mode == MODE_RECORD:
            self._try_capture_revealed(ocr_result, answer_data)

        # 点击下一题
        if AUTO_NEXT_ENABLED:
            self._try_click_next()
            self.cooldown_until = time.time() + POST_NEXT_WAIT

    # ------------------------------------------------------------------
    def _try_capture_revealed(self, ocr_result: Dict, answer_data: Dict):
        """点击后截屏，看是否出现“正确答案/解析”，用于校验题库。"""
        try:
            from PIL import Image
            screenshot = self.sct.grab(self.region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra,
                                  "raw", "BGRX")
            result2 = self.ocr.recognize(img)
            raw = result2.get('raw_text', '')
            if not raw:
                return

            revealed_ans = None
            revealed_detail = ""

            m = re.search(r'正确答案[：:]\s*([A-Za-z]+)', raw)
            if not m:
                m = re.search(r'答案[：:]\s*([A-Za-z]{1,3})', raw)
            if m:
                revealed_ans = m.group(1).upper()

            m = re.search(r'解析[：:]\s*(.+?)(?:\n|$)', raw, re.DOTALL)
            if m:
                revealed_detail = m.group(1).strip()[:200]

            if revealed_ans:
                logger.info(f"🖥 屏幕捕获答案: {revealed_ans}")
                if revealed_ans != answer_data.get('answer'):
                    logger.warning(
                        f"⚠️ 屏幕答案 {revealed_ans} 与 AI 答案 "
                        f"{answer_data.get('answer')} 不一致，用屏幕答案覆盖题库"
                    )
                # 更新题库
                match_id = answer_data.get('match_id')
                if match_id:
                    self.kb.update_by_id(match_id, revealed_ans, revealed_detail)
                else:
                    # 之前没有匹配记录，写入新记录
                    self.kb.add(
                        question=ocr_result.get('question', ''),
                        options=ocr_result.get('options', {}),
                        question_type=ocr_result.get('question_type', 'choice'),
                        answer=revealed_ans,
                        detail=revealed_detail,
                        source='屏幕捕获',
                        verified=True,
                    )
        except Exception as e:
            logger.error(f"捕获屏幕答案失败: {e}")

    # ------------------------------------------------------------------
    def _try_click_next(self):
        try:
            from PIL import Image
            screenshot = self.sct.grab(self.region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra,
                                  "raw", "BGRX")
            result = self.ocr.recognize(img)
            btn = result.get('button_boxes', {}) or {}
            target = None
            if 'next' in btn:
                target = btn['next']['center']
            elif 'submit' in btn:
                target = btn['submit']['center']
            if target is None:
                logger.info("未找到『下一题/提交』按钮，跳过")
                return
            self.operator.click_at(target)
            logger.info("已点击『下一题/提交』")
        except Exception as e:
            logger.error(f"点击下一题失败: {e}")

    # ------------------------------------------------------------------
    def update_region(self, new_region: Dict):
        self.region = new_region
        self.last_text = ""
        if self.operator:
            self.operator.update_region(new_region)
        logger.info(f"捕获区域已更新: {new_region}")

    def pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def stop(self):
        self.running = False

    def trigger_capture(self):
        self.single_shot = True

    def toggle_mode(self) -> str:
        self.mode = MODE_RECORD if self.mode == MODE_QUIZ else MODE_QUIZ
        return self.mode


# ====================== 主应用 ======================

class AutoAnswerApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("答题助手")

        self.overlay = AnswerOverlay()

        self.thread = CaptureThread(mode=DEFAULT_MODE)
        self.thread.result_signal.connect(self.overlay.show_answer)
        self.thread.status_signal.connect(self.overlay.update_status)
        self.thread.mode_signal.connect(self._on_mode_changed)

        self.hotkey_handler = HotkeyHandler()
        self.hotkey_handler.select_region_signal.connect(self._on_select_region)
        self.hotkey_handler.toggle_pause_signal.connect(self._on_toggle_pause)
        self.hotkey_handler.capture_signal.connect(self._on_capture)
        self.hotkey_handler.toggle_mode_signal.connect(self._on_toggle_mode)
        self.hotkey_handler.exit_signal.connect(self._on_exit)
        self.hotkey_handler.setup_hotkeys()

        self.thread.start()
        self.overlay.update_status("按 Ctrl+F3 手动识别 | Ctrl+F4 切换模式")

    # ------------------------------------------------------------------
    def _on_mode_changed(self, mode: str):
        label = "做题模式" if mode == MODE_QUIZ else "题库纪录模式"
        self.overlay.show_message(f"已切换到: {label}")

    def _on_capture(self):
        self.overlay.show_message("正在识别...")
        self.thread.trigger_capture()

    def _on_select_region(self):
        logger.info("触发区域选择...")
        was_paused = self.thread.paused
        self.thread.paused = True
        try:
            import subprocess
            subprocess.run(
                [sys.executable, REGION_SELECTOR_SCRIPT],
                timeout=60
            )
            try:
                with open(REGION_JSON, "r", encoding="utf-8") as f:
                    new_region = json.load(f)
                self.thread.update_region(new_region)
                self.overlay.show_message("区域已更新 ✓")
            except Exception as e:
                logger.error(f"加载新区域失败: {e}")
                self.overlay.show_message("区域加载失败 ✗")
        except subprocess.TimeoutExpired:
            self.overlay.show_message("选择超时 ✗")
        except Exception as e:
            logger.error(f"区域选择失败: {e}")
            self.overlay.show_message("选择失败 ✗")
        finally:
            if not was_paused:
                self.thread.paused = False

    def _on_toggle_pause(self):
        is_paused = self.thread.pause()
        self.overlay.show_message(
            "已暂停 ⏸ | 按 Ctrl+F2 继续" if is_paused
            else "连续扫描中 ▶"
        )

    def _on_toggle_mode(self):
        new_mode = self.thread.toggle_mode()
        label = "做题模式" if new_mode == MODE_QUIZ else "题库纪录模式"
        self.overlay.show_message(f"模式切换 → {label}")
        logger.info(f"模式切换 → {label}")

    def _on_exit(self):
        runtime = time.time() - self.thread.start_time
        logger.info(f"运行 {int(runtime)} 秒 | 题库命中 {self.thread.kb_hit_count} "
                    f"| AI {self.thread.ai_count} | 新录 {self.thread.recorded_count}")
        self.thread.stop()
        self.thread.wait(3000)
        self.app.quit()

    def run(self):
        sys.exit(self.app.exec_())


def main():
    print("\n" + "=" * 50)
    print("  自动答题助手 v3.0 (题库 + 自动刷题)")
    print("=" * 50 + "\n")

    check_dependencies()

    if not DEEPSEEK_API_KEY:
        print("❌ 未配置 DEEPSEEK_API_KEY")
        input("按回车退出...")
        sys.exit(1)

    try:
        app = AutoAnswerApp()
        app.run()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"应用异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
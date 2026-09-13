# ================== DeepSeek / LMStudio 生成模型 ==================

DEEPSEEK_API_KEY = "sk-local"
DEEPSEEK_BASE_URL = "http://127.0.0.1:1234/v1"
DEEPSEEK_MODEL = "qwen3.8-9b-heretic-uncensored-nvfp4"
DEEPSEEK_ENABLE_SEARCH = False

# ================== 屏幕捕获配置 ==================

CAPTURE_REGION = {
    "left": 100,
    "top": 100,
    "width": 800,
    "height": 600
}

# ================== 快捷键 ==================

HOTKEY_SELECT = "ctrl+f1"       # 框选区域
HOTKEY_PAUSE = "ctrl+f2"        # 暂停/继续扫描
HOTKEY_CAPTURE = "ctrl+f3"      # 手动识别一次
HOTKEY_TOGGLE_MODE = "ctrl+f4"  # 切换 做题/纪录 模式
HOTKEY_EXIT = "ctrl+q"          # 退出

# ================== 扫描 / 自动化 ==================

SCAN_INTERVAL = 1.0             # 循环扫描间隔（秒）
POST_CLICK_WAIT = 0.8           # 点击选项或确认按钮后等待界面更新
POST_NEXT_WAIT = 1.2            # 点击下一题后等待秒数
OPTION_CLICK_INTERVAL = 0.12    # 多选相邻选项点击间隔
FEEDBACK_POLL_INTERVAL = 0.35   # 等待正确答案/解析时的轮询间隔
FEEDBACK_TIMEOUT = 4.0          # 等待正确答案/解析的最长时间
MIN_AUTO_ANSWER_CONFIDENCE = 0.55  # 低于该 OCR 置信度时只展示不点击

AUTO_CLICK_ENABLED = True       # 是否自动点击选项
AUTO_NEXT_ENABLED = True        # 是否自动点击下一题

# ================== 模式 ==================

MODE_QUIZ = "quiz"              # 做题模式：查库 → LLM → 点击；不写库
MODE_RECORD = "record"          # 题库纪录模式：在 quiz 基础上写入新题
DEFAULT_MODE = MODE_QUIZ

# ================== 题库 / Embedding ==================

KB_DB_PATH = "question_bank.db"
EMBEDDING_BASE_URL = "http://127.0.0.1:1234/v1"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"
EMBEDDING_API_KEY = "sk-local"
SIMILARITY_THRESHOLD = 0.85     # 语义命中阈值

# ================== 按钮关键词（OCR 识别） ==================

BUTTON_KEYWORDS = {
    "confirm": ["确认答案"],
    "next": ["下一题", "下一页", "下一章", "继续答题", "下一节"],
    "submit": ["提交答案", "提交"],
}

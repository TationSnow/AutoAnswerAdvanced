# ================== DeepSeek / LMStudio 生成模型 ==================

DEEPSEEK_API_KEY = "sk-local"
DEEPSEEK_BASE_URL = "http://127.0.0.1:1234/v1"
DEEPSEEK_MODEL = "qwen3.8-9b-heretic-uncensored-nvfp4"
DEEPSEEK_ENABLE_SEARCH = False


# 生成参数
# 注意：推理型/思考型模型（模型名常带 reasoner、thinking、flash 等）会先输出思考内容，
# max_tokens 过小会把额度全部消耗在思考上，导致正文为空（finish_reason=length），
# 表现为“模型返回空响应”。这里默认给足额度，并在检测到截断时自动加倍重试。
DEEPSEEK_MAX_TOKENS = 2048          # 单次生成上限
DEEPSEEK_MAX_TOKENS_LIMIT = 8192    # 因输出截断自动加倍时的硬上限
DEEPSEEK_TEMPERATURE = 0.0
DEEPSEEK_TIMEOUT = 60              # 单次请求超时（秒）；推理型模型需要更大值

# 是否优先尝试 json_schema 强约束输出。
# 不支持该格式的接口会返回 400，程序会自动降级为 json_object → 纯文本并记住结果；
# 已知接口不支持时可直接置 False，省掉一次注定失败的请求。
DEEPSEEK_JSON_SCHEMA_ENABLED = False

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

# ================== OCR 识别阈值与鲁棒性 ==================
# 说明：RapidOCR 默认把“识别置信度 < text_score”的文字块直接丢弃。
# 实测题库页面上的孤立单字选项（判断题的“对/错”）识别得分只有 0.499 左右，
# 恰好压在主通道阈值 0.5 之下，会被整块丢弃，导致判断题永远解析不出选项、
# 程序在同一题上无限重复识别。主通道保持严格阈值以保证结果干净，
# 解析失败时再由补救通道用放宽阈值把这类文字救回来。

OCR_TEXT_SCORE = 0.5            # 主通道：识别置信度阈值
OCR_DET_BOX_THRESH = 0.5        # 主通道：检测框置信度阈值
OCR_DET_UNCLIP_RATIO = 1.6      # 主通道：检测框扩张比例
OCR_MIN_ITEM_SCORE = 0.2        # 主通道：文字块保留的最低置信度

OCR_RESCUE_TEXT_SCORE = 0.1     # 补救通道：识别置信度阈值（救回“对/错”等低分单字）
OCR_RESCUE_DET_BOX_THRESH = 0.2  # 补救通道：检测框置信度阈值
OCR_RESCUE_DET_UNCLIP_RATIO = 1.6  # 补救通道：检测框扩张比例
OCR_RESCUE_MIN_ITEM_SCORE = 0.1  # 补救通道：文字块保留的最低置信度
OCR_RESCUE_COOLDOWN = 3.0       # 同一画面重复补救的最小间隔（秒）

OCR_MAX_PIXELS = 6000000        # 截图超过该像素数时先缩放，避免检测耗时爆炸
OCR_MAX_ASPECT_RATIO = 8.0      # 极端长宽比会触发检测模型内部放大，先补边规避
OCR_WARMUP_SIZE = None          # OCR 预热图尺寸 (高, 宽)；None 时使用默认值

OCR_SLOW_WARN_SECONDS = 5.0     # 单帧推理超过该秒数输出告警
OCR_SLOW_ERROR_SECONDS = 30.0   # 单帧推理超过该秒数输出错误与排查建议

# ================== 无进展退避 ==================
# 画面内容不变且始终无法解析时，逐步放慢扫描频率，避免以 1 秒间隔
# 无限重复识别并刷新同一批日志。

FAIL_BACKOFF_THRESHOLD = 3      # 连续无进展次数达到该值后开始退避
FAIL_BACKOFF_MAX_INTERVAL = 5.0  # 退避后的扫描间隔上限（秒）
FAIL_BACKOFF_LOG_EVERY = 10     # 退避期间每隔多少次无进展再提醒一次

# ================== 页面装饰关键词（OCR 识别） ==================

PAGE_CHROME_KEYWORDS = ["题库练习", "纠"]  # 页面标题、悬浮纠错按钮等非题目文字

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

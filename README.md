# 🧠 AutoAnswer 智答助手 — AI 智能屏幕答题工具

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![RapidOCR](https://img.shields.io/badge/OCR-RapidOCR-orange?logo=opencv&logoColor=white)
![DeepSeek](https://img.shields.io/badge/AI-DeepSeek-green?logo=openai&logoColor=white)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-purple?logo=qt&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **AutoAnswer — AI-Powered Screen OCR Quiz Solver**
>
> **智答助手 — 基于 AI 的屏幕 OCR 智能答题工具**

一款基于 AI 的桌面答题神器。框选屏幕区域，程序自动截图、识别题目文字、解析选项，秒级调用大模型推理出答案，结果直接显示在半透明悬浮窗上——全程无需手动操作。内置 50+ 条本地知识库，命中即返回，响应低至毫秒级；未命中则调用 DeepSeek API 联网搜索作答，准确率极高。OCR 引擎采用 RapidOCR，体积仅 50MB，无需安装庞大的深度学习框架，开箱即用。支持全局快捷键一键框选、暂停、手动触发识别，操作丝滑不打断工作流。适用于线上知识竞赛、时政学习、模拟测验、趣味问答等场景。轻量、快速、安静，藏在后台帮你拿分的那个队友。

---

## 📖 目录

- [项目简介](#-项目简介)
- [功能特性](#-功能特性)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [快捷键](#-快捷键)
- [项目结构](#-项目结构)
- [性能指标](#-性能指标)
- [常见问题](#-常见问题)
- [开发计划](#-开发计划)
- [许可证](#-许可证)

---

## 🔍 项目简介

**AutoAnswer 智答助手** 是一款结合实时屏幕捕获、光学字符识别（OCR）和大语言模型（LLM）推理的桌面智能答题工具。

它在后台静默运行，持续监控用户指定的屏幕区域。当检测到新题目时，自动提取文字、解析题干与选项、查询 AI 模型，并在悬浮窗中展示答案——全程仅需数秒。

### 使用平台
- Windows🖥 （目前没推出exe）
- Android📱（apk）

### 适用场景

- 📚 线上知识竞赛与答题活动
- 🏛️ 时政学习与模拟测验
- 📝 限时考试练习与自我评估
- 🎮 趣味问答与互动竞猜

---

## ✨ 功能特性

### 🔤 智能 OCR 识别
- 基于 **RapidOCR**（ONNX Runtime 推理后端），轻量高效，无需安装 PaddlePaddle 框架
- 自动检测并结构化解析题干与 A/B/C/D 选项
- 支持多种选项格式：`A.` / `A、` / `A)` / `A：` / `①` 等
- 基于置信度过滤低质量识别结果，减少误判
- 完整支持中文、中英混排等复杂文本版式

### 🤖 AI 智能推理
- 集成 **DeepSeek API**，支持快速准确的答案推理
- 可选 **联网搜索增强**（`enable_search`），实时获取最新事实信息
- 内置 **本地知识库**，预置 50+ 条目，命中即返回，无需调用 API
- 双层应答策略：本地缓存优先 → API 兜底
- 自动重试机制（最多 3 次），含指数退避策略

### 🖥️ 无感桌面体验
- **半透明悬浮窗**（置顶显示、圆角设计），展示答案不遮挡工作界面
- **可视化区域选择器**，拖拽框选任意屏幕区域作为监控目标
- **全局快捷键** 控制，无需切换窗口
- **翻题回退机制**：平台没有“上一题 / 下一题”按钮时，自动改用左滑 / 右滑手势模拟翻页
  （按钮优先、手势兜底，两种方式共用同一条导航链路）
- 线程安全架构：UI 线程与捕获线程通过 Qt 信号完全解耦
- 内容去重：屏幕内容未变化时自动跳过，避免重复处理

### ⚡ 性能优化
- 启动时模型预热，消除首次推理延迟
- 轻量 ONNX 推理，CPU 占用与内存消耗极低
- 可配置扫描间隔，灵活平衡响应速度与资源占用
- 精简 Prompt 设计（`max_tokens=20`，`temperature=0.0`），确保 API 响应快速且确定

---

## 🏗️ 系统架构

```mermaid
graph TB
    subgraph 主线程 - UI
        HK[HotkeyHandler<br/>全局热键]
        OV[AnswerOverlay<br/>悬浮窗]
        RS[RegionSelector<br/>区域选择]
        APP[AutoAnswerApp]
    end

    subgraph 捕获线程 - 后台
        CT[CaptureThread]
        MSS[mss<br/>屏幕截图]
        OCR[OCREngine<br/>RapidOCR]
        AI[DeepSeekSolver<br/>本地库 + API]
    end

    HK -- Qt信号 --> APP
    APP -- Qt信号 --> OV
    APP -- 启动 --> RS
    APP -- 管理 --> CT
    MSS -- 图像 --> OCR
    OCR -- 题目数据 --> AI
    AI -- 答案 --> APP

    style HK fill:#4a9eff,stroke:#fff,color:#fff
    style OV fill:#00c853,stroke:#fff,color:#fff
    style RS fill:#ff9800,stroke:#fff,color:#fff
    style APP fill:#7c4dff,stroke:#fff,color:#fff
    style CT fill:#455a64,stroke:#fff,color:#fff
    style MSS fill:#78909c,stroke:#fff,color:#fff
    style OCR fill:#ef6c00,stroke:#fff,color:#fff
    style AI fill:#2e7d32,stroke:#fff,color:#fff
```

**数据流：**
1. `mss` 对用户指定的屏幕区域进行截图
2. `OCREngine`（RapidOCR）提取文字并解析为结构化题目数据
3. `DeepSeekSolver` 优先查询本地知识库，未命中则调用 API
4. 结果通过 Qt 信号传递至 `AnswerOverlay` 悬浮窗展示


---

## 🚀 快速开始

### 环境要求

- Python 3.8 及以上
- Windows 10 / 11（主要适配平台；Linux / macOS 需针对 `keyboard` 和 `mss` 做适配调整）

### 安装依赖


# 克隆仓库
git clone https://github.com/imjianglee1/AutoAnswer_ai.git
cd AutoAnswer

# 安装依赖
pip install PyQt5 mss Pillow keyboard rapidocr_onnxruntime openai


### 配置 API 密钥

打开 `config.py`，填入你的 DeepSeek API 密钥：


DEEPSEEK_API_KEY = "sk-your-api-key-here"


> 🔑 前往 [DeepSeek 开放平台](https://platform.deepseek.com/) 获取密钥

### 启动运行


python main.py


首次启动：
1. 按 `Ctrl+F1` 框选包含题目的屏幕区域
2. 按 `Ctrl+F3` 手动触发一次识别，或按 `Ctrl+F2` 开启连续扫描
3. 答案将在 1–3 秒内显示在悬浮窗中

---


## 配置说明

- **修改提示词**：请编辑 `ai_solver.py` 文件第 149 行，调整 `prompt` 变量内容即可。
- **更换模型提供商**：请编辑 `config.py` 文件，修改 `MODEL_PROVIDER` 相关配置项。

```python
# ai_solver.py 第149行示例
prompt = "请按以下规则求解..."  # 在此处修改你的提示词

# config.py 示例
MODEL_PROVIDER = "openai"  # 可选: openai, anthropic, deepseek

```
---


## ⚙️ 配置说明

所有配置项集中在 `config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | `""` | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API 端点地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型选择：`deepseek-chat`（快速）或 `deepseek-reasoner`（推理更强） |
| `DEEPSEEK_ENABLE_SEARCH` | `False` | 是否启用联网搜索以获取更准确的事实性答案 |
| `DEEPSEEK_MAX_TOKENS` | `2048` | 单次生成上限；推理型模型额度不足会导致正文为空 |
| `DEEPSEEK_MAX_TOKENS_LIMIT` | `8192` | 检测到输出截断时自动加倍的硬上限 |
| `DEEPSEEK_TIMEOUT` | `60` | 单次请求超时（秒） |
| `DEEPSEEK_JSON_SCHEMA_ENABLED` | `True` | 是否优先尝试 json_schema；已知接口不支持时置 `False` 可省掉一次失败请求 |
| `SCAN_INTERVAL` | `1.0` | 屏幕扫描间隔（秒） |
| `CAPTURE_REGION` | `{left, top, width, height}` | 默认捕获区域（可通过区域选择器覆盖） |
| `OCR_TEXT_SCORE` | `0.5` | 主通道识别置信度阈值（低于该值的文字块被丢弃） |
| `OCR_RESCUE_TEXT_SCORE` | `0.1` | 补救通道识别阈值，用于救回“对/错”等低分单字选项 |
| `OCR_RESCUE_COOLDOWN` | `3.0` | 同一画面重复补救识别的最小间隔（秒） |
| `OCR_MAX_PIXELS` | `6000000` | 截图超过该像素数时先缩放，避免识别耗时爆炸 |
| `OCR_SLOW_WARN_SECONDS` | `5.0` | 单帧 OCR 超过该耗时输出告警 |
| `FAIL_BACKOFF_MAX_INTERVAL` | `5.0` | 画面无进展时的最大扫描间隔（秒） |
| `SWIPE_FALLBACK_ENABLED` | `True` | 未找到“下一题”按钮时，是否用滑动手势替代点击 |
| `SWIPE_PREV_ENABLED` | `True` | 是否允许用滑动回退到“上一题” |
| `SWIPE_NEXT_DIRECTION` | `left` | 左滑 = 下一题 |
| `SWIPE_PREV_DIRECTION` | `right` | 右滑 = 上一题 |
| `SWIPE_PROFILES` | 3 档 | 手势档位：`(名称, 拖动距离比例, 纵向锚点比例, 时长秒, 中间步数)`，滑动无效时逐档升级 |
| `SWIPE_SETTLE_WAIT` | `1.2` | 滑动后等待页面刷新的秒数 |
| `SWIPE_MAX_ATTEMPTS` | `3` | 同一题最多滑动推进次数 |
| `SWIPE_SKIP_STUCK_ENABLED` | `False` | 画面长期无法解析且无按钮时，是否滑动跳走（默认关闭） |
| `SWIPE_SKIP_STUCK_FRAMES` | `8` | 连续无进展多少帧后触发跳走 |
| `SWIPE_SKIP_MAX_ESCAPES` | `2` | 连续最多跳走几帧，仍解析不出则回退一帧并放弃跳走 |
| `SWIPE_SKIP_ROLLBACK_ENABLED` | `True` | 跳走失败时是否反向滑动回退一帧 |

---

## ⌨️ 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + F1` | 打开区域选择器 — 拖拽框选屏幕监控区域 |
| `Ctrl + F2` | 暂停 / 恢复连续扫描 |
| `Ctrl + F3` | 手动触发一次识别 |
| `Ctrl + F4` | 切换 做题 / 题库纪录 模式 |
| `Ctrl + F5` | 手动上一题 — 有按钮点按钮，没有按钮则右滑模拟 |
| `Ctrl + F6` | 手动下一题 — 有按钮点按钮，没有按钮则左滑模拟 |
| `Ctrl + Q` | 退出程序 |

---

## 📁 项目结构


AutoAnswer/
├── main.py               # 程序入口，线程调度，快捷键管理
├── ocr_engine.py         # RapidOCR 封装 — 图像预处理、文字提取、题目解析
├── ai_solver.py          # DeepSeek API 客户端 — 本地知识库查询、API 调用、响应解析
├── navigation.py         # 翻题策略 — 按钮优先、滑动手势兜底（无按钮平台）
├── automation.py         # 动作计划与同一题状态机 — 防重复作答、限制重试次数
├── answer_strategy.py    # 题型策略 — 答案规范化与校验
├── knowledge_base.py     # 本地题库 — 向量检索、反馈回写
├── models.py             # 领域模型 — 题目快照、导航指令、动作计划
├── screen_operator.py    # 屏幕操作 — 点击与滑动手势原语
├── overlay.py            # PyQt5 悬浮窗 — 答案展示、自适应尺寸、自动淡出
├── region_selector.py    # 可视化屏幕区域选择器，支持实时坐标反馈
├── config.py             # 集中配置 — API 密钥、快捷键、阈值、导航与手势参数
├── capture_region.json   # 持久化的捕获区域坐标（自动生成）
├── tests/                # 单元测试与集成测试（TDD）
└── README.md             # 本文件


---

## 📊 性能指标

以下数据在典型桌面环境（Intel i5-12400, 16 GB RAM, 无 GPU）下测得：

| 阶段 | 典型耗时 |
|------|----------|
| 屏幕截图（`mss`） | ~10–30 ms |
| OCR 推理（RapidOCR ONNX） | ~100–300 ms |
| 本地知识库查询 | < 1 ms |
| DeepSeek API 调用（单次） | ~0.5–2.0 s |
| **端到端（调用 API）** | **~1–3 s** |
| **端到端（本地命中）** | **~150–400 ms** |

> 💡 **提示：** 在 `config.py` 的 `LOCAL_KNOWLEDGE` 中添加领域相关条目，可以最大化本地命中率，减少 API 调用次数与延迟。

---

## ❓ 常见问题

<details>
<summary><b>为什么选择 RapidOCR 而不是 PaddleOCR？</b></summary>

RapidOCR 使用 ONNX Runtime 作为推理后端，无需安装庞大的 PaddlePaddle 框架（约 1.5 GB+），优势包括：
- **安装体积大幅缩小**（约 50 MB vs 约 1.5 GB）
- **启动速度更快** — 无框架初始化开销
- **跨平台兼容性更好** — ONNX Runtime 在 Windows、Linux、macOS 上均可稳定运行
- **识别精度相当** — RapidOCR 底层使用与 PaddleOCR 相同的 PP-OCR 模型架构（ONNX 格式）

</details>

<details>
<summary><b>可以使用其他大模型服务商吗？</b></summary>

可以。`DeepSeekSolver` 使用 OpenAI 兼容的 API 格式，你可以将其指向任何支持相同接口的服务商（如 OpenAI、Moonshot / Kimi、智谱 GLM、通过 Ollama / vLLM 部署的本地模型等），只需修改 `config.py` 中的 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_API_KEY` 即可。

</details>

<details>
<summary><b>支持 Linux 或 macOS 吗？</b></summary>

核心逻辑是跨平台的，但需注意：
- `keyboard` 库在 Linux 上需要 **root / sudo** 权限
- `mss` 屏幕截图在所有平台均可正常工作
- PyQt5 悬浮窗在非 Windows 窗口合成器下的渲染效果可能略有差异

</details>

<details>
<summary><b>如何向本地知识库添加自定义题目？</b></summary>

编辑 `config.py` 中的 `LOCAL_KNOWLEDGE` 字典：


"你的条目名称": {
    "keys": ["关键词1", "关键词2", "关键词3"],  # 匹配触发词
    "answer": "A",                               # 正确选项字母
    "detail": "简要解释说明"                       # 可选的解析
},


识别出的题目文本中只要包含至少一个关键词，即可触发本地匹配。

</details>

<details>
<summary><b>线上 API 一直提示“模型返回空响应”，答案拿不到怎么办？</b></summary>

先看日志里的 `finish_reason` 与 `usage`（程序已把它们写进失败原因）：

- **`finish_reason=length` + `reasoning_tokens` 等于全部额度**：
  说明用的是**推理型/思考型模型**（模型名常带 reasoner、thinking、flash 等），
  token 全部被思考过程消耗，正文（`content`）为空。处理方式：
  提高 `DEEPSEEK_MAX_TOKENS`（程序也会在检测到截断时自动加倍重试，
  上限为 `DEEPSEEK_MAX_TOKENS_LIMIT`），或改用非推理模型。
- **`content` 为空但 `reasoning_content` 有内容**：
  程序会自动改用 `reasoning_content` 解析答案，无需额外配置。
- **`HTTP 400` 与输出格式无关**（例如 `Model not found`、鉴权失败）：
  这类属于配置问题，程序会原样上报错误且不重试，请检查 `DEEPSEEK_BASE_URL` /
  `DEEPSEEK_MODEL` / `DEEPSEEK_API_KEY` 是否匹配（注意别把线上模型名指向本机服务，或反之）。
- **接口不支持 `json_schema`**：程序会自动降级 `json_schema → json_object → 纯文本`，
  并记住结果，同一次运行内不再重复试探；已知接口不支持时可直接把
  `DEEPSEEK_JSON_SCHEMA_ENABLED` 设为 `False`。
- **请求超时**：推理型模型单次耗时可能超过一分钟，可提高 `DEEPSEEK_TIMEOUT`。

启动日志里会打印 `生成模型` 一行，包含 endpoint、模型名、max_tokens 与协议梯度，
可先据此确认配置没有错配。

</details>

<details>
<summary><b>OCR 识别部分文字不准确怎么办？</b></summary>

- 适当放大捕获区域，确保文字没有被裁切
- 确保监控区域有足够的对比度（避免透明或重叠窗口干扰）
- 调整 `config.py` 中的识别阈值：`OCR_TEXT_SCORE`（主通道识别阈值，默认 `0.5`）
  与 `OCR_RESCUE_TEXT_SCORE`（补救通道，默认 `0.1`）— 调高可过滤噪声，调低可捕获更多文字

</details>

<details>
<summary><b>判断题/单选题一直卡在同一题不动，日志重复输出同一行 OCR 结果怎么办？</b></summary>

这类现象说明 **OCR 识别结果无法解析成有效题目**（典型原因是选项没被识别出来，
例如判断题的“对/错”是孤立单字，识别得分约 0.499，会被 RapidOCR 的默认阈值 0.5 丢掉），
主循环因此每帧都判定“题目无效”并静默返回。

程序内置的处理机制：

- **补救通道**：主通道解析失败时，自动用放宽阈值（`OCR_RESCUE_*`）再识别一遍，
  把这类低分单字选项救回来；同一画面在 `OCR_RESCUE_COOLDOWN` 秒内只补救一次。
- **失败原因日志**：无法解析时会输出具体原因（题干长度、选项数量），
  可用于判断是“题干没识别到”还是“选项没识别到”。
- **选项结构自适应**：选项标签是圆形徽章里的单个字母、与文字之间**没有任何标点**
  （形如 `A 工作证`）时，程序会自动补上标准分隔符后再解析，避免整题选项全部丢失。
- **无进展退避**：画面内容不变且始终无法推进时，扫描间隔会逐步放慢到
  `FAIL_BACKOFF_MAX_INTERVAL`，避免持续刷同一批日志并占满 CPU。
  画面是否“没变”以**画面签名**为准，不受 OCR 文本抖动影响。
- **慢推理告警**：单帧 OCR 超过 `OCR_SLOW_WARN_SECONDS` 会输出告警，
  超过 `OCR_SLOW_ERROR_SECONDS` 会提示缩小捕获区域（截图越大、画面越花，识别越慢）。

排查建议：先用 `python debug_ocr.py --image 截图.png` 查看识别出的文字块与解析结果，
确认捕获区域是否完整覆盖题干与全部选项。

</details>

<details>
<summary><b>平台没有“上一题 / 下一题”按钮，程序答完一题就卡住不动怎么办？</b></summary>

部分答题平台把题目做成了手势翻页，页面上根本没有翻题按钮。
这类平台上旧版本会一直找不到“下一题”按钮，于是卡在同一题上反复识别。

现在程序内置**导航回退机制**，翻题逻辑按优先级选择落地方式：

| 优先级 | 方式 | 触发条件 |
|--------|------|----------|
| 1 | 点击按钮 | 页面上识别到明确的“下一题 / 上一题”按钮 |
| 2 | 滑动手势 | **没有这类按钮时**，在捕获区域内做一次鼠标拖拽 |

方向映射由配置决定（默认左滑 = 下一题，右滑 = 上一题）：

```python
# config.py
SWIPE_NEXT_DIRECTION = "left"     # 左滑 = 下一题
SWIPE_PREV_DIRECTION = "right"    # 右滑 = 上一题
```

使用要点：

- **自动推进**：答完一题后找不到按钮，会自动左滑翻到下一题；
  页面没推进时按 `SWIPE_PROFILES` 逐档升级手势（拉长距离、上移锚点），
  最多重试 `SWIPE_MAX_ATTEMPTS` 次，不会无限滑动。
- **手动翻题**：`Ctrl+F5` 上一题 / `Ctrl+F6` 下一题。同一条导航链路，
  有按钮就点按钮，没有按钮就滑动，因此没有按钮的平台也能手动翻页。
- **手势安全**：拖动起止点始终限制在捕获区域内并留出边缘余量，
  不会拖到答题选项上误选；手势期间临时关闭鼠标角落保护，
  异常时也会松开鼠标按键，不会出现“鼠标被按住不放”。
- **画面识别不出来且没有按钮**（默认关闭）：把 `SWIPE_SKIP_STUCK_ENABLED`
  设为 `True` 后，程序会在连续 `SWIPE_SKIP_STUCK_FRAMES` 帧无法解析时滑动跳走；
  连续跳走超过 `SWIPE_SKIP_MAX_ESCAPES` 帧仍解析不出题目时，
  会反向滑动回退一帧并停止跳走，避免越滑越远。

想让滑动更容易被页面识别，可以调大拖动距离与步数：

```python
# config.py
SWIPE_PROFILES = (
    ("标准", 0.55, 0.50, 0.32, 8),      # (名称, 距离比例, 纵向锚点比例, 时长, 步数)
    ("长距", 0.75, 0.50, 0.38, 10),
    ("长距上移", 0.75, 0.70, 0.38, 12),
)
```

启动日志里会打印实际生效的导航方式，便于确认配置是否按预期生效：

```
导航策略: 按钮导航 → 滑动导航（左滑→下一题 / 右滑→上一题，共 3 档）
```

</details>

---

## 🔮 开发计划

- [ ] 多显示器支持
- [ ] OCR GPU 加速（ONNX CUDA / DirectML）
- [ ] 答题历史记录与导出
- [ ] 大模型服务商插件化
- [ ] 系统托盘集成与通知推送
- [ ] 本地知识库在线更新

---

## 📜 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

<p align="center">
  <sub>使用 RapidOCR、DeepSeek 和 PyQt5 构建 ❤️</sub>
</p>


---

## 修复要点

| # | 问题 | 修复 |
|---|------|------|
| 1 | 徽章用 `<p align="center">` + `<img>` 标签，部分 Markdown 渲染器不识别 HTML | 改为纯 Markdown `![alt](url)` 语法，GitHub 100% 兼容 |
| 2 | 双语标题用 `<p><b>` 嵌套，容易被吞 | 改为 `>` 引用块语法 |
| 3 | 系统架构 ASCII 图表没有被 `  ` 包裹 | 加上 `    ` 代码块，确保等宽渲染 |
| 4 | 顶部简介段落加入了之前你认可的吸引人的中文短描述 | 直接内嵌在标题下方 |
```

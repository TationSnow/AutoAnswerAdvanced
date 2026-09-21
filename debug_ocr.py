"""OCR 调试工具。

按需截图或读取已有截图，打印识别出的文字块与解析结果，
用于定位“题目解析失败 / 选项缺失 / 单帧推理异常缓慢”一类问题。

用法：
    python debug_ocr.py                     # 3 秒后按 capture_region.json 截屏识别
    python debug_ocr.py --image 截图.png     # 直接识别已有截图（推荐用于问题复现）
    python debug_ocr.py --region 0,0,800,600 # 临时指定捕获区域
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from PIL import Image  # noqa: E402

from models import QuestionSnapshot  # noqa: E402
from ocr_engine import OCREngine  # noqa: E402

REGION_JSON = os.path.join(BASE_DIR, "capture_region.json")


def load_region(explicit: Optional[str] = None) -> Dict[str, int]:
    """读取捕获区域：命令行优先，其次读取持久化配置。"""
    if explicit:
        parts = [int(value) for value in explicit.split(",")]
        if len(parts) != 4:
            raise ValueError("--region 需要 4 个整数：left,top,width,height")
        return {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    try:
        with open(REGION_JSON, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print("读取 %s 失败（%s），使用默认区域" % (REGION_JSON, exc))
        return {"left": 100, "top": 100, "width": 800, "height": 600}


def capture_test_image(region: Dict[str, int]) -> Image.Image:
    """按指定区域截屏并保存到磁盘。"""
    import mss

    print("正在捕获截图（区域: %s）..." % region)
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    image.save("debug_capture.png")
    print("已保存到: debug_capture.png")
    return image


def dump_raw_items(items: List[Dict], elapsed: float) -> None:
    """打印每个文字块的文本、置信度与坐标，便于逐块排查。"""
    print("\n" + "=" * 62)
    print("主通道识别到的文字块（共 %d 块，耗时 %.2fs）" % (len(items), elapsed))
    print("=" * 62)
    for index, item in enumerate(items):
        center = OCREngine._box_center(item.get("box"))
        print(
            "  #%02d score=%.3f center=(%4d,%4d) | %s"
            % (index, float(item.get("score", 0.0)), center[0], center[1], item.get("text", ""))
        )
    if not items:
        print("  （主通道未识别到任何文字，请检查捕获区域与截图内容）")


def dump_snapshot(engine: OCREngine, snapshot: QuestionSnapshot) -> None:
    """打印解析结果概览。"""
    print("\n" + "=" * 62)
    print("解析结果")
    print("=" * 62)
    print("is_valid     : %s" % snapshot.is_valid)
    print("error        : %s" % (snapshot.error or "-"))
    print("external_id  : %s" % (snapshot.external_id or "-"))
    print("题型          : %s" % snapshot.question_type.label)
    print("平均置信度    : %.3f" % snapshot.confidence)
    print("\n--- 题干 ---")
    print(snapshot.question or "（未识别到题干）")
    print("\n--- 选项 ---")
    if snapshot.options:
        for option in snapshot.options:
            print("  %s: %s    center=%s" % (option.label, option.text, option.center))
    else:
        print("  （未识别到任何选项，常见原因是选项文字过小/过短，被识别阈值过滤）")
    print("\n--- 按钮 ---")
    if snapshot.button_boxes:
        for role, button in snapshot.button_boxes.items():
            print("  %s: %s    center=%s" % (role, button.text, button.center))
    else:
        print("  （未识别到操作按钮，将不会自动翻页）")
    print("\n--- 补救识别 ---")
    print("  使用策略: %s" % (engine.last_rescue_name or "未启用"))
    print("=" * 62)


def main() -> None:
    """入口。"""
    parser = argparse.ArgumentParser(description="AutoAnswer OCR 调试工具")
    parser.add_argument("--image", help="已有截图路径，指定后不再实时截图")
    parser.add_argument("--region", help="临时捕获区域，格式 left,top,width,height")
    parser.add_argument("--countdown", type=int, default=3, help="实时截图前的倒计时秒数")
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("OCR 调试工具")
    print("=" * 62)

    engine = OCREngine(use_gpu=False, lang="ch", show_log=True)
    print("版本信息: %s" % engine.get_version_info())

    if args.image:
        image = Image.open(args.image).convert("RGB")
        print("已加载截图: %s（%dx%d）" % (args.image, image.width, image.height))
    else:
        for index in range(args.countdown, 0, -1):
            print(index, end="...")
            time.sleep(1)
        print()
        image = capture_test_image(load_region(args.region))

    # 先单独跑主通道，逐块展示识别结果，再由完整流程给出解析结论。
    items, elapsed = engine.recognize_raw(image)
    dump_raw_items(items, elapsed)
    dump_snapshot(engine, engine.recognize(image))


if __name__ == "__main__":
    main()

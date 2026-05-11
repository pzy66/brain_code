from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


UI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = UI_ROOT.parent
DEFAULT_OUTPUT = UI_ROOT / "artifacts" / "workbench_static_preview.png"

if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from softcopyright_workbench.state import collect_workbench_state  # noqa: E402


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _round(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=1 if outline else 0)


def _pill(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: str, outline: str, font: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    w = box[2] - box[0] + 22
    h = 28
    _round(draw, (x, y, x + w, y + h), 5, fill, outline)
    draw.text((x + 11, y + 5), text, fill=outline, font=font)


def render(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = collect_workbench_state()
    cards = state.status_cards()
    material_count = sum(1 for item in state.materials if item.exists)
    img = Image.new("RGB", (1440, 900), "#F1F4F7")
    draw = ImageDraw.Draw(img)
    title = _font(26, bold=True)
    subtitle = _font(14)
    body = _font(15)
    small = _font(13)
    mono = _font(13)

    _round(draw, (16, 14, 1424, 88), 7, "#FFFFFF", "#CBD3DC")
    draw.text((32, 27), "基于混合脑机接口的智能机械臂协同控制软件 V1.0", fill="#17202A", font=title)
    draw.text((32, 61), "软著 V1.0 UI 原型 | MI、SSVEP、视觉抓取、机械臂控制与材料冻结工作台", fill="#52616F", font=subtitle)
    _round(draw, (1060, 38, 1228, 65), 4, "#FFFFFF", "#8EA4B8")
    draw.text((1072, 42), "软著演示模式", fill="#25313D", font=small)
    _round(draw, (1242, 34, 1318, 69), 4, "#FFFFFF", "#8EA4B8")
    draw.text((1260, 42), "刷新", fill="#25313D", font=small)
    _round(draw, (1330, 34, 1408, 69), 4, "#FFFFFF", "#8EA4B8")
    draw.text((1346, 42), "截图", fill="#25313D", font=small)

    _round(draw, (16, 102, 224, 782), 6, "#FFFFFF", "#CBD3DC")
    nav_items = ["总览", "数据采集", "训练评估", "在线控制", "视觉机械臂", "软著材料"]
    for i, item in enumerate(nav_items):
        y = 108 + i * 45
        fill = "#DCEAF7" if i == 0 else "#FFFFFF"
        _round(draw, (24, y, 216, y + 38), 4, fill, None)
        draw.text((44, y + 9), item, fill="#1E5B8E" if i == 0 else "#25313D", font=body)

    colors = {
        "good": ("#E9F7EF", "#1F7A4D"),
        "warn": ("#FFF3DF", "#A35A00"),
        "bad": ("#FCEAEA", "#A33838"),
        "neutral": ("#EEF2F6", "#4C5B6B"),
    }
    for i, card in enumerate(cards):
        col = i % 3
        row = i // 3
        x = 246 + col * 364
        y = 110 + row * 102
        fill, outline = colors.get(card.level, colors["neutral"])
        _round(draw, (x, y, x + 352, y + 90), 6, "#FFFFFF", "#CBD3DC")
        draw.text((x + 12, y + 12), card.name, fill="#17202A", font=body)
        _pill(draw, x + 286, y + 9, card.state, fill, outline, small)
        draw.text((x + 12, y + 47), card.detail[:48], fill="#52616F", font=small)

    _round(draw, (246, 334, 1414, 516), 6, "#FFFFFF", "#CBD3DC")
    draw.text((258, 344), "V1.0 闭环流程", fill="#17202A", font=body)
    steps = [("采集", "#4F7DB8"), ("训练", "#4E8A62"), ("发布", "#8C6D31"), ("识别", "#7A5CA8"), ("视觉", "#3F7F87"), ("执行", "#9A4F4F"), ("报告", "#59636F")]
    x0, x1, cy = 286, 1374, 424
    draw.line((x0, cy, x1, cy), fill="#CBD3DC", width=3)
    for i, (label, color) in enumerate(steps):
        x = int(x0 + (x1 - x0) * i / (len(steps) - 1))
        draw.ellipse((x - 21, cy - 21, x + 21, cy + 21), fill=color, outline="#FFFFFF", width=2)
        draw.text((x - 5, cy - 11), str(i + 1), fill="#FFFFFF", font=body)
        draw.text((x - 18, cy + 34), label, fill="#25313D", font=small)

    _round(draw, (246, 548, 1414, 644), 6, "#FFFFFF", "#CBD3DC")
    draw.text((258, 558), "当前最重要的工程任务", fill="#17202A", font=body)
    tasks = [
        "新版 MI 分类器入库后补齐训练入口、实时入口和 current_mi_profile.json。",
        "SSVEP、视觉和机械臂侧只读已有入口/profile，不复制训练和执行逻辑。",
        "UI 按钮只打开目录、定位文件、显示命令，保持硬件无关演示模式。",
        "冻结 softcopyright-v1.0 源码边界，排除数据、日志、缓存和临时输出。",
    ]
    for i, task in enumerate(tasks):
        draw.text((258, 586 + i * 19), "• " + task, fill="#25313D", font=small)

    _round(draw, (16, 794, 1424, 886), 6, "#18202A", "#2F3B48")
    draw.text((28, 808), "Workbench loaded in software-copyright demo mode.", fill="#E7EDF5", font=mono)
    draw.text((28, 832), "Repository root: C:/Users/P1233/Desktop/brain/brain_code", fill="#E7EDF5", font=mono)
    draw.text((28, 856), f"Materials: {material_count}/{len(state.materials)} present. Next: publish MI profile and freeze source-deposit scope.", fill="#E7EDF5", font=mono)

    img.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a static preview for the soft-copyright workbench.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

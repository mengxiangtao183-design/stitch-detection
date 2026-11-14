"""Render a restrained academic-style main flow diagram for the TSR unit."""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(r"H:\C3Netqing")
FIG_DIR = ROOT / "docs/figures/trajectory_self_reference_unit"
SVG_PATH = FIG_DIR / "图1_轨迹自参照缺陷检测单元_主流程图.svg"

W, H = 1800, 560
NAVY = "#17324D"
INK = "#1F2933"
MUTED = "#687681"
LINE = "#98A6B3"
PALE = "#F4F7F9"
PALE_BLUE = "#EAF1F6"
WHITE = "#FFFFFF"


def text(x, y, value, *, size=28, weight=400, fill=INK, anchor="middle", family="Microsoft YaHei, SimSun, sans-serif"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f'{escape(value)}</text>'
    )


def block(x, y, w, h, title, symbol, note="", *, fill=WHITE, stroke=NAVY, symbol_size=27):
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>',
        text(x + w / 2, y + 54, title, size=25, weight=600, fill=NAVY),
        f'<line x1="{x + 22}" y1="{y + 72}" x2="{x + w - 22}" y2="{y + 72}" stroke="#D6DEE4" stroke-width="1.4"/>',
        text(x + w / 2, y + 116, symbol, size=symbol_size, weight=500, fill=INK, family="Times New Roman, Cambria Math, serif"),
    ]
    if note:
        parts.append(text(x + w / 2, y + 153, note, size=18, fill=MUTED))
    return "\n".join(parts)


def arrow(x1, y, x2):
    return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{MUTED}" stroke-width="2.2" marker-end="url(#arrow)"/>'


def group_label(x1, x2, label):
    cx = (x1 + x2) / 2
    return "\n".join([
        f'<line x1="{x1}" y1="92" x2="{x2}" y2="92" stroke="{LINE}" stroke-width="1.5"/>',
        f'<line x1="{x1}" y1="92" x2="{x1}" y2="108" stroke="{LINE}" stroke-width="1.5"/>',
        f'<line x1="{x2}" y1="92" x2="{x2}" y2="108" stroke="{LINE}" stroke-width="1.5"/>',
        text(cx, 72, label, size=20, weight=500, fill=MUTED),
    ])


def render():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = [42, 260, 478, 696, 914, 1132, 1350, 1568]
    bw, bh, by = 184, 184, 145
    cy = by + bh / 2
    blocks = [
        ("ACHNet 掩码", "B(x,y)", "320 × 320"),
        ("掩码规范化", "B̃(x,y)", "去除孤立噪声"),
        ("中心轨迹构建", "y(x)", "纵向质量中心"),
        ("全局趋势分离", "r(x)=y(x)−(ax+b)", "方向与局部形变解耦"),
        ("六单元定位", "{uᵢ}ᵢ₌₁⁶", "首尾截断保护"),
        ("轨迹特征编码", "fᵢ ∈ ℝ¹²", "连续性·长度·形态·方向"),
        ("分类与互斥融合", "qₖ=sₖ/τₖ", "最大置信比决策"),
        ("类别与位置输出", "(ĉ, Ω̂)", "正常/四类缺陷"),
    ]

    body = [
        '<defs>',
        f'<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>',
        '</defs>',
        f'<rect width="{W}" height="{H}" fill="{WHITE}"/>',
        group_label(xs[0], xs[1] + bw, "输入与规范化"),
        group_label(xs[2], xs[4] + bw, "轨迹建模与单元定位"),
        group_label(xs[5], xs[6] + bw, "特征表征与缺陷判定"),
        group_label(xs[7], xs[7] + bw, "输出"),
    ]

    for i, (x, data) in enumerate(zip(xs, blocks)):
        title, symbol, note = data
        block_fill = PALE_BLUE if i in (0, 7) else PALE if i in (3, 6) else WHITE
        body.append(block(x, by, bw, bh, title, symbol, note, fill=block_fill, symbol_size=22 if i == 3 else 27))
        if i < len(xs) - 1:
            body.append(arrow(x + bw + 8, cy, xs[i + 1] - 10))

    body.extend([
        f'<line x1="42" y1="390" x2="1752" y2="390" stroke="#D6DEE4" stroke-width="1.4"/>',
        text(42, 435, "输入", size=18, weight=600, fill=NAVY, anchor="start"),
        text(99, 435, "ACHNet 输出的二值线迹掩码", size=18, fill=MUTED, anchor="start"),
        text(665, 435, "自参照", size=18, weight=600, fill=NAVY, anchor="start"),
        text(742, 435, "长度与形态均相对于同图线迹统计量归一化", size=18, fill=MUTED, anchor="start"),
        text(1302, 435, "输出", size=18, weight=600, fill=NAVY, anchor="start"),
        text(1359, 435, "缺陷类型、置信比及异常单元范围", size=18, fill=MUTED, anchor="start"),
        text(W / 2, 505, "B(x,y) → y(x) → r(x) → {uᵢ} → fᵢ → qₖ → (ĉ, Ω̂)", size=25, weight=500, fill=INK, family="Times New Roman, Cambria Math, serif"),
    ])

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    SVG_PATH.write_text(svg, encoding="utf-8")
    print(SVG_PATH)


if __name__ == "__main__":
    render()

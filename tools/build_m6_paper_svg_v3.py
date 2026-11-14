"""Generate compact, publication-style SVG diagrams for the exact M6 network.

The diagrams are derived from:
  configs/ablations/m6_afm_guided_highres_full_150_seed42.yaml
  models/c3net.py
  models/edge_branch.py
  models/localization_branch.py
  models/fusion_head.py
  models/utils.py

The visual language deliberately follows common paper figures: short English
operator labels, tensor slabs, small operator boxes, orthogonal connectors and
dashed enlargement callouts.  No external drawing dependency is required.
"""

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "m6_network_vectors_v3"
OUT.mkdir(parents=True, exist_ok=True)


# restrained paper palette
INK = "#20252B"
MUTED = "#64707D"
LINE = "#515B66"
BLUE = "#82B7DA"
BLUE_D = "#397DAF"
PEACH = "#F6D9C7"
PEACH_D = "#D78A5F"
YELLOW = "#F5E4A8"
YELLOW_D = "#C29B2E"
GREEN = "#CFE8CE"
GREEN_D = "#4E9A66"
PURPLE = "#E5D7EE"
PURPLE_D = "#8761A8"
ROSE = "#F4CBD2"
ROSE_D = "#C85E72"
CYAN = "#CFE8E8"
CYAN_D = "#3A9295"
WHITE = "#FFFFFF"


def start_svg(w: int, h: int, title: str, caption: str) -> list[str]:
    return [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<title>{escape(title)}</title>
<desc>{escape(caption)}</desc>
<defs>
  <marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="{LINE}"/>
  </marker>
  <marker id="arrow-blue" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="{BLUE_D}"/>
  </marker>
  <marker id="arrow-purple" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="{PURPLE_D}"/>
  </marker>
  <marker id="arrow-rose" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L9,3.5 L0,7 Z" fill="{ROSE_D}"/>
  </marker>
  <filter id="soft-shadow" x="-15%" y="-20%" width="135%" height="145%">
    <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#111827" flood-opacity="0.12"/>
  </filter>
  <style>
    .title {{ font-family:'Times New Roman','Microsoft YaHei',sans-serif; font-size:30px; font-weight:700; fill:{INK}; }}
    .subtitle {{ font-family:'Times New Roman','Microsoft YaHei',sans-serif; font-size:16px; fill:{MUTED}; }}
    .module {{ font-family:'Times New Roman',Arial,sans-serif; font-size:18px; font-weight:700; fill:{INK}; }}
    .label {{ font-family:'Times New Roman',Arial,sans-serif; font-size:15px; fill:{INK}; }}
    .small {{ font-family:'Times New Roman',Arial,sans-serif; font-size:13px; fill:{INK}; }}
    .tiny {{ font-family:'Times New Roman',Arial,sans-serif; font-size:11px; fill:{MUTED}; }}
    .tensor-label {{ font-family:'Times New Roman',Arial,sans-serif; font-size:17px; font-weight:700; fill:{INK}; }}
    .tensor-shape {{ font-family:'Times New Roman',Arial,sans-serif; font-size:12px; fill:{INK}; }}
    .caption {{ font-family:'Times New Roman','Microsoft YaHei',sans-serif; font-size:19px; font-weight:700; fill:{INK}; }}
  </style>
</defs>
<rect width="100%" height="100%" fill="{WHITE}"/>
<text x="{w/2}" y="42" text-anchor="middle" class="title">{escape(title)}</text>
<text x="{w/2}" y="66" text-anchor="middle" class="subtitle">{escape(caption)}</text>
''']


def finish_svg(parts: list[str], w: int, h: int, fig_no: str, caption: str) -> None:
    parts.append(f'<text x="{w/2}" y="{h-22}" text-anchor="middle" class="caption">Fig. {escape(fig_no)}. {escape(caption)}</text>')
    parts.append('</svg>')


def text(x: float, y: float, value: str, cls: str = "label", anchor: str = "middle", color: str | None = None) -> str:
    fill = f' fill="{color}"' if color else ""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}"{fill}>{escape(value)}</text>'


def box(x: float, y: float, w: float, h: float, title: str, lines=(), fill=WHITE,
        stroke=LINE, radius: int = 8, shadow: bool = False, title_color: str | None = None) -> str:
    filt = ' filter="url(#soft-shadow)"' if shadow else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.7"{filt}/>']
    out.append(text(x + w / 2, y + 24, title, "module", color=title_color))
    if lines:
        step = 17
        start = y + 45
        for i, line in enumerate(lines):
            out.append(text(x + w / 2, start + i * step, line, "small"))
    return "".join(out)


def opbox(x: float, y: float, w: float, label: str, fill=WHITE, stroke=LINE, h: float = 42) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.35"/>'
            + text(x + w / 2, y + h / 2 + 5, label, "small"))


def tensor(x: float, y: float, w: float, h: float, name: str, shape: str, fill=BLUE, stroke=BLUE_D) -> str:
    skew = 12
    depth = 7
    pts = f'{x+skew},{y} {x+w},{y} {x+w-skew},{y+h} {x},{y+h}'
    side = f'{x+w},{y} {x+w+depth},{y+depth} {x+w-skew+depth},{y+h+depth} {x+w-skew},{y+h}'
    bottom = f'{x},{y+h} {x+w-skew},{y+h} {x+w-skew+depth},{y+h+depth} {x+depth},{y+h+depth}'
    return (f'<polygon points="{side}" fill="{stroke}" opacity="0.45"/>'
            f'<polygon points="{bottom}" fill="{stroke}" opacity="0.28"/>'
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.7" filter="url(#soft-shadow)"/>'
            + text(x + w/2, y + 25, name, "tensor-label")
            + text(x + w/2, y + 44, shape, "tensor-shape"))


def feature_stack(x: float, y: float, w: float, h: float, name: str, shape: str, fill=BLUE, stroke=BLUE_D) -> str:
    out = []
    for dx, dy, opacity in [(16, -12, .25), (8, -6, .45), (0, 0, 1.0)]:
        out.append(f'<rect x="{x+dx}" y="{y+dy}" width="{w}" height="{h}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.2"/>')
    out.append(text(x+w/2, y+h/2-2, name, "tensor-label"))
    out.append(text(x+w/2, y+h/2+17, shape, "tensor-shape"))
    return ''.join(out)


def circle_op(cx: float, cy: float, symbol: str, stroke=LINE, r: float = 15) -> str:
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{WHITE}" stroke="{stroke}" stroke-width="1.8"/>'
            + text(cx, cy + 5, symbol, "module", color=stroke))


def arrow(points, color=LINE, marker="arrow", width: float = 1.8, dash: str | None = None) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#{marker})"{d}/>'


def line(points, color=LINE, width: float = 1.6, dash: str | None = None) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{d}/>'


def callout(x: float, y: float, w: float, h: float, stroke: str, label: str) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{stroke}" fill-opacity="0.055" stroke="{stroke}" stroke-width="2" stroke-dasharray="8 6"/>'
            f'<rect x="{x+18}" y="{y-15}" width="{max(150, len(label)*9.2)}" height="30" rx="15" fill="{stroke}"/>'
             + text(x+31, y+6, label, "small", anchor="start", color=WHITE))


def section_tag(x: float, y: float, label: str, fill: str, width: float) -> str:
    """Compact section label used where a full enclosing frame adds dead space."""
    return (f'<rect x="{x}" y="{y}" width="{width}" height="32" rx="16" fill="{fill}"/>'
            + text(x + 18, y + 22, label, "small", anchor="start", color=WHITE))


def legend(parts: list[str], x: float, y: float) -> None:
    items = [
        (PEACH, PEACH_D, "Encoder"),
        (YELLOW, YELLOW_D, "LMFR / FPN"),
        (ROSE, ROSE_D, "Edge path"),
        (PURPLE, PURPLE_D, "Localization / prior"),
        (GREEN, GREEN_D, "Refinement"),
    ]
    for i, (fill, stroke, label) in enumerate(items):
        xx = x + i * 142
        parts.append(f'<rect x="{xx}" y="{y}" width="20" height="12" fill="{fill}" stroke="{stroke}"/>')
        parts.append(text(xx + 27, y + 11, label, "tiny", anchor="start"))


def figure1_overall() -> None:
    w, h = 2200, 1450
    p = start_svg(w, h, "APH-StitchNet (M6)", "Compact end-to-end architecture and explicit feature routes")

    # input and stem
    p.append(feature_stack(80, 125, 105, 70, "RGB", "320×320×3", PEACH, PEACH_D))
    p.append(box(250, 125, 145, 70, "Stem", ["3×3 / s2", "32 ch"], PEACH, PEACH_D, shadow=True))
    p.append(arrow([(185,160),(250,160)]))

    # backbone stages: deliberately stacked vertically to avoid an overlong main figure
    p.append(callout(70, 245, 430, 610, PEACH_D, "MobileNetV4-Hybrid-Medium"))
    stages = [
        ("C2", "80×80×48", "EdgeResidual", 295),
        ("C3", "40×40×80", "UIB ×2", 425),
        ("C4", "20×20×160", "UIB ×8 + MQA ×4", 555),
        ("C5", "10×10×960", "UIB ×12 + MQA ×4", 685),
    ]
    for i, (name, shape, desc, yy) in enumerate(stages):
        p.append(tensor(120, yy, 150, 60, name, shape, PEACH, PEACH_D))
        p.append(opbox(310, yy+9, 145, desc, WHITE, PEACH_D))
        p.append(arrow([(270,yy+30),(310,yy+30)], PEACH_D, "arrow"))
        if i == 0:
            p.append(arrow([(322,195),(322,265),(195,265),(195,295)], PEACH_D, "arrow"))
        else:
            prev_y = stages[i-1][3]
            p.append(arrow([(195,prev_y+60),(195,yy)], PEACH_D, "arrow"))

    # compact LMFR / FPN, with each level visible
    p.append(callout(570, 245, 540, 610, YELLOW_D, "LMFR: Mobile-FPN"))
    fpn_rows = [("C2","L2","P2","80×80",295),("C3","L3","P3","40×40",425),("C4","L4","P4","20×20",555),("C5","L5","P5","10×10",685)]
    for cname, lname, pname, size, yy in fpn_rows:
        p.append(opbox(610, yy+9, 118, f"{lname}: 1×1→192", YELLOW, YELLOW_D))
        p.append(circle_op(775, yy+30, "·" if pname == "P5" else "+", YELLOW_D, 13))
        p.append(opbox(820, yy+9, 120, "DW3 + PW1", GREEN, GREEN_D))
        p.append(tensor(975, yy, 100, 60, pname, f"{size}×192", YELLOW, YELLOW_D))
        p.append(arrow([(455,yy+30),(610,yy+30)], PEACH_D))
        p.append(arrow([(728,yy+30),(762,yy+30)], YELLOW_D))
        p.append(arrow([(788,yy+30),(820,yy+30)], YELLOW_D))
        p.append(arrow([(940,yy+30),(975,yy+30)], GREEN_D))
    # P5 has no incoming top-down addition; mark identity at the add node
    p.append(text(775, 745, "identity", "tiny", color=MUTED))
    # Top-down routes use the clear inter-row corridors.  Each route exits from
    # the right side of its tensor and enters the next add node from below, so
    # no connector crosses a tensor or a DW/PW operator box.
    top_down_routes = [
        # source centre y, corridor y, destination add-node bottom y, label y
        (715, 650, 598, 636),  # P5 -> P4
        (585, 520, 468, 506),  # P4 -> P3
        (455, 390, 338, 376),  # P3 -> P2
    ]
    for y_from, y_corridor, y_dest, y_label in top_down_routes:
        p.append(arrow([(1075,y_from),(1125,y_from),(1125,y_corridor),
                        (775,y_corridor),(775,y_dest)], BLUE_D, "arrow-blue"))
        p.append(text(1135, y_label, "Up×2", "tiny", anchor="start", color=BLUE_D))

    # edge pathway
    p.append(callout(1190, 245, 420, 275, ROSE_D, "Boundary pathway"))
    p.append(tensor(1220, 295, 105, 55, "P3", "40×40×192", YELLOW, YELLOW_D))
    p.append(opbox(1360, 302, 92, "Up×2", WHITE, BLUE_D))
    p.append(tensor(1220, 405, 105, 55, "P2", "80×80×192", YELLOW, YELLOW_D))
    p.append(circle_op(1505, 382, "C", ROSE_D, 14))
    p.append(box(1538, 338, 52, 88, "", ["3×3", "384→128"], ROSE, ROSE_D))
    p.append(arrow([(1325,322),(1360,322)], BLUE_D,"arrow-blue"))
    p.append(arrow([(1452,322),(1475,322),(1475,372),(1491,372)], ROSE_D,"arrow-rose"))
    p.append(arrow([(1325,432),(1475,432),(1475,392),(1491,392)], ROSE_D,"arrow-rose"))
    p.append(arrow([(1519,382),(1538,382)], ROSE_D,"arrow-rose"))
    p.append(box(1640, 330, 180, 104, "EdgeEnhance", ["Context | Sobel | Lap.", "Concat → ECA", "+ residual"], ROSE, ROSE_D, shadow=True))
    p.append(arrow([(1590,382),(1640,382)], ROSE_D,"arrow-rose"))
    p.append(tensor(1860, 350, 125, 64, "F_edge", "80×80×128", ROSE, ROSE_D))
    p.append(arrow([(1820,382),(1860,382)], ROSE_D,"arrow-rose"))

    # localization pathway
    p.append(callout(1190, 565, 795, 305, PURPLE_D, "Contextual localization pathway"))
    # M6 localization starts from P4 directly; P5 already contributes through FPN.
    p.append(tensor(1220, 660, 100, 55, "P4", "20×20×192", YELLOW, YELLOW_D))
    p.append(box(1530, 645, 145, 90, "SEU", ["DW/PW", "SA → ECA", "+ residual"], PURPLE, PURPLE_D))
    p.append(box(1715, 625, 210, 130, "ICG", ["P_init + RGB-CNN", "Guided Contrast", "AG-1 → AG-2"], PURPLE, PURPLE_D, shadow=True))
    p.append(arrow([(1320,687),(1530,690)], PURPLE_D,"arrow-purple"))
    p.append(arrow([(1675,690),(1715,690)], PURPLE_D,"arrow-purple"))
    p.append(box(1430, 795, 118, 46, "Resize → 40²", [], WHITE, BLUE_D))
    p.append(box(1590, 795, 118, 46, "Resize → 80²", [], WHITE, BLUE_D))
    p.append(circle_op(1727,818,"+P2",PURPLE_D,16))
    p.append(box(1750, 790, 150, 58, "Loc refine", ["3×3 → DW/PW"], PURPLE, PURPLE_D))
    p.append(arrow([(1820,755),(1820,775),(1489,775),(1489,795)], PURPLE_D,"arrow-purple"))
    p.append(arrow([(1548,818),(1590,818)], PURPLE_D,"arrow-purple"))
    p.append(arrow([(1708,818),(1711,818)], PURPLE_D,"arrow-purple"))
    p.append(arrow([(1743,818),(1750,818)], PURPLE_D,"arrow-purple"))
    p.append(arrow([(1075,325),(1140,325),(1140,865),(1727,865),(1727,834)],PURPLE_D,"arrow-purple",dash="6 5"))
    p.append(tensor(1915, 790, 115, 58, "F_loc", "80×80×128", PURPLE, PURPLE_D))
    p.append(arrow([(1900,818),(1915,818)], PURPLE_D,"arrow-purple"))

    # AFM and high-resolution refinement below, keeping overall aspect compact
    p.append(callout(1000, 930, 1030, 340, BLUE_D, "AFM-guided high-resolution decoding"))
    p.append(box(1040, 995, 205, 96, "AFM", ["A = σ(Conv(F_loc))", "Concat[A ⊙ E, L]", "1×1 → Z_afm"], BLUE, BLUE_D, shadow=True))
    # Long branch routes are deliberately kept outside the localization frame.
    # Edge enters AFM from the left; localization enters from the bottom.  This
    # avoids the former vertical strokes through ICG, F_loc and the section tag.
    p.append(arrow([(1988,382),(2070,382),(2070,900),(980,900),
                    (980,1043),(1040,1043)], ROSE_D,"arrow-rose"))
    p.append(arrow([(2032,818),(2050,818),(2050,1135),(1200,1135),
                    (1200,1091)], PURPLE_D,"arrow-purple"))
    p.append(tensor(1300, 1012, 120, 62, "Z_afm", "80×80×1", BLUE, BLUE_D))
    p.append(arrow([(1245,1043),(1300,1043)], BLUE_D,"arrow-blue"))
    p.append(box(1480, 970, 250, 145, "APG-HR", ["F_edge: 128→48, Up×4", "F_loc: 128→48, Up×4", "RGB: 3→16", "Up(Z) → p, u", "Concat 114→48"], GREEN, GREEN_D, shadow=True))
    p.append(arrow([(1420,1043),(1480,1043)], BLUE_D,"arrow-blue"))
    p.append(box(1780, 990, 185, 105, "HR refine", ["DW3 → PW1", "1×1 prediction", "320×320×1"], GREEN, GREEN_D))
    p.append(arrow([(1730,1043),(1780,1043)], GREEN_D,"arrow"))
    p.append(feature_stack(1815, 1165, 115, 72, "Mask", "320×320×1", BLUE, BLUE_D))
    p.append(arrow([(1872,1095),(1872,1165)], BLUE_D,"arrow-blue"))

    # RGB guidance uses two separate perimeter lanes.  The ICG route enters at
    # the lower-left edge of ICG; the APG-HR route enters from below the decoder.
    p.append(arrow([(132,195),(45,195),(45,770),(1800,770),
                    (1800,755)], PEACH_D,"arrow",dash="7 6"))
    p.append(text(55, 760, "RGB guidance", "tiny", anchor="start", color=PEACH_D))
    p.append(arrow([(100,195),(25,195),(25,1290),(1555,1290),
                    (1555,1115)], PEACH_D,"arrow",dash="7 6"))

    legend(p, 80, 1325)
    finish_svg(p, w, h, "1", "Overall architecture of APH-StitchNet (M6)")
    (OUT / "fig01_m6_overall_compact.svg").write_text("".join(p), encoding="utf-8")


def figure2_encoder_lmfr() -> None:
    w, h = 2350, 1600
    p = start_svg(w, h, "Encoder and LMFR", "Expanded C2–C5 stages and top-down lightweight reconstruction")

    # stage rows, with exact timm block counts
    rows = [
        ("Stem", "320²×3 → 160²×32", [("Conv 3×3 / s2",150),("BN + ReLU",120)], 115),
        ("C2", "80²×48", [("Conv 3×3 / s2\n32→128",165),("BN + ReLU",120),("Conv 1×1\n128→48",145),("BN",90)], 265),
        ("C3", "40²×80", [("UIB-1 / s2\n48→80",155),("DW3→PW192",135),("DW5/s2→PW80",155),("UIB-2\n80→80",135)], 415),
        ("C4", "20²×160", [("UIB-DS\n80→160",145),("UIB ×3",110),("MQA",90),("UIB",90),("MQA",90),("UIB",90),("MQA",90),("UIB",90),("MQA",90),("UIB",90)], 565),
        ("C5", "10²×960", [("UIB-DS\n160→256",155),("UIB ×7",110),("MQA",90),("UIB",90),("MQA",90),("UIB",90),("MQA",90),("UIB",90),("MQA",90),("UIB",90),("Conv 1×1\n256→960",145)], 715),
    ]
    for name, shape, blocks, yy in rows:
        p.append(tensor(70, yy, 135, 58, name, shape, PEACH, PEACH_D))
        x = 255
        for j, (label, bw) in enumerate(blocks):
            lines = label.split("\n")
            if len(lines) == 1:
                p.append(opbox(x, yy+8, bw, lines[0], WHITE if "MQA" not in label else PURPLE, PURPLE_D if "MQA" in label else PEACH_D))
            else:
                p.append(box(x, yy, bw, 58, lines[0], [lines[1]], WHITE, PEACH_D, 5))
            if j == 0:
                p.append(arrow([(205,yy+29),(x,yy+29)], PEACH_D))
            else:
                p.append(arrow([(x_prev+w_prev,yy+29),(x,yy+29)], PEACH_D))
            x_prev, w_prev = x, bw
            x += bw + 28
        p.append(feature_stack(min(x+10,2210), yy+4, 78, 50, name, shape, PEACH, PEACH_D))
        p.append(arrow([(x_prev+w_prev,yy+29),(min(x+10,2210),yy+29)], PEACH_D))

    # enlarged primitive blocks, like the reference figure's operator insets
    p.append(callout(80, 905, 720, 250, PEACH_D, "UIB primitive"))
    x0, yy = 120, 985
    labels = [("DW-start",110),("PW-expand",120),("DW-mid",110),("PW-project",120),("LayerScale",110)]
    for i,(lab,bw) in enumerate(labels):
        p.append(opbox(x0,yy,bw,lab,PEACH if "PW" in lab else WHITE,PEACH_D))
        if i: p.append(arrow([(prev_x+prev_w,yy+21),(x0,yy+21)],PEACH_D))
        prev_x,prev_w=x0,bw;x0+=bw+22
    p.append(line([(118,1006),(100,1006),(100,1090),(690,1090),(690,1027)],PEACH_D,1.5))
    p.append(circle_op(690,1006,"+",PEACH_D,12))
    p.append(text(440,1125,"BN/Act are attached to convolutional operators", "tiny"))

    p.append(callout(845, 905, 560, 250, PURPLE_D, "MQA primitive"))
    p.append(opbox(885,985,105,"BN",WHITE,PURPLE_D))
    p.append(opbox(1030,950,105,"Q: 1×1",PURPLE,PURPLE_D))
    p.append(opbox(1030,1005,105,"K: DW/s2",PURPLE,PURPLE_D))
    p.append(opbox(1030,1060,105,"V: DW/s2",PURPLE,PURPLE_D))
    p.append(circle_op(1195,1006,"Attn",PURPLE_D,24))
    p.append(opbox(1250,985,110,"Proj 1×1",PURPLE,PURPLE_D))
    p.append(arrow([(990,1006),(1012,1006),(1012,971),(1030,971)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(990,1006),(1030,1026)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(990,1006),(1012,1006),(1012,1081),(1030,1081)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(1135,971),(1171,993)],PURPLE_D,"arrow-purple"));p.append(arrow([(1135,1026),(1171,1006)],PURPLE_D,"arrow-purple"));p.append(arrow([(1135,1081),(1171,1019)],PURPLE_D,"arrow-purple"));p.append(arrow([(1219,1006),(1250,1006)],PURPLE_D,"arrow-purple"))

    # LMFR data flow: exact lateral projection and fixed bilinear top-down fusion
    p.append(callout(1450, 905, 820, 525, YELLOW_D, "LMFR / Mobile-FPN"))
    levels=[("C5","10²×960","P5","10²×192",975),("C4","20²×160","P4","20²×192",1080),("C3","40²×80","P3","40²×192",1185),("C2","80²×48","P2","80²×192",1290)]
    for i,(cn,cs,pn,ps,yy) in enumerate(levels):
        p.append(tensor(1485,yy,100,50,cn,cs,PEACH,PEACH_D))
        p.append(opbox(1625,yy+4,120,"1×1 → 192",YELLOW,YELLOW_D))
        p.append(circle_op(1800,yy+25,"·" if i == 0 else "+",YELLOW_D,12))
        p.append(opbox(1840,yy+4,120,"DW3 → PW1",GREEN,GREEN_D))
        p.append(tensor(2000,yy,105,50,pn,ps,YELLOW,YELLOW_D))
        p.append(arrow([(1585,yy+25),(1625,yy+25)],PEACH_D))
        p.append(arrow([(1745,yy+25),(1788,yy+25)],YELLOW_D))
        p.append(arrow([(1812,yy+25),(1840,yy+25)],YELLOW_D))
        p.append(arrow([(1960,yy+25),(2000,yy+25)],GREEN_D))
        if i < 3:
            next_y = levels[i+1][4]
            p.append(arrow([(2053,yy+50),(2145,yy+50),(2145,next_y+25),(1812,next_y+25)],BLUE_D,"arrow-blue"))
            p.append(text(2155,(yy+next_y)/2+15,"Bilinear ×2","tiny",anchor="start",color=BLUE_D))
    p.append(text(1800, 1018, "identity", "tiny", color=MUTED))
    p.append(text(1485, 1380, "Lateral: Conv1×1 → GN → ReLU", "tiny", anchor="start", color=YELLOW_D))
    p.append(text(1485, 1402, "DSR: DW3 → GN/ReLU → PW1 → GN/ReLU", "tiny", anchor="start", color=GREEN_D))

    legend(p, 80, 1490)
    finish_svg(p,w,h,"2","Expanded MobileNetV4 encoder and LMFR")
    (OUT / "fig02_encoder_lmfr_detailed.svg").write_text("".join(p), encoding="utf-8")


def figure3_dual_branch_icg() -> None:
    w, h = 2400, 1810
    p = start_svg(w,h,"Dual-branch decoder and ICG","Boundary evidence, semantic localization and two-stage image-guided gating")

    # Boundary pathway
    p.append(section_tag(65,110,"A  Boundary pathway",ROSE_D,235))
    p.append(line([(65,155),(2335,155)],ROSE_D,1.2,dash="5 6"))
    p.append(tensor(95,175,105,58,"P3","40²×192",YELLOW,YELLOW_D))
    p.append(opbox(245,183,90,"Up×2",WHITE,BLUE_D))
    p.append(tensor(95,285,105,58,"P2","80²×192",YELLOW,YELLOW_D))
    p.append(circle_op(395,260,"C",ROSE_D,14))
    p.append(box(445,215,150,88,"ConvGNReLU",["3×3", "384→128"],ROSE,ROSE_D))
    p.append(arrow([(200,204),(245,204)],BLUE_D,"arrow-blue"));p.append(arrow([(335,204),(365,204),(365,250),(381,250)],ROSE_D,"arrow-rose"));p.append(arrow([(200,314),(365,314),(365,270),(381,270)],ROSE_D,"arrow-rose"));p.append(arrow([(409,260),(445,260)],ROSE_D,"arrow-rose"))
    p.append(tensor(625,230,110,60,"E0","80²×128",ROSE,ROSE_D));p.append(arrow([(595,260),(625,260)],ROSE_D,"arrow-rose"))
    # three exact EdgeEnhance paths
    p.append(box(790,145,205,92,"Context path",["DW3 / g=128", "GN-ReLU", "PW1 →128"],WHITE,ROSE_D))
    p.append(box(790,255,205,92,"Gradient path",["Sobel Gx/Gy DW3", "256 ch → GN-ReLU", "PW1 →32"],WHITE,ROSE_D))
    p.append(box(790,365,205,92,"Laplacian path",["Laplacian DW3", "128 ch", "GN-ReLU"],WHITE,ROSE_D))
    for yy in (191,301,411): p.append(arrow([(735,260),(760,260),(760,yy),(790,yy)],ROSE_D,"arrow-rose"))
    p.append(circle_op(1055,301,"C",ROSE_D,15))
    for yy in (191,301,411): p.append(arrow([(995,yy),(1025,yy),(1025,301),(1040,301)],ROSE_D,"arrow-rose"))
    p.append(box(1110,250,160,102,"Channel mix",["1×1: 288→128", "GN-ReLU", "ECA"],ROSE,ROSE_D));p.append(arrow([(1070,301),(1110,301)],ROSE_D,"arrow-rose"))
    p.append(circle_op(1325,301,"+",ROSE_D,15));p.append(arrow([(1270,301),(1310,301)],ROSE_D,"arrow-rose"));p.append(arrow([(680,290),(680,505),(1325,505),(1325,316)],ROSE_D,"arrow-rose"))
    p.append(opbox(1380,280,105,"GN + ReLU",WHITE,ROSE_D));p.append(arrow([(1340,301),(1380,301)],ROSE_D,"arrow-rose"))
    p.append(tensor(1525,270,125,62,"F_edge","80²×128",ROSE,ROSE_D));p.append(arrow([(1485,301),(1525,301)],ROSE_D,"arrow-rose"))
    p.append(opbox(1700,280,105,"1×1 →1",WHITE,ROSE_D));p.append(arrow([(1650,301),(1700,301)],ROSE_D,"arrow-rose"));p.append(tensor(1850,270,120,62,"Z_edge","80²×1",ROSE,ROSE_D));p.append(arrow([(1805,301),(1850,301)],ROSE_D,"arrow-rose"))
    p.append(text(2170, 310, "Edge supervision", "small", color=ROSE_D))

    # Localization input and SEU.  The main lane is intentionally unframed to
    # avoid a large empty rectangle; only the expanded primitives are framed.
    p.append(section_tag(65,680,"B  Contextual localization pathway",PURPLE_D,330))
    p.append(line([(65,725),(2335,725)],PURPLE_D,1.2,dash="5 6"))
    p.append(callout(405,745,1085,245,PURPLE_D,"SEU  (spatial + channel enhancement)"))
    # P4 is the sole input of context_refine in the implemented M6 decoder.
    p.append(tensor(95,790,105,58,"P4","20²×192",YELLOW,YELLOW_D));p.append(arrow([(200,820),(445,820)],PURPLE_D,"arrow-purple"))
    # SEU exact chain
    p.append(opbox(445,799,105,"DW3",PURPLE,PURPLE_D));p.append(opbox(575,799,105,"GN-ReLU",WHITE,PURPLE_D));p.append(opbox(705,799,105,"PW1",PURPLE,PURPLE_D));p.append(opbox(835,779,135,"Spatial Attn",PURPLE,PURPLE_D,82));p.append(opbox(995,779,120,"ECA",PURPLE,PURPLE_D,82));p.append(opbox(1140,799,105,"DW3→PW1",PURPLE,PURPLE_D));p.append(circle_op(1295,820,"+",PURPLE_D,14));p.append(tensor(1345,790,120,60,"F_seu","20²×192",PURPLE,PURPLE_D))
    xs=[(550,575),(680,705),(810,835),(970,995),(1115,1140),(1245,1281),(1309,1345)]
    for a,b in xs:p.append(arrow([(a,820),(b,820)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(200,848),(225,848),(225,950),(1295,950),(1295,834)],PURPLE_D,"arrow-purple"))
    p.append(text(902, 875, "Avg/Max → Conv7×7 → σ → ⊙", "tiny", color=PURPLE_D));p.append(text(1055, 895, "GAP → Conv1D → σ → ⊙", "tiny", color=PURPLE_D))

    # ICG: initial mask and RGB appearance encoder
    p.append(callout(1500,725,790,770,PURPLE_D,"ICG  (RGB contrast + iterative AG)"))
    p.append(tensor(1530,785,115,58,"F_seu","20²×192",PURPLE,PURPLE_D));p.append(arrow([(1465,820),(1530,814)],PURPLE_D,"arrow-purple"))
    p.append(opbox(1690,793,110,"1×1 →1",WHITE,PURPLE_D));p.append(opbox(1835,793,90,"Sigmoid",WHITE,PURPLE_D));p.append(tensor(1965,785,110,58,"P_init","20²×1",PURPLE,PURPLE_D));p.append(arrow([(1645,814),(1690,814)],PURPLE_D,"arrow-purple"));p.append(arrow([(1800,814),(1835,814)],PURPLE_D,"arrow-purple"));p.append(arrow([(1925,814),(1965,814)],PURPLE_D,"arrow-purple"))
    p.append(tensor(1530,905,115,58,"RGB↓","20²×3",PEACH,PEACH_D));p.append(opbox(1690,885,115,"Conv3×3\n3→16",PEACH,PEACH_D,82));p.append(opbox(1840,885,115,"Conv3×3\n16→32",PEACH,PEACH_D,82));p.append(tensor(1990,905,115,58,"F_app","20²×32",PEACH,PEACH_D));p.append(arrow([(1645,934),(1690,926)],PEACH_D));p.append(arrow([(1805,926),(1840,926)],PEACH_D));p.append(arrow([(1955,926),(1990,934)],PEACH_D))
    # Guided contrast compact expansion
    p.append(box(1535,1035,150,92,"FG local",["F_app ⊙ P_init", "DW5 → PW1"],WHITE,PURPLE_D));p.append(box(1720,1035,150,92,"BG local",["F_app ⊙ (1-P)", "DW5 → PW1"],WHITE,PURPLE_D));p.append(box(1905,1035,150,92,"BG global",["P_init < 0.1", "masked GAP"],WHITE,PURPLE_D))
    p.append(arrow([(2047,963),(2047,1000),(1610,1000),(1610,1035)],PEACH_D));p.append(arrow([(2047,963),(2047,1015),(1795,1015),(1795,1035)],PEACH_D));p.append(arrow([(2047,963),(2047,1035)],PEACH_D));p.append(arrow([(2020,843),(2020,1018),(1610,1018),(1610,1035)],PURPLE_D,"arrow-purple",dash="6 5"));p.append(arrow([(2020,843),(2020,1025),(1795,1025),(1795,1035)],PURPLE_D,"arrow-purple",dash="6 5"))
    p.append(circle_op(1740,1195,"C",PURPLE_D,14));p.append(arrow([(1795,1127),(1795,1160),(1754,1185)],PURPLE_D,"arrow-purple"));p.append(arrow([(1980,1127),(1980,1170),(1754,1205)],PURPLE_D,"arrow-purple"));p.append(opbox(1800,1174,120,"BG fuse 1×1",PURPLE,PURPLE_D));p.append(arrow([(1754,1195),(1800,1195)],PURPLE_D,"arrow-purple"));p.append(circle_op(1985,1215,"|−|",PURPLE_D,16));p.append(arrow([(1920,1195),(1969,1210)],PURPLE_D,"arrow-purple"));p.append(arrow([(1610,1127),(1610,1215),(1969,1215)],PURPLE_D,"arrow-purple"));p.append(opbox(2040,1194,120,"1×1 →16",PURPLE,PURPLE_D));p.append(arrow([(2001,1215),(2040,1215)],PURPLE_D,"arrow-purple"));p.append(tensor(2190,1186,80,58,"G","20²×16",PURPLE,PURPLE_D));p.append(arrow([(2160,1215),(2190,1215)],PURPLE_D,"arrow-purple"))
    # two attention gates
    p.append(box(1540,1310,235,108,"Attention Gate 1",["Wx: 192→96", "Wg: 16→96", "Add→GN-ReLU→1×1→σ", "α1 ⊙ F_seu"],PURPLE,PURPLE_D));p.append(box(1820,1310,235,108,"Attention Gate 2",["Wx: 192→96", "Wg: 16→96", "reuse G", "α2 ⊙ F1"],PURPLE,PURPLE_D));p.append(tensor(2110,1334,125,60,"F_icg","20²×192",PURPLE,PURPLE_D))
    p.append(arrow([(1590,843),(1505,843),(1505,1364),(1540,1364)],PURPLE_D,"arrow-purple"));p.append(arrow([(2230,1215),(2260,1215),(2260,1280),(1660,1280),(1660,1310)],PURPLE_D,"arrow-purple"));p.append(arrow([(1775,1364),(1820,1364)],PURPLE_D,"arrow-purple"));p.append(arrow([(2260,1280),(1938,1280),(1938,1310)],PURPLE_D,"arrow-purple"));p.append(arrow([(2055,1364),(2110,1364)],PURPLE_D,"arrow-purple"))
    # return to P2 outside ICG, routed below without crossings
    p.append(opbox(145,1515,120,"Resize → 40²",WHITE,BLUE_D));p.append(tensor(300,1508,95,56,"F_40","40²×192",PURPLE,PURPLE_D));p.append(opbox(435,1515,120,"Resize → 80²",WHITE,BLUE_D));p.append(circle_op(625,1536,"+",PURPLE_D,14));p.append(tensor(585,1600,95,52,"P2","80²×192",YELLOW,YELLOW_D));p.append(box(705,1495,165,82,"Loc refine",["Conv3×3: 192→128", "DW3 → PW1"],PURPLE,PURPLE_D));p.append(tensor(915,1505,130,62,"F_loc","80²×128",PURPLE,PURPLE_D));p.append(opbox(1090,1515,105,"1×1 →1",WHITE,PURPLE_D));p.append(tensor(1240,1505,120,62,"Z_loc","80²×1",PURPLE,PURPLE_D))
    p.append(arrow([(2172,1394),(2172,1465),(145,1465),(145,1515)],PURPLE_D,"arrow-purple"));p.append(arrow([(265,1536),(300,1536)],PURPLE_D,"arrow-purple"));p.append(arrow([(395,1536),(435,1536)],PURPLE_D,"arrow-purple"));p.append(arrow([(555,1536),(611,1536)],PURPLE_D,"arrow-purple"));p.append(arrow([(632,1600),(632,1550)],PURPLE_D,"arrow-purple"));p.append(arrow([(639,1536),(705,1536)],PURPLE_D,"arrow-purple"));p.append(arrow([(870,1536),(915,1536)],PURPLE_D,"arrow-purple"));p.append(arrow([(1045,1536),(1090,1536)],PURPLE_D,"arrow-purple"));p.append(arrow([(1195,1536),(1240,1536)],PURPLE_D,"arrow-purple"))

    legend(p, 1450, 1700)
    finish_svg(p,w,h,"3","Boundary and localization pathways with the complete ICG module")
    (OUT / "fig03_dual_branch_icg_detailed.svg").write_text("".join(p), encoding="utf-8")


def figure4_afm_apghr() -> None:
    w, h = 2250, 1300
    p = start_svg(w,h,"AFM-guided High-resolution Refinement","Exact attentive fusion, probability–ambiguity prior and pixel-scale reconstruction")

    # AFM lane
    p.append(callout(60,115,1010,870,BLUE_D,"A  AFM (80×80)"))
    p.append(tensor(95,205,130,62,"F_edge","80²×128",ROSE,ROSE_D));p.append(opbox(285,215,145,"1×1+GN/ReLU",ROSE,ROSE_D));p.append(tensor(455,205,120,62,"E","80²×128",ROSE,ROSE_D));p.append(arrow([(225,236),(285,236)],ROSE_D,"arrow-rose"));p.append(arrow([(430,236),(455,236)],ROSE_D,"arrow-rose"))
    p.append(tensor(95,390,130,62,"F_loc","80²×128",PURPLE,PURPLE_D));p.append(opbox(285,400,145,"1×1+GN/ReLU",PURPLE,PURPLE_D));p.append(tensor(455,390,120,62,"L","80²×128",PURPLE,PURPLE_D));p.append(arrow([(225,421),(285,421)],PURPLE_D,"arrow-purple"));p.append(arrow([(430,421),(455,421)],PURPLE_D,"arrow-purple"))
    p.append(opbox(635,365,130,"Conv3+GN\n128→64",PURPLE,PURPLE_D,72));p.append(opbox(810,380,100,"Conv1×1\n64→1",PURPLE,PURPLE_D,72));p.append(opbox(945,395,80,"Sigmoid",WHITE,PURPLE_D));p.append(arrow([(575,421),(635,401)],PURPLE_D,"arrow-purple"));p.append(arrow([(765,401),(810,416)],PURPLE_D,"arrow-purple"));p.append(arrow([(910,416),(945,416)],PURPLE_D,"arrow-purple"))
    # Keep the edge gate on the E row.  The previous drawing sent E vertically
    # through the L tensor and then shared part of L's direct-to-concat lane.
    # Here E enters the product from the left, while A reaches it through a
    # separate upper return lane; the two signals never cross a feature box.
    p.append(circle_op(650,236,"⊙",PURPLE_D,16))
    p.append(arrow([(575,236),(634,236)],ROSE_D,"arrow-rose"))
    p.append(arrow([(985,395),(985,300),(650,300),(650,252)],PURPLE_D,"arrow-purple"))
    p.append(text(975,290,"A", "tiny", anchor="end", color=PURPLE_D))
    p.append(tensor(720,205,130,62,"E_g","80²×128",ROSE,ROSE_D))
    p.append(arrow([(666,236),(720,236)],ROSE_D,"arrow-rose"))
    p.append(circle_op(690,680,"C",BLUE_D,16));p.append(arrow([(850,236),(1060,236),(1060,640),(700,664)],ROSE_D,"arrow-rose"));p.append(arrow([(515,452),(515,680),(674,680)],PURPLE_D,"arrow-purple"));p.append(opbox(765,659,155,"Conv3×3\n256→128",BLUE,BLUE_D,72));p.append(arrow([(706,680),(765,695)],BLUE_D,"arrow-blue"));p.append(opbox(945,674,85,"Conv1×1",BLUE,BLUE_D));p.append(arrow([(920,695),(945,695)],BLUE_D,"arrow-blue"));p.append(tensor(790,790,145,66,"Z_afm","80²×1",BLUE,BLUE_D));p.append(arrow([(987,716),(987,760),(862,760),(862,790)],BLUE_D,"arrow-blue"))

    # APG-HR lane
    p.append(callout(1140,115,1050,1030,GREEN_D,"B  APG-HR (320×320)"))
    inputs=[("F_edge","128→48",ROSE,ROSE_D,190),("F_loc","128→48",PURPLE,PURPLE_D,330),("RGB","3→16",PEACH,PEACH_D,470)]
    for name,chs,fill,stroke,yy in inputs:
        p.append(tensor(1175,yy,115,58,name,"80²" if name!="RGB" else "320²",fill,stroke));p.append(opbox(1330,yy+8,110,chs,fill,stroke));
        if name!="RGB": p.append(opbox(1475,yy+8,95,"Up×4",WHITE,BLUE_D))
        else: p.append(opbox(1475,yy+8,120,"DW3→PW1",WHITE,PEACH_D))
        p.append(arrow([(1290,yy+29),(1330,yy+29)],stroke,"arrow-rose" if stroke==ROSE_D else "arrow-purple" if stroke==PURPLE_D else "arrow"));p.append(arrow([(1440,yy+29),(1475,yy+29)],stroke,"arrow-rose" if stroke==ROSE_D else "arrow-purple" if stroke==PURPLE_D else "arrow"))
    p.append(tensor(1175,620,115,58,"Z_afm","80²×1",BLUE,BLUE_D));p.append(opbox(1330,628,90,"Up×4",WHITE,BLUE_D));p.append(opbox(1455,600,95,"Sigmoid\np",PURPLE,PURPLE_D,72));p.append(opbox(1580,600,115,"4p(1−p)\nu",YELLOW,YELLOW_D,72));p.append(arrow([(1290,649),(1330,649)],BLUE_D,"arrow-blue"));p.append(arrow([(1420,649),(1455,636)],BLUE_D,"arrow-blue"));p.append(arrow([(1550,636),(1580,636)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(935,823),(1100,823),(1100,649),(1175,649)],BLUE_D,"arrow-blue"))
    p.append(circle_op(1755,505,"C",GREEN_D,17))
    # separate buses approach concat from the left and do not cross boxes
    p.append(arrow([(1570,219),(1710,219),(1710,485),(1738,495)],ROSE_D,"arrow-rose"));p.append(arrow([(1570,359),(1685,359),(1685,495),(1738,503)],PURPLE_D,"arrow-purple"));p.append(arrow([(1595,499),(1660,499),(1738,507)],PEACH_D));p.append(arrow([(1503,672),(1503,735),(1720,735),(1720,515),(1738,511)],PURPLE_D,"arrow-purple"));p.append(arrow([(1638,672),(1638,760),(1740,760),(1740,522)],YELLOW_D))
    p.append(box(1815,430,190,150,"Prior fusion",["Concat: 48+48+16+1+1", "114 channels", "Conv1×1: 114→48", "GN + ReLU"],GREEN,GREEN_D));p.append(arrow([(1772,505),(1815,505)],GREEN_D))
    p.append(box(1815,650,190,115,"HR refine",["DW3 / groups=48", "GN + ReLU", "PW1: 48→48", "GN + ReLU"],GREEN,GREEN_D));p.append(arrow([(1910,580),(1910,650)],GREEN_D))
    p.append(tensor(1845,830,130,62,"F_hr","320²×48",GREEN,GREEN_D));p.append(arrow([(1910,765),(1910,830)],GREEN_D))
    p.append(opbox(1680,950,110,"Final 1×1",BLUE,BLUE_D));p.append(opbox(1845,950,110,"Edge 1×1",ROSE,ROSE_D));p.append(opbox(2010,950,130,"Centerline 1×1",PURPLE,PURPLE_D));p.append(arrow([(1910,892),(1910,920),(1735,920),(1735,950)],BLUE_D,"arrow-blue"));p.append(arrow([(1910,892),(1910,950)],ROSE_D,"arrow-rose"));p.append(arrow([(1910,892),(1910,920),(2075,920),(2075,950)],PURPLE_D,"arrow-purple"))
    p.append(feature_stack(1675,1055,120,66,"Mask","320²×1",BLUE,BLUE_D));p.append(arrow([(1735,992),(1735,1055)],BLUE_D,"arrow-blue"));p.append(feature_stack(1840,1055,120,66,"Edge","320²×1",ROSE,ROSE_D));p.append(arrow([(1900,992),(1900,1055)],ROSE_D,"arrow-rose"));p.append(feature_stack(2020,1055,120,66,"Center","interface",PURPLE,PURPLE_D));p.append(arrow([(2075,992),(2075,1055)],PURPLE_D,"arrow-purple",dash="6 5"))

    p.append(text(420,1045,"A is generated only from localization features; E_g = A ⊙ E", "small", anchor="start", color=BLUE_D))
    p.append(text(420,1070,"M6 uses one AFM refine block (256→128) before Z_afm.", "tiny", anchor="start"))
    p.append(text(80,1180,"Solid: active prediction route", "tiny", anchor="start"));p.append(text(80,1200,"Dashed: retained auxiliary interface", "tiny", anchor="start"))
    legend(p, 1160, 1190)
    finish_svg(p,w,h,"4","AFM-guided probability–ambiguity high-resolution refinement")
    (OUT / "fig04_afm_apghr_detailed.svg").write_text("".join(p), encoding="utf-8")


def figure4a_afm_only() -> None:
    """AFM subfigure shared by M0/M4/M6 before high-resolution decoding."""
    w, h = 1420, 920
    p = start_svg(
        w, h, "Attentive Fusion Module (AFM)",
        "Localization-guided boundary fusion and coarse mask prediction"
    )

    p.append(tensor(70,145,135,62,"F_edge","80²×128",ROSE,ROSE_D))
    p.append(opbox(255,155,150,"Conv1×1+GN/ReLU",ROSE,ROSE_D))
    p.append(tensor(445,145,125,62,"E","80²×128",ROSE,ROSE_D))
    p.append(arrow([(205,176),(255,176)],ROSE_D,"arrow-rose"))
    p.append(arrow([(405,176),(445,176)],ROSE_D,"arrow-rose"))

    p.append(tensor(70,335,135,62,"F_loc","80²×128",PURPLE,PURPLE_D))
    p.append(opbox(255,345,150,"Conv1×1+GN/ReLU",PURPLE,PURPLE_D))
    p.append(tensor(445,335,125,62,"L","80²×128",PURPLE,PURPLE_D))
    p.append(arrow([(205,366),(255,366)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(405,366),(445,366)],PURPLE_D,"arrow-purple"))

    p.append(opbox(640,305,140,"Conv3×3+GN/ReLU\n128→64",PURPLE,PURPLE_D,72))
    p.append(opbox(825,320,105,"Conv1×1\n64→1",PURPLE,PURPLE_D,72))
    p.append(opbox(970,335,85,"Sigmoid",WHITE,PURPLE_D))
    p.append(tensor(1100,325,120,62,"A","80²×1",PURPLE,PURPLE_D))
    p.append(arrow([(570,366),(640,341)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(780,341),(825,356)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(930,356),(970,356)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(1055,356),(1100,356)],PURPLE_D,"arrow-purple"))

    # Explicit edge-gating lane: E stays on the upper row and enters ⊙ from
    # the left.  A returns through the empty corridor between the two rows.
    # This removes the former E line through L and the overlapping E/L segment.
    p.append(circle_op(665,176,"⊙",PURPLE_D,17))
    p.append(arrow([(570,176),(648,176)],ROSE_D,"arrow-rose"))
    p.append(arrow([(1160,325),(1160,250),(665,250),(665,193)],PURPLE_D,"arrow-purple"))
    p.append(tensor(735,145,135,62,"E_g","80²×128",ROSE,ROSE_D))
    p.append(arrow([(682,176),(735,176)],ROSE_D,"arrow-rose"))

    p.append(circle_op(705,640,"C",BLUE_D,17))
    p.append(arrow([(870,176),(1325,176),(1325,600),(717,600),(717,623)],ROSE_D,"arrow-rose"))
    p.append(arrow([(508,397),(508,640),(688,640)],PURPLE_D,"arrow-purple"))
    p.append(opbox(790,619,170,"Conv3×3+GN/ReLU\n256→128",BLUE,BLUE_D,72))
    p.append(opbox(1010,634,95,"Conv1×1",BLUE,BLUE_D))
    p.append(tensor(1155,605,145,66,"Z_afm","80²×1",BLUE,BLUE_D))
    p.append(arrow([(722,640),(790,655)],BLUE_D,"arrow-blue"))
    p.append(arrow([(960,655),(1010,655)],BLUE_D,"arrow-blue"))
    p.append(arrow([(1105,655),(1155,638)],BLUE_D,"arrow-blue"))

    p.append(text(70,760,"A = sigmoid(Conv1×1(Conv3×3(L)));  E_g = A ⊙ E", "small", anchor="start", color=BLUE_D))
    p.append(text(70,785,"C: channel concatenation.  AFM refine blocks: 1.", "tiny", anchor="start"))
    finish_svg(p,w,h,"4(a)","Localization-guided attentive fusion module")
    (OUT / "fig04a_afm_fusion_detailed.svg").write_text("".join(p), encoding="utf-8")


def figure4b_m4_highres_only() -> None:
    """Exact M4 HighResStitchRefineHead without the M6 AFM prior inputs."""
    w, h = 1780, 1060
    p = start_svg(
        w, h, "HighResStitchRefineHead (M4)",
        "Pixel-scale reconstruction from boundary, localization and RGB detail features"
    )

    # Three input streams. M4 has no Z_afm, probability or uncertainty input.
    streams = [
        ("F_edge", "80²×128", "Conv1×1", "128→48", ROSE, ROSE_D, 155),
        ("F_loc",  "80²×128", "Conv1×1", "128→48", PURPLE, PURPLE_D, 315),
        ("RGB",    "320²×3",  "Conv3×3", "3→16", PEACH, PEACH_D, 475),
    ]
    for name, shape, op, ch, fill, stroke, yy in streams:
        p.append(tensor(75,yy,135,62,name,shape,fill,stroke))
        p.append(opbox(265,yy+10,115,op,fill,stroke))
        p.append(opbox(425,yy+10,105,ch,fill,stroke))
        marker = "arrow-rose" if stroke == ROSE_D else "arrow-purple" if stroke == PURPLE_D else "arrow"
        p.append(arrow([(210,yy+31),(265,yy+31)],stroke,marker))
        p.append(arrow([(380,yy+31),(425,yy+31)],stroke,marker))

    p.append(opbox(585,165,115,"GN + ReLU",WHITE,ROSE_D))
    p.append(opbox(750,165,105,"Bilinear\nUp×4",WHITE,BLUE_D,62))
    p.append(tensor(905,155,130,62,"E_hr","320²×48",ROSE,ROSE_D))
    p.append(arrow([(530,186),(585,186)],ROSE_D,"arrow-rose"))
    p.append(arrow([(700,186),(750,196)],ROSE_D,"arrow-rose"))
    p.append(arrow([(855,196),(905,186)],ROSE_D,"arrow-rose"))

    p.append(opbox(585,325,115,"GN + ReLU",WHITE,PURPLE_D))
    p.append(opbox(750,325,105,"Bilinear\nUp×4",WHITE,BLUE_D,62))
    p.append(tensor(905,315,130,62,"L_hr","320²×48",PURPLE,PURPLE_D))
    p.append(arrow([(530,346),(585,346)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(700,346),(750,356)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(855,356),(905,346)],PURPLE_D,"arrow-purple"))

    p.append(opbox(585,485,115,"GN + ReLU",WHITE,PEACH_D))
    p.append(opbox(750,465,120,"DW3+GN/ReLU",PEACH,PEACH_D,42))
    p.append(opbox(750,520,120,"PW1+GN/ReLU",PEACH,PEACH_D,42))
    p.append(tensor(920,475,130,62,"I_rgb","320²×16",PEACH,PEACH_D))
    p.append(arrow([(530,506),(585,506)],PEACH_D))
    p.append(arrow([(700,506),(730,506),(730,486),(750,486)],PEACH_D))
    p.append(arrow([(810,507),(810,520)],PEACH_D))
    p.append(arrow([(870,541),(895,541),(895,506),(920,506)],PEACH_D))

    # Dedicated non-crossing buses into channel concatenation.
    p.append(circle_op(1125,350,"C",GREEN_D,18))
    p.append(arrow([(1035,186),(1080,186),(1080,335),(1107,340)],ROSE_D,"arrow-rose"))
    p.append(arrow([(1035,346),(1107,350)],PURPLE_D,"arrow-purple"))
    p.append(arrow([(1050,506),(1080,506),(1080,365),(1107,360)],PEACH_D))

    p.append(box(1190,280,210,140,"HR fusion",[
        "Concat: 48+48+16", "112 channels", "Conv1×1: 112→48", "GN + ReLU"
    ],GREEN,GREEN_D,shadow=True))
    p.append(arrow([(1143,350),(1190,350)],GREEN_D))

    p.append(box(1190,500,210,125,"DS refine ×1",[
        "DW3 / groups=48", "GN + ReLU", "PW1: 48→48", "GN + ReLU"
    ],GREEN,GREEN_D))
    p.append(arrow([(1295,420),(1295,500)],GREEN_D))
    p.append(tensor(1230,690,140,66,"F_hr","320²×48",GREEN,GREEN_D))
    p.append(arrow([(1295,625),(1295,690)],GREEN_D))

    p.append(opbox(990,810,115,"Final 1×1",BLUE,BLUE_D))
    p.append(opbox(1225,810,115,"Edge 1×1",ROSE,ROSE_D))
    p.append(opbox(1450,810,145,"Centerline 1×1",PURPLE,PURPLE_D))
    p.append(arrow([(1295,756),(1295,780),(1048,780),(1048,810)],BLUE_D,"arrow-blue"))
    p.append(arrow([(1295,756),(1295,810)],ROSE_D,"arrow-rose"))
    p.append(arrow([(1295,756),(1295,780),(1522,780),(1522,810)],PURPLE_D,"arrow-purple"))
    p.append(feature_stack(988,900,120,62,"Mask","320²×1",BLUE,BLUE_D))
    p.append(feature_stack(1235,900,120,62,"Edge","320²×1",ROSE,ROSE_D))
    p.append(feature_stack(1462,900,120,62,"Center","interface",PURPLE,PURPLE_D))
    p.append(arrow([(1048,852),(1048,900)],BLUE_D,"arrow-blue"))
    p.append(arrow([(1282,852),(1295,900)],ROSE_D,"arrow-rose"))
    p.append(arrow([(1522,852),(1522,900)],PURPLE_D,"arrow-purple",dash="6 5"))

    p.append(text(75,760,"M4 inputs: F_edge + F_loc + RGB detail only", "small", anchor="start", color=GREEN_D))
    p.append(text(75,785,"No AFM probability/uncertainty prior; output_stride=1; refine blocks=1.", "tiny", anchor="start"))
    finish_svg(p,w,h,"4(b)","High-resolution stitch refinement head introduced in M4")
    (OUT / "fig04b_m4_highres_stitch_refine_head.svg").write_text("".join(p), encoding="utf-8")


if __name__ == "__main__":
    figure1_overall()
    figure2_encoder_lmfr()
    figure3_dual_branch_icg()
    figure4_afm_apghr()
    figure4a_afm_only()
    figure4b_m4_highres_only()
    print(OUT)

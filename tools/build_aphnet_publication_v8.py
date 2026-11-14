"""APHNet/M6 publication schematic v8.

Editorial layout: a compact main graph in the upper-left and four clearly
separated operator panels around it.  The main graph is intentionally terse;
all implementation detail is delegated to anchored panels (b)-(e).
"""
from pathlib import Path
import json
import re
import cairosvg
from PIL import Image

import build_aphnet_compact_v6 as v6
from build_aphnet_publication_v4 import Scene, svg


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_publication_v8"
OUT.mkdir(parents=True, exist_ok=True)

FILL = {
    "enc": "#F2BE9B", "fpn": "#EDCB52", "edge": "#E893AA",
    "loc": "#B69BD5", "prior": "#87B7D6", "ref": "#91C889",
    "op": "#E4EBF0", "white": "#FFFFFF",
}
STROKE = {
    "enc": "#A95831", "fpn": "#8D7100", "edge": "#9F3653",
    "loc": "#60408D", "prior": "#28688F", "ref": "#36733E",
    "op": "#405565", "ink": "#202B38",
}
PANEL = {
    "fpn": "#FFF9D9", "edge": "#FCE8EE", "loc": "#F1EBFA",
    "ref": "#EAF6E8", "main": "#FCFCFC",
}


def node(s, key, x, y, w, text, family="op", h=50, fs=21, kind="box"):
    return s.node(key, x, y, w, h, text, FILL[family], kind=kind, fs=fs,
                  stroke=STROKE[family])


def tensor(s, key, x, y, w, text, family, h=56, fs=20):
    return node(s, key, x, y, w, text, family, h, fs, "tensor")


def panel(s, title, family="main"):
    stroke = STROKE.get(family, "#71808B")
    s.node("panel_frame", 6, 6, s.w - 12, s.h - 12, "", PANEL[family],
           kind="frame", stroke=stroke, dash=True)
    s.label(22, 13, title, 23, s.w - 44, color=STROKE["ink"])


def add(s, key, x, y):
    return s.node(key, x - 17, y - 17, 34, 34, "+", FILL["white"],
                  kind="circle", fs=22, stroke=STROKE["op"])


def mul(s, key, x, y):
    return s.node(key, x - 17, y - 17, 34, 34, "×", FILL["white"],
                  kind="circle", fs=21, stroke=STROKE["op"])


def cat(s, key, x, y, w=50):
    return node(s, key, x - w / 2, y - 20, w, "CAT", "prior", 40, 17)


def main_graph():
    s = Scene("01_main_network", 1500, 880)
    panel(s, "(a) APHNet/M6: end-to-end stitch segmentation", "main")
    s.label(24, 55, "Encoder", 22, 180)
    s.label(292, 55, "LMFR pyramid", 22, 260)
    s.label(808, 55, "Dual-path decoder", 22, 300)
    s.label(1250, 55, "Output", 22, 180)

    # Encoder and pyramid are aligned by scale.
    tensor(s, "rgb", 24, 92, 92, "RGB\n320²×3", "enc", 56, 20)
    node(s, "stem", 144, 95, 88, "Stem", "enc", 50, 21)
    s.link("rgb", "stem", color=STROKE["enc"])
    rows = [
        ("2", 175, "80²×48"),
        ("3", 295, "40²×80"),
        ("4", 415, "20²×160"),
        ("5", 535, "10²×960"),
    ]
    prev = "stem"
    for n, y, dims in rows:
        tensor(s, "c" + n, 140, y, 112, f"C{n}\n{dims}", "enc", 60, 22)
        s.link(prev, "c" + n, "b", "t", color=STROKE["enc"])
        prev = "c" + n

        node(s, "l" + n, 292, y + 3, 72, f"L{n}", "fpn", 52, 21)
        s.link("c" + n, "l" + n)
        if n != "5":
            add(s, "add" + n, 411, y + 29)
            s.link("l" + n, "add" + n)
            node(s, "dsr" + n, 451, y + 3, 90, "DSR (b)", "ref", 52, 19)
            s.link("add" + n, "dsr" + n)
        else:
            node(s, "dsr" + n, 402, y + 3, 90, "DSR (b)", "ref", 52, 19)
            s.link("l" + n, "dsr" + n)
        tensor(s, "p" + n, 578, y, 106, f"P{n}\n{dims.split('×')[0]}²×192", "fpn", 60, 22)
        s.link("dsr" + n, "p" + n)

    # One dedicated top-down rail; no branch line is allowed into this corridor.
    rail = 724
    for src, dst in [("p5", "add4"), ("p4", "add3"), ("p3", "add2")]:
        x0, y0 = s.point(src, "r")
        x1, y1 = s.point(dst, "b")
        s.route([(x0, y0), (rail, y0), (rail, y1 + 24), (x1, y1 + 24), (x1, y1)],
                color=STROKE["prior"], start=(src, "r"), end=(dst, "b"))

    # Edge lane. P3 and P2 enter CAT from separate, parallel orthogonal routes.
    node(s, "p3up", 804, 162, 90, "P3 ↑2", "edge", 50, 20)
    cat(s, "ecat", 940, 188, 52)
    node(s, "econv", 987, 162, 74, "Conv", "edge", 50, 20)
    node(s, "eeb", 1098, 162, 90, "EEB", "edge", 50, 21)
    tensor(s, "fe", 1218, 157, 112, "F_edge\n80²×128", "edge", 60, 21)
    s.route([s.point("p3", "r"), (760, 324), (760, 187), s.point("p3up", "l")],
            color=STROKE["edge"], start=("p3", "r"), end=("p3up", "l"))
    s.link("p3up", "ecat", color=STROKE["edge"])
    s.route([s.point("p2", "r"), (744, 204), (914, 204), s.point("ecat", "b")],
            color=STROKE["edge"], start=("p2", "r"), end=("ecat", "b"))
    for a, b in [("ecat", "econv"), ("econv", "eeb"), ("eeb", "fe")]:
        s.link(a, b, color=STROKE["edge"])

    # Localization lane. P2 uses its own vertical rail and enters only at the add node.
    node(s, "seu", 804, 416, 76, "SEU", "loc", 50, 20)
    node(s, "icg", 916, 416, 96, "ICG*", "loc", 50, 21)
    node(s, "up4", 1040, 416, 76, "↑4", "loc", 50, 21)
    add(s, "ladd", 1160, 441)
    node(s, "lref", 1200, 416, 88, "LocRef", "loc", 50, 19)
    tensor(s, "fl", 1320, 411, 112, "F_loc\n80²×128", "loc", 60, 21)
    s.route([s.point("p4", "r"), (780, 444), s.point("seu", "l")],
            color=STROKE["loc"], start=("p4", "r"), end=("seu", "l"))
    for a, b in [("seu", "icg"), ("icg", "up4"), ("up4", "ladd"), ("ladd", "lref"), ("lref", "fl")]:
        s.link(a, b, color=STROKE["loc"])
    s.route([s.point("p2", "r"), (746, 204), (746, 500), (1160, 500), s.point("ladd", "b")],
            color=STROKE["loc"], start=("p2", "r"), end=("ladd", "b"))

    # Fusion is a vertical terminal stack, avoiding red/purple line crossings.
    node(s, "afm", 1244, 625, 104, "AFM", "prior", 54, 22)
    tensor(s, "z", 1260, 704, 78, "Z_afm", "prior", 50, 19)
    node(s, "apg", 1232, 778, 128, "APG-HR*", "ref", 54, 21)
    tensor(s, "mask", 1392, 777, 78, "Mask\n320²", "prior", 56, 18)
    s.route([s.point("fe", "r"), (1370, 187), (1370, 642), s.point("afm", "r")],
            color=STROKE["edge"], start=("fe", "r"), end=("afm", "r"))
    s.route([s.point("fl", "b"), (1373, 570), (1296, 570), s.point("afm", "t")],
            color=STROKE["loc"], start=("fl", "b"), end=("afm", "t"))
    s.link("afm", "z", "b", "t", color=STROKE["prior"])
    s.link("z", "apg", "b", "t", color=STROKE["prior"])
    s.link("apg", "mask", color=STROKE["ref"])

    # Centralized notation and abbreviation strip.
    s.label(24, 675, "Module key", 21, 180)
    keys = [("enc", "Encoder"), ("fpn", "Pyramid"), ("edge", "Edge"),
            ("loc", "Localization"), ("prior", "Prior"), ("ref", "Refine")]
    x = 24
    for i, (fam, label) in enumerate(keys):
        node(s, f"key{i}", x, 712, 24, "", fam, 18, 16)
        s.label(x + 31, 707, label, 18, 110)
        x += 118
    s.label(24, 756, "+ add   CAT concatenate   ↑ bilinear upsample   * receives the same RGB input", 20, 820)
    return s


def lmfr_panel():
    s = Scene("02_lmfr", 820, 300)
    panel(s, "(b) Lightweight multi-scale feature reconstruction (LMFR)", "fpn")
    tensor(s, "coarse", 34, 88, 100, "P(i+1)", "fpn", 50, 19)
    node(s, "up", 172, 88, 82, "↑2", "op", 50, 21)
    tensor(s, "lat", 34, 180, 100, "L_i", "fpn", 50, 20)
    add(s, "sum", 312, 158)
    node(s, "dw", 370, 133, 112, "DW 3×3", "ref", 50, 20)
    node(s, "pw", 520, 133, 112, "PW 1×1", "ref", 50, 20)
    tensor(s, "out", 678, 133, 92, "P_i", "fpn", 50, 20)
    s.link("coarse", "up"); s.link("up", "sum", via=[(282, 113), (282, 141)], sb="t")
    s.link("lat", "sum", via=[(282, 205), (282, 175)], sb="b")
    for a, b in [("sum", "dw"), ("dw", "pw"), ("pw", "out")]:
        s.link(a, b)
    return s


def edge_panel():
    s = Scene("03_edge_enhance", 820, 500)
    panel(s, "(c) Edge enhancement block (EEB)", "edge")
    tensor(s, "x", 26, 226, 62, "X", "edge", 54, 21)
    lanes = [
        (86, "DW 3×3", "PW 1×1", "128"),
        (206, "Sobel X/Y", "PW 1×1", "32"),
        (326, "Laplacian", "Identity", "128"),
    ]
    for i, (y, a, b, ch) in enumerate(lanes):
        node(s, f"a{i}", 130, y, 132, a, "edge", 52, 20)
        node(s, f"b{i}", 300, y, 118, b, "edge", 52, 20)
        tensor(s, f"t{i}", 454, y, 72, ch + " ch", "edge", 52, 18)
        s.route([s.point("x", "r"), (106, 253), (106, y + 26), s.point(f"a{i}", "l")],
                color=STROKE["edge"], end=(f"a{i}", "l"))
        s.link(f"a{i}", f"b{i}", color=STROKE["edge"])
        s.link(f"b{i}", f"t{i}", color=STROKE["edge"])
    node(s, "cat", 566, 79, 44, "CAT", "prior", 332, 17)
    for i in range(3):
        s.link(f"t{i}", "cat")
    node(s, "mix", 648, 128, 116, "PW 1×1", "edge", 52, 20)
    node(s, "eca", 648, 220, 116, "ECA", "edge", 52, 21)
    add(s, "res", 706, 344)
    s.link("cat", "mix"); s.link("mix", "eca", "b", "t", color=STROKE["edge"])
    s.link("eca", "res", "b", "t", color=STROKE["edge"])
    s.route([s.point("x", "b"), (57, 430), (706, 430), s.point("res", "b")],
            color=STROKE["edge"], start=("x", "b"), end=("res", "b"))
    return s


def icg_panel():
    s = Scene("04_icg", 900, 600)
    panel(s, "(d) Image-based context guidance (ICG)", "loc")
    tensor(s, "rgb", 30, 86, 76, "RGB", "enc", 50, 20)
    node(s, "cnn", 146, 86, 116, "Conv×2", "enc", 50, 20)
    tensor(s, "v", 302, 86, 62, "V", "enc", 50, 21)
    s.link("rgb", "cnn", color=STROKE["enc"]); s.link("cnn", "v", color=STROKE["enc"])
    tensor(s, "f", 30, 174, 88, "F_seu", "loc", 50, 19)
    node(s, "p", 158, 174, 108, "P_init", "loc", 50, 20)
    s.link("f", "p", color=STROKE["loc"])

    node(s, "fg", 54, 286, 120, "FG mean", "loc", 50, 19)
    node(s, "bg", 224, 286, 120, "BG local", "loc", 50, 19)
    node(s, "bgm", 394, 286, 120, "BG global", "loc", 50, 19)
    cat(s, "bcat", 568, 311, 54)
    node(s, "contrast", 630, 286, 126, "|FG−BG|", "loc", 50, 19)
    tensor(s, "g", 796, 286, 70, "G", "loc", 50, 20)
    s.route([s.point("v", "b"), (333, 248), (114, 248), s.point("fg", "t")], color=STROKE["enc"])
    s.route([(333, 248), (284, 248), s.point("bg", "t")], color=STROKE["enc"])
    s.route([(333, 248), (454, 248), s.point("bgm", "t")], color=STROKE["enc"])
    s.route([s.point("p", "b"), (212, 258), (114, 258), s.point("fg", "t")], color=STROKE["loc"])
    s.route([(212, 258), (284, 258), s.point("bg", "t")], color=STROKE["loc"])
    s.route([(212, 258), (454, 258), s.point("bgm", "t")], color=STROKE["loc"])
    s.link("bg", "bcat", color=STROKE["loc"]); s.link("bgm", "bcat", color=STROKE["loc"])
    s.link("fg", "contrast", via=[(194, 311), (194, 374), (692, 374), (692, 336)], color=STROKE["loc"])
    s.link("bcat", "contrast", color=STROKE["loc"]); s.link("contrast", "g", color=STROKE["loc"])

    node(s, "ag1", 180, 470, 126, "AG-1", "loc", 54, 21)
    node(s, "ag2", 390, 470, 126, "AG-2", "loc", 54, 21)
    tensor(s, "out", 602, 470, 92, "F_icg", "loc", 54, 20)
    s.route([s.point("f", "b"), (74, 444), (154, 444), (154, 497), s.point("ag1", "l")], color=STROKE["loc"])
    s.link("ag1", "ag2", color=STROKE["loc"]); s.link("ag2", "out", color=STROKE["loc"])
    s.route([s.point("g", "b"), (831, 430), (243, 430), (243, 470)], color=STROKE["loc"])
    s.route([(453, 430), (453, 470)], color=STROKE["loc"]); s.dot(453, 430, STROKE["loc"])
    return s


def fusion_panel():
    s = Scene("05_afm_apghr", 1410, 600)
    panel(s, "(e) Attentive fusion and ambiguity-prior-guided high-resolution refinement", "ref")
    s.label(28, 60, "AFM", 22, 120)
    tensor(s, "fe", 28, 106, 92, "F_edge", "edge", 50, 19)
    node(s, "pe", 154, 106, 94, "Proj E", "edge", 50, 19)
    tensor(s, "fl", 28, 214, 92, "F_loc", "loc", 50, 19)
    node(s, "pl", 154, 214, 94, "Proj L", "loc", 50, 19)
    node(s, "att", 292, 214, 90, "Attn", "loc", 50, 20)
    mul(s, "mul", 432, 131)
    cat(s, "cat", 432, 239, 50)
    node(s, "ref", 484, 214, 92, "Refine", "prior", 50, 19)
    tensor(s, "z", 616, 214, 82, "Z_afm", "prior", 50, 19)
    s.link("fe", "pe", color=STROKE["edge"]); s.link("fl", "pl", color=STROKE["loc"])
    s.link("pl", "att", color=STROKE["loc"]); s.link("pe", "mul", color=STROKE["edge"])
    s.link("att", "mul", "t", "b", via=[(337, 181), (432, 181)], color=STROKE["loc"])
    s.link("mul", "cat", "b", "t", color=STROKE["edge"])
    s.link("pl", "cat", via=[(270, 239)], color=STROKE["loc"])
    s.link("cat", "ref"); s.link("ref", "z")

    s.label(760, 60, "APG-HR", 22, 180)
    node(s, "p", 760, 106, 94, "p", "prior", 50, 21)
    node(s, "u", 884, 106, 128, "u=4p(1−p)", "fpn", 50, 22)
    s.route([s.point("z", "r"), (728, 239), (728, 131), s.point("p", "l")], color=STROKE["prior"])
    s.link("p", "u", color=STROKE["prior"])

    sources = [("he", 204, "F_edge ↑4", "edge"), ("hl", 286, "F_loc ↑4", "loc"),
               ("rgb", 368, "RGB feat", "enc")]
    for key, y, label, fam in sources:
        node(s, key, 760, y, 126, label, fam, 50, 19)
    cat(s, "hcat", 1046, 311, 58)
    for key, _, _, _ in sources:
        s.link(key, "hcat")
    s.route([s.point("p", "b"), (807, 174), (1018, 174), (1018, 294), s.point("hcat", "t")], color=STROKE["prior"])
    s.route([s.point("u", "b"), (942, 190), (1062, 190), (1062, 294), s.point("hcat", "t")], color=STROKE["fpn"])
    node(s, "fuse", 1110, 286, 98, "Fuse", "ref", 50, 20)
    node(s, "dsr", 1244, 286, 92, "DSR", "ref", 50, 20)
    tensor(s, "mask", 1272, 406, 88, "Mask\n320²", "prior", 56, 19)
    s.link("hcat", "fuse"); s.link("fuse", "dsr")
    s.link("dsr", "mask", "b", "t", color=STROKE["ref"])

    return s


def add_anchor(full, source, target_xy, letter, color):
    sx, sy = full.point(source, "r")
    tx, ty = target_xy
    # Callouts are thin, dashed, arrow-free and live only in panel gutters.
    full.route([(sx, sy), (tx - 26, sy), (tx - 26, ty), (tx, ty)],
               color=color, dash=True, arrow=False)
    full.node(f"anchor_{letter}", tx - 15, ty - 15, 30, 30, letter, FILL["white"],
              kind="circle", fs=18, stroke=color)


def local_badge(full, source, letter, side, dx, dy):
    """Short gray reference mark; never crosses a feature-flow region."""
    sx, sy = full.point(source, side)
    cx, cy = sx + dx, sy + dy
    full.route([(sx, sy), (cx, cy)], color="#7B858C", dash=True, arrow=False)
    full.node(f"anchor_{letter}", cx - 15, cy - 15, 30, 30, letter, FILL["white"],
              kind="circle", fs=18, stroke="#7B858C")


def apply_hatch(svg_path):
    """Add subtle family-specific textures for monochrome-print robustness."""
    text = svg_path.read_text(encoding="utf-8")
    defs = """
    <pattern id="h_enc" patternUnits="userSpaceOnUse" width="9" height="9">
      <rect width="9" height="9" fill="#F2BE9B"/><path d="M-2,9 L9,-2 M4,11 L11,4" stroke="#A95831" stroke-width="0.8" opacity="0.20"/>
    </pattern>
    <pattern id="h_fpn" patternUnits="userSpaceOnUse" width="9" height="9">
      <rect width="9" height="9" fill="#EDCB52"/><circle cx="2" cy="2" r="0.8" fill="#8D7100" opacity="0.25"/>
    </pattern>
    <pattern id="h_edge" patternUnits="userSpaceOnUse" width="9" height="9">
      <rect width="9" height="9" fill="#E893AA"/><path d="M0,3 H9" stroke="#9F3653" stroke-width="0.7" opacity="0.18"/>
    </pattern>
    <pattern id="h_loc" patternUnits="userSpaceOnUse" width="9" height="9">
      <rect width="9" height="9" fill="#B69BD5"/><path d="M3,0 V9" stroke="#60408D" stroke-width="0.7" opacity="0.18"/>
    </pattern>
    <pattern id="h_prior" patternUnits="userSpaceOnUse" width="10" height="10">
      <rect width="10" height="10" fill="#87B7D6"/><path d="M0,0 L10,10 M10,0 L0,10" stroke="#28688F" stroke-width="0.55" opacity="0.13"/>
    </pattern>
    <pattern id="h_ref" patternUnits="userSpaceOnUse" width="9" height="9">
      <rect width="9" height="9" fill="#91C889"/><path d="M0,-2 L11,9 M-2,4 L5,11" stroke="#36733E" stroke-width="0.8" opacity="0.17"/>
    </pattern>
    <pattern id="p_main" patternUnits="userSpaceOnUse" width="18" height="18">
      <rect width="18" height="18" fill="#FCFCFC"/><circle cx="2" cy="2" r="0.45" fill="#71808B" opacity="0.10"/>
    </pattern>
    <pattern id="p_fpn" patternUnits="userSpaceOnUse" width="15" height="15">
      <rect width="15" height="15" fill="#FFF9D9"/><circle cx="3" cy="3" r="0.55" fill="#8D7100" opacity="0.12"/>
    </pattern>
    <pattern id="p_edge" patternUnits="userSpaceOnUse" width="16" height="16">
      <rect width="16" height="16" fill="#FCE8EE"/><path d="M0,4 H16" stroke="#9F3653" stroke-width="0.5" opacity="0.08"/>
    </pattern>
    <pattern id="p_loc" patternUnits="userSpaceOnUse" width="16" height="16">
      <rect width="16" height="16" fill="#F1EBFA"/><path d="M4,0 V16" stroke="#60408D" stroke-width="0.5" opacity="0.08"/>
    </pattern>
    <pattern id="p_ref" patternUnits="userSpaceOnUse" width="16" height="16">
      <rect width="16" height="16" fill="#EAF6E8"/><path d="M-2,16 L16,-2 M8,18 L18,8" stroke="#36733E" stroke-width="0.5" opacity="0.08"/>
    </pattern>
    """
    for color, pattern in [
        (FILL["enc"], "h_enc"), (FILL["fpn"], "h_fpn"),
        (FILL["edge"], "h_edge"), (FILL["loc"], "h_loc"),
        (FILL["prior"], "h_prior"), (FILL["ref"], "h_ref"),
        (PANEL["main"], "p_main"), (PANEL["fpn"], "p_fpn"),
        (PANEL["edge"], "p_edge"), (PANEL["loc"], "p_loc"),
        (PANEL["ref"], "p_ref"),
    ]:
        text = text.replace(f'fill="{color}"', f'fill="url(#{pattern})"')
    # Insert definitions only after replacing scene fills, otherwise each
    # pattern's own base rectangle would recursively reference itself.
    text = text.replace("</defs>", defs + "</defs>", 1)
    def normalize_wire(match):
        tag = match.group(0)
        dashed = "stroke-dasharray" in tag
        tag = re.sub(r'stroke="[^"]+"',
                     'stroke="#7B858C"' if dashed else 'stroke="#202B38"', tag, count=1)
        tag = re.sub(r'stroke-width="[^"]+"',
                     'stroke-width="1.0"' if dashed else 'stroke-width="2.0"', tag, count=1)
        if dashed:
            tag = re.sub(r' marker-end="[^"]+"', '', tag)
        return tag
    text = re.sub(r'<polyline\b[^>]*/>', normalize_wire, text)
    svg_path.write_text(text, encoding="utf-8")


def build():
    panels = [main_graph(), lmfr_panel(), edge_panel(), icg_panel(), fusion_panel()]
    full = Scene("APHNet_M6_publication_v8", 2360, 1570)
    full.paste(panels[0], 10, 10, "a_")
    full.paste(panels[1], 1530, 10, "b_")
    full.paste(panels[2], 1530, 330, "c_")
    full.paste(panels[3], 10, 900, "d_")
    full.paste(panels[4], 930, 900, "e_")

    # Local gray badges map main-graph blocks to panels without cross-panel lines.
    local_badge(full, "a_dsr2", "b", "t", 0, -34)
    local_badge(full, "a_eeb", "c", "t", 0, -34)
    local_badge(full, "a_icg", "d", "t", 0, -34)
    local_badge(full, "a_apg", "e", "l", -34, 0)

    # Shared footer: all notational guidance is consolidated here.
    full.label(18, 1510, "+ addition | CAT concatenation | × multiplication | ↑ bilinear upsample | * same RGB input", 18, 1700)
    full.label(18, 1540, "DW/PW → GN/ReLU | Sobel/Laplacian trainable | AG-1/2 share G | p probability | u ambiguity", 18, 1800)

    # Normalize hierarchy and line semantics before both SVG and Visio export.
    for sc in [full, *panels]:
        for item in sc.items:
            if item["kind"] == "line":
                item["stroke"] = "#7B858C" if item.get("dash") else "#202B38"
                if item.get("dash"):
                    item["arrow"] = False
            elif item["kind"] == "text":
                item["fs"] = max(item.get("fs", 22), 22)
            else:
                item["fs"] = max(item.get("fs", 24), 24)

    for sc in [full, *panels]:
        path = OUT / f"{sc.name}.svg"
        svg(sc, path)
        apply_hatch(path)
        kwargs = {"output_width": 183 / 25.4 * 96} if sc is full else {}
        cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix(".pdf")), **kwargs)
        cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix(".png")),
                         output_width=4720 if sc is full else max(sc.w * 2, 1800))

    with Image.open(OUT / "APHNet_M6_publication_v8.png") as image:
        image.save(OUT / "APHNet_M6_publication_v8_600dpi.tiff", dpi=(600, 600), compression="tiff_lzw")
        image.convert("L").save(OUT / "APHNet_M6_publication_v8_grayscale_preview.png")
    v6.export_vsdx(full, OUT / "APHNet_M6_publication_v8_editable.vsdx")
    (OUT / "editable_scene.json").write_text(
        json.dumps([{"name": x.name, "width": x.w, "height": x.h, "items": x.items}
                    for x in [full, *panels]], ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()

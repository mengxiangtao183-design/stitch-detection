"""Generate two corrected APHNet/M6 publication schematics.

Variant A combines the code-audited detailed panels with the compact overview.
Variant B follows a PSSNet-like U-shaped overview with surrounding module insets.
Only SVG and PNG are exported, as requested.
"""
from pathlib import Path
import copy
import re

import cairosvg

import build_aphnet_publication_v4 as v4


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_two_styles"
OUT.mkdir(parents=True, exist_ok=True)


def lmfr_panel():
    s = v4.Scene("02_lmfr", 820, 300)
    s.frame(8, 8, 804, 284, "(b) LMFR cell — top-down lightweight reconstruction", "fpn")
    s.node("coarse", 34, 80, 100, 50, "P(i+1)", "fpn", kind="tensor", fs=18)
    v4.block(s, "up", 170, 80, 82, "↑2", "prior", 50, 19)
    s.node("lat", 34, 180, 100, 50, "L_i", "fpn", kind="tensor", fs=18)
    s.plus("sum", 310, 155)
    v4.block(s, "dw", 365, 130, 112, "DW 3×3", "hr", 50, 18)
    v4.block(s, "pw", 515, 130, 112, "PW 1×1", "hr", 50, 18)
    s.node("out", 674, 130, 92, 50, "P_i", "fpn", kind="tensor", fs=18)
    s.link("coarse", "up")
    s.link("up", "sum", via=[(280, 105), (280, 138)], sb="t")
    s.link("lat", "sum", via=[(280, 205), (280, 172)], sb="b")
    for a, b in [("sum", "dw"), ("dw", "pw"), ("pw", "out")]:
        s.link(a, b)
    s.label(365, 218, "DW/PW followed by GN + ReLU", 16, 420)
    return s


def compact_edge_panel():
    s = v4.Scene("03_edge_compact", 820, 500)
    s.frame(8, 8, 804, 484, "(c) Edge enhancement block (EEB)", "edge")
    s.node("x", 26, 220, 64, 54, "X", "edge", kind="port", fs=19)
    rows = [(78, "DW 3×3", "PW 1×1", "128"),
            (198, "Sobel X/Y", "PW 1×1", "32"),
            (318, "Laplacian", "Identity", "128")]
    for i, (y, a, b, ch) in enumerate(rows):
        v4.block(s, f"a{i}", 130, y, 132, a, "edge", 52, 18)
        v4.block(s, f"b{i}", 300, y, 118, b, "edge", 52, 18)
        s.node(f"t{i}", 454, y, 72, 52, ch + " ch", "edge", kind="tensor", fs=16)
        s.route([s.point("x", "r"), (106, 247), (106, y + 26), s.point(f"a{i}", "l")])
        s.link(f"a{i}", f"b{i}"); s.link(f"b{i}", f"t{i}")
    v4.block(s, "cat", 560, 75, 48, "CAT", "prior", 330, 16)
    for i in range(3): s.link(f"t{i}", "cat")
    v4.block(s, "mix", 645, 125, 116, "PW 1×1", "edge", 52, 18)
    v4.block(s, "eca", 645, 215, 116, "ECA", "edge", 52, 19)
    s.plus("res", 703, 340)
    s.link("cat", "mix"); s.link("mix", "eca", "b", "t"); s.link("eca", "res", "b", "t")
    s.route([s.point("x", "b"), (58, 430), (703, 430), s.point("res", "b")])
    v4.block(s, "act", 650, 400, 82, "ReLU", "edge", 42, 16)
    s.node("out", 755, 400, 42, 42, "Y", "edge", kind="port", fs=17)
    s.link("res", "act", "b", "t"); s.link("act", "out")
    s.label(125, 456, "Sobel/Laplacian kernels remain trainable after initialization.", 14, 560)
    return s


def compact_fusion_panel():
    s = v4.Scene("05_afm_apghr", 1410, 600)
    s.frame(8, 8, 1394, 584, "(f) AFM + ambiguity-prior-guided HighRes refinement", "hr")
    s.label(28, 58, "AFM", 20, 120)
    s.node("fe", 28, 100, 92, 50, "F_edge", "edge", kind="port", fs=17)
    v4.block(s, "pe", 154, 100, 94, "Proj E", "edge", 50, 17)
    s.node("fl", 28, 208, 92, 50, "F_loc", "loc", kind="port", fs=17)
    v4.block(s, "pl", 154, 208, 94, "Proj L", "loc", 50, 17)
    v4.block(s, "att", 292, 208, 90, "Attn", "loc", 50, 18)
    s.plus("mul", 430, 125, "×"); s.plus("cat", 430, 233, "CAT")
    s.nodes["cat"]["fs"] = 12
    v4.block(s, "ref", 482, 208, 92, "Refine", "prior", 50, 17)
    s.node("z", 612, 208, 82, 50, "Z_afm", "prior", kind="tensor", fs=17)
    s.link("fe", "pe"); s.link("fl", "pl"); s.link("pl", "att"); s.link("pe", "mul")
    s.link("att", "mul", "t", "b", via=[(337, 175), (430, 175)])
    s.link("mul", "cat", "b", "t"); s.link("pl", "cat", via=[(270, 233)])
    s.link("cat", "ref"); s.link("ref", "z")

    s.label(750, 58, "APG-HR", 20, 180)
    v4.block(s, "sig", 720, 100, 82, "Sigmoid", "prior", 50, 16)
    v4.block(s, "p", 825, 100, 62, "p", "prior", 50, 19)
    v4.block(s, "u", 910, 100, 110, "4p(1−p)", "fpn", 50, 17)
    s.route([s.point("z", "r"), (704, 233), (704, 125), s.point("sig", "l")])
    s.link("sig", "p"); s.link("p", "u")
    sources = [("he", 198, "F_edge ↑4", "edge"),
               ("hl", 280, "F_loc ↑4", "loc"),
               ("rgb", 362, "RGB feat", "enc")]
    for key, y, label, family in sources:
        v4.block(s, key, 750, y, 126, label, family, 50, 17)
    v4.block(s, "hcat", 1034, 180, 58, "CAT", "prior", 260, 16)
    for key, _, _, _ in sources: s.link(key, "hcat")
    s.route([s.point("p", "b"), (856, 170), (1010, 170), (1010, 230), s.point("hcat", "l")])
    s.route([s.point("u", "b"), (965, 184), (1000, 184), (1000, 390), s.point("hcat", "l")])
    v4.block(s, "fuse", 1140, 248, 105, "1×1 fuse", "hr", 52, 17)
    v4.block(s, "dsr", 1280, 248, 88, "DSR", "hr", 52, 18)
    v4.block(s, "pred", 1248, 342, 112, "Conv 1×1", "hr", 46, 16)
    s.node("mask", 1254, 420, 105, 60, "Mask\n320²", "prior", kind="tensor", fs=18)
    s.link("hcat", "fuse"); s.link("fuse", "dsr")
    s.link("dsr", "pred", "b", "t"); s.link("pred", "mask", "b", "t")
    s.label(750, 505, "Five inputs: 48 + 48 + 16 + 1 + 1 = 114 channels", 17, 620)
    return s


def rename_panel(scene, title):
    for item in scene.items:
        if item["kind"] == "text":
            item["text"] = title
            return


def paste_scaled(dst, src, x, y, scale, prefix):
    """Paste a Scene with uniform scaling while retaining vector editability."""
    for original in src.items:
        item = copy.deepcopy(original)
        item["id"] = prefix + item["id"]
        if item["kind"] == "line":
            item["points"] = [(x + px * scale, y + py * scale)
                              for px, py in item["points"]]
            for tag in ("start", "end"):
                if item.get(tag):
                    item[tag] = (prefix + item[tag][0], item[tag][1])
        else:
            item["x"] = x + item["x"] * scale
            item["y"] = y + item["y"] * scale
            item["w"] *= scale
            item["h"] *= scale
            item["fs"] = max(13, item.get("fs", 17) * scale)
        dst.items.append(item)
        if item["kind"] not in ("line", "text"):
            dst.nodes[item["id"]] = item


def soften_operator_boxes(scene, tint):
    """Avoid stark white operator tiles while keeping circles and ports crisp."""
    for item in scene.items:
        if item["kind"] == "box" and item.get("fill") == v4.COL["white"]:
            item["fill"] = tint
        if item["kind"] == "line":
            item["stroke"] = "#34424F" if not item.get("dash") else "#7B858C"


def add_defs_and_normalize(path):
    """Uniform arrows/lines and subtle panel textures for grayscale robustness."""
    text = path.read_text(encoding="utf-8")
    defs = """
    <pattern id="pPeach" patternUnits="userSpaceOnUse" width="14" height="14">
      <rect width="14" height="14" fill="#F8DFCF"/><path d="M-2,14 L14,-2 M7,16 L16,7" stroke="#A95831" stroke-width="0.45" opacity="0.10"/>
    </pattern>
    <pattern id="pPink" patternUnits="userSpaceOnUse" width="14" height="14">
      <rect width="14" height="14" fill="#F5D6DF"/><path d="M0,4 H14" stroke="#9F3653" stroke-width="0.45" opacity="0.10"/>
    </pattern>
    <pattern id="pPurple" patternUnits="userSpaceOnUse" width="14" height="14">
      <rect width="14" height="14" fill="#E5DDF2"/><path d="M4,0 V14" stroke="#60408D" stroke-width="0.45" opacity="0.10"/>
    </pattern>
    <pattern id="pGreen" patternUnits="userSpaceOnUse" width="14" height="14">
      <rect width="14" height="14" fill="#DDEEDB"/><circle cx="3" cy="3" r="0.55" fill="#36733E" opacity="0.12"/>
    </pattern>
    <pattern id="pBlue" patternUnits="userSpaceOnUse" width="14" height="14">
      <rect width="14" height="14" fill="#D8EAF6"/><path d="M0,0 L14,14" stroke="#28688F" stroke-width="0.4" opacity="0.09"/>
    </pattern>
    """
    text = text.replace("</defs>", defs + "</defs>", 1)
    # Apply texture only to large panel frames; compact operator blocks stay flat.
    text = re.sub(r'(<rect[^>]+width="(?:[8-9]\d\d|1\d{3,})"[^>]+)fill="#F8DFCF"', r'\1fill="url(#pPeach)"', text)
    text = re.sub(r'(<rect[^>]+width="(?:[8-9]\d\d|1\d{3,})"[^>]+)fill="#F5D6DF"', r'\1fill="url(#pPink)"', text)
    text = re.sub(r'(<rect[^>]+width="(?:[8-9]\d\d|1\d{3,})"[^>]+)fill="#E5DDF2"', r'\1fill="url(#pPurple)"', text)
    text = re.sub(r'(<rect[^>]+width="(?:[8-9]\d\d|1\d{3,})"[^>]+)fill="#DDEEDB"', r'\1fill="url(#pGreen)"', text)
    text = re.sub(r'(<rect[^>]+width="(?:[8-9]\d\d|1\d{3,})"[^>]+)fill="#D8EAF6"', r'\1fill="url(#pBlue)"', text)

    def wire(m):
        tag = m.group(0)
        dashed = "stroke-dasharray" in tag
        tag = re.sub(r'stroke="[^"]+"',
                     'stroke="#7B858C"' if dashed else 'stroke="#34424F"',
                     tag, count=1)
        tag = re.sub(r'stroke-width="[^"]+"',
                     'stroke-width="1.15"' if dashed else 'stroke-width="1.8"',
                     tag, count=1)
        if dashed:
            tag = re.sub(r' marker-end="[^"]+"', '', tag)
        return tag

    text = re.sub(r'<polyline\b[^>]*/>', wire, text)
    path.write_text(text, encoding="utf-8")


def export(scene, basename):
    svg_path = OUT / f"{basename}.svg"
    png_path = OUT / f"{basename}.png"
    v4.svg(scene, svg_path)
    add_defs_and_normalize(svg_path)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                     output_width=scene.w, output_height=scene.h)


def corrected_overview():
    """Use the audited overview; stage operations precede C2-C5 outputs."""
    scene = v4.overview()
    scene.name = "corrected_overview"
    # Clarify that the three feature routes really enter the HighRes head.
    for item in scene.items:
        if item.get("id") == "er":
            item["text"] = ("Edge enhancement (c)\nContext / Sobel / Lap.\n"
                            "Channel attention + skip")
        if item.get("id") == "s2":
            item["text"] = "Stage 2\nEdgeResidual"
        if item.get("id") == "icg":
            item["text"] = "Image-context guidance\nICG (e); 192 ch"
        if item.get("id") == "afm":
            item["text"] = "Attentive fusion\nAFM (d)\n80×80×1 logits"
        if item.get("id") == "head":
            item["text"] = ("AFM-guided HighRes head (h)\n"
                            "F_edge + F_loc + RGB + p + u\n"
                            "CAT(114) → 1×1(48) → DSR → logits")
        if item.get("id") == "sig":
            item["text"] = "Sigmoid → threshold\n320×320 stitch mask"
        if item.get("id") == "ec":
            item["text"] = "CAT"
            item["fs"] = 11
        if item.get("id") == "cat":
            item["text"] = "CAT"
            item["fs"] = 11
        if (item.get("kind") == "text" and
                str(item.get("text", "")).startswith("Main mask shown")):
            item["text"] = ("Main prediction shown; detailed modules are expanded in (b)–(h). "
                            "Disabled centerline/topology heads are omitted.")
        if item.get("kind") == "text" and item.get("text", "").startswith("Main mask shown"):
            item["text"] = "Detailed modules are shown in (b)–(h); disabled centerline/topology heads are omitted."
    return scene


def build_variant_a():
    """Detailed-content + compact-overview fusion in a reference-paper layout."""
    main = corrected_overview()
    enc = v4.encoder(); soften_operator_boxes(enc, "#FCECE1")
    for item in enc.items:
        if item["kind"] == "text" and "Fused inverted block / Stage 2" in item.get("text", ""):
            item["text"] = "EdgeResidual / Stage 2"
    lmfr = lmfr_panel(); soften_operator_boxes(lmfr, "#FFF6CF")
    edge = v4.edge(); soften_operator_boxes(edge, "#FBE9EE")
    afm = v4.afm(); soften_operator_boxes(afm, "#EAF3F8")
    seu = v4.seu(); soften_operator_boxes(seu, "#F0EAF8")
    icg = v4.icg(); soften_operator_boxes(icg, "#F0EAF8")
    hr = v4.highres(); soften_operator_boxes(hr, "#EAF5E8")
    rename_panel(edge, "(c) Edge enhancement — 80×80×128")
    rename_panel(afm, "(d) Attentive fusion — localization-guided coarse logits")
    rename_panel(icg, "(e) Image-based context guidance — explicit contrast and shared gates")
    rename_panel(hr, "(h) AFM-guided HighRes head — five explicit feature inputs")

    full = v4.Scene("APHNet_M6_fusion_reference_style", 3840, 2910)
    full.paste(main, 10, 10, "a_")
    full.paste(enc, 2590, 10, "g_")
    paste_scaled(full, lmfr, 2610, 625, 1.45, "b_")
    full.paste(edge, 10, 950, "c_")
    full.paste(afm, 1305, 950, "d_")
    full.paste(seu, 2590, 950, "f_")
    paste_scaled(full, icg, 10, 1460, 1.45, "e_")
    paste_scaled(full, hr, 1935, 1570, 1.45, "h_")
    export(full, "APHNet_M6_variant_A_fusion_layout")


def pss_main():
    """PSSNet-like U-shaped main graph, while preserving the implemented M6."""
    s = v4.Scene("01_pss_style_main", 2300, 1280)
    s.frame(8, 8, 2284, 1264, "(a) APHNet / M6 — corrected end-to-end architecture", "hr")
    s.label(42, 55, "MobileNetV4 Hybrid Medium", 20, 440)
    s.label(515, 55, "LMFR / Mobile-FPN", 20, 450)
    s.label(1130, 55, "Dual-path decoding and full-resolution refinement", 20, 1050)

    # Encoder: stage operator first, then its named output tensor.
    v4.block(s, "rgb", 42, 98, 105, "RGB\n320²×3", "enc", 60, 17)
    v4.block(s, "stem", 185, 98, 135, "Stem\n3×3 / s2", "enc", 60, 17)
    s.link("rgb", "stem")
    rows = [
        (2, 210, "EdgeResidual", "80²×48"),
        (3, 380, "UIB ×2", "40²×80"),
        (4, 550, "UIB ×8 + MQA ×4", "20²×160"),
        (5, 720, "UIB ×12 + MQA ×4\nPW 1×1→960", "10²×960"),
    ]
    for n, y, stage, shape in rows:
        v4.block(s, f"s{n}", 75, y, 170, f"Stage {n}\n{stage}", "enc", 66, 16)
        s.node(f"c{n}", 285, y + 5, 105, 56, f"C{n}\n{shape}", "enc", kind="tensor", fs=16)
        if n == 2:
            y_mid = s.point("stem", "b")[1] + 26
            s.link("stem", "s2", "b", "t", via=[(252, y_mid), (160, y_mid)])
        else:
            s.link(f"s{n-1}", f"s{n}", "b", "t")
        s.link(f"s{n}", f"c{n}")

        v4.block(s, f"l{n}", 500, y + 2, 122, f"L{n}\n1×1→192", "fpn", 60, 16)
        s.link(f"c{n}", f"l{n}")
        if n < 5:
            s.plus(f"add{n}", 685, y + 32)
            s.link(f"l{n}", f"add{n}")
            v4.block(s, f"dsr{n}", 730, y + 2, 118, "DSR (b)", "hr", 60, 17)
            s.link(f"add{n}", f"dsr{n}")
        else:
            v4.block(s, f"dsr{n}", 665, y + 2, 118, "DSR (b)", "hr", 60, 17)
            s.link(f"l{n}", f"dsr{n}")
        s.node(f"p{n}", 895, y + 5, 104, 56,
               f"P{n}\n{shape.split('×')[0]}²×192", "fpn", kind="tensor", fs=16)
        s.link(f"dsr{n}", f"p{n}")

    # Ascending FPN spine, analogous to the reference decoder ladder.
    for src, dst in [(5, 4), (4, 3), (3, 2)]:
        x0, y0 = s.point(f"p{src}", "r")
        x1, y1 = s.point(f"add{dst}", "b")
        s.route([(x0, y0), (1045, y0), (1045, y1 + 28), (x1, y1 + 28), (x1, y1)],
                color="#377DAB")
        s.label(1015, (y0 + y1) / 2 - 5, "↑2", 15, 45)

    # Boundary branch at P2 scale.
    s.node("bp3", 1135, 130, 85, 48, "P3", "fpn", kind="port", fs=17)
    v4.block(s, "bup", 1250, 125, 90, "↑2", "prior", 56, 18)
    s.node("bp2", 1135, 230, 85, 48, "P2", "fpn", kind="port", fs=17)
    s.plus("bcat", 1395, 205, "CAT")
    s.nodes["bcat"]["fs"] = 11
    v4.block(s, "bconv", 1440, 172, 125, "Conv 3×3\n384→128", "edge", 66, 16)
    v4.block(s, "eeb", 1605, 172, 125, "EEB (c)", "edge", 66, 18)
    s.node("fe", 1770, 177, 118, 56, "F_edge\n80²×128", "edge", kind="tensor", fs=16)
    s.link("bp3", "bup"); s.link("bup", "bcat", via=[(1370, 153), (1370, 188)], sb="t")
    s.link("bp2", "bcat", via=[(1370, 254), (1370, 222)], sb="b")
    for a, b in [("bcat", "bconv"), ("bconv", "eeb"), ("eeb", "fe")]: s.link(a, b)
    s.route([s.point("p3", "r"), (1080, 412), (1080, 154), s.point("bp3", "l")])
    s.route([s.point("p2", "r"), (1095, 242), s.point("bp2", "l")])

    # Localization branch: P4 -> SEU -> ICG -> two 2x steps -> +P2.
    s.node("lp4", 1135, 365, 85, 48, "P4", "fpn", kind="port", fs=17)
    v4.block(s, "seu", 1250, 356, 112, "SEU", "loc", 66, 18)
    v4.block(s, "icg", 1400, 356, 112, "ICG* (d)", "loc", 66, 18)
    v4.block(s, "lup1", 1550, 356, 82, "↑2", "loc", 66, 18)
    v4.block(s, "lup2", 1670, 356, 82, "↑2", "loc", 66, 18)
    s.plus("ladd", 1800, 389)
    v4.block(s, "lref", 1845, 356, 118, "LocRef", "loc", 66, 18)
    s.node("fl", 2000, 361, 118, 56, "F_loc\n80²×128", "loc", kind="tensor", fs=16)
    for a, b in [("lp4", "seu"), ("seu", "icg"), ("icg", "lup1"),
                 ("lup1", "lup2"), ("lup2", "ladd"), ("ladd", "lref"), ("lref", "fl")]:
        s.link(a, b)
    s.route([s.point("p4", "r"), (1080, 582), (1080, 389), s.point("lp4", "l")])
    s.route([s.point("p2", "r"), (1105, 242), (1105, 485), (1800, 485), s.point("ladd", "b")])

    # AFM and APG-HR: explicit branch forks prevent the omission in the old figure.
    v4.block(s, "afm", 1365, 675, 165, "AFM (e)\nE gated by L", "prior", 78, 18)
    s.node("z", 1580, 686, 104, 56, "Z_afm", "prior", kind="tensor", fs=17)
    s.link("afm", "z")
    # Feature rails fork once: one branch to AFM, another directly to HighRes.
    s.route([s.point("fe", "r"), (2180, 205), (2180, 630), (1410, 630), s.point("afm", "t")])
    s.dot(2180, 630)
    s.route([s.point("fl", "r"), (2150, 389), (2150, 770), s.point("afm", "r")])
    s.dot(2150, 770)

    v4.block(s, "head", 1510, 935, 410,
             "APG-HR (e)\nF_edge + F_loc + RGB* + p + u\nCAT 114 → 1×1/48 → DSR → 1×1",
             "hr", 105, 18)
    s.link("z", "head", "b", "l", via=[(1632, 855), (1470, 855), (1470, 987)])
    # Explicit direct feature inputs into the full-resolution head.
    s.route([(2180, 630), (2220, 630), (2220, 900), (1600, 900), (1600, 935)])
    s.route([(2150, 770), (2190, 770), (2190, 885), (1720, 885), (1720, 935)])
    s.node("rgbh", 1945, 845, 88, 46, "RGB*", "enc", kind="port", fs=17)
    s.link("rgbh", "head", "b", "t")
    v4.block(s, "sig", 1980, 948, 145, "Sigmoid\nthreshold", "prior", 78, 17)
    s.node("mask", 2160, 955, 100, 64, "Mask\n320²", "prior", kind="tensor", fs=17)
    s.link("head", "sig"); s.link("sig", "mask")

    # Compact legend, echoing the reference figure.
    legend = [("enc", "Encoder"), ("fpn", "LMFR"), ("edge", "Edge"),
              ("loc", "Localization"), ("prior", "Prior"), ("hr", "Refinement")]
    x = 45
    for i, (fam, label) in enumerate(legend):
        s.node(f"k{i}", x, 1132, 24, 18, "", fam, fs=12)
        s.label(x + 32, 1128, label, 16, 125)
        x += 155
    s.label(45, 1182, "+ add   CAT concatenate   ↑ bilinear upsample   RGB*: the same input image", 17, 1050)
    s.label(1135, 1182, "Stage operator → C_i output; P_i denotes the completed pyramid feature.", 17, 1080)
    return s


def build_variant_b():
    """Closer imitation of the reference: one dominant U-shaped graph + insets."""
    main = pss_main()
    lmfr = lmfr_panel(); soften_operator_boxes(lmfr, "#FFF6CF")
    edge = compact_edge_panel(); soften_operator_boxes(edge, "#FBE9EE")
    icg = v4.icg(); soften_operator_boxes(icg, "#F0EAF8")
    fusion = compact_fusion_panel(); soften_operator_boxes(fusion, "#EAF5E8")
    rename_panel(icg, "(d) Image-based context guidance — explicit contrast and shared gates")
    rename_panel(fusion, "(e) AFM + ambiguity-prior-guided HighRes refinement")

    full = v4.Scene("APHNet_M6_PSSNet_inspired", 3600, 2360)
    full.paste(main, 10, 10, "a_")
    paste_scaled(full, lmfr, 2325, 10, 1.50, "b_")
    paste_scaled(full, edge, 2325, 480, 1.50, "c_")
    paste_scaled(full, icg, 2325, 1245, 1.00, "d_")
    paste_scaled(full, fusion, 10, 1320, 1.62, "f_")
    full.label(22, 2320,
               "Solid arrows: active feature flow | thin dashed marks: module reference | only enabled M6 outputs are shown",
               18, 2200)
    export(full, "APHNet_M6_variant_B_PSSNet_inspired")


def main():
    build_variant_a()
    build_variant_b()
    print(OUT)


if __name__ == "__main__":
    main()

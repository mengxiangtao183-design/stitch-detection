"""APHNet/M6 six-panel publication figure.

Layout: one corrected end-to-end graph and five compact detail panels.  The
active M6 inference path is used as the source of truth; disabled loss-only
interfaces are intentionally absent.
"""
from pathlib import Path
import re

import cairosvg

import build_aphnet_publication_v4 as v4
from build_aphnet_publication_two_styles import add_defs_and_normalize


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_publication_v9"
OUT.mkdir(parents=True, exist_ok=True)


def b(s, key, x, y, w, text, fam, h=58, fs=18):
    return v4.block(s, key, x, y, w, text, fam, h, fs)


def tensor(s, key, x, y, w, text, fam, h=56, fs=18):
    return s.node(key, x, y, w, h, text, fam, kind="tensor", fs=fs)


def main_panel():
    s = v4.Scene("01_main", 2300, 1220)
    s.frame(8, 8, 2284, 1204, "(a) APHNet / M6 — complete active feature route", "hr")
    s.label(34, 55, "MobileNetV4 encoder", 21, 390)
    s.label(445, 55, "LMFR / Mobile-FPN", 21, 430)
    s.label(1090, 55, "Dual-path decoder", 21, 470)
    s.label(1570, 55, "Fusion and reconstruction", 21, 650)

    b(s, "rgb", 35, 95, 105, "RGB\n320²×3", "enc", 62, 18)
    b(s, "stem", 178, 95, 128, "Stem\n3×3 / s2", "enc", 62, 18)
    s.link("rgb", "stem")

    rows = [
        (2, 205, "S2\nFusedIB", "80²×48"),
        (3, 365, "S3\nUIB×2", "40²×80"),
        (4, 525, "S4\nUIB×8 + MQA×4", "20²×160"),
        (5, 685, "S5\nUIB×12 + MQA×4", "10²×960"),
    ]
    prev = "stem"
    for n, y, stage, shape in rows:
        b(s, f"s{n}", 70, y, 170, stage, "enc", 70, 17)
        tensor(s, f"c{n}", 280, y + 7, 112, f"C{n}\n{shape}", "enc", 56, 17)
        if n == 2:
            s.link(prev, f"s{n}", "b", "t", via=[(242, 180), (155, 180)])
        else:
            s.link(prev, f"s{n}", "b", "t")
        s.link(f"s{n}", f"c{n}")
        prev = f"s{n}"

        b(s, f"l{n}", 450, y + 5, 112, f"L{n}\n1×1 / 192", "fpn", 60, 17)
        s.link(f"c{n}", f"l{n}")
        if n < 5:
            s.plus(f"add{n}", 625, y + 35)
            b(s, f"dsr{n}", 672, y + 5, 105, "DSR (b)", "hr", 60, 18)
            s.link(f"l{n}", f"add{n}")
            s.link(f"add{n}", f"dsr{n}")
        else:
            b(s, f"dsr{n}", 608, y + 5, 105, "DSR (b)", "hr", 60, 18)
            s.link(f"l{n}", f"dsr{n}")
        tensor(s, f"p{n}", 825, y + 7, 112,
               f"P{n}\n{shape.split('×')[0]}²×192", "fpn", 56, 17)
        s.link(f"dsr{n}", f"p{n}")

    # A single outside rail keeps the top-down pyramid free of crossings.
    for src, dst in [(5, 4), (4, 3), (3, 2)]:
        x0, y0 = s.point(f"p{src}", "r")
        x1, y1 = s.point(f"add{dst}", "b")
        s.route([(x0, y0), (1000, y0), (1000, y1 + 25),
                 (x1, y1 + 25), (x1, y1)], color="#377DAB")
        s.label(965, (y0 + y1) / 2 - 5, "↑2", 16, 45)

    # Boundary path.
    tensor(s, "ep3", 1090, 130, 82, "P3", "fpn", 48, 18)
    b(s, "eup", 1205, 126, 82, "↑2", "prior", 56, 19)
    tensor(s, "ep2", 1090, 225, 82, "P2", "fpn", 48, 18)
    b(s, "ecat", 1330, 171, 64, "CAT", "prior", 54, 17)
    b(s, "econv", 1430, 165, 112, "3×3 Conv", "edge", 66, 18)
    b(s, "eeb", 1580, 165, 112, "EEB (c)", "edge", 66, 19)
    tensor(s, "fe", 1730, 170, 118, "F_edge\n80²×128", "edge", 56, 17)
    s.link("ep3", "eup")
    s.link("eup", "ecat", "r", "t", via=[(1310, 154), (1362, 154)])
    s.link("ep2", "ecat", "r", "b", via=[(1310, 249), (1362, 249)])
    for x, y in [("ecat", "econv"), ("econv", "eeb"), ("eeb", "fe")]:
        s.link(x, y)
    s.route([s.point("p3", "r"), (1045, 397), (1045, 154), s.point("ep3", "l")])
    s.route([s.point("p2", "r"), (1055, 237), s.point("ep2", "l")])

    # Localization path.
    tensor(s, "lp4", 1090, 365, 82, "P4", "fpn", 48, 18)
    b(s, "seu", 1205, 356, 96, "SEU (c)", "loc", 66, 18)
    b(s, "icg", 1338, 356, 104, "ICG* (e)", "loc", 66, 18)
    b(s, "lup", 1480, 356, 82, "↑4", "loc", 66, 19)
    s.plus("ladd", 1607, 389)
    b(s, "lref", 1650, 356, 104, "LocRef", "loc", 66, 18)
    tensor(s, "fl", 1792, 361, 118, "F_loc\n80²×128", "loc", 56, 17)
    for x, y in [("lp4", "seu"), ("seu", "icg"), ("icg", "lup"),
                 ("lup", "ladd"), ("ladd", "lref"), ("lref", "fl")]:
        s.link(x, y)
    s.route([s.point("p4", "r"), (1045, 557), (1045, 389), s.point("lp4", "l")])
    s.route([s.point("p2", "r"), (1065, 237), (1065, 470),
             (1607, 470), s.point("ladd", "b")])

    # The actual branch outputs feed AFM; duplicated named ports below feed HSR.
    b(s, "afm", 1665, 610, 170, "AFM (d)", "prior", 76, 21)
    tensor(s, "z", 1880, 620, 105, "Z_afm\n80²×1", "prior", 58, 17)
    s.route([s.point("fe", "r"), (2050, 198), (2050, 630), s.point("afm", "t")])
    s.route([s.point("fl", "b"), (1851, 545), (1750, 545), s.point("afm", "t")])
    s.link("afm", "z")

    tensor(s, "hfe", 1110, 890, 115, "F_edge", "edge", 52, 17)
    tensor(s, "hfl", 1110, 970, 115, "F_loc", "loc", 52, 17)
    tensor(s, "hrgb", 1110, 1050, 115, "RGB*", "enc", 52, 17)
    b(s, "head", 1390, 905, 420,
      "HSR + APG-HR  (f)\nCAT 114 → Fuse → DSR", "hr", 155, 21)
    for key in ("hfe", "hfl", "hrgb"):
        s.link(key, "head")
    s.link("z", "head", "b", "t", via=[(1932, 820), (1600, 820)])
    b(s, "sig", 1860, 935, 145, "Sigmoid", "prior", 78, 19)
    tensor(s, "mask", 2050, 942, 105, "Mask\n320²", "prior", 64, 18)
    s.link("head", "sig")
    s.link("sig", "mask")

    # Centralized key replaces repetitive explanations inside blocks.
    s.label(35, 890, "Module key", 20, 180)
    keys = [("enc", "Encoder"), ("fpn", "Pyramid"), ("edge", "Boundary"),
            ("loc", "Localization"), ("prior", "Prior"), ("hr", "Refine")]
    x = 35
    for i, (fam, label) in enumerate(keys):
        s.node(f"key{i}", x, 930, 24, 18, "", fam, fs=11)
        s.label(x + 32, 924, label, 17, 120)
        x += 142
    s.label(35, 1008, "+: add    CAT: concatenate    ↑: bilinear upsample", 17, 720)
    s.label(35, 1052, "Repeated tensor names denote the same feature; RGB* is the input image.", 17, 950)
    s.label(35, 1100, "C2–C5: stage outputs    P2–P5: completed pyramid features", 17, 950)
    return s


def lmfr_panel():
    s = v4.Scene("02_lmfr", 1260, 585)
    s.frame(8, 8, 1244, 569, "(b) LMFR cell — top-down lightweight reconstruction", "fpn")
    tensor(s, "pc", 45, 120, 135, "P(i+1)", "fpn", 58, 19)
    b(s, "up", 225, 120, 110, "↑2", "prior", 58, 20)
    tensor(s, "li", 45, 245, 135, "L_i", "fpn", 58, 20)
    s.plus("sum", 405, 210)
    b(s, "dw", 475, 181, 145, "DW 3×3", "hr", 62, 20)
    b(s, "pw", 675, 181, 145, "PW 1×1", "hr", 62, 20)
    tensor(s, "pi", 880, 184, 125, "P_i", "fpn", 58, 20)
    s.link("pc", "up")
    s.link("up", "sum", "r", "t", via=[(370, 149), (405, 149)])
    s.link("li", "sum", "r", "b", via=[(370, 274), (405, 274)])
    for x, y in [("sum", "dw"), ("dw", "pw"), ("pw", "pi")]:
        s.link(x, y)
    s.label(45, 385, "L_i: lateral 1×1 projection, 192 ch", 18, 560)
    s.label(655, 385, "i = 2, 3, 4", 18, 300)
    s.label(45, 438, "DW/PW are each followed by GN + ReLU", 18, 650)
    s.label(655, 438, "P5 = DSR(L5)", 18, 360)
    return s


def enhance_panel():
    s = v4.Scene("03_enhancement", 1260, 615)
    s.frame(8, 8, 1244, 599, "(c) Branch enhancement primitives — EEB and SEU", "edge")
    s.label(35, 65, "EEB", 20, 110)
    tensor(s, "x", 35, 145, 75, "X", "edge", 56, 20)
    labels = [(105, "DW/PW"), (175, "Sobel/PW"), (245, "Lap./Id")]
    for i, (y, text) in enumerate(labels):
        b(s, f"eb{i}", 175, y, 145, text, "edge", 54, 18)
        s.route([s.point("x", "r"), (140, 173), (140, y + 27), s.point(f"eb{i}", "l")])
    b(s, "ecat", 385, 158, 80, "CAT", "prior", 90, 19)
    for i in range(3):
        s.link(f"eb{i}", "ecat")
    b(s, "mix", 505, 174, 110, "PW 1×1", "edge", 58, 18)
    b(s, "eca", 655, 174, 100, "ECA", "edge", 58, 20)
    s.plus("eres", 820, 203)
    s.link("ecat", "mix")
    s.link("mix", "eca")
    s.link("eca", "eres")
    s.route([s.point("x", "b"), (72, 305), (820, 305), s.point("eres", "b")])
    tensor(s, "eo", 875, 175, 105, "F_edge", "edge", 56, 18)
    s.link("eres", "eo")

    s.label(35, 360, "SEU", 20, 110)
    tensor(s, "p4", 35, 430, 75, "P4", "loc", 56, 19)
    steps = [(155, 120, "DW/PW"), (320, 95, "SA"), (460, 95, "ECA"),
             (600, 120, "DW/PW")]
    prev = "p4"
    for i, (x, w, text) in enumerate(steps):
        b(s, f"su{i}", x, 430, w, text, "loc", 56, 19)
        s.link(prev, f"su{i}")
        prev = f"su{i}"
    s.plus("sres", 775, 458)
    s.link(prev, "sres")
    s.route([s.point("p4", "b"), (72, 530), (775, 530), s.point("sres", "b")])
    tensor(s, "so", 835, 430, 105, "F_seu", "loc", 56, 18)
    s.link("sres", "so")
    s.label(1010, 150, "3 branches\n+ residual", 18, 210)
    s.label(985, 430, "SA: spatial\nECA: channel", 18, 230)
    return s


def afm_panel():
    s = v4.Scene("04_afm", 1180, 820)
    s.frame(8, 8, 1164, 804, "(d) AFM — localization-guided boundary fusion", "prior")
    tensor(s, "fe", 45, 125, 125, "F_edge", "edge", 58, 19)
    b(s, "pe", 220, 125, 135, "Proj E", "edge", 58, 19)
    tensor(s, "e", 405, 125, 80, "E", "edge", 58, 20)
    tensor(s, "fl", 45, 315, 125, "F_loc", "loc", 58, 19)
    b(s, "pl", 220, 315, 135, "Proj L", "loc", 58, 19)
    tensor(s, "l", 405, 315, 80, "L", "loc", 58, 20)
    b(s, "att", 545, 315, 145, "Attention", "loc", 58, 19)
    s.plus("mul", 770, 154, "×")
    tensor(s, "eg", 830, 125, 105, "E_g", "edge", 58, 19)
    for x, y in [("fe", "pe"), ("pe", "e"), ("fl", "pl"),
                 ("pl", "l"), ("l", "att"), ("e", "mul"), ("mul", "eg")]:
        s.link(x, y)
    s.link("att", "mul", "t", "b", via=[(618, 255), (770, 255)])
    b(s, "cat", 545, 505, 90, "CAT", "prior", 70, 19)
    b(s, "ref", 700, 505, 150, "3×3 Refine", "prior", 70, 19)
    b(s, "pred", 915, 505, 135, "1×1 Conv", "prior", 70, 19)
    tensor(s, "z", 915, 640, 135, "Z_afm\n80²×1", "prior", 64, 18)
    s.link("l", "cat", "b", "l", via=[(445, 540)])
    s.link("eg", "cat", "r", "t", via=[(1010, 154), (1010, 450), (590, 450)])
    s.link("cat", "ref")
    s.link("ref", "pred")
    s.link("pred", "z", "b", "t")
    s.label(45, 705, "A is generated only from L; E_g = A × E.", 18, 620)
    s.label(45, 745, "CAT(E_g, L) → one refinement block → coarse logits.", 18, 820)
    return s


def icg_panel():
    s = v4.Scene("05_icg", 1180, 820)
    s.frame(8, 8, 1164, 804, "(e) ICG — image-context guidance with shared gates", "loc")
    tensor(s, "rgb", 38, 95, 95, "RGB*", "enc", 56, 19)
    b(s, "cnn", 180, 95, 165, "Resize + Conv×2", "enc", 56, 18)
    tensor(s, "v", 390, 95, 80, "V", "enc", 56, 20)
    tensor(s, "fs", 38, 205, 105, "F_seu", "loc", 56, 18)
    b(s, "pi", 190, 205, 140, "P_init", "loc", 56, 19)
    s.link("rgb", "cnn"); s.link("cnn", "v")
    s.link("fs", "pi")

    branches = [(80, 350, "FG"), (295, 350, "BG local"), (510, 350, "BG global")]
    for i, (x, y, text) in enumerate(branches):
        b(s, f"br{i}", x, y, 150, text, "loc", 62, 18)
    s.route([s.point("v", "b"), (405, 300), (155, 300), s.point("br0", "t")])
    s.route([(405, 300), (370, 300), s.point("br1", "t")])
    s.route([(405, 300), (585, 300), s.point("br2", "t")])
    s.route([s.point("pi", "b"), (260, 320), (155, 320), s.point("br0", "t")])
    s.route([(260, 320), (370, 320), s.point("br1", "t")])
    s.route([(260, 320), (585, 320), s.point("br2", "t")])
    b(s, "contrast", 740, 350, 165, "|FG − BG|", "loc", 62, 19)
    tensor(s, "g", 960, 353, 80, "G", "loc", 56, 20)
    for i in range(3): s.link(f"br{i}", "contrast")
    s.link("contrast", "g")

    b(s, "ag1", 225, 560, 150, "AG-1", "loc", 64, 20)
    b(s, "ag2", 470, 560, 150, "AG-2", "loc", 64, 20)
    tensor(s, "out", 715, 564, 120, "F_icg", "loc", 56, 19)
    s.route([s.point("fs", "b"), (90, 520), (190, 520), (190, 592), s.point("ag1", "l")])
    s.link("ag1", "ag2"); s.link("ag2", "out")
    s.route([s.point("g", "b"), (1000, 510), (300, 510), (300, 560)])
    s.route([(545, 510), (545, 560)]); s.dot(545, 510)
    s.label(45, 695, "FG/BG contrast produces guidance G; AG-1 and AG-2 use independent weights.", 18, 1030)
    return s


def highres_panel():
    s = v4.Scene("06_highres", 1180, 820)
    s.frame(8, 8, 1164, 804, "(f) HSR + APG-HR — full-resolution reconstruction", "hr")
    entries = [("fe", "F_edge", "1×1 / 48", "Up ×4", "edge", 105),
               ("fl", "F_loc", "1×1 / 48", "Up ×4", "loc", 205),
               ("rgb", "RGB*", "3×3 / 16", "DSR", "enc", 305)]
    for key, name, proj, post, fam, y in entries:
        tensor(s, key, 35, y, 105, name, fam, 54, 18)
        b(s, key + "p", 180, y, 125, proj, fam, 54, 17)
        b(s, key + "u", 345, y, 105, post, fam, 54, 18)
        s.link(key, key + "p"); s.link(key + "p", key + "u")

    tensor(s, "z", 35, 470, 105, "Z_afm", "prior", 54, 18)
    b(s, "up", 180, 470, 105, "Up ×4", "prior", 54, 18)
    b(s, "sig", 325, 470, 105, "Sigmoid", "prior", 54, 18)
    tensor(s, "p", 470, 470, 62, "p", "prior", 54, 20)
    b(s, "u", 380, 565, 152, "u=4p(1−p)", "fpn", 58, 18)
    for x, y in [("z", "up"), ("up", "sig"), ("sig", "p")]: s.link(x, y)
    s.link("p", "u", "b", "t")

    b(s, "cat", 620, 220, 90, "CAT\n114", "prior", 250, 20)
    for key, *_ in entries:
        cy = s.point(key + "u", "r")[1]
        s.route([s.point(key + "u", "r"), (580, cy), (580, 345), s.point("cat", "l")])
    s.route([s.point("p", "r"), (570, 497), (570, 390), s.point("cat", "l")])
    s.route([s.point("u", "r"), (590, 594), (590, 420), s.point("cat", "l")])
    b(s, "fuse", 770, 255, 145, "1×1 Fuse", "hr", 70, 19)
    b(s, "ref", 770, 365, 145, "DSR", "hr", 70, 21)
    b(s, "pred", 970, 310, 135, "1×1 Conv", "prior", 78, 19)
    tensor(s, "mask", 970, 445, 135, "Mask\n320²", "prior", 64, 18)
    s.link("cat", "fuse"); s.link("fuse", "ref", "b", "t")
    s.link("ref", "pred"); s.link("pred", "mask", "b", "t")
    s.label(35, 675, "HSR uses 48 + 48 + 16 channels; M6 appends probability p and ambiguity u.", 18, 1080)
    s.label(35, 720, "The final five-way fusion contains 114 channels.", 18, 760)
    return s


def paste(dst, src, x, y, prefix):
    dst.paste(src, x, y, prefix)


def finish_svg(path):
    add_defs_and_normalize(path)
    text = path.read_text(encoding="utf-8")
    text = text.replace('rx="5"', 'rx="14"').replace('rx="2"', 'rx="18"')
    # A softer edge hierarchy: rounded blocks, uniform forward arrows.
    text = re.sub(r'stroke-width="1\.3"', 'stroke-width="1.45"', text)
    path.write_text(text, encoding="utf-8")


def build():
    full = v4.Scene("APHNet_M6_publication_v9", 3600, 2140)
    paste(full, main_panel(), 10, 10, "a_")
    paste(full, lmfr_panel(), 2330, 10, "b_")
    paste(full, enhance_panel(), 2330, 605, "c_")
    paste(full, afm_panel(), 10, 1260, "d_")
    paste(full, icg_panel(), 1210, 1260, "e_")
    paste(full, highres_panel(), 2410, 1260, "f_")
    full.label(24, 2102,
               "DW: depthwise convolution   PW: pointwise convolution   DSR: DW3×3 → PW1×1, each followed by GN/ReLU",
               18, 2100)

    svg_path = OUT / "APHNet_M6_publication_v9.svg"
    png_path = OUT / "APHNet_M6_publication_v9.png"
    v4.svg(full, svg_path)
    finish_svg(svg_path)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                     output_width=3600, output_height=2140)
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    build()

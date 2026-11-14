"""Build the compact, innovation-focused APHNet/M6 manuscript figure.

The figure deliberately keeps only the active inference route and the three
ablation-aligned innovations: LMFR, HSR and APG-HR.  Disabled centerline,
topology and standalone boundary-loss branches are not drawn.
"""
from pathlib import Path

import cairosvg

import build_aphnet_publication_v4 as v4
from build_aphnet_publication_two_styles import add_defs_and_normalize


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_innovation_focused"
OUT.mkdir(parents=True, exist_ok=True)


def block(s, key, x, y, w, text, family, h=66, fs=19):
    return v4.block(s, key, x, y, w, text, family, h, fs)


def main_flow():
    s = v4.Scene("main_flow", 2980, 650)
    s.frame(8, 8, 2964, 632, "(a) APHNet / M6 — active inference route", "hr")

    s.label(38, 62, "Feature extraction", 20, 520)
    s.label(665, 62, "Cross-scale reconstruction", 20, 560)
    s.label(1110, 62, "Dual-path decoding", 20, 720)
    s.label(1805, 62, "Prior-guided full-resolution prediction", 20, 930)

    s.node("rgb", 38, 262, 112, 66, "RGB\n320²×3", "enc", kind="tensor", fs=19)
    block(s, "enc", 205, 135, 350,
          "MobileNetV4 Hybrid Medium\nC2  80²×48     C3  40²×80\nC4  20²×160   C5  10²×960",
          "enc", 265, 19)
    block(s, "lmfr", 660, 135, 330,
          "LMFR  (b)\nC2–C5  →  P2–P5\n192 channels / level",
          "fpn", 265, 20)
    s.link("rgb", "enc")
    s.link("enc", "lmfr")

    # Two decoder lanes are vertically separated; no crossing is required.
    block(s, "edge", 1090, 126, 365,
          "Boundary path\nP3 ↑2  CAT  P2  →  EEB",
          "edge", 106, 19)
    s.node("fe", 1500, 146, 132, 66, "F_edge\n80²×128", "edge", kind="tensor", fs=18)
    block(s, "loc", 1090, 326, 365,
          "Localization path\nP4 → SEU → ICG* → ↑4 + P2",
          "loc", 106, 19)
    s.node("fl", 1500, 346, 132, 66, "F_loc\n80²×128", "loc", kind="tensor", fs=18)
    s.link("lmfr", "edge", "r", "l", via=[(1040, 267), (1040, 179)])
    s.link("lmfr", "loc", "r", "l", via=[(1040, 267), (1040, 379)])
    s.link("edge", "fe")
    s.link("loc", "fl")

    block(s, "afm", 1730, 234, 190, "AFM\nattention fusion", "prior", 92, 20)
    s.node("z", 1965, 247, 112, 66, "Z_afm\n80²×1", "prior", kind="tensor", fs=18)
    s.link("fe", "afm", "r", "l", via=[(1670, 179), (1670, 260)])
    s.link("fl", "afm", "r", "l", via=[(1670, 379), (1670, 300)])
    s.link("afm", "z")

    block(s, "apg", 2165, 195, 360,
          "APG-HR  (d)\nHSR features + AFM prior\n320² reconstruction",
          "hr", 170, 21)
    # Direct feature rails stay local to the fusion section.
    s.route([s.point("fe", "t"), (1566, 100), (2260, 100), (2260, 195)])
    s.route([s.point("fl", "b"), (1566, 468), (2415, 468), (2415, 365)])
    s.link("z", "apg")
    s.node("rgb2", 2305, 113, 92, 48, "RGB*", "enc", kind="port", fs=18)
    s.link("rgb2", "apg", "b", "t")

    block(s, "logit", 2580, 242, 128, "1×1 Conv\nlogits", "prior", 76, 18)
    block(s, "sig", 2748, 242, 102, "Sigmoid", "prior", 76, 18)
    s.node("mask", 2880, 247, 76, 66, "Mask\n320²", "prior", kind="tensor", fs=17)
    s.link("apg", "logit")
    s.link("logit", "sig")
    s.link("sig", "mask")

    s.label(38, 512,
            "EEB: edge enhancement block    SEU: semantic enhancement unit    ICG: image-based context guidance",
            17, 1420)
    s.label(38, 548,
            "Solid arrows: active feature flow    CAT: channel concatenation    ↑: bilinear upsampling",
            17, 1420)
    s.label(1805, 512,
            "RGB* denotes the same input image; no duplicate RGB stream is created.",
            17, 1100)
    s.label(1805, 548,
            "AFM supplies a coarse prior; APG-HR reconstructs the final 320×320 mask.",
            17, 1100)
    return s


def lmfr_panel():
    s = v4.Scene("lmfr_panel", 960, 700)
    s.frame(8, 8, 944, 684, "(b) LMFR — lightweight multiscale feature reconstruction", "fpn")

    s.label(35, 62, "Top-down reconstruction", 19, 420)
    s.node("upin", 45, 125, 145, 62, "P_(i+1)", "fpn", kind="tensor", fs=20)
    block(s, "up", 235, 125, 120, "Up ×2", "prior", 62, 19)
    s.node("li", 45, 255, 145, 62, "L_i", "fpn", kind="tensor", fs=20)
    s.plus("add", 420, 222)
    block(s, "dsr", 495, 179, 190, "DSR", "hr", 86, 24)
    s.node("pi", 748, 191, 145, 62, "P_i", "fpn", kind="tensor", fs=20)
    s.link("upin", "up")
    s.link("up", "add", "r", "t", via=[(385, 156), (420, 156)])
    s.link("li", "add", "r", "b", via=[(385, 286), (420, 286)])
    s.link("add", "dsr")
    s.link("dsr", "pi")

    s.label(35, 365, "DSR cell", 19, 300)
    block(s, "dw", 85, 427, 170, "DW 3×3", "hr", 66, 20)
    block(s, "gn1", 300, 427, 150, "GN + ReLU", "hr", 66, 18)
    block(s, "pw", 495, 427, 170, "PW 1×1", "hr", 66, 20)
    block(s, "gn2", 710, 427, 150, "GN + ReLU", "hr", 66, 18)
    for a, b in [("dw", "gn1"), ("gn1", "pw"), ("pw", "gn2")]:
        s.link(a, b)

    s.label(45, 552, "L_i = lateral 1×1 projection, 192 ch", 18, 520)
    s.label(495, 552, "i = 2, 3, 4", 18, 300)
    s.label(45, 592, "P5 = DSR(L5)", 18, 360)
    s.label(495, 592, "Output: P2–P5", 18, 360)
    return s


def hsr_panel():
    s = v4.Scene("hsr_panel", 960, 700)
    s.frame(8, 8, 944, 684, "(c) HSR — high-resolution stitch refinement", "edge")

    entries = [
        ("fe", "F_edge", "1×1 / 48", "Up ×4", "edge", 118),
        ("fl", "F_loc", "1×1 / 48", "Up ×4", "loc", 235),
        ("rgb", "RGB", "3×3 / 16", "DSR", "enc", 352),
    ]
    for key, name, proj, post, fam, y in entries:
        s.node(key, 40, y, 115, 58, name, fam, kind="tensor", fs=19)
        block(s, key + "p", 205, y, 145, proj, fam, 58, 18)
        block(s, key + "u", 402, y, 125, post, fam, 58, 18)
        s.link(key, key + "p")
        s.link(key + "p", key + "u")

    block(s, "cat", 580, 190, 104, "CAT\n112 ch", "edge", 225, 20)
    for key, *_ in entries:
        cy = s.point(key + "u", "r")[1]
        s.route([s.point(key + "u", "r"), (555, cy), (555, 302), s.point("cat", "l")])
    block(s, "fuse", 735, 198, 160, "1×1 Fuse\n112 → 48", "hr", 82, 19)
    block(s, "ref", 735, 328, 160, "DSR", "hr", 72, 22)
    block(s, "pred", 735, 448, 160, "1×1 Conv\nlogits", "prior", 76, 18)
    s.link("cat", "fuse")
    s.link("fuse", "ref", "b", "t")
    s.link("ref", "pred", "b", "t")

    s.label(40, 575, "Three complementary cues are reconstructed directly at 320×320.", 18, 850)
    return s


def apg_panel():
    s = v4.Scene("apg_panel", 960, 700)
    s.frame(8, 8, 944, 684, "(d) APG-HR — AFM-prior-guided high-resolution refinement", "prior")

    s.node("fe", 38, 105, 112, 56, "F_edge", "edge", kind="tensor", fs=18)
    s.node("fl", 38, 190, 112, 56, "F_loc", "loc", kind="tensor", fs=18)
    block(s, "afm", 205, 126, 148, "AFM", "prior", 88, 23)
    s.node("z", 402, 142, 112, 58, "Z_afm", "prior", kind="tensor", fs=18)
    block(s, "up", 560, 139, 100, "Up ×4", "prior", 64, 18)
    block(s, "sig", 705, 139, 105, "Sigmoid", "prior", 64, 18)
    s.node("p", 850, 143, 65, 56, "p", "prior", kind="tensor", fs=21)
    s.link("fe", "afm", "r", "l", via=[(178, 133), (178, 156)])
    s.link("fl", "afm", "r", "l", via=[(178, 218), (178, 184)])
    for a, b in [("afm", "z"), ("z", "up"), ("up", "sig"), ("sig", "p")]:
        s.link(a, b)

    block(s, "unc", 775, 252, 140, "u = 4p(1−p)", "fpn", 66, 19)
    s.link("p", "unc", "b", "t")

    block(s, "base", 45, 350, 300,
          "HSR feature set\n48 + 48 + 16 channels",
          "hr", 100, 19)
    block(s, "cat", 418, 337, 128, "CAT\n114 ch", "prior", 126, 21)
    block(s, "fuse", 610, 348, 135, "1×1 Fuse\n114 → 48", "hr", 82, 18)
    block(s, "ref", 795, 348, 125, "DSR", "hr", 82, 22)
    s.link("base", "cat")
    s.link("cat", "fuse")
    s.link("fuse", "ref")
    s.route([s.point("p", "b"), (882, 295), (500, 295), (500, 337)])
    s.route([s.point("unc", "b"), (845, 315), (465, 315), (465, 337)])

    block(s, "pred", 795, 498, 125, "1×1 Conv\nlogits", "prior", 76, 18)
    s.link("ref", "pred", "b", "t")
    s.label(45, 540, "p: foreground probability", 18, 380)
    s.label(435, 540, "u: boundary ambiguity", 18, 350)
    s.label(45, 590, "M6 appends only p and u; the HSR reconstruction core remains unchanged.", 18, 860)
    return s


def export():
    full = v4.Scene("APHNet_M6_innovation_focused", 3000, 1460)
    full.paste(main_flow(), 10, 10, "a_")
    full.paste(lmfr_panel(), 10, 680, "b_")
    full.paste(hsr_panel(), 1020, 680, "c_")
    full.paste(apg_panel(), 2030, 680, "d_")
    full.label(25, 1410,
               "Common operators: DW = depthwise convolution; PW = pointwise convolution; DSR = DW3×3 → PW1×1 with GN/ReLU.",
               18, 1900)

    svg_path = OUT / "APHNet_M6_innovation_focused.svg"
    png_path = OUT / "APHNet_M6_innovation_focused.png"
    v4.svg(full, svg_path)
    add_defs_and_normalize(svg_path)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                     output_width=full.w, output_height=full.h)
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    export()

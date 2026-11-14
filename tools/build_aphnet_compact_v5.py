"""Build the compact, paper-facing APHNet/M6 architecture figure.

The figure deliberately uses one dominant end-to-end panel and four compact
mechanism insets.  All dimensions and active routes are audited against the
selected M6 configuration.  Disabled centreline/topology/standalone-boundary
objectives are intentionally omitted.
"""
from pathlib import Path
import json

import cairosvg

from build_aphnet_publication_v4 import Scene, block, COL, svg


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_compact_v5"
OUT.mkdir(parents=True, exist_ok=True)


def main_architecture():
    s = Scene("panel_a_main_architecture", 1540, 900)
    s.frame(8, 8, 1524, 884, "(a) APHNet — attention-prior-guided high-resolution stitch segmentation")

    # Input and encoder: stage outputs are tensors, not extra blocks placed after C2--C5.
    s.node("rgb", 34, 94, 112, 58, "RGB\n320²×3", "enc", kind="tensor", fs=16)
    block(s, "stem", 34, 188, 112, "Stem\n3×3 / s2\n32 ch", "enc", 68, 15)
    s.link("rgb", "stem", "b", "t")

    enc = [
        ("c2", 300, "C2\n80²×48", "FusedIB"),
        ("c3", 412, "C3\n40²×80", "UIB ×2"),
        ("c4", 524, "C4\n20²×160", "UIB ×8\nMQA ×4"),
        ("c5", 648, "C5\n10²×960", "UIB ×12\nMQA ×4"),
    ]
    previous = "stem"
    for key, y, label, detail in enc:
        block(s, key + "b", 34, y - 10, 112, detail, "enc", 58 if key != "c5" else 70, 14)
        s.node(key, 178, y - 4, 96, 50, label, "enc", kind="tensor", fs=15)
        s.link(previous, key + "b", "b", "t")
        s.link(key + "b", key)
        previous = key + "b"

    # LMFR / Mobile-FPN.  P5 is refined L5; P4--P2 are top-down additions followed by DSR.
    fpn_y = {"2": 300, "3": 412, "4": 524, "5": 648}
    for n in "2345":
        y = fpn_y[n]
        block(s, "l" + n, 330, y - 7, 118, "1×1 Conv\nGN/ReLU\n→192", "fpn", 56, 14)
        s.link("c" + n, "l" + n)
        if n != "5":
            s.plus("a" + n, 488, y + 21)
            s.link("l" + n, "a" + n)
            block(s, "r" + n, 532, y - 7, 105, "DSR (b)", "hr", 56, 15)
            s.link("a" + n, "r" + n)
        else:
            block(s, "r5", 471, y - 7, 105, "DSR (b)", "hr", 56, 15)
            s.link("l5", "r5")
        px = 674
        s.node("p" + n, px, y - 4, 92, 50, f"P{n}\n{80 // (2 ** (int(n)-2))}²×192", "fpn", kind="tensor", fs=15)
        s.link("r" + n, "p" + n)

    for upper, lower in [("p5", "a4"), ("p4", "a3"), ("p3", "a2")]:
        x0, y0 = s.point(upper, "r")
        x1, y1 = s.point(lower, "b")
        bus = 795
        s.route([(x0, y0), (bus, y0), (bus, y1 + 30), (x1, y1 + 30), (x1, y1)],
                color=COL["accent"], start=(upper, "r"), end=(lower, "b"))
        s.label(bus - 72, (y0 + y1) / 2 - 6, "↑2", 14, 50)

    # Boundary path.
    s.port("ep3", 845, 112, "P3", "fpn", 72)
    block(s, "up3", 950, 106, 84, "↑2", "white", 50, 16)
    s.port("ep2", 845, 190, "P2", "fpn", 72)
    s.plus("ec", 1080, 168, "C")
    s.link("ep3", "up3")
    s.link("up3", "ec", via=[(1055, 131), (1055, 151)], sb="t")
    s.link("ep2", "ec", via=[(1055, 213), (1055, 185)], sb="b")
    block(s, "eproj", 1120, 136, 112, "3×3 Conv\n384→128", "edge", 64, 15)
    block(s, "er", 1268, 136, 126, "EdgeEnhance\n(c)", "edge", 64, 15)
    s.port("fe", 1427, 145, "F_edge", "edge", 86)
    for a, b in [("ec", "eproj"), ("eproj", "er"), ("er", "fe")]:
        s.link(a, b, color="#C65D78")

    # Contextual localization path. RGB guidance is explicit because it is active in M6.
    s.port("lp4", 845, 307, "P4", "fpn", 72)
    block(s, "seu", 948, 296, 102, "SEU\nDW/PW\nSA→ECA", "loc", 72, 14)
    block(s, "icg", 1084, 296, 112, "ICG (d)", "loc", 72, 16)
    s.port("rgbg", 1088, 232, "RGB", "enc", 76)
    s.link("rgbg", "icg", "b", "t", color="#D1815B")
    block(s, "u40", 1230, 296, 78, "↑2\n40²", "loc", 72, 14)
    block(s, "u80", 1340, 296, 78, "↑2\n80²", "loc", 72, 14)
    s.plus("la", 1453, 332)
    s.port("lp2", 1414, 398, "P2", "fpn", 78)
    s.link("lp2", "la", "t", "b", color="#8564A9")
    for a, b in [("lp4", "seu"), ("seu", "icg"), ("icg", "u40"), ("u40", "u80"), ("u80", "la")]:
        s.link(a, b, color="#8564A9")

    block(s, "lref", 1250, 408, 168, "Loc refine\n3×3: 192→128\nDSR: 128→128", "loc", 76, 14)
    s.port("fl", 1450, 421, "F_loc", "loc", 70)
    s.link("la", "lref", "b", "t", via=[(1453, 389), (1334, 389)], color="#8564A9")
    s.link("lref", "fl", color="#8564A9")

    # AFM coarse prediction and M6 final head.  Direct feature routes and the coarse-prior route
    # are separated to remove the ambiguity present in the old drawing.
    s.label(845, 520, "Attentive fusion and pixel-scale refinement", 18, 660)
    s.port("afe", 850, 580, "F_edge", "edge", 82)
    s.port("afl", 850, 660, "F_loc", "loc", 82)
    block(s, "afm", 982, 597, 130, "AFM (e)\nA(L) ⊙ E + L\n→ Z_afm", "prior", 94, 15)
    s.link("afe", "afm", via=[(955, 603), (955, 620)])
    s.link("afl", "afm", via=[(955, 683), (1047, 683)], sb="b")
    s.port("z", 1150, 621, "Z_afm", "prior", 88)
    s.link("afm", "z")

    block(s, "hr", 1268, 568, 170,
          "APG-HR (e)\nF_edge / F_loc / RGB\np, u → concat → DSR\n→ final logits", "hr", 140, 15)
    s.link("z", "hr", color=COL["accent"])
    s.route([s.point("afe", "t"), (891, 548), (1305, 548), (1305, 568)], color="#C65D78")
    s.route([s.point("afl", "b"), (891, 735), (1350, 735), (1350, 708)], color="#8564A9")
    s.port("rgbh", 1180, 770, "RGB", "enc", 78)
    s.link("rgbh", "hr", "t", "b", via=[(1219, 742), (1395, 742)], color="#D1815B")
    block(s, "mask", 1450, 606, 70, "Mask\n320²×1", "prior", 66, 15)
    s.link("hr", "mask")

    # Main-panel annotations are intentionally compact and ablation-aligned.
    s.label(330, 91, "LMFR / M0", 17, 300)
    s.label(845, 80, "Dual-path decoder", 17, 300)
    s.label(1268, 536, "M4: HSR   |   M6: + AFM prior", 15, 250)
    s.label(34, 820, "Solid lines: active prediction routes.  Named ports denote the same completed pyramid tensor.", 14, 1100)
    s.label(34, 850, "Disabled centreline, topology and standalone-boundary objectives are omitted.", 14, 1000)
    return s


def lmfr_inset():
    s = Scene("panel_b_lmfr", 770, 410)
    s.frame(8, 8, 754, 394, "(b) LMFR cell — repeated for P4, P3 and P2", "fpn")
    s.port("coarse", 32, 82, "Pᵢ₊₁", "fpn", 78)
    block(s, "up", 143, 77, 86, "Bilinear ↑2", "white", 54, 14)
    s.port("lat", 32, 170, "Lᵢ", "fpn", 78)
    s.plus("add", 285, 139)
    s.link("coarse", "up")
    s.link("up", "add", via=[(258, 104), (258, 122)], sb="t")
    s.link("lat", "add", via=[(258, 193), (258, 156)], sb="b")
    block(s, "dw", 342, 111, 135, "DW 3×3\nGN → ReLU", "hr", 62, 15)
    block(s, "pw", 510, 111, 135, "PW 1×1\nGN → ReLU", "hr", 62, 15)
    s.port("out", 678, 120, "Pᵢ", "fpn", 62)
    for a, b in [("add", "dw"), ("dw", "pw"), ("pw", "out")]:
        s.link(a, b)
    s.label(32, 260, "P5 = DSR(L5)", 15, 220)
    s.label(263, 260, "P4 = DSR(L4 + ↑P5)", 15, 260)
    s.label(32, 302, "P3 = DSR(L3 + ↑P4)", 15, 260)
    s.label(330, 302, "P2 = DSR(L2 + ↑P3)", 15, 260)
    s.label(32, 351, "One shared topology; different learned weights at each level.", 14, 680)
    return s


def edge_inset():
    s = Scene("panel_c_edge", 770, 450)
    s.frame(8, 8, 754, 434, "(c) EdgeEnhance — complementary boundary evidence", "edge")
    s.port("x", 28, 183, "X\n128 ch", "edge", 72)
    branches = [
        (66, "Context", "DW 3×3 → PW 1×1\n128→128"),
        (166, "Gradient", "Sobel X/Y Conv 3×3\n128→256 → PW1→32"),
        (274, "Curvature", "Laplacian Conv 3×3\n128→128"),
    ]
    for i, (y, name, ops) in enumerate(branches):
        block(s, f"b{i}", 137, y, 154, name, "white", 48, 15)
        block(s, f"o{i}", 326, y - 7, 210, ops + "\nGN/ReLU", "white", 64, 14)
        s.link("x", f"b{i}", via=[(114, 210), (114, y + 24)])
        s.link(f"b{i}", f"o{i}")
    s.node("cat", 574, 61, 38, 295, "C\no\nn\nc\na\nt", "white", fs=15)
    for i, (y, _, _) in enumerate(branches):
        s.route([s.point(f"o{i}", "r"), (574, y + 25)])
    block(s, "mix", 646, 91, 92, "PW 1×1\n288→128", "edge", 60, 14)
    block(s, "eca", 646, 184, 92, "ECA\nk=5", "edge", 60, 14)
    s.plus("add", 692, 305)
    s.link("cat", "mix")
    s.link("mix", "eca", "b", "t")
    s.link("eca", "add", "b", "t")
    s.link("x", "add", "b", "l", via=[(64, 366), (650, 366)])
    s.label(627, 382, "ReLU → F_edge", 14, 130)
    return s


def icg_inset():
    s = Scene("panel_d_icg", 780, 500)
    s.frame(8, 8, 764, 484, "(d) ICG — RGB-conditioned foreground/background contrast", "loc")
    s.port("rgb", 28, 66, "RGB", "enc", 70)
    block(s, "cnn", 132, 56, 175, "Resize 20²\nConv 3×3: 3→16→32", "enc", 66, 14)
    s.port("v", 346, 67, "V\n32 ch", "enc", 72)
    s.link("rgb", "cnn"); s.link("cnn", "v")
    s.port("f", 28, 166, "F_seu\n192 ch", "loc", 82)
    block(s, "p", 150, 157, 178, "Conv 1×1 → Sigmoid\nP_init", "loc", 64, 14)
    s.link("f", "p")

    block(s, "fg", 55, 275, 158, "V ⊙ P_init\nFG: DW5 → PW1", "white", 62, 14)
    block(s, "bg", 239, 275, 158, "V ⊙ (1−P_init)\nBG: DW5 → PW1", "white", 62, 14)
    block(s, "gb", 423, 275, 158, "P_init < 0.1\nmasked BG mean", "white", 62, 14)
    s.plus("cat", 616, 306, "C")
    s.link("bg", "cat", via=[(410, 306), (599, 306)])
    s.link("gb", "cat")
    block(s, "dif", 659, 267, 90, "|FG−BG|\nPW1→G", "loc", 78, 14)
    s.link("fg", "dif", via=[(222, 306), (222, 246), (704, 246), (704, 267)], sb="t")
    s.link("cat", "dif")

    block(s, "g1", 210, 393, 150, "Attention gate 1", "loc", 54, 15)
    block(s, "g2", 423, 393, 150, "Attention gate 2", "loc", 54, 15)
    s.port("out", 628, 397, "F_icg", "loc", 92)
    s.link("f", "g1", "b", "l", via=[(69, 365), (185, 365), (185, 420)])
    s.link("dif", "g1", "b", "t", via=[(704, 372), (285, 372)])
    s.route([(498, 372), (498, 393)], start=("dif", "b"), end=("g2", "t")); s.dot(498, 372)
    s.link("g1", "g2"); s.link("g2", "out")
    s.label(28, 458, "Both gates use the same contrast guidance G but have independent parameters.", 14, 720)
    return s


def afm_highres_inset():
    s = Scene("panel_e_afm_highres", 1540, 500)
    s.frame(8, 8, 1524, 484, "(e) AFM-guided HighRes — M4 reconstruction plus the M6 prior path", "hr")

    # AFM on the left.
    s.port("fe", 30, 92, "F_edge", "edge", 88)
    block(s, "e", 153, 84, 122, "1×1 Conv\n→ E (128)", "edge", 62, 14)
    s.port("fl", 30, 200, "F_loc", "loc", 88)
    block(s, "l", 153, 192, 122, "1×1 Conv\n→ L (128)", "loc", 62, 14)
    block(s, "att", 318, 179, 155, "3×3: 128→64\n1×1: 64→1\nSigmoid → A", "loc", 88, 14)
    s.plus("mul", 536, 119, "×")
    s.plus("cat", 536, 244, "C")
    s.link("fe", "e"); s.link("fl", "l"); s.link("l", "att")
    s.link("e", "mul")
    s.link("att", "mul", "t", "b", via=[(395, 143), (536, 143)])
    s.link("mul", "cat", "b", "t", color="#C65D78")
    s.link("l", "cat", via=[(294, 223), (294, 244)])
    block(s, "afmr", 586, 216, 146, "3×3 Conv\n256→128", "prior", 62, 14)
    block(s, "afmp", 768, 216, 112, "1×1 Conv\n→ Z_afm", "prior", 62, 14)
    s.link("cat", "afmr"); s.link("afmr", "afmp")

    # M6 prior path is highlighted above the reconstruction inputs.
    block(s, "prob", 914, 80, 130, "↑4 → Sigmoid\np", "prior", 62, 14)
    block(s, "amb", 1082, 80, 130, "4p(1−p)\nu", "fpn", 62, 14)
    s.link("afmp", "prob", "t", "l", via=[(824, 170), (895, 170), (895, 111)], color=COL["accent"])
    s.link("prob", "amb", color=COL["accent"])

    # HSR / HighRes reconstruction route.  Three feature projections plus p and u.
    entries = [
        ("he", 317, "F_edge\n128→48 → ↑4", "edge"),
        ("hl", 382, "F_loc\n128→48 → ↑4", "loc"),
        ("hi", 447, "RGB\n3→16 → DSR", "enc"),
    ]
    for key, y, label, col in entries:
        block(s, key, 914, y, 178, label, col, 48, 14)
    s.node("hcat", 1136, 292, 38, 190, "C\no\nn\nc\na\nt", "white", fs=14)
    for key, y, _, _ in entries:
        s.route([s.point(key, "r"), (1136, y + 24)], start=(key, "r"), end=("hcat", "l"))
    s.route([s.point("prob", "b"), (979, 260), (1155, 260), (1155, 292)], color=COL["accent"])
    s.route([s.point("amb", "b"), (1147, 278), (1168, 278), (1168, 292)], color=COL["accent"])
    block(s, "fuse", 1212, 319, 132, "1×1: 114→48\nGN/ReLU", "hr", 66, 14)
    block(s, "dsr", 1382, 319, 126, "DSR 48→48\n1×1 → logits", "hr", 66, 14)
    s.link("hcat", "fuse"); s.link("fuse", "dsr")
    s.label(902, 34, "M6 prior", 15, 200)
    s.label(902, 284, "M4 HighRes reconstruction", 15, 270)
    s.label(30, 434, "AFM: localization generates A; only the gated edge E_g is concatenated with L.", 14, 760)
    s.label(828, 434, "M6 adds p and ambiguity u to the M4 five-source reconstruction input.", 14, 680)
    return s


def build():
    panels = [main_architecture(), lmfr_inset(), edge_inset(), icg_inset(), afm_highres_inset()]
    full = Scene("APHNet_M6_compact_publication", 2360, 1435)
    full.paste(panels[0], 10, 10, "a_")
    full.paste(panels[1], 1575, 10, "b_")
    full.paste(panels[2], 1575, 435, "c_")
    full.paste(panels[3], 10, 920, "d_")
    full.paste(panels[4], 810, 920, "e_")

    for scene in [full, *panels]:
        svg_path = OUT / f"{scene.name}.svg"
        svg(scene, svg_path)
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(svg_path.with_suffix(".pdf")))
        png_width = 4720 if scene is full else max(scene.w * 2, 1800)
        cairosvg.svg2png(url=str(svg_path), write_to=str(svg_path.with_suffix(".png")), output_width=png_width)

    (OUT / "editable_scene.json").write_text(
        json.dumps([{"name": p.name, "width": p.w, "height": p.h, "items": p.items} for p in [full, *panels]],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()

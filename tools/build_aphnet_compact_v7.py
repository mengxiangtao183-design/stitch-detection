"""APHNet/M6 publication diagram v7.

Layout: one fully connected main pipeline across the top and four compact
operator insets below.  Main-flow boxes intentionally contain only short
module names and essential tensor sizes.
"""
from pathlib import Path
import json
import cairosvg
from PIL import Image

import build_aphnet_compact_v6 as v6
from build_aphnet_publication_v4 import Scene, block, COL, svg


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_compact_v7"
OUT.mkdir(parents=True, exist_ok=True)
P = v6.P


def box(s, key, x, y, w, text, fill, h=52, fs=23):
    return block(s, key, x, y, w, text, fill, h, fs)


def tensor(s, key, x, y, w, text, fill, h=54, fs=22):
    return s.node(key, x, y, w, h, text, fill, kind="tensor", fs=fs)


def hero():
    """A single, explicitly connected end-to-end APHNet dataflow."""
    s = Scene("01_main_network", 2180, 650)
    s.frame(6, 6, 2168, 638, "(a) APHNet / M6", "white")

    # Section labels.
    s.label(24, 48, "Input / encoder", 25, 260)
    s.label(334, 48, "LMFR / Mobile-FPN", 25, 360)
    s.label(1030, 48, "Dual-path decoder", 25, 340)
    s.label(1638, 48, "Fusion & refinement", 25, 430)

    # Input and encoder ladder.
    tensor(s, "rgb", 24, 92, 96, "RGB\n320²×3", P["enc2"], 58, 21)
    box(s, "stem", 148, 96, 92, "Stem", P["enc2"], 50, 23)
    s.link("rgb", "stem")
    rows = [
        ("2", 148, "C2", "80²×48"),
        ("3", 258, "C3", "40²×80"),
        ("4", 368, "C4", "20²×160"),
        ("5", 478, "C5", "10²×960"),
    ]
    prev = "stem"
    for n, y, name, dim in rows:
        tensor(s, "c" + n, 148, y, 106, f"{name}\n{dim}", P["enc2"], 58, 21)
        if n == "2":
            s.link(prev, "c" + n, "b", "t", via=[(194, 130), (194, 138)])
        else:
            s.link(prev, "c" + n, "b", "t")
        prev = "c" + n

    # Four aligned LMFR cells. P5 is the seed; P5->P4->P3->P2 uses one clear bus.
    for n, y, _, dim in rows:
        box(s, "l" + n, 334, y + 3, 78, f"L{n}", P["fpn2"], 52, 23)
        s.link("c" + n, "l" + n)
        if n != "5":
            s.plus("add" + n, 468, y + 29)
            s.link("l" + n, "add" + n)
            x_dsr = 508
        else:
            x_dsr = 452
        box(s, "dsr" + n, x_dsr, y + 3, 84, "DSR", P["hr2"], 52, 22)
        if n != "5":
            s.link("add" + n, "dsr" + n)
        else:
            s.link("l" + n, "dsr" + n)
        tensor(s, "p" + n, 628, y, 98, f"P{n}\n{dim.split('×')[0]}²×192", P["fpn2"], 58, 21)
        s.link("dsr" + n, "p" + n)

    # Top-down feature rail, kept entirely in the empty corridor.
    rail_x = 770
    for src, dst in [("p5", "add4"), ("p4", "add3"), ("p3", "add2")]:
        x0, y0 = s.point(src, "r")
        x1, y1 = s.point(dst, "b")
        s.route([(x0, y0), (rail_x, y0), (rail_x, y1 + 24), (x1, y1 + 24), (x1, y1)],
                color=COL["accent"], start=(src, "r"), end=(dst, "b"))

    # Boundary pathway: P3 and P2 are connected directly from LMFR.
    box(s, "ejoin", 1042, 142, 126, "P3↑2 + P2", P["edge2"], 54, 22)
    box(s, "econv", 1206, 142, 86, "Conv", P["edge2"], 54, 22)
    box(s, "eeb", 1330, 142, 92, "EEB (c)", P["edge2"], 54, 21)
    tensor(s, "fe", 1460, 142, 98, "F_edge", P["edge2"], 54, 21)
    for a, b in [("ejoin", "econv"), ("econv", "eeb"), ("eeb", "fe")]:
        s.link(a, b, color="#B75070")
    s.route([s.point("p3", "r"), (840, 287), (840, 160), s.point("ejoin", "l")],
            color="#B75070", start=("p3", "r"), end=("ejoin", "l"))
    s.route([s.point("p2", "r"), (812, 177), (812, 184), s.point("ejoin", "l")],
            color="#B75070", start=("p2", "r"), end=("ejoin", "l"))

    # Localization pathway: P4 supplies semantics; P2 restores detail.
    box(s, "seu", 920, 358, 82, "SEU", P["loc2"], 54, 22)
    box(s, "icg", 1038, 358, 94, "ICG (d)", P["loc2"], 54, 21)
    box(s, "lup", 1168, 358, 82, "Up4x", P["loc2"], 54, 21)
    box(s, "ladd", 1286, 358, 98, "+ P2", P["loc2"], 54, 22)
    box(s, "lref", 1420, 358, 94, "LocRef", P["loc2"], 54, 21)
    tensor(s, "fl", 1548, 358, 88, "F_loc", P["loc2"], 54, 21)
    for a, b in [("seu", "icg"), ("icg", "lup"), ("lup", "ladd"), ("ladd", "lref"), ("lref", "fl")]:
        s.link(a, b, color="#785AA0")
    s.route([s.point("p4", "r"), (836, 397), s.point("seu", "l")],
            color="#785AA0", start=("p4", "r"), end=("seu", "l"))
    s.route([s.point("p2", "r"), (800, 177), (800, 440), (1335, 440), s.point("ladd", "b")],
            color="#785AA0", start=("p2", "r"), end=("ladd", "b"))

    # Fusion/refinement: both decoder outputs enter AFM and also feed APG-HR.
    box(s, "afm", 1690, 244, 106, "AFM (e)", P["prior2"], 64, 23)
    tensor(s, "z", 1832, 249, 88, "Z_afm", P["prior2"], 54, 21)
    box(s, "apg", 1954, 225, 112, "APG-HR\n(e)", P["hr2"], 100, 22)
    tensor(s, "mask", 2092, 248, 70, "Mask", P["prior2"], 58, 21)
    s.route([s.point("fe", "r"), (1622, 169), (1622, 267), s.point("afm", "l")],
            color="#B75070", start=("fe", "r"), end=("afm", "l"))
    s.route([s.point("fl", "r"), (1660, 385), (1660, 285), s.point("afm", "l")],
            color="#785AA0", start=("fl", "r"), end=("afm", "l"))
    s.link("afm", "z", color=COL["accent"])
    s.link("z", "apg", color=COL["accent"])
    s.link("apg", "mask", color=COL["accent"])
    s.route([s.point("fe", "b"), (1509, 332), (1982, 332), (1982, 325)], color="#B75070")
    s.route([s.point("fl", "b"), (1592, 470), (2028, 470), (2028, 325)], color="#785AA0")

    # One unobtrusive RGB guidance rail along the bottom.
    s.route([s.point("rgb", "b"), (72, 604), (1085, 604), (1085, 412)],
            color="#C9734D", dash=True, arrow=True)
    s.route([(1085, 604), (2010, 604), (2010, 325)], color="#C9734D", dash=True, arrow=True)
    s.label(82, 574, "RGB guidance", 21, 190, color="#C9734D")

    # Compact legend only; detailed operators are delegated to panels b-e.
    legend = [(P["enc2"], "Enc"), (P["fpn2"], "LMFR"), (P["edge2"], "Edge"),
              (P["loc2"], "Loc"), (P["hr2"], "Refine")]
    x = 1600
    for i, (fill, label) in enumerate(legend):
        s.node(f"leg{i}", x, 558, 25, 18, "", fill, kind="box", fs=20)
        s.label(x + 32, 555, label, 20, 90)
        x += 108
    s.label(1600, 592, "↑: bilinear upsample   +: addition", 20, 500)
    return s


def compact_icg():
    s = Scene("04_icg", 560, 470)
    s.frame(6, 6, 548, 458, "(d) ICG", "loc")
    s.port("rgb", 22, 60, "RGB", P["enc2"], 64)
    box(s, "cnn", 112, 60, 112, "Conv×2", P["enc2"], 48, 21)
    tensor(s, "v", 254, 60, 60, "V", P["enc2"], 48, 22)
    s.link("rgb", "cnn"); s.link("cnn", "v")
    s.port("f", 22, 142, "F_seu", P["loc2"], 78)
    box(s, "pin", 126, 142, 100, "P_init", P["loc2"], 48, 21)
    s.link("f", "pin")
    box(s, "fg", 54, 236, 112, "FG mean", "#DCD0EC", 48, 20)
    box(s, "bg", 200, 236, 132, "BG mean", "#DCD0EC", 48, 20)
    box(s, "g", 374, 236, 132, "|FG−BG|", P["loc2"], 48, 20)
    s.route([s.point("v", "b"), (284, 204), (110, 204), s.point("fg", "t")], end=("fg", "t"))
    s.route([s.point("v", "b"), (284, 204), (266, 204), s.point("bg", "t")], end=("bg", "t"))
    s.route([s.point("pin", "b"), (176, 212), (110, 212), s.point("fg", "t")], color="#785AA0")
    s.route([s.point("pin", "b"), (176, 212), (266, 212), s.point("bg", "t")], color="#785AA0")
    s.link("fg", "g"); s.link("bg", "g")
    box(s, "ag1", 72, 354, 112, "AG-1", "#C4B1DF", 50, 22)
    box(s, "ag2", 228, 354, 112, "AG-2", "#C4B1DF", 50, 22)
    s.port("out", 386, 354, "F_icg", P["loc2"], 84)
    s.link("f", "ag1", "b", "l", via=[(61, 330), (52, 330), (52, 379)], color="#785AA0")
    s.route([s.point("g", "b"), (440, 328), (128, 328), (128, 354)], color="#785AA0")
    s.route([(284, 328), (284, 354)], color="#785AA0"); s.dot(284, 328)
    s.link("ag1", "ag2", color="#785AA0"); s.link("ag2", "out", color="#785AA0")
    s.label(26, 424, "P-weighted FG/BG; shared G", 20, 450)
    return s


def wide_afm_hr():
    s = Scene("05_afm_apghr", 2180, 480)
    s.frame(6, 6, 2168, 468, "(e) AFM-guided high-resolution refinement", "hr")
    s.label(28, 50, "AFM", 25, 180)
    s.port("fe", 28, 96, "F_edge", P["edge2"], 86)
    box(s, "ep", 148, 96, 88, "Proj E", P["edge2"], 50, 21)
    s.port("fl", 28, 198, "F_loc", P["loc2"], 86)
    box(s, "lp", 148, 198, 88, "Proj L", P["loc2"], 50, 21)
    box(s, "att", 286, 186, 100, "Attn", P["loc2"], 58, 22)
    s.plus("mul", 438, 121, "×"); s.plus("cat", 438, 225, "C")
    s.link("fe", "ep"); s.link("fl", "lp"); s.link("lp", "att")
    s.link("ep", "mul"); s.link("att", "mul", "t", "b", via=[(336, 154), (438, 154)])
    s.link("mul", "cat", "b", "t", color="#B75070")
    s.link("lp", "cat", via=[(260, 223), (260, 225)])
    box(s, "ref", 492, 198, 96, "Refine", P["prior2"], 54, 22)
    tensor(s, "z", 630, 198, 88, "Z_afm", P["prior2"], 54, 21)
    s.link("cat", "ref"); s.link("ref", "z")

    s.label(810, 50, "Probability / ambiguity prior", 25, 430)
    box(s, "p", 814, 116, 100, "Up4x → p", P["prior2"], 52, 20)
    box(s, "u", 960, 116, 108, "4p(1−p)", P["fpn2"], 52, 20)
    s.route([s.point("z", "r"), (766, 225), (766, 142), s.point("p", "l")], color=COL["accent"])
    s.link("p", "u", color=COL["accent"])

    s.label(1168, 50, "Pixel-scale reconstruction", 25, 430)
    sources = [("he", 96, "F_edge↑4", P["edge2"]), ("hl", 176, "F_loc↑4", P["loc2"]),
               ("hi", 256, "RGB feat", P["enc2"])]
    for key, y, label, fill in sources:
        box(s, key, 1172, y, 126, label, fill, 50, 20)
    s.node("hc", 1350, 82, 42, 256, "C", P["prior2"], kind="port", fs=26)
    for key, _, _, _ in sources:
        s.link(key, "hc")
    s.route([s.point("p", "b"), (864, 302), (1366, 302), (1366, 338)], color=COL["accent"])
    s.route([s.point("u", "b"), (1014, 324), (1382, 324), (1382, 338)], color=COL["accent"])
    box(s, "fuse", 1450, 180, 110, "Fuse 1×1", P["hr2"], 56, 21)
    box(s, "dsr", 1610, 180, 100, "DSR", P["hr2"], 56, 22)
    box(s, "pred", 1760, 180, 104, "Pred 1×1", P["prior2"], 56, 21)
    tensor(s, "mask", 1918, 180, 88, "Mask", P["prior2"], 56, 21)
    for a, b in [("hc", "fuse"), ("fuse", "dsr"), ("dsr", "pred"), ("pred", "mask")]:
        s.link(a, b)
    s.label(28, 390, "A ← L;  E_g = A × E;  Z_afm ← concat(E_g, L)", 21, 690)
    s.label(1168, 390, "M6 adds p and u to the M4 high-resolution inputs", 21, 700)
    return s


def build():
    panels = [hero(), v6.lmfr(), v6.edge(), compact_icg(), wide_afm_hr()]
    full = Scene("APHNet_M6_publication_v7", 2200, 1635)
    full.paste(panels[0], 10, 10, "a_")
    full.paste(panels[1], 10, 670, "b_")
    full.paste(panels[2], 820, 670, "c_")
    full.paste(panels[3], 1630, 670, "d_")
    full.paste(panels[4], 10, 1155, "e_")

    # At 183 mm final width, 22 px on the 2200 px canvas is about 5.2 pt.
    for sc in [full, *panels]:
        for item in sc.items:
            if item.get("fs", 22) < 22:
                item["fs"] = 22

    for sc in [full, *panels]:
        path = OUT / f"{sc.name}.svg"
        svg(sc, path)
        kwargs = {"output_width": 183 / 25.4 * 96} if sc is full else {}
        cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix(".pdf")), **kwargs)
        cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix(".png")),
                         output_width=4400 if sc is full else max(sc.w * 2, 1800))

    with Image.open(OUT / "APHNet_M6_publication_v7.png") as im:
        im.save(OUT / "APHNet_M6_publication_v7_600dpi.tiff", dpi=(600, 600), compression="tiff_lzw")
    v6.export_vsdx(full, OUT / "APHNet_M6_publication_v7_editable.vsdx")
    (OUT / "editable_scene.json").write_text(
        json.dumps([{"name": x.name, "width": x.w, "height": x.h, "items": x.items}
                    for x in [full, *panels]], ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()

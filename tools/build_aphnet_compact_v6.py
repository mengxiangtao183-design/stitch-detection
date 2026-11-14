"""Compact publication diagram for APHNet/M6.

One hero architecture plus four compact insets.  Text is kept above a 5 pt
floor at 183 mm final width.  White operator boxes are intentionally avoided.
"""
from pathlib import Path
import json
import cairosvg
import xml.etree.ElementTree as ET
import shutil
import vsdx
from PIL import Image

from build_aphnet_publication_v4 import Scene, block, COL, svg

OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_compact_v6"
OUT.mkdir(parents=True, exist_ok=True)

P = {
    "enc2": "#F1C9B2", "fpn2": "#F0D77D", "edge2": "#EFAFC0",
    "loc2": "#CDBDE5", "hr2": "#B9DDB8", "prior2": "#AFCFE5",
    "neutral": "#E7EDF1", "ink": "#263445",
}


def box(s, key, x, y, w, text, fill, h=54, fs=24):
    return block(s, key, x, y, w, text, fill, h, fs)


def tensor(s, key, x, y, w, text, fill, h=54, fs=24):
    return s.node(key, x, y, w, h, text, fill, kind="tensor", fs=fs)


def hero():
    s = Scene("01_main_network", 1370, 795)
    s.frame(6, 6, 1358, 783, "(a) APHNet / M6", "white")
    s.label(24, 50, "MobileNetV4 encoder", 23, 250)
    s.label(300, 50, "LMFR / Mobile-FPN", 23, 300)
    s.label(758, 50, "Dual-path decoder", 23, 300)

    tensor(s, "rgb", 28, 78, 104, "RGB\n320²×3", P["enc2"], 58, 22)
    box(s, "stem", 28, 162, 104, "Stem", P["enc2"], 50, 24)
    s.link("rgb", "stem", "b", "t")

    rows = [
        ("2", 260, "S2", "C2", "80²×48"),
        ("3", 355, "S3", "C3", "40²×80"),
        ("4", 450, "S4", "C4", "20²×160"),
        ("5", 545, "S5", "C5", "10²×960"),
    ]
    prev = "stem"
    for n, y, st, ct, dim in rows:
        box(s, "s" + n, 28, y, 104, st, P["enc2"], 50, 24)
        tensor(s, "c" + n, 162, y - 2, 92, f"{ct}\n{dim}", P["enc2"], 54, 21)
        box(s, "l" + n, 300, y, 82, f"L{n}", P["fpn2"], 50, 24)
        s.link(prev, "s" + n, "b", "t")
        s.link("s" + n, "c" + n)
        s.link("c" + n, "l" + n)
        if n == "5":
            box(s, "dsr" + n, 431, y, 91, "DSR", P["hr2"], 50, 23)
            s.link("l" + n, "dsr" + n)
        else:
            s.plus("add" + n, 428, y + 25)
            box(s, "dsr" + n, 468, y, 91, "DSR", P["hr2"], 50, 23)
            s.link("l" + n, "add" + n)
            s.link("add" + n, "dsr" + n)
        tensor(s, "p" + n, 596, y - 2, 86, f"P{n}\n{dim.split('×')[0]}×192", P["fpn2"], 54, 21)
        s.link("dsr" + n, "p" + n)
        prev = "s" + n

    # Clean top-down bus: all bends live in the empty corridor x=712.
    for src, dst in [("p5", "add4"), ("p4", "add3"), ("p3", "add2")]:
        x0, y0 = s.point(src, "r"); x1, y1 = s.point(dst, "b")
        bus = 716
        s.route([(x0, y0), (bus, y0), (bus, y1 + 22), (x1, y1 + 22), (x1, y1)],
                color=COL["accent"], start=(src, "r"), end=(dst, "b"))

    # Boundary path.
    s.port("ep3", 760, 98, "P3", P["fpn2"], 68)
    box(s, "eup", 857, 98, 78, "Up2x", P["neutral"], 46, 21)
    s.port("ep2", 760, 168, "P2", P["fpn2"], 68)
    s.plus("ec", 978, 145, "C")
    s.link("ep3", "eup")
    s.link("eup", "ec", via=[(953, 121), (953, 128)], sb="t")
    s.link("ep2", "ec", via=[(953, 191), (953, 162)], sb="b")
    box(s, "estem", 1016, 119, 100, "3×3 Conv", P["edge2"], 52, 21)
    box(s, "eeb", 1144, 119, 100, "EEB (c)", P["edge2"], 52, 21)
    s.port("fe", 1270, 122, "F_edge", P["edge2"], 82)
    for a, b in [("ec", "estem"), ("estem", "eeb"), ("eeb", "fe")]:
        s.link(a, b, color="#B75070")

    # Localization path.
    s.port("lp4", 760, 278, "P4", P["fpn2"], 68)
    box(s, "seu", 855, 278, 86, "SEU", P["loc2"], 50, 23)
    box(s, "icg", 970, 278, 92, "ICG (d)", P["loc2"], 50, 22)
    s.port("rgbg", 970, 217, "RGB", P["enc2"], 78)
    s.link("rgbg", "icg", "b", "t", color="#C9734D")
    box(s, "u1", 1090, 278, 70, "Up2x", P["loc2"], 50, 20)
    box(s, "u2", 1188, 278, 70, "Up2x", P["loc2"], 50, 20)
    s.plus("la", 1300, 303)
    s.port("lp2", 1292, 352, "P2", P["fpn2"], 62)
    s.link("lp2", "la", "t", "b", color="#785AA0")
    for a, b in [("lp4", "seu"), ("seu", "icg"), ("icg", "u1"), ("u1", "u2"), ("u2", "la")]:
        s.link(a, b, color="#785AA0")
    box(s, "lref", 1110, 401, 145, "LocRef", P["loc2"], 52, 23)
    s.port("fl", 1284, 404, "F_loc", P["loc2"], 70)
    s.link("la", "lref", "b", "t", via=[(1300, 382), (1182, 382)], color="#785AA0")
    s.link("lref", "fl", color="#785AA0")

    # Fusion and high-resolution refinement.  Direct routes use dedicated upper/lower rails.
    s.label(758, 484, "AFM-guided high-resolution decoding", 26, 560)
    s.port("afe", 760, 544, "F_edge", P["edge2"], 80)
    s.port("afl", 760, 628, "F_loc", P["loc2"], 80)
    box(s, "afm", 886, 576, 100, "AFM (e)", P["prior2"], 62, 23)
    s.link("afe", "afm", via=[(860, 567), (860, 592)])
    s.link("afl", "afm", via=[(860, 651), (936, 651)], sb="b")
    tensor(s, "z", 1020, 580, 86, "Z_afm", P["prior2"], 54, 22)
    s.link("afm", "z")
    box(s, "apg", 1160, 548, 118, "APG-HR\n(e)", P["hr2"], 96, 23)
    s.link("z", "apg", color=COL["accent"])
    # branch rails to APG-HR, separated from AFM route
    s.route([s.point("afe", "t"), (800, 518), (1189, 518), (1189, 548)], color="#B75070")
    s.route([s.point("afl", "b"), (800, 692), (1220, 692), (1220, 644)], color="#785AA0")
    s.port("rgbh", 1040, 704, "RGB", P["enc2"], 76)
    s.link("rgbh", "apg", "t", "b", via=[(1078, 684), (1248, 684)], color="#C9734D")
    tensor(s, "mask", 1306, 569, 56, "Mask", P["prior2"], 58, 20)
    s.link("apg", "mask")

    # Compact legend / notation, per user request.
    s.label(24, 658, "Stages: S2 FusedIB | S3 UIB×2 | S4 UIB×8 + MQA×4", 22, 700)
    s.label(24, 686, "S5 UIB×12 + MQA×4", 22, 300)
    y = 716
    legend = [(P["enc2"], "Encoder"), (P["fpn2"], "LMFR"), (P["edge2"], "Edge"),
              (P["loc2"], "Location"), (P["hr2"], "Refine")]
    x = 24
    for i, (fill, label) in enumerate(legend):
        s.node(f"leg{i}", x, y, 28, 20, "", fill, kind="box", fs=20)
        s.label(x + 38, y - 1, label, 20, 100)
        x += 136
    s.label(24, 748, "C: concat   +: add   DSR: DW3×3 → PW1×1   Up2x: bilinear", 22, 720)
    return s


def lmfr():
    s = Scene("02_lmfr_cell", 800, 300)
    s.frame(6, 6, 788, 288, "(b) LMFR cell", "fpn")
    s.port("pc", 28, 70, "P(i+1)", P["fpn2"], 92)
    box(s, "up", 158, 70, 82, "Up2x", P["neutral"], 48, 21)
    s.port("li", 28, 154, "L_i", P["fpn2"], 92)
    s.plus("add", 298, 132)
    box(s, "dw", 358, 105, 116, "DW3×3", P["hr2"], 54, 23)
    box(s, "pw", 512, 105, 116, "PW1×1", P["hr2"], 54, 23)
    s.port("pi", 674, 109, "P_i", P["fpn2"], 76)
    s.link("pc", "up")
    s.link("up", "add", via=[(270, 94), (270, 115)], sb="t")
    s.link("li", "add", via=[(270, 177), (270, 149)], sb="b")
    for a, b in [("add", "dw"), ("dw", "pw"), ("pw", "pi")]: s.link(a, b)
    s.label(28, 225, "GN + ReLU after DW/PW", 22, 330)
    s.label(390, 225, "i = 2, 3, 4;  P5 = DSR(L5)", 22, 370)
    return s


def edge():
    s = Scene("03_edge_enhance", 800, 470)
    s.frame(6, 6, 788, 458, "(c) EdgeEnhance block", "edge")
    s.port("x", 28, 198, "X", P["edge2"], 58)
    # No white boxes: each lane is a compact, colored operator stack.
    lanes = [
        (66, "DW3×3", "PW1×1", "128"),
        (176, "Sobel X/Y", "PW1×1", "32"),
        (286, "Laplacian", "Identity", "128"),
    ]
    for i, (y, a, b, c) in enumerate(lanes):
        box(s, f"a{i}", 126, y, 142, a, P["edge2"], 54, 22)
        box(s, f"b{i}", 308, y, 126, b, "#F4C9D4", 54, 22)
        tensor(s, f"t{i}", 470, y, 74, c + " ch", "#F7DCE3", 54, 20)
        s.link("x", f"a{i}", via=[(98, 225), (98, y + 27)])
        s.link(f"a{i}", f"b{i}"); s.link(f"b{i}", f"t{i}")
    s.node("cat", 582, 54, 42, 318, "C", P["prior2"], kind="port", fs=26)
    for i, (y, _, _, _) in enumerate(lanes):
        s.route([s.point(f"t{i}", "r"), (582, y + 27)])
    box(s, "mix", 662, 100, 104, "PW1×1", P["edge2"], 54, 22)
    box(s, "eca", 662, 190, 104, "ECA", P["edge2"], 54, 22)
    s.plus("add", 714, 306)
    s.link("cat", "mix"); s.link("mix", "eca", "b", "t"); s.link("eca", "add", "b", "t")
    s.link("x", "add", "b", "l", via=[(57, 389), (676, 389)])
    s.label(28, 420, "Conv → GN/ReLU;  ECA k=5;  skip + ReLU", 21, 610)
    return s


def icg():
    s = Scene("04_icg", 780, 490)
    s.frame(6, 6, 768, 478, "(d) ICG", "loc")
    s.port("rgb", 25, 62, "RGB", P["enc2"], 68)
    box(s, "cnn", 128, 62, 126, "Conv3×3 ×2", P["enc2"], 50, 21)
    tensor(s, "v", 292, 62, 76, "V", P["enc2"], 50, 23)
    s.link("rgb", "cnn"); s.link("cnn", "v")
    s.port("f", 25, 146, "F_seu", P["loc2"], 82)
    box(s, "pin", 158, 146, 118, "P_init", P["loc2"], 50, 22)
    s.link("f", "pin")

    box(s, "route", 316, 158, 132, "V, P_init", "#DCD0EC", 46, 20)
    s.link("v", "route", "b", "t", via=[(330, 132), (382, 132)])
    s.link("pin", "route")

    labels = [(35, 248, "V × P", "FG"), (235, 248, "V × (1-P)", "BG"), (435, 248, "P < 0.1", "BG mean")]
    for i, (x, y, top, bottom) in enumerate(labels):
        box(s, f"m{i}", x, y, 142, top, "#DCD0EC", 46, 20)
        box(s, f"q{i}", x, y + 62, 142, bottom, P["loc2"], 46, 21)
        s.link(f"m{i}", f"q{i}", "b", "t")
        s.route([s.point("route", "b"), (382, 226), (x + 71, 226), (x + 71, y)],
                color="#785AA0", start=("route", "b"), end=(f"m{i}", "t"))
    s.plus("cat", 626, 334, "C")
    s.link("q1", "cat"); s.link("q2", "cat")
    box(s, "g", 677, 306, 78, "|FG-BG|\n→ G", P["loc2"], 70, 20)
    s.link("q0", "g", via=[(192, 333), (192, 226), (716, 226), (716, 306)], sb="t")
    s.link("cat", "g")
    box(s, "ag1", 164, 403, 126, "AG-1", "#C4B1DF", 48, 22)
    box(s, "ag2", 356, 403, 126, "AG-2", "#C4B1DF", 48, 22)
    s.port("out", 553, 404, "F_icg", P["loc2"], 84)
    s.link("f", "ag1", "b", "l", via=[(66, 387), (136, 387), (136, 427)])
    s.link("g", "ag1", "b", "t", via=[(716, 386), (227, 386)])
    s.route([(419, 386), (419, 403)], color="#785AA0"); s.dot(419, 386)
    s.link("ag1", "ag2"); s.link("ag2", "out")
    s.label(647, 421, "shared G", 20, 110)
    return s


def afm_hr():
    s = Scene("05_afm_apghr", 1390, 490)
    s.frame(6, 6, 1378, 478, "(e) AFM + APG-HR", "hr")
    s.label(24, 48, "AFM", 25, 160)
    s.port("fe", 24, 88, "F_edge", P["edge2"], 82)
    box(s, "ep", 140, 88, 90, "Proj E", P["edge2"], 50, 21)
    s.port("fl", 24, 182, "F_loc", P["loc2"], 82)
    box(s, "lp", 140, 182, 90, "Proj L", P["loc2"], 50, 21)
    box(s, "att", 278, 170, 104, "Attn → A", P["loc2"], 58, 21)
    s.plus("mul", 433, 113, "×"); s.plus("cat", 433, 218, "C")
    s.link("fe", "ep"); s.link("fl", "lp"); s.link("lp", "att")
    s.link("ep", "mul"); s.link("att", "mul", "t", "b", via=[(330, 141), (433, 141)])
    s.link("mul", "cat", "b", "t", color="#B75070"); s.link("lp", "cat", via=[(254, 207), (254, 218)])
    box(s, "ref", 486, 190, 98, "Refine", P["prior2"], 54, 22)
    tensor(s, "z", 620, 190, 82, "Z_afm", P["prior2"], 54, 21)
    s.link("cat", "ref"); s.link("ref", "z")

    s.label(754, 48, "APG-HR", 25, 200)
    box(s, "pp", 756, 82, 100, "Up4x → p", P["prior2"], 50, 20)
    box(s, "uu", 892, 82, 104, "4p(1-p)", P["fpn2"], 50, 20)
    s.link("z", "pp", "t", "l", via=[(661, 160), (734, 160), (734, 107)], color=COL["accent"])
    s.link("pp", "uu", color=COL["accent"])

    sources = [("he", 208, "F_edge ↑4", P["edge2"]), ("hl", 274, "F_loc ↑4", P["loc2"]),
               ("hi", 340, "RGB feat", P["enc2"])]
    for key, y, label, fill in sources:
        box(s, key, 756, y, 134, label, fill, 48, 20)
    s.node("hc", 930, 184, 40, 222, "C", P["prior2"], kind="port", fs=26)
    for key, _, _, _ in sources: s.link(key, "hc")
    s.route([s.point("pp", "b"), (806, 160), (950, 160), (950, 184)], color=COL["accent"])
    s.route([s.point("uu", "b"), (944, 148), (966, 148), (966, 184)], color=COL["accent"])
    box(s, "fuse", 1012, 250, 112, "Fuse 1×1", P["hr2"], 54, 21)
    box(s, "dsr", 1162, 250, 104, "DSR", P["hr2"], 54, 22)
    tensor(s, "out", 1304, 250, 66, "Mask", P["prior2"], 54, 20)
    s.link("hc", "fuse"); s.link("fuse", "dsr"); s.link("dsr", "out")
    s.label(24, 412, "A is generated from L; E_g = A × E; concat(E_g, L) → Z_afm.", 21, 660)
    s.label(754, 412, "M4: three feature sources   |   M6: add p and uncertainty u", 21, 610)
    return s


def export_vsdx(scene, path):
    """Export every box, tensor, text item and polyline as a native Visio shape."""
    template = Path(vsdx.__file__).resolve().parent / "media" / "media.vsdx"
    work = path.with_suffix(".template.vsdx")
    shutil.copyfile(template, work)
    ns = "{http://schemas.microsoft.com/office/visio/2012/main}"
    scale = 100.0  # scene pixels per Visio inch

    def cell(parent, name, value, unit=None, formula=None):
        a = {"N": name, "V": str(value)}
        if unit: a["U"] = unit
        if formula: a["F"] = formula
        ET.SubElement(parent, ns + "Cell", a)

    def shape_cells(sh, x, y, w, h, fill, stroke, fs, nofill=False, noline=False, dash=False):
        pinx = (x + w / 2) / scale
        piny = (scene.h - (y + h / 2)) / scale
        wi, hi = max(w / scale, 0.01), max(h / scale, 0.01)
        cell(sh, "PinX", pinx); cell(sh, "PinY", piny)
        cell(sh, "Width", wi); cell(sh, "Height", hi)
        cell(sh, "LocPinX", wi / 2, formula="Width*0.5")
        cell(sh, "LocPinY", hi / 2, formula="Height*0.5")
        cell(sh, "Angle", 0)
        cell(sh, "FillForegnd", fill); cell(sh, "FillPattern", 0 if nofill else 1)
        cell(sh, "LineColor", stroke); cell(sh, "LineWeight", 0.012, "IN")
        cell(sh, "LinePattern", 0 if noline else (2 if dash else 1))
        cell(sh, "Rounding", 0.06, "IN")
        ch = ET.SubElement(sh, ns + "Section", {"N": "Character"})
        row = ET.SubElement(ch, ns + "Row", {"IX": "0"})
        cell(row, "Font", 0); cell(row, "Size", max(fs * 72 / scale, 5.2), "PT")
        cell(row, "Color", P["ink"])
        pa = ET.SubElement(sh, ns + "Section", {"N": "Paragraph"})
        prow = ET.SubElement(pa, ns + "Row", {"IX": "0"})
        cell(prow, "HorzAlign", 1)
        tb = ET.SubElement(sh, ns + "Section", {"N": "TextBlock"})
        cell(tb, "VerticalAlign", 1)
        for m in ("LeftMargin", "RightMargin", "TopMargin", "BottomMargin"):
            cell(tb, m, 0.035, "IN")

    def geometry(sh, kind, w, h):
        sec = ET.SubElement(sh, ns + "Section", {"N": "Geometry", "IX": "0"})
        if kind in ("circle", "dot"):
            row = ET.SubElement(sec, ns + "Row", {"T": "Ellipse", "IX": "1"})
            for n, v in [("X", w/2), ("Y", h/2), ("A", w), ("B", h/2), ("C", w/2), ("D", h)]:
                cell(row, n, v / scale)
            return
        pts = [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]
        if kind == "tensor": pts = [(9, 0), (w, 0), (w-9, h), (0, h), (9, 0)]
        for idx, (xx, yy) in enumerate(pts, 1):
            row = ET.SubElement(sec, ns + "Row", {"T": "MoveTo" if idx == 1 else "LineTo", "IX": str(idx)})
            cell(row, "X", xx / scale); cell(row, "Y", (h - yy) / scale)

    with vsdx.VisioFile(str(work)) as vis:
        page = vis.pages[0]
        page.name = "APHNet M6"
        page.width = scene.w / scale
        page.height = scene.h / scale
        root = page.xml.getroot()
        shapes = root.find(ns + "Shapes")
        if shapes is None: shapes = ET.SubElement(root, ns + "Shapes")
        shapes.clear()
        con = root.find(ns + "Connects")
        if con is not None: root.remove(con)

        ordered = sorted(scene.items, key=lambda i: 0 if i["kind"] == "frame" else 1 if i["kind"] == "line" else 2)
        sid = 1
        for item in ordered:
            if item["kind"] == "line":
                pts = item["points"]
                xs = [p[0] for p in pts]; ys = [scene.h - p[1] for p in pts]
                minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
                wi, hi = max(maxx-minx, 1), max(maxy-miny, 1)
                sh = ET.SubElement(shapes, ns + "Shape", {"ID": str(sid), "NameU": f"Connector.{sid}", "Type": "Shape"})
                shape_cells(sh, minx, scene.h-maxy, wi, hi, "#FFFFFF", item["stroke"], 20,
                            nofill=True, dash=item.get("dash", False))
                cell(sh, "BeginArrow", 0); cell(sh, "EndArrow", 13 if item.get("arrow", True) else 0)
                sec = ET.SubElement(sh, ns + "Section", {"N": "Geometry", "IX": "0"})
                cell(sec, "NoFill", 1)
                for ix, (x, y) in enumerate(zip(xs, ys), 1):
                    row = ET.SubElement(sec, ns + "Row", {"T": "MoveTo" if ix == 1 else "LineTo", "IX": str(ix)})
                    cell(row, "X", (x-minx)/scale); cell(row, "Y", (y-miny)/scale)
                sid += 1
                continue

            x, y, w, h = item["x"], item["y"], item["w"], item["h"]
            sh = ET.SubElement(shapes, ns + "Shape", {"ID": str(sid), "NameU": item["id"], "Type": "Shape"})
            is_text = item["kind"] == "text"
            shape_cells(sh, x, y, w, h, item.get("fill", "#FFFFFF"), item["stroke"], item["fs"],
                        nofill=is_text, noline=is_text, dash=item.get("dash", False))
            geometry(sh, "box" if is_text else item["kind"], w, h)
            if item.get("text"):
                t = ET.SubElement(sh, ns + "Text")
                t.text = item["text"]
            sid += 1
        vis.save_vsdx(str(path))
    work.unlink(missing_ok=True)
    # Structural validation: package opens and all items survived as independent shapes.
    with vsdx.VisioFile(str(path)) as check:
        count = len(check.pages[0].all_shapes)
        if count < len(scene.items) * 0.95:
            raise RuntimeError(f"Visio export lost shapes: {count}/{len(scene.items)}")


def build():
    panels = [hero(), lmfr(), edge(), icg(), afm_hr()]
    full = Scene("APHNet_M6_publication_v6", 2200, 1310)
    full.paste(panels[0], 10, 10, "a_")
    full.paste(panels[1], 1390, 10, "b_")
    full.paste(panels[2], 1390, 320, "c_")
    full.paste(panels[3], 10, 810, "d_")
    full.paste(panels[4], 800, 810, "e_")
    # 22 px on a 2200 px-wide canvas corresponds to about 5.2 pt at 183 mm.
    for sc in [full, *panels]:
        for item in sc.items:
            if item.get("fs", 22) < 22:
                item["fs"] = 22
    for sc in [full, *panels]:
        p = OUT / f"{sc.name}.svg"
        svg(sc, p)
        pdf_kwargs = {"output_width": 183 / 25.4 * 96} if sc is full else {}
        cairosvg.svg2pdf(url=str(p), write_to=str(p.with_suffix(".pdf")), **pdf_kwargs)
        cairosvg.svg2png(url=str(p), write_to=str(p.with_suffix(".png")),
                         output_width=4400 if sc is full else max(sc.w * 2, 1800))
    with Image.open(OUT / "APHNet_M6_publication_v6.png") as im:
        im.save(OUT / "APHNet_M6_publication_v6_600dpi.tiff", dpi=(600, 600), compression="tiff_lzw")
    export_vsdx(full, OUT / "APHNet_M6_publication_v6_editable.vsdx")
    (OUT / "editable_scene.json").write_text(
        json.dumps([{"name": x.name, "width": x.w, "height": x.h, "items": x.items}
                    for x in [full, *panels]], ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()

"""APHNet/M6 publication figure v11: larger type, six focused panels."""
from pathlib import Path
import re

import cairosvg

import build_aphnet_publication_v4 as v4
import build_aphnet_publication_v9 as v9
import build_aphnet_publication_v10 as v10
from build_aphnet_publication_two_styles import add_defs_and_normalize


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_publication_v11"
OUT.mkdir(parents=True, exist_ok=True)


def b(s, key, x, y, w, text, fam, h=64, fs=24):
    return v4.block(s, key, x, y, w, text, fam, h, fs)


def tensor(s, key, x, y, w, text, fam, h=62, fs=24):
    return s.node(key, x, y, w, h, text, fam, kind="tensor", fs=fs)


def main_panel():
    s = v10.compact_main()
    s.name = "01_main_v11"
    s.h = 1240
    for it in s.items:
        if it.get("kind") == "frame":
            it["h"] = 1224
        if it.get("text"):
            it["text"] = it["text"].replace("RGB*", "RGB")
            if it.get("kind") == "text":
                it["fs"] = max(it.get("fs", 17), 20)
            elif it.get("kind") != "line":
                it["fs"] = max(it.get("fs", 17), 21)
        if it.get("id") in {"hfe", "hfl", "hrgb"}:
            positions = {"hfe": 850, "hfl": 930, "hrgb": 1010}
            it.update(x=1090, y=positions[it["id"]], w=132, h=60, fs=22)
        if it.get("id") == "head":
            it.update(x=1400, y=835, w=430, h=290, fs=25,
                      text="HSR + APG-HR (f)\n48 + 48 + 16 + 1 + 1 = 114 ch\n1×1 Fuse → DSR")
        if it.get("id") == "sig":
            it.update(x=1875, y=930, w=150, h=82, fs=23)
        if it.get("id") == "mask":
            it.update(x=2070, y=936, w=118, h=70, fs=22)
        if it.get("kind") == "text" and "Repeated tensor names" in it.get("text", ""):
            it["text"] = "Repeated tensor names denote the same feature; RGB denotes the same input image."
            it["fs"] = 21

    # Remove the old four-to-one convergence and the stale output links.
    s.items = [it for it in s.items if not (
        it.get("kind") == "line" and
        ((it.get("start") or (None,))[0] in {"hfe", "hfl", "hrgb", "z", "head", "sig"}
         or (it.get("end") or (None,))[0] in {"head", "sig", "mask"})
    )]

    tensor(s, "hza", 1090, 1090, 132, "Z_afm", "prior", 60, 22)
    ports = [("hfe", 890), ("hfl", 955), ("hrgb", 1020), ("hza", 1085)]
    for key, py in ports:
        x0, y0 = s.point(key, "r")
        s.route([(x0, y0), (1320, y0), (1320, py), (1400, py)],
                start=(key, "r"), end=("head", "l"))
    s.route([(1830, 980), (1875, 971)], start=("head", "r"), end=("sig", "l"))
    s.route([(2025, 971), (2070, 971)], start=("sig", "r"), end=("mask", "l"))

    # Keep the stage note and explanatory key in separate horizontal bands.
    for it in s.items:
        if it.get("kind") == "text" and it.get("text", "").startswith("Stages:"):
            it["y"] = 785
        if it.get("kind") == "text" and it.get("text") == "Module key":
            it.update(y=855, fs=22)
        if it.get("id", "").startswith("key"):
            it["y"] = 900
        if it.get("kind") == "text" and it.get("y") == 924:
            it["y"] = 894
    return s


def backbone_panel():
    s = v4.Scene("02_backbone_cells", 1260, 600)
    s.frame(8, 8, 1244, 584, "(b) MobileNetV4 backbone cells", "enc")

    s.label(30, 64, "Universal inverted bottleneck (UIB)", 27, 560)
    tensor(s, "ux", 35, 120, 72, "X", "enc")
    ops = [(145, 130, "DW-start*"), (320, 142, "PW-expand"),
           (508, 130, "DW-middle*"), (685, 140, "PW-project"),
           (875, 120, "LayerScale")]
    prev = "ux"
    for i, (x, w, text) in enumerate(ops):
        b(s, f"u{i}", x, 120, w, text, "enc", 62, 23)
        s.link(prev, f"u{i}"); prev = f"u{i}"
    s.plus("uadd", 1045, 151)
    tensor(s, "uy", 1100, 120, 92, "Y", "enc")
    s.link(prev, "uadd"); s.link("uadd", "uy")
    s.route([s.point("ux", "b"), (71, 224), (1045, 224), s.point("uadd", "b")])
    s.label(805, 242, "Optional DW layers; residual requires matching shape", 20, 430)

    s.label(30, 315, "Mobile multi-query attention (MQA, 4 heads)", 27, 680)
    tensor(s, "mx", 35, 390, 72, "X", "loc")
    b(s, "bn", 145, 390, 82, "BN", "loc")
    b(s, "q", 275, 350, 138, "Q: PW 1×1", "loc", 62, 22)
    b(s, "kv", 275, 430, 190, "K,V: DW↓* + PW", "loc", 62, 21)
    b(s, "att", 520, 374, 190, "QKᵀ → Softmax\nAttention × V", "prior", 92, 22)
    b(s, "out", 760, 390, 130, "PW-out", "loc")
    b(s, "ls", 935, 390, 120, "LayerScale", "loc", 62, 22)
    s.plus("madd", 1095, 421)
    tensor(s, "my", 1140, 390, 72, "Y", "loc")
    s.link("mx", "bn")
    s.link("bn", "q", via=[(250, 421), (250, 381)])
    s.link("bn", "kv", via=[(250, 421), (250, 461)])
    s.link("q", "att", via=[(490, 381), (490, 404)])
    s.link("kv", "att", via=[(490, 461), (490, 436)])
    for a, c in [("att", "out"), ("out", "ls"), ("ls", "madd"), ("madd", "my")]:
        s.link(a, c)
    s.route([s.point("mx", "b"), (71, 545), (1095, 545), s.point("madd", "b")])
    s.label(888, 526, "* K/V downsample only in Stage 4", 20, 320)
    return s


def lmfr_panel():
    s = v4.Scene("03_lmfr", 1260, 620)
    s.frame(8, 8, 1244, 604, "(c) LMFR — lightweight multiscale reconstruction", "fpn")
    tensor(s, "pc", 55, 145, 155, "P(i+1)", "fpn", 70, 25)
    b(s, "up", 270, 145, 125, "Bilinear ↑2", "prior", 70, 23)
    tensor(s, "li", 55, 300, 155, "L_i", "fpn", 70, 26)
    s.plus("sum", 480, 265)
    b(s, "dw", 555, 230, 155, "DW 3×3", "hr", 72, 25)
    b(s, "pw", 770, 230, 155, "PW 1×1", "hr", 72, 25)
    tensor(s, "pi", 985, 235, 145, "P_i", "fpn", 66, 26)
    s.link("pc", "up")
    s.link("up", "sum", via=[(440, 180), (440, 238), (480, 238)])
    s.link("li", "sum", via=[(440, 335), (440, 292), (480, 292)])
    for a, c in [("sum", "dw"), ("dw", "pw"), ("pw", "pi")]: s.link(a, c)
    s.label(55, 430, "L_i = lateral 1×1 projection (192 channels)", 23, 620)
    s.label(720, 430, "i = 2, 3, 4", 23, 310)
    s.label(55, 495, "DW/PW are each followed by GN + ReLU", 23, 690)
    s.label(720, 495, "P5 = DSR(L5)", 23, 350)
    return s


def afm_panel():
    s = v9.afm_panel()
    s.name = "04_afm_v11"; s.h = 700
    for it in s.items:
        if it.get("kind") == "frame": it["h"] = 684
        if it.get("text") and it.get("kind") != "line": it["fs"] = max(it.get("fs", 18), 23)
        if it.get("id") in {"cat", "ref", "pred"}: it["y"] -= 55
        if it.get("id") == "z": it["y"] -= 85
        if it.get("kind") == "text" and it.get("y", 0) > 680: it["y"] -= 85
    # Rebuild only the lower prediction route after moving its boxes.
    s.items = [it for it in s.items if not (it.get("kind") == "line" and
        ((it.get("start") or (None,))[0] in {"l", "eg", "cat", "ref", "pred"} or
         (it.get("end") or (None,))[0] in {"cat", "ref", "pred", "z"}))]
    s.route([s.point("l", "b"), (445, 490), s.point("cat", "l")], start=("l", "b"), end=("cat", "l"))
    s.route([s.point("eg", "r"), (1010, 154), (1010, 410), (590, 410), s.point("cat", "t")], start=("eg", "r"), end=("cat", "t"))
    s.link("cat", "ref"); s.link("ref", "pred"); s.link("pred", "z", "b", "t")
    return s


def icg_panel():
    s = v9.icg_panel()
    s.name = "05_icg_v11"; s.h = 700
    for it in s.items:
        if it.get("kind") == "frame": it["h"] = 684
        if it.get("text"):
            it["text"] = it["text"].replace("RGB*", "RGB")
            if it.get("kind") != "line": it["fs"] = max(it.get("fs", 18), 23)
        ident = it.get("id", "")
        if ident in {"br0", "br1", "br2", "contrast", "g"}: it["y"] -= 65
        if ident in {"ag1", "ag2", "out"}: it["y"] -= 85
        if it.get("kind") == "text" and it.get("y", 0) > 650: it["y"] -= 80
    # Move existing routes with the same vertical offsets to remove the empty band.
    for it in s.items:
        if it.get("kind") != "line": continue
        pts = it.get("points", [])
        shifted = []
        for x, y in pts:
            if y >= 540: y -= 85
            elif y >= 300: y -= 65
            shifted.append((x, y))
        it["points"] = shifted
    return s


def highres_panel():
    s = v4.Scene("06_highres_v11", 1180, 700)
    s.frame(8, 8, 1164, 684, "(f) HSR + APG-HR — full-resolution reconstruction", "hr")
    entries = [("fe", "F_edge", "1×1 / 48", "Up ×4", "edge", 100),
               ("fl", "F_loc", "1×1 / 48", "Up ×4", "loc", 190),
               ("rgb", "RGB", "3×3 / 16", "DSR", "enc", 280)]
    for key, name, proj, post, fam, y in entries:
        tensor(s, key, 30, y, 120, name, fam, 60, 22)
        b(s, key + "p", 185, y, 135, proj, fam, 60, 22)
        b(s, key + "u", 355, y, 115, post, fam, 60, 22)
        s.link(key, key + "p"); s.link(key + "p", key + "u")
    tensor(s, "z", 30, 415, 120, "Z_afm", "prior", 60, 22)
    b(s, "up", 185, 415, 115, "Up ×4", "prior", 60, 22)
    b(s, "sig", 335, 415, 115, "Sigmoid", "prior", 60, 22)
    tensor(s, "p", 485, 415, 66, "p", "prior", 60, 25)
    b(s, "u", 365, 505, 186, "u = 4p(1−p)\nambiguity prior", "fpn", 78, 21)
    for a, c in [("z", "up"), ("up", "sig"), ("sig", "p")]: s.link(a, c)
    s.link("p", "u", "b", "t")

    b(s, "cat", 620, 145, 92, "CAT\n114", "prior", 360, 24)
    port_y = {"feu": 190, "flu": 245, "rgbu": 300, "p": 385, "u": 455}
    for key, py in port_y.items():
        x0, y0 = s.point(key, "r")
        s.route([(x0, y0), (585, y0), (585, py), (620, py)], start=(key, "r"), end=("cat", "l"))
    b(s, "fuse", 770, 205, 150, "1×1 Fuse", "hr", 76, 24)
    b(s, "ref", 770, 335, 150, "DSR", "hr", 76, 25)
    b(s, "pred", 970, 270, 150, "1×1 Conv", "prior", 82, 23)
    tensor(s, "mask", 970, 410, 150, "Mask\n320²", "prior", 76, 23)
    s.link("cat", "fuse"); s.link("fuse", "ref", "b", "t")
    s.link("ref", "pred"); s.link("pred", "mask", "b", "t")
    s.label(30, 615, "48 + 48 + 16 channels; p and ambiguity prior u form a five-way 114-channel fusion.", 21, 1110)
    return s


def style_svg(path):
    add_defs_and_normalize(path)
    text = path.read_text(encoding="utf-8")
    text = text.replace('rx="5"', 'rx="16"').replace('rx="2"', 'rx="18"')
    text = text.replace('width="3600" height="2020"', 'width="280mm" height="157.1mm"', 1)

    def wire(match):
        tag = match.group(0)
        dashed = "stroke-dasharray" in tag
        color = "#7B858C" if dashed else "#344653"
        width = "1.0" if dashed else "1.75"
        tag = re.sub(r'stroke="[^"]+"', f'stroke="{color}"', tag, count=1)
        tag = re.sub(r'stroke-width="[^"]+"', f'stroke-width="{width}"', tag, count=1)
        return tag
    text = re.sub(r'<polyline\b[^>]*/>', wire, text)
    path.write_text(text, encoding="utf-8")


def build():
    full = v4.Scene("APHNet_M6_publication_v11", 3600, 2020)
    full.paste(main_panel(), 10, 10, "a_")
    full.paste(backbone_panel(), 2330, 10, "b_")
    full.paste(lmfr_panel(), 2330, 620, "c_")
    full.paste(afm_panel(), 10, 1260, "d_")
    full.paste(icg_panel(), 1210, 1260, "e_")
    full.paste(highres_panel(), 2410, 1260, "f_")
    full.label(24, 1984,
               "Line key: solid = feature flow | thin gray = module-internal route | CAT = channel concatenation | + = addition",
               23, 2600)

    svg_path = OUT / "APHNet_M6_publication_v11.svg"
    png_path = OUT / "APHNet_M6_publication_v11.png"
    pdf_path = OUT / "APHNet_M6_publication_v11_QA.pdf"
    v4.svg(full, svg_path)
    style_svg(svg_path)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=3600, output_height=2020)
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    print(svg_path); print(png_path); print(pdf_path)


if __name__ == "__main__":
    build()

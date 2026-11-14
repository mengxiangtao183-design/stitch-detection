"""APHNet/M6 compact six-panel figure with audited UIB and MQA cells."""
from pathlib import Path
import re

import cairosvg

import build_aphnet_publication_v4 as v4
import build_aphnet_publication_v9 as v9
from build_aphnet_publication_two_styles import add_defs_and_normalize


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_publication_v10"
OUT.mkdir(parents=True, exist_ok=True)


def b(s, key, x, y, w, text, fam, h=54, fs=21):
    return v4.block(s, key, x, y, w, text, fam, h, fs)


def tensor(s, key, x, y, w, text, fam, h=52, fs=21):
    return s.node(key, x, y, w, h, text, fam, kind="tensor", fs=fs)


def compact_main():
    s = v9.main_panel()
    s.name = "01_main_compact"
    s.h = 1160
    for item in s.items:
        if item.get("kind") == "frame":
            item["h"] = 1144
        if item.get("id") in {"s2", "s3", "s4", "s5"}:
            item["text"] = item["id"].upper()
            item["fs"] = 24
        if item.get("id") in {"l2", "l3", "l4", "l5"}:
            item["text"] = item["id"].upper()
            item["fs"] = 23
        if item.get("id") in {"dsr2", "dsr3", "dsr4", "dsr5"}:
            item["text"] = "DSR"
            item["fs"] = 22
        if item.get("kind") == "text":
            item["text"] = {
                "Encoder": "Enc.",
                "Boundary": "Edge",
                "Localization": "Loc.",
            }.get(item.get("text"), item.get("text"))
        if item.get("kind") == "line":
            item["stroke"] = "#202B38"
    s.label(35, 820,
            "Stages: S2 = FusedIB | S3 = UIB×2 | S4 = UIB×8 + MQA×4\n"
            "S5 = UIB×12 + MQA×4 + PW1×1 → 960",
            20, 1050)
    return s


def backbone_lmfr_panel():
    s = v4.Scene("02_backbone_lmfr", 1260, 700)
    s.frame(8, 8, 1244, 684, "(b) MobileNetV4 cells and LMFR", "fpn")

    # Universal inverted residual. DW-start and DW-middle are configuration-dependent.
    s.label(30, 62, "Universal inverted bottleneck (UIB)", 22, 520)
    tensor(s, "ux", 34, 105, 62, "X", "enc")
    ops = [(135, 120, "DW-s*"), (295, 128, "PW-exp"), (463, 120, "DW-m*"),
           (623, 128, "PW-proj"), (790, 105, "LayerScale")]
    prev = "ux"
    for i, (x, w, text) in enumerate(ops):
        b(s, f"u{i}", x, 105, w, text, "enc")
        s.link(prev, f"u{i}")
        prev = f"u{i}"
    s.plus("uadd", 955, 132)
    s.link(prev, "uadd")
    s.route([s.point("ux", "b"), (65, 190), (955, 190), s.point("uadd", "b")])
    tensor(s, "uo", 1010, 106, 88, "Y", "enc")
    s.link("uadd", "uo")
    s.label(1110, 104, "* optional", 19, 125)
    s.label(1110, 142, "skip if shape", 19, 135)
    s.label(1110, 177, "matches", 19, 120)

    # Mobile multi-query attention used in Stages 4 and 5.
    s.label(30, 225, "Mobile multi-query attention (MQA, 4 heads)", 22, 570)
    tensor(s, "mx", 34, 295, 62, "X", "loc")
    b(s, "mbn", 135, 295, 78, "BN", "loc")
    b(s, "mq", 260, 255, 142, "Q: PW 1×1", "loc")
    b(s, "mkv", 260, 330, 190, "K,V: DW↓* + PW", "loc", 54, 19)
    b(s, "matt", 505, 278, 180, "QKᵀ → Softmax\nAttention × V", "prior", 78, 19)
    b(s, "mout", 735, 295, 135, "PW-out", "loc")
    b(s, "mls", 915, 295, 110, "LayerScale", "loc", 54, 19)
    s.plus("madd", 1080, 322)
    s.link("mx", "mbn")
    s.link("mbn", "mq", "r", "l", via=[(235, 322), (235, 282)])
    s.link("mbn", "mkv", "r", "l", via=[(235, 322), (235, 357)])
    s.link("mq", "matt", "r", "l", via=[(475, 282), (475, 302)])
    s.link("mkv", "matt", "r", "l", via=[(475, 357), (475, 332)])
    for a, c in [("matt", "mout"), ("mout", "mls"), ("mls", "madd")]:
        s.link(a, c)
    s.route([s.point("mx", "b"), (65, 410), (1080, 410), s.point("madd", "b")])
    tensor(s, "mo", 1130, 296, 82, "Y", "loc")
    s.link("madd", "mo")
    s.label(960, 365, "* K/V downsample in S4", 19, 255)

    # LMFR cell.
    s.label(30, 455, "LMFR cell", 22, 240)
    tensor(s, "pc", 34, 505, 110, "P(i+1)", "fpn")
    b(s, "up", 185, 505, 90, "↑2", "prior")
    tensor(s, "li", 34, 605, 110, "L_i", "fpn")
    s.plus("sum", 340, 557)
    b(s, "dw", 405, 530, 125, "DW 3×3", "hr")
    b(s, "pw", 575, 530, 125, "PW 1×1", "hr")
    tensor(s, "pi", 750, 531, 105, "P_i", "fpn")
    s.link("pc", "up")
    s.link("up", "sum", "r", "t", via=[(310, 532), (340, 532)])
    s.link("li", "sum", "r", "b", via=[(310, 632), (340, 632)])
    for a, c in [("sum", "dw"), ("dw", "pw"), ("pw", "pi")]:
        s.link(a, c)
    s.label(900, 510, "L_i: 1×1 / 192 ch", 19, 310)
    s.label(900, 555, "i = 2, 3, 4", 19, 250)
    s.label(900, 600, "P5 = DSR(L5)", 19, 270)
    s.label(900, 645, "DW/PW → GN + ReLU", 19, 315)
    return s


def enhancement_panel():
    s = v4.Scene("03_enhancement", 1260, 440)
    s.frame(8, 8, 1244, 424, "(c) Boundary and semantic enhancement", "edge")
    s.label(28, 58, "EEB", 21, 90)
    tensor(s, "x", 30, 115, 65, "X", "edge")
    lanes = [(65, "DW/PW"), (115, "Sobel/PW"), (165, "Lap./Id")]
    for i, (y, text) in enumerate(lanes):
        b(s, f"eb{i}", 150, y, 135, text, "edge", 46, 19)
        s.route([s.point("x", "r"), (120, 141), (120, y + 23), s.point(f"eb{i}", "l")])
    b(s, "cat", 335, 91, 74, "CAT", "prior", 120, 20)
    for i in range(3): s.link(f"eb{i}", "cat")
    b(s, "mix", 458, 124, 102, "PW 1×1", "edge")
    b(s, "eca", 605, 124, 85, "ECA", "edge")
    s.plus("add", 745, 151)
    tensor(s, "fo", 795, 125, 105, "F_edge", "edge")
    for a, c in [("cat", "mix"), ("mix", "eca"), ("eca", "add"), ("add", "fo")]: s.link(a, c)
    s.route([s.point("x", "b"), (62, 230), (745, 230), s.point("add", "b")])

    s.label(28, 275, "SEU", 21, 90)
    tensor(s, "p4", 30, 330, 65, "P4", "loc")
    seq = [(140, 112, "DW/PW"), (295, 76, "SA"), (414, 76, "ECA"),
           (535, 112, "DW/PW")]
    prev = "p4"
    for i, (x, w, text) in enumerate(seq):
        b(s, f"su{i}", x, 330, w, text, "loc")
        s.link(prev, f"su{i}"); prev = f"su{i}"
    s.plus("sadd", 700, 357)
    tensor(s, "fseu", 750, 331, 105, "F_seu", "loc")
    s.link(prev, "sadd"); s.link("sadd", "fseu")
    s.route([s.point("p4", "b"), (62, 408), (700, 408), s.point("sadd", "b")])
    s.label(960, 94, "EEB", 20, 120)
    s.label(960, 132, "3 branches + residual", 19, 250)
    s.label(930, 312, "SA: spatial", 19, 250)
    s.label(930, 350, "ECA: channel", 19, 250)
    return s


def compact_bottom(scene):
    scene.h = 780
    for item in scene.items:
        if item.get("kind") == "frame": item["h"] = 764
    return scene


def increase_readability(scene, main=False):
    for item in scene.items:
        kind = item.get("kind")
        if item.get("text"):
            if kind == "text":
                item["fs"] = max(item.get("fs", 20), 22)
            else:
                # Boxes contain short operator names; use a strong minimum.
                item["fs"] = max(item.get("fs", 20), 21)
        if kind == "line":
            item["stroke"] = "#202B38" if main else "#526477"


def style_svg(path):
    # Reuse the subdued family textures, then restore the requested line hierarchy.
    add_defs_and_normalize(path)
    text = path.read_text(encoding="utf-8")
    text = text.replace('rx="5"', 'rx="15"').replace('rx="2"', 'rx="18"')

    def wire(match):
        tag = match.group(0)
        pts = re.search(r'points="([^"]+)"', tag)
        coords = []
        if pts:
            for pair in pts.group(1).split():
                try:
                    x, y = pair.split(','); coords.append((float(x), float(y)))
                except ValueError:
                    pass
        is_main = coords and max(x for x, _ in coords) < 2310 and max(y for _, y in coords) < 1170
        dashed = "stroke-dasharray" in tag
        color = "#7B858C" if dashed else ("#202B38" if is_main else "#526477")
        width = "1.0" if dashed else ("2.20" if is_main else "1.45")
        tag = re.sub(r'stroke="[^"]+"', f'stroke="{color}"', tag, count=1)
        tag = re.sub(r'stroke-width="[^"]+"', f'stroke-width="{width}"', tag, count=1)
        if dashed: tag = re.sub(r' marker-end="[^"]+"', '', tag)
        return tag

    text = re.sub(r'<polyline\b[^>]*/>', wire, text)
    # Give the SVG a real landscape-page size while preserving the vector viewBox.
    text = text.replace('width="3600" height="2025"', 'width="280mm" height="157.5mm"', 1)
    path.write_text(text, encoding="utf-8")


def build():
    main = compact_main()
    bpanel = backbone_lmfr_panel()
    cpanel = enhancement_panel()
    dpanel = compact_bottom(v9.afm_panel())
    epanel = compact_bottom(v9.icg_panel())
    fpanel = compact_bottom(v9.highres_panel())
    for sc in (main,): increase_readability(sc, main=True)
    for sc in (bpanel, cpanel, dpanel, epanel, fpanel): increase_readability(sc)

    full = v4.Scene("APHNet_M6_publication_v10", 3600, 2025)
    full.paste(main, 10, 10, "a_")
    full.paste(bpanel, 2330, 10, "b_")
    full.paste(cpanel, 2330, 720, "c_")
    full.paste(dpanel, 10, 1180, "d_")
    full.paste(epanel, 1210, 1180, "e_")
    full.paste(fpanel, 2410, 1180, "f_")
    full.label(24, 1986,
               "Line key: thick solid = main feature flow | thin solid = module-internal flow | dashed gray = reference/auxiliary link | CAT = channel concatenation",
               22, 2800)

    svg_path = OUT / "APHNet_M6_publication_v10.svg"
    png_path = OUT / "APHNet_M6_publication_v10.png"
    pdf_path = OUT / "APHNet_M6_publication_v10_QA.pdf"
    v4.svg(full, svg_path)
    style_svg(svg_path)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                     output_width=3600, output_height=2025)
    # Preserve the SVG's physical 280 mm × 157.5 mm canvas in the QA PDF.
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    print(svg_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    build()

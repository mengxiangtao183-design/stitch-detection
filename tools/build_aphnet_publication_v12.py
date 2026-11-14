"""APHNet/M6 v12: semantic labels replace the most abstract tensor symbols."""
from pathlib import Path
import re

import cairosvg

import build_aphnet_publication_v4 as v4
import build_aphnet_publication_v11 as v11
from build_aphnet_publication_two_styles import add_defs_and_normalize


OUT = Path(__file__).resolve().parents[1] / "outputs" / "aphnet_publication_v12"
OUT.mkdir(parents=True, exist_ok=True)


def relabel(scene, mapping):
    for item in scene.items:
        key = item.get("id")
        if key in mapping:
            item["text"] = mapping[key]
    return scene


def main_panel():
    s = v11.main_panel()
    relabel(s, {
        "icg": "ICG (e)",
        "z": "Coarse\nlogits 80²×1",
        "hza": "Coarse\nlogits",
        "head": "APG-HR head (f)\nFive-way CAT = 114 ch\n1×1 Fuse → DSR",
    })

    # Replace the acronym-heavy colour key with a shape/semantics key.
    remove_ids = {f"key{i}" for i in range(6)}
    remove_text = {
        "Module key", "Enc.", "Pyramid", "Edge", "Loc.", "Prior", "Refine",
    }
    remove_prefixes = (
        "+: add", "Repeated tensor names", "C2–C5:",
    )
    s.items = [it for it in s.items if not (
        it.get("id") in remove_ids or
        (it.get("kind") == "text" and (
            it.get("text") in remove_text or
            it.get("text", "").startswith(remove_prefixes)
        ))
    )]

    s.label(35, 855, "Notation", 23, 180)
    s.node("leg_feat", 35, 895, 115, 44, "Feature", "prior", kind="tensor", fs=20)
    s.node("leg_op", 190, 895, 125, 44, "Operation", "hr", kind="box", fs=20)
    s.node("leg_gate", 355, 895, 135, 44, "Gate / prior", "loc", kind="box", fs=20)
    s.plus("leg_add", 545, 917)
    s.label(575, 900, "element-wise addition", 19, 300)
    s.label(35, 970,
            "C_i: encoder output    L_i: lateral feature    P_i: reconstructed pyramid feature",
            20, 950)
    s.label(35, 1020,
            "F_edge / F_loc: branch features    Coarse logits: AFM prediction before Sigmoid",
            20, 990)
    s.label(35, 1070,
            "80²×128 = 80×80 spatial size with 128 channels    ↑k = k-fold bilinear upsampling",
            20, 1040)
    s.label(35, 1120,
            "Repeated labels denote the same tensor; RGB always denotes the same input image.",
            20, 1020)
    return s


def backbone_panel():
    s = v11.backbone_panel()
    relabel(s, {
        "ux": "In", "uy": "Out",
        "mx": "In", "my": "Out",
        "q": "Query Q", "kv": "Shared K/V",
    })
    for item in s.items:
        if item.get("kind") == "text" and item.get("text", "").startswith("Optional DW"):
            item["text"] = "Optional DW; residual if shapes match"
    return s


def lmfr_panel():
    s = v11.lmfr_panel()
    relabel(s, {
        "pc": "P(i+1)",
        "li": "L_i",
        "pi": "P_i",
    })
    return s


def afm_panel():
    s = v11.afm_panel()
    for it in s.items:
        if it.get("kind") == "text" and it.get("text", "").startswith("(d) AFM"):
            it["text"] = "(d) Attentive fusion (AFM) — localization-guided boundary fusion"
    relabel(s, {
        "pe": "Edge proj.",
        "e": "Edge",
        "pl": "Loc. proj.",
        "l": "Loc. feat.",
        "att": "Gate A",
        "eg": "Gated\nedge",
        "z": "Coarse logits\n80²×1",
    })
    for it in s.items:
        if it.get("kind") == "text" and it.get("text", "").startswith("A is generated"):
            it["text"] = "Localization generates spatial gate A; the gate suppresses unsupported edges."
        if it.get("kind") == "text" and it.get("text", "").startswith("CAT(E_g"):
            it["text"] = "Gated edge and localization features are concatenated to predict coarse logits."
    return s


def icg_panel():
    s = v11.icg_panel()
    for it in s.items:
        if it.get("kind") == "text" and it.get("text", "").startswith("(e) ICG"):
            it["text"] = "(e) Image-context guidance (ICG) — contrast and shared gates"
    relabel(s, {
        "v": "RGB\nfeat.",
        "pi": "Init. mask",
        "br0": "FG",
        "contrast": "FG−BG",
        "g": "Guide",
        "ag1": "Gate 1",
        "ag2": "Gate 2",
        "out": "ICG feat.",
    })
    for it in s.items:
        if it.get("kind") == "text" and it.get("text", "").startswith("FG/BG contrast"):
            it["text"] = "Foreground–background contrast forms one guide shared by two independent gates."
    return s


def highres_panel():
    s = v11.highres_panel()
    for it in s.items:
        if it.get("kind") == "text" and it.get("text", "").startswith("(f) HSR"):
            it["text"] = "(f) AFM-prior-guided high-resolution head (APG-HR)"
    relabel(s, {
        "z": "Coarse\nlogits",
        "p": "Prob.",
        "u": "Ambiguity u\n4p(1−p)",
    })
    return s


def style_svg(path):
    add_defs_and_normalize(path)
    text = path.read_text(encoding="utf-8")
    text = text.replace('rx="5"', 'rx="16"').replace('rx="2"', 'rx="18"')
    text = text.replace('width="3600" height="2020"', 'width="280mm" height="157.1mm"', 1)

    def wire(match):
        tag = match.group(0)
        dashed = "stroke-dasharray" in tag
        tag = re.sub(r'stroke="[^"]+"',
                     'stroke="#7B858C"' if dashed else 'stroke="#344653"', tag, count=1)
        tag = re.sub(r'stroke-width="[^"]+"',
                     'stroke-width="1.0"' if dashed else 'stroke-width="1.75"', tag, count=1)
        return tag
    text = re.sub(r'<polyline\b[^>]*/>', wire, text)
    path.write_text(text, encoding="utf-8")


def build():
    full = v4.Scene("APHNet_M6_publication_v12", 3600, 2020)
    full.paste(main_panel(), 10, 10, "a_")
    full.paste(backbone_panel(), 2330, 10, "b_")
    full.paste(lmfr_panel(), 2330, 620, "c_")
    full.paste(afm_panel(), 10, 1260, "d_")
    full.paste(icg_panel(), 1210, 1260, "e_")
    full.paste(highres_panel(), 2410, 1260, "f_")
    full.label(24, 1984,
               "Line key: solid = feature flow | thin gray = module-internal route | CAT = channel concatenation | × = gating",
               23, 2600)

    svg_path = OUT / "APHNet_M6_publication_v12.svg"
    png_path = OUT / "APHNet_M6_publication_v12.png"
    pdf_path = OUT / "APHNet_M6_publication_v12_QA.pdf"
    v4.svg(full, svg_path)
    style_svg(svg_path)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=3600, output_height=2020)
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    print(svg_path); print(png_path); print(pdf_path)


if __name__ == "__main__":
    build()

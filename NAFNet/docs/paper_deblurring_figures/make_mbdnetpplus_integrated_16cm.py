from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


OUT_DIR = Path(__file__).resolve().parent
OUT = OUT_DIR / "MBDNetPPlus_integrated_architecture"

# Final-size contract: 160 mm wide and 232 mm high.
MM = 1 / 25.4
FIG_W_MM, FIG_H_MM = 160, 232

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.linewidth": 0.8,
    }
)


COL = {
    "ink": "#263746",
    "muted": "#687580",
    "line": "#40515F",
    "guide": "#8A9AA6",
    "blue": "#DCEFF8",
    "blue_edge": "#4E9EC6",
    "blue_side": "#B7DCEA",
    "orange": "#FCE3D5",
    "orange_edge": "#E97948",
    "orange_side": "#EEC1A9",
    "yellow": "#FFF0B8",
    "yellow_edge": "#D39A13",
    "yellow_side": "#E5C564",
    "green": "#DCF1DD",
    "green_edge": "#58A76B",
    "green_side": "#A7D6AE",
    "purple": "#E8DDF3",
    "purple_edge": "#8E67AE",
    "purple_side": "#CDBAE0",
    "pink": "#F6DFE9",
    "pink_edge": "#C86691",
    "panel_blue": "#F4FAFD",
    "panel_orange": "#FFF8F3",
    "panel_green": "#F5FBF5",
    "panel_grey": "#F7F8F9",
    "panel_grey_edge": "#83919B",
    "white": "#FFFFFF",
}


def panel(ax, x, y, w, h, title, subtitle="", fc="#FFFFFF", ec="#AAB5BD", label=None):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.85, edgecolor=ec, facecolor=fc,
        transform=ax.transAxes, clip_on=False, zorder=0,
    )
    ax.add_patch(p)
    prefix = f"({label}) " if label else ""
    ax.text(x + 0.012, y + h - 0.020, prefix + title, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2, fontweight="bold", color=COL["ink"])
    if subtitle:
        ax.text(x + 0.012, y + h - 0.042, subtitle, transform=ax.transAxes,
                ha="left", va="top", fontsize=6.6, color=COL["muted"])
    return p


def arrow(ax, x1, y1, x2, y2, color=None, lw=0.9, style="-|>", ls="-", z=4,
          connectionstyle="arc3"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7.5,
        linewidth=lw, color=color or COL["line"], linestyle=ls,
        connectionstyle=connectionstyle, transform=ax.transAxes,
        clip_on=False, zorder=z,
    )
    ax.add_patch(a)
    return a


def orth_arrow(ax, points, color=None, lw=0.85, ls="--", z=3):
    c = color or COL["guide"]
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], transform=ax.transAxes, color=c,
                lw=lw, ls=ls, zorder=z, clip_on=False)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    arrow(ax, x1, y1, x2, y2, color=c, lw=lw, ls=ls, z=z)


def box(ax, x, y, w, h, text, fc=COL["white"], ec=COL["line"], fs=6.5,
        weight="normal", radius=0.004, z=5, italic=False):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        linewidth=0.75, edgecolor=ec, facecolor=fc,
        transform=ax.transAxes, zorder=z,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fs, fontweight=weight,
            fontstyle="italic" if italic else "normal", color=COL["ink"], zorder=z + 1)
    return p


def chip(ax, x, y, w, text, family="orange", fs=5.8, z=8):
    return box(ax, x, y, w, 0.018, text,
               fc=COL[family], ec=COL[f"{family}_edge"], fs=fs,
               weight="bold", radius=0.003, z=z)


def feature_block(ax, x, y, w, h, title, ops, family="orange", dim="", depth=0.007,
                  fs_title=7.2, fs_ops=5.7):
    # Pseudo-3D feature volume: front, top and side remain independent SVG patches.
    top = Polygon(
        [(x, y + h), (x + depth, y + h + depth), (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True, facecolor=COL[f"{family}_side"], edgecolor=COL[f"{family}_edge"],
        linewidth=0.7, transform=ax.transAxes, zorder=4,
    )
    side = Polygon(
        [(x + w, y), (x + w + depth, y + depth), (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True, facecolor=COL[f"{family}_side"], edgecolor=COL[f"{family}_edge"],
        linewidth=0.7, transform=ax.transAxes, zorder=4,
    )
    front = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.005",
        facecolor=COL[family], edgecolor=COL[f"{family}_edge"], linewidth=0.9,
        transform=ax.transAxes, zorder=5,
    )
    for p in (top, side, front):
        ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.015, title, transform=ax.transAxes,
            ha="center", va="top", fontsize=fs_title, fontweight="bold", color=COL["ink"], zorder=7)
    if ops:
        ax.text(x + w / 2, y + h * 0.42, ops, transform=ax.transAxes,
                ha="center", va="center", fontsize=fs_ops, color=COL["ink"], zorder=7)
    if dim:
        ax.text(x + w / 2, y - 0.010, dim, transform=ax.transAxes,
                ha="center", va="top", fontsize=6.2, fontstyle="italic", color=COL["ink"], zorder=7)
    return front


def stack_icon(ax, x, y, w=0.018, h=0.038, family="blue", n=3, z=5):
    for i in range(n - 1, -1, -1):
        dx = i * 0.004
        poly = Polygon(
            [(x + dx, y + dx * 0.45), (x + w + dx, y + 0.005 + dx * 0.45),
             (x + w + dx, y + h + 0.005 + dx * 0.45), (x + dx, y + h + dx * 0.45)],
            closed=True, facecolor=COL[family], edgecolor=COL[f"{family}_edge"],
            linewidth=0.65, transform=ax.transAxes, zorder=z + (n - i),
        )
        ax.add_patch(poly)


def op_circle(ax, x, y, text="+", r=0.008, fs=7.2, z=8):
    c = Circle((x, y), r, transform=ax.transAxes, facecolor=COL["white"],
               edgecolor=COL["line"], linewidth=0.75, zorder=z)
    ax.add_patch(c)
    ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=fs, color=COL["ink"], zorder=z + 1)
    return c


def label_outside(ax, x, y, text, ha="center", va="center", fs=6.2, italic=True):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            fontstyle="italic" if italic else "normal", color=COL["ink"], zorder=9)


def make_figure():
    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title strip.
    ax.text(0.5, 0.983, "MBDNet-P+: Architecture and Core Blocks",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=COL["ink"])
    ax.text(0.5, 0.964, "Three-scale RGB restoration with locally routed motion bias",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=7.0, color=COL["muted"])

    # ------------------------------------------------------------------
    # (a) Main architecture: dominant panel and the only pseudo-3D panel.
    # ------------------------------------------------------------------
    panel(ax, 0.025, 0.675, 0.950, 0.270,
          "Overall MBDNet-P+ network",
          "Feature dimensions are placed outside the blocks; operators inside each stage follow the actual implementation.",
          fc=COL["panel_grey"], ec="#9FAAB3", label="a")

    y_top, y_mid, y_bot = 0.835, 0.775, 0.720
    stack_icon(ax, 0.045, y_mid - 0.022, w=0.020, h=0.045, family="purple")
    label_outside(ax, 0.055, y_mid - 0.038, "Iᴮ ∈ ℝ³×ᴴ×ᵂ", fs=5.9)
    arrow(ax, 0.075, y_mid, 0.092, y_mid)

    feature_block(ax, 0.092, y_mid - 0.030, 0.075, 0.060,
                  "Stem", "3×3 Conv", family="blue", dim="H×W×64")
    arrow(ax, 0.174, y_mid, 0.191, y_mid)

    feature_block(ax, 0.191, y_mid - 0.037, 0.110, 0.074,
                  "Encoder E0", "DWPW × 3", family="orange", dim="H×W×64")

    # Down-1 shown as a 3-D operator prism with the transformation outside.
    arrow(ax, 0.308, y_mid, 0.327, y_mid)
    feature_block(ax, 0.327, y_mid - 0.027, 0.070, 0.054,
                  "Down-1", "DW s=2\nPW", family="purple", dim="64 → 128", fs_title=6.8, fs_ops=5.4)
    arrow(ax, 0.404, y_mid, 0.423, y_mid)

    feature_block(ax, 0.423, y_mid - 0.042, 0.130, 0.084,
                  "Encoder E1", "DWPW × 4  →  MBD-E", family="orange", dim="H/2×W/2×128")

    # Down-2 bends to the bottleneck row.
    orth_arrow(ax, [(0.560, y_mid), (0.580, y_mid), (0.580, y_bot), (0.595, y_bot)],
               color=COL["line"], lw=0.9, ls="-")
    feature_block(ax, 0.595, y_bot - 0.027, 0.072, 0.054,
                  "Down-2", "DW s=2\nPW", family="purple", dim="128 → 256", fs_title=6.8, fs_ops=5.4)
    arrow(ax, 0.674, y_bot, 0.691, y_bot)

    # Bottleneck is wider and presents the exact internal order as an arrowed strip.
    bx, by, bw, bh = 0.691, y_bot - 0.045, 0.260, 0.090
    feature_block(ax, bx, by, bw, bh, "Bottleneck B", "", family="yellow", dim="H/4×W/4×256")
    chips = [
        (0.705, 0.055, "DWPW × 12", "orange"),
        (0.765, 0.034, "LRC", "green"),
        (0.804, 0.040, "MBD-1", "yellow"),
        (0.849, 0.047, "DWPW × 2", "orange"),
        (0.901, 0.040, "MBD-2", "yellow"),
    ]
    cy = by + 0.032
    for i, (cx, cw, txt, fam) in enumerate(chips):
        chip(ax, cx, cy, cw, txt, fam, fs=5.2)
        if i < len(chips) - 1:
            arrow(ax, cx + cw, cy + 0.009, chips[i + 1][0] - 0.002, cy + 0.009, lw=0.65, z=9)

    # Decoder reads right-to-left on the upper row, preserving a compact U geometry.
    orth_arrow(ax, [(0.951, y_bot), (0.963, y_bot), (0.963, y_top), (0.944, y_top)],
               color=COL["line"], lw=0.9, ls="-")
    feature_block(ax, 0.868, y_top - 0.027, 0.076, 0.054,
                  "Up-1", "PW → PS(2)", family="purple", dim="256 → 128", fs_title=6.8, fs_ops=5.4)
    arrow(ax, 0.868, y_top, 0.847, y_top)
    op_circle(ax, 0.837, y_top, "+", r=0.008)
    arrow(ax, 0.829, y_top, 0.808, y_top)
    feature_block(ax, 0.665, y_top - 0.040, 0.143, 0.080,
                  "Decoder D1", "DWPW × 4  →  MBD-D", family="orange", dim="H/2×W/2×128")
    arrow(ax, 0.665, y_top, 0.644, y_top)
    feature_block(ax, 0.568, y_top - 0.027, 0.076, 0.054,
                  "Up-2", "PW → PS(2)", family="purple", dim="128 → 64", fs_title=6.8, fs_ops=5.4)
    arrow(ax, 0.568, y_top, 0.547, y_top)
    op_circle(ax, 0.537, y_top, "+", r=0.008)
    arrow(ax, 0.529, y_top, 0.508, y_top)
    feature_block(ax, 0.393, y_top - 0.037, 0.115, 0.074,
                  "Decoder D0", "DWPW × 3", family="orange", dim="H×W×64")
    arrow(ax, 0.393, y_top, 0.374, y_top)
    feature_block(ax, 0.298, y_top - 0.030, 0.076, 0.060,
                  "Head", "3×3 Conv", family="blue", dim="64 → 3")
    arrow(ax, 0.298, y_top, 0.278, y_top)
    op_circle(ax, 0.268, y_top, "+", r=0.008)
    arrow(ax, 0.260, y_top, 0.235, y_top)
    stack_icon(ax, 0.205, y_top - 0.021, w=0.021, h=0.043, family="blue")
    label_outside(ax, 0.216, y_top - 0.039, "Î ∈ ℝ³×ᴴ×ᵂ", fs=5.9)

    # Skip paths attach to addition nodes. Labels are short and live outside stage bodies.
    orth_arrow(ax, [(0.246, y_mid + 0.038), (0.246, 0.912), (0.537, 0.912), (0.537, y_top + 0.009)],
               color=COL["blue_edge"], lw=0.75, ls="--")
    label_outside(ax, 0.390, 0.916, "full-resolution skip  F₀", fs=5.8)
    orth_arrow(ax, [(0.488, y_mid + 0.044), (0.488, 0.892), (0.837, 0.892), (0.837, y_top + 0.009)],
               color=COL["blue_edge"], lw=0.75, ls="--")
    label_outside(ax, 0.660, 0.896, "half-resolution skip  F₁", fs=5.8)
    orth_arrow(ax, [(0.055, y_mid - 0.026), (0.055, 0.691), (0.268, 0.691), (0.268, y_top - 0.009)],
               color=COL["guide"], lw=0.65, ls="--")
    label_outside(ax, 0.155, 0.686, "global image residual", fs=5.7)

    # Callout anchors: integrated visual bridges instead of detached explanation boxes.
    orth_arrow(ax, [(0.248, y_mid - 0.038), (0.248, 0.667), (0.180, 0.645)],
               color=COL["orange_edge"], lw=0.65, ls=":")
    orth_arrow(ax, [(0.488, y_mid - 0.046), (0.488, 0.660), (0.650, 0.645)],
               color=COL["yellow_edge"], lw=0.65, ls=":")
    orth_arrow(ax, [(0.765, by - 0.003), (0.765, 0.660), (0.815, 0.645)],
               color=COL["green_edge"], lw=0.65, ls=":")

    # ------------------------------------------------------------------
    # (b) DWPW block: no prose box, explicit channel transformations.
    # ------------------------------------------------------------------
    panel(ax, 0.025, 0.435, 0.950, 0.210,
          "DWPW generic restoration block",
          "Two residual sublayers; SG denotes channel split followed by element-wise multiplication.",
          fc=COL["panel_blue"], ec="#82BBD5", label="b")

    def dwpw_row(y, left, labels, widths, families, residual_label):
        x = left
        centres = []
        for txt, w, fam in zip(labels, widths, families):
            box(ax, x, y, w, 0.031, txt, fc=COL[fam], ec=COL[f"{fam}_edge"], fs=5.9,
                radius=0.003)
            centres.append((x, w))
            x += w
            if txt != labels[-1]:
                arrow(ax, x + 0.002, y + 0.0155, x + 0.014, y + 0.0155, lw=0.65)
                x += 0.016
        op_circle(ax, x + 0.012, y + 0.0155, "+", r=0.007)
        arrow(ax, x + 0.019, y + 0.0155, x + 0.041, y + 0.0155, lw=0.65)
        box(ax, x + 0.043, y, 0.045, 0.031, residual_label, fs=6.4, italic=True)
        # identity path
        ax.plot([left - 0.012, left - 0.012, x + 0.012],
                [y + 0.0155, y - 0.012, y - 0.012], transform=ax.transAxes,
                color=COL["guide"], lw=0.65, zorder=3)
        arrow(ax, x + 0.012, y - 0.012, x + 0.012, y + 0.008, color=COL["guide"], lw=0.65, z=3)

    ax.text(0.045, 0.575, "Spatial mixing", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.0, fontweight="bold", color=COL["ink"])
    dwpw_row(
        0.535, 0.057,
        ["LN", "PW  C→4C", "DW 3×3", "SimpleGate\n4C→2C", "PW  2C→C", "β ⊙"],
        [0.045, 0.082, 0.070, 0.090, 0.082, 0.045],
        ["panel_grey", "blue", "orange", "pink", "blue", "yellow"],
        "Y",
    )
    label_outside(ax, 0.518, 0.516, "Y = X + β ⊙ PW[SG(DW(PW(LN(X))))]", ha="left", fs=6.3)

    ax.text(0.045, 0.492, "Channel mixing", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.0, fontweight="bold", color=COL["ink"])
    dwpw_row(
        0.454, 0.057,
        ["LN", "PW  C→4C", "SimpleGate\n4C→2C", "PW  2C→C", "γ ⊙"],
        [0.045, 0.082, 0.090, 0.082, 0.045],
        ["panel_grey", "blue", "pink", "blue", "yellow"],
        "Z",
    )
    label_outside(ax, 0.518, 0.463, "Z = Y + γ ⊙ PW[SG(PW(LN(Y)))]", ha="left", fs=6.3)

    # Compact visual definition of SG replaces the long prose sentence.
    sx, sy = 0.700, 0.527
    ax.text(sx, 0.575, "SimpleGate", transform=ax.transAxes, ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=COL["ink"])
    box(ax, sx, sy, 0.055, 0.031, "V", fs=6.5, italic=True)
    arrow(ax, sx + 0.057, sy + 0.0155, sx + 0.078, sy + 0.0155, lw=0.65)
    box(ax, sx + 0.080, sy + 0.075, 0.031, 0.026, "V₁", fs=6.1, italic=True)
    box(ax, sx + 0.080, sy - 0.012, 0.031, 0.026, "V₂", fs=6.1, italic=True)
    arrow(ax, sx + 0.078, sy + 0.0155, sx + 0.080, sy + 0.090, lw=0.6)
    arrow(ax, sx + 0.078, sy + 0.0155, sx + 0.080, sy + 0.003, lw=0.6)
    op_circle(ax, sx + 0.137, sy + 0.039, "×", r=0.008)
    arrow(ax, sx + 0.111, sy + 0.090, sx + 0.130, sy + 0.047, lw=0.6)
    arrow(ax, sx + 0.111, sy + 0.003, sx + 0.130, sy + 0.031, lw=0.6)
    arrow(ax, sx + 0.145, sy + 0.039, sx + 0.169, sy + 0.039, lw=0.65)
    box(ax, sx + 0.171, sy + 0.023, 0.066, 0.031, "V₁ ⊙ V₂", fc=COL["pink"], ec=COL["pink_edge"], fs=6.1)
    label_outside(ax, sx + 0.118, 0.493, "split along channels", fs=5.8)

    # ------------------------------------------------------------------
    # (c) MBD: descriptive variables and three grouped stages.
    # ------------------------------------------------------------------
    panel(ax, 0.025, 0.165, 0.950, 0.250,
          "Local motion-biased deblurring block",
          "Generic restoration and horizontal-support estimation are mixed by pixel-wise routing.",
          fc=COL["panel_orange"], ec="#EEA17F", label="c")

    # Group 1: response extraction.
    gx0, gy0, gh = 0.045, 0.205, 0.150
    ax.text(gx0, 0.365, "1  Response extraction", transform=ax.transAxes,
            fontsize=7.0, fontweight="bold", color=COL["ink"])
    box(ax, gx0, 0.273, 0.050, 0.035, "X", fs=6.7, italic=True)
    arrow(ax, gx0 + 0.052, 0.2905, gx0 + 0.073, 0.2905)
    box(ax, gx0 + 0.075, 0.273, 0.060, 0.035, "LN", fc=COL["panel_grey"], ec="#83919B", fs=6.1)
    arrow(ax, gx0 + 0.137, 0.2905, gx0 + 0.158, 0.2905)
    box(ax, gx0 + 0.160, 0.273, 0.057, 0.035, "X̄", fs=6.7, italic=True)
    ax.plot([gx0 + 0.219, gx0 + 0.219], [0.225, 0.347], transform=ax.transAxes,
            color=COL["line"], lw=0.8, zorder=3)
    branches = [
        (0.337, "DW 3×3", "G", "isotropic response"),
        (0.295, "DW 1×5", "Bₛ", "short horizontal support"),
        (0.253, "DW 1×5, d=3", "Bₗ", "long horizontal support"),
        (0.211, "Router DW 3×3", "R", "routing context"),
    ]
    for yy, op, out, note in branches:
        arrow(ax, gx0 + 0.219, yy + 0.0155, gx0 + 0.240, yy + 0.0155, lw=0.65)
        box(ax, gx0 + 0.242, yy, 0.100, 0.031, op,
            fc=COL["green"] if "Router" in op else COL["orange"],
            ec=COL["green_edge"] if "Router" in op else COL["orange_edge"], fs=5.7)
        arrow(ax, gx0 + 0.344, yy + 0.0155, gx0 + 0.365, yy + 0.0155, lw=0.65)
        box(ax, gx0 + 0.367, yy, 0.042, 0.031, out, fs=6.2, italic=True)
        label_outside(ax, gx0 + 0.415, yy + 0.0155, note, ha="left", fs=5.2, italic=False)

    # Group 2: router. Variables are named by role instead of anonymous H/F/U.
    rx = 0.515
    ax.text(rx, 0.365, "2  Pixel-wise router", transform=ax.transAxes,
            fontsize=7.0, fontweight="bold", color=COL["ink"])
    box(ax, rx, 0.310, 0.073, 0.033, "PW  C→2", fc=COL["purple"], ec=COL["purple_edge"], fs=5.8)
    arrow(ax, rx + 0.075, 0.3265, rx + 0.095, 0.3265, lw=0.65)
    box(ax, rx + 0.097, 0.310, 0.070, 0.033, "Softmax", fc=COL["purple"], ec=COL["purple_edge"], fs=5.8)
    arrow(ax, rx + 0.169, 0.3265, rx + 0.190, 0.3265, lw=0.65)
    box(ax, rx + 0.192, 0.310, 0.070, 0.033, "αₛ, αₗ", fs=6.2, italic=True)
    label_outside(ax, rx + 0.131, 0.298, "support weights", fs=5.3, italic=False)

    box(ax, rx, 0.235, 0.073, 0.033, "PW  C→1", fc=COL["pink"], ec=COL["pink_edge"], fs=5.8)
    arrow(ax, rx + 0.075, 0.2515, rx + 0.095, 0.2515, lw=0.65)
    box(ax, rx + 0.097, 0.235, 0.070, 0.033, "Sigmoid", fc=COL["pink"], ec=COL["pink_edge"], fs=5.8)
    arrow(ax, rx + 0.169, 0.2515, rx + 0.190, 0.2515, lw=0.65)
    box(ax, rx + 0.192, 0.235, 0.070, 0.033, "a", fs=6.5, italic=True)
    label_outside(ax, rx + 0.131, 0.223, "motion confidence", fs=5.3, italic=False)

    # Group 3: explicit fusion with short equations typeset in Times-like math style.
    fx = 0.795
    ax.text(fx, 0.365, "3  Adaptive fusion", transform=ax.transAxes,
            fontsize=7.0, fontweight="bold", color=COL["ink"])
    box(ax, fx, 0.315, 0.158, 0.036, "Bₕ = αₛ⊙Bₛ + αₗ⊙Bₗ",
        fc=COL["yellow"], ec=COL["yellow_edge"], fs=6.1, italic=True)
    arrow(ax, fx + 0.079, 0.313, fx + 0.079, 0.288, lw=0.65)
    box(ax, fx, 0.248, 0.158, 0.036, "M = (1−a)⊙G + a⊙Bₕ",
        fc=COL["yellow"], ec=COL["yellow_edge"], fs=6.1, italic=True)
    arrow(ax, fx + 0.079, 0.246, fx + 0.079, 0.221, lw=0.65)
    box(ax, fx, 0.181, 0.158, 0.036, "Y = X + ρ⊙PW(M)",
        fc=COL["blue"], ec=COL["blue_edge"], fs=6.1, italic=True)

    # Routing lines connect semantically, avoiding repeated ambiguous tensor labels.
    orth_arrow(ax, [(0.455, 0.2265), (0.490, 0.2265), (0.490, 0.2515), (rx, 0.2515)],
               color=COL["green_edge"], lw=0.65, ls="-")
    orth_arrow(ax, [(0.455, 0.2265), (0.480, 0.2265), (0.480, 0.3265), (rx, 0.3265)],
               color=COL["green_edge"], lw=0.65, ls="-")
    orth_arrow(ax, [(rx + 0.262, 0.3265), (0.765, 0.3265), (0.765, 0.333), (fx, 0.333)],
               color=COL["purple_edge"], lw=0.65, ls="-")
    orth_arrow(ax, [(rx + 0.262, 0.2515), (0.775, 0.2515), (0.775, 0.266), (fx, 0.266)],
               color=COL["pink_edge"], lw=0.65, ls="-")

    # ------------------------------------------------------------------
    # (d) LRC + operators + legend: all notation guidance grouped together.
    # ------------------------------------------------------------------
    panel(ax, 0.025, 0.025, 0.950, 0.120,
          "Context, scale operators and notation",
          "All symbols used in the overview are defined here; implementation constraints are moved to the manuscript text.",
          fc=COL["panel_green"], ec="#8DBB96", label="d")

    # LRC compact graph.
    ax.text(0.045, 0.105, "LRC at H/4", transform=ax.transAxes,
            fontsize=6.8, fontweight="bold", color=COL["ink"])
    box(ax, 0.045, 0.058, 0.043, 0.028, "LN", fc=COL["panel_grey"], ec="#83919B", fs=5.7)
    arrow(ax, 0.090, 0.072, 0.106, 0.072, lw=0.6)
    box(ax, 0.108, 0.083, 0.055, 0.026, "DW d=1", fc=COL["green"], ec=COL["green_edge"], fs=5.3)
    box(ax, 0.108, 0.045, 0.055, 0.026, "DW d=2", fc=COL["green"], ec=COL["green_edge"], fs=5.3)
    arrow(ax, 0.163, 0.096, 0.184, 0.076, lw=0.55)
    arrow(ax, 0.163, 0.058, 0.184, 0.068, lw=0.55)
    op_circle(ax, 0.193, 0.072, "C", r=0.007, fs=5.6)
    arrow(ax, 0.200, 0.072, 0.217, 0.072, lw=0.6)
    box(ax, 0.219, 0.058, 0.060, 0.028, "PW 2C→C", fc=COL["blue"], ec=COL["blue_edge"], fs=5.3)
    arrow(ax, 0.281, 0.072, 0.297, 0.072, lw=0.6)
    box(ax, 0.299, 0.058, 0.043, 0.028, "ξ⊙", fc=COL["yellow"], ec=COL["yellow_edge"], fs=5.7, italic=True)
    arrow(ax, 0.344, 0.072, 0.360, 0.072, lw=0.6)
    op_circle(ax, 0.369, 0.072, "+", r=0.007)

    # Scale operators grouped beside symbol key.
    ax.text(0.405, 0.105, "Scale operators", transform=ax.transAxes,
            fontsize=6.8, fontweight="bold", color=COL["ink"])
    box(ax, 0.405, 0.073, 0.052, 0.025, "Down", fc=COL["purple"], ec=COL["purple_edge"], fs=5.4, weight="bold")
    ax.text(0.463, 0.0855, "DW 3×3, s=2  →  PW 1×1", transform=ax.transAxes,
            ha="left", va="center", fontsize=5.3, color=COL["ink"])
    box(ax, 0.405, 0.039, 0.052, 0.025, "Up", fc=COL["purple"], ec=COL["purple_edge"], fs=5.4, weight="bold")
    ax.text(0.463, 0.0515, "PW 1×1  →  PixelShuffle(2)", transform=ax.transAxes,
            ha="left", va="center", fontsize=5.3, color=COL["ink"])

    ax.text(0.650, 0.105, "Notation", transform=ax.transAxes,
            fontsize=6.8, fontweight="bold", color=COL["ink"])
    legend_items = [
        (0.650, 0.077, "orange", "DWPW"),
        (0.750, 0.077, "yellow", "MBD"),
        (0.840, 0.077, "green", "LRC"),
        (0.650, 0.044, "purple", "Down / Up"),
    ]
    for lx, ly, fam, txt in legend_items:
        box(ax, lx, ly, 0.038, 0.022, "", fc=COL[fam], ec=COL[f"{fam}_edge"], fs=5.0)
        ax.text(lx + 0.044, ly + 0.011, txt, transform=ax.transAxes,
                ha="left", va="center", fontsize=5.4, color=COL["ink"])
    ax.text(0.840, 0.055, "+  addition     C  concat.     ⊙  multiply",
            transform=ax.transAxes, ha="left", va="center", fontsize=5.1, color=COL["ink"])

    fig.savefig(OUT.with_suffix(".svg"), format="svg", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), format="pdf", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), format="png", dpi=360, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()

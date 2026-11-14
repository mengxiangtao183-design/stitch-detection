from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


OUT_DIR = Path(r"H:\C3Netqing\NAFNet\docs\paper_deblurring_figures")
OUT_PREFIX = OUT_DIR / "MBDNetPPlus_01_main_network_nature"
SKILL_SCRIPTS = Path(r"C:\Users\user\.codex\skills\nature-figure\scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


COLORS = {
    "ink": "#2D333A",
    "muted": "#69727D",
    "guide": "#CBD1D8",
    "lane": "#FBFCFE",
    "stem_fill": "#CFE8F6",
    "stem_edge": "#4F8FB3",
    "dwpw_fill": "#F8D9CC",
    "dwpw_edge": "#C97959",
    "mbd_fill": "#F8E7A1",
    "mbd_edge": "#B88A1C",
    "lrc_fill": "#D8ECD7",
    "lrc_edge": "#5B9460",
    "sample_fill": "#F4E7F0",
    "sample_edge": "#956B89",
    "scale_fill": "#E7DCF0",
    "scale_edge": "#87669D",
    "skip": "#447BA8",
}


def rounded(ax, x, y, w, h, face, edge, radius=0.6, lw=0.9, z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.08,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=None, lw=0.9, style="-", rad=0.0, z=2, ms=8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=style,
        color=color or COLORS["ink"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def poly_icon(ax, x, y, w=1.7, h=1.25, kind="dwpw", z=5):
    if kind == "dwpw":
        face, edge = COLORS["dwpw_fill"], COLORS["dwpw_edge"]
    elif kind == "mbd":
        face, edge = COLORS["mbd_fill"], COLORS["mbd_edge"]
    elif kind == "lrc":
        face, edge = COLORS["lrc_fill"], COLORS["lrc_edge"]
    else:
        raise ValueError(kind)
    slant = 0.22 * w
    patch = Polygon(
        [(x + slant, y + h), (x + w, y + h), (x + w - slant, y), (x, y)],
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.8,
        zorder=z,
    )
    ax.add_patch(patch)
    if kind == "dwpw":
        for frac in (0.28, 0.52, 0.76):
            yy = y + frac * h
            ax.plot(
                [x + 0.22 * w, x + 0.73 * w],
                [yy, yy],
                color=edge,
                linewidth=0.45,
                zorder=z + 1,
            )
    elif kind == "mbd":
        for frac in (0.34, 0.68):
            xx = x + frac * w
            ax.plot([xx, xx - 0.12 * w], [y + 0.14 * h, y + 0.86 * h], color=edge, lw=0.45, zorder=z + 1)
        for frac in (0.33, 0.66):
            yy = y + frac * h
            ax.plot([x + 0.15 * w, x + 0.8 * w], [yy, yy], color=edge, lw=0.45, zorder=z + 1)
    else:
        for dx, dy in ((0.30, 0.68), (0.58, 0.68), (0.24, 0.32), (0.68, 0.32)):
            ax.add_patch(Circle((x + dx * w, y + dy * h), 0.07 * h, fc=edge, ec="none", zorder=z + 1))
    return patch


def image_stack(ax, x, y, w=2.6, h=5.0, restored=False):
    edge = "#608CB5" if restored else COLORS["sample_edge"]
    faces = ["#E7F2FF", "#F4F9FF"] if restored else ["#F4E7F0", "#FFF8FC"]
    for i in range(2):
        dx, dy = i * 0.55, i * 0.45
        ax.add_patch(
            Polygon(
                [(x + dx + 0.35, y + dy + h), (x + dx + w, y + dy + h), (x + dx + w - 0.35, y + dy), (x + dx, y + dy)],
                closed=True,
                facecolor=faces[i],
                edgecolor=edge,
                linewidth=0.8,
                zorder=5 + i,
            )
        )


def stage_group(ax, x, y, w, h, title, shape_text, icons, edge):
    rounded(ax, x, y, w, h, "white", edge, radius=0.65, lw=0.9)
    ax.text(x + 0.7, y + h - 0.9, title, fontsize=7.3, weight="bold", color=COLORS["ink"], va="top", zorder=7)
    ax.text(x + w - 0.6, y + h - 0.9, shape_text, fontsize=5.4, color=COLORS["muted"], ha="right", va="top", zorder=7)
    total_w = sum(item[1] for item in icons) + 0.8 * (len(icons) - 1)
    cursor = x + (w - total_w) / 2
    center_y = y + 1.45
    last_right = None
    for kind, icon_w in icons:
        poly_icon(ax, cursor, center_y, icon_w, 1.35, kind)
        if last_right is not None:
            arrow(ax, (last_right + 0.1, center_y + 0.68), (cursor - 0.12, center_y + 0.68), lw=0.65, ms=6, z=4)
        last_right = cursor + icon_w
        cursor += icon_w + 0.8


def add_node(ax, x, y):
    ax.add_patch(Circle((x, y), 0.75, fc="white", ec=COLORS["ink"], lw=0.8, zorder=8))
    ax.text(x, y - 0.02, "+", ha="center", va="center", fontsize=8, color=COLORS["ink"], zorder=9)


def build_figure():
    fig = plt.figure(figsize=(7.09, 4.15), facecolor="white")
    ax = fig.add_axes([0.012, 0.025, 0.976, 0.95])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 58)
    ax.axis("off")

    # Title and compact inference note.
    ax.text(60, 56.8, "MBDNet-P+ architecture", ha="center", va="center", fontsize=10.5, weight="bold", color=COLORS["ink"])
    ax.text(60, 55.25, "Three-scale encoder–decoder with locally routed motion-biased restoration", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])
    ax.text(1.8, 53.8, "RGB-only inference · no mask/domain label/degradation parameter", fontsize=5.8, color=COLORS["muted"], va="center")

    # Resolution lanes. Very pale boundaries avoid confusion with feature flow.
    for y, h, label, xmax in ((42.5, 10.3, "H × W", 118.5), (26.0, 12.2, "H/2 × W/2", 105.5), (9.4, 12.0, "H/4 × W/4", 75.5)):
        rounded(ax, 1.2, y, xmax - 1.2, h, COLORS["lane"], COLORS["guide"], radius=0.8, lw=0.55, z=0)
        ax.text(2.1, y + h - 1.1, label, fontsize=6.8, weight="bold", color=COLORS["muted"], va="top")

    # Main topology arrows are drawn before boxes.
    arrow(ax, (5.8, 47.3), (7.3, 47.3), lw=0.95)
    arrow(ax, (13.0, 47.3), (15.1, 47.3), lw=0.95)
    arrow(ax, (28.0, 47.3), (30.2, 33.0), lw=0.95)
    arrow(ax, (34.6, 33.0), (36.1, 33.0), lw=0.95)
    arrow(ax, (54.0, 33.0), (55.8, 19.0), lw=0.95)
    arrow(ax, (60.2, 19.0), (26.6, 15.2), lw=0.95, rad=-0.04)
    arrow(ax, (69.0, 15.2), (71.0, 20.4), lw=0.95)
    arrow(ax, (75.6, 20.4), (77.2, 33.0), lw=0.95)
    arrow(ax, (78.0, 33.0), (80.0, 33.0), lw=0.95)
    arrow(ax, (98.0, 33.0), (100.0, 42.2), lw=0.95)
    arrow(ax, (104.6, 42.2), (106.0, 47.3), lw=0.95)
    arrow(ax, (106.8, 47.3), (108.0, 47.3), lw=0.95)
    arrow(ax, (114.0, 47.3), (115.0, 47.3), lw=0.95)
    arrow(ax, (118.0, 47.3), (119.1, 47.3), lw=0.95)

    # Skip connections and global residual use separate visual grammars.
    arrow(ax, (27.0, 50.6), (78.0, 33.8), color=COLORS["skip"], lw=0.8, style=(0, (4, 3)), rad=-0.14, ms=7)
    ax.text(51.5, 51.45, "F₀ skip", fontsize=5.6, color=COLORS["skip"], ha="center")
    arrow(ax, (52.8, 35.8), (78.0, 33.8), color=COLORS["skip"], lw=0.8, style=(0, (4, 3)), rad=-0.08, ms=7)
    ax.text(65.0, 37.4, "F₁ skip", fontsize=5.6, color=COLORS["skip"], ha="center")
    arrow(ax, (3.5, 44.2), (116.5, 46.5), color=COLORS["muted"], lw=0.7, style=(0, (1.5, 2.5)), rad=-0.2, ms=7)
    ax.text(58.0, 23.15, "Global image residual:  Î = Iᴮ + R", fontsize=5.8, color=COLORS["muted"], ha="center")

    # Input and stem.
    image_stack(ax, 2.0, 44.2, 2.6, 5.4)
    ax.text(3.6, 43.7, "Blurred RGB", ha="center", fontsize=6.3, weight="bold")
    ax.text(3.6, 42.9, "B×3×H×W", ha="center", fontsize=5.3, color=COLORS["muted"])
    rounded(ax, 7.3, 44.5, 5.7, 5.6, COLORS["stem_fill"], COLORS["stem_edge"], radius=0.65, lw=0.9)
    ax.text(10.15, 48.9, "Stem", ha="center", fontsize=7.3, weight="bold")
    ax.text(10.15, 47.3, "3×3 Conv", ha="center", fontsize=6.2)
    ax.text(10.15, 45.8, "3 → 64", ha="center", fontsize=5.8, color=COLORS["muted"])

    # Encoder and decoder stages. Every repeated block is shown as an individual glyph.
    stage_group(ax, 15.1, 44.0, 12.9, 6.7, "Encoder E0", "B×64×H×W", [("dwpw", 2.1)] * 3, COLORS["stem_edge"])
    ax.add_patch(Circle((27.0, 47.3), 0.22, fc=COLORS["ink"], ec="none", zorder=8))
    ax.text(27.0, 44.65, "F₀", fontsize=5.5, ha="center", color=COLORS["muted"])

    rounded(ax, 30.2, 30.3, 4.4, 5.4, COLORS["scale_fill"], COLORS["scale_edge"], radius=0.55, lw=0.9)
    ax.text(32.4, 34.7, "Down-1", ha="center", fontsize=6.8, weight="bold")
    ax.text(32.4, 33.25, "DW 3×3, s=2", ha="center", fontsize=5.3)
    ax.text(32.4, 32.15, "PW 1×1", ha="center", fontsize=5.3)
    ax.text(32.4, 31.05, "64 → 128", ha="center", fontsize=5.2, color=COLORS["muted"])

    stage_group(ax, 36.1, 29.3, 17.9, 7.4, "Encoder E1", "B×128×H/2×W/2", [("dwpw", 1.9)] * 4 + [("mbd", 2.2)], COLORS["dwpw_edge"])
    ax.add_patch(Circle((52.8, 33.0), 0.22, fc=COLORS["ink"], ec="none", zorder=8))
    ax.text(52.8, 30.0, "F₁", fontsize=5.5, ha="center", color=COLORS["muted"])

    rounded(ax, 55.8, 16.3, 4.4, 5.4, COLORS["scale_fill"], COLORS["scale_edge"], radius=0.55, lw=0.9)
    ax.text(58.0, 20.7, "Down-2", ha="center", fontsize=6.8, weight="bold")
    ax.text(58.0, 19.25, "DW 3×3, s=2", ha="center", fontsize=5.3)
    ax.text(58.0, 18.15, "PW 1×1", ha="center", fontsize=5.3)
    ax.text(58.0, 17.05, "128 → 256", ha="center", fontsize=5.2, color=COLORS["muted"])

    # Quarter-resolution bottleneck: 12 DWPW + LRC + MBD + 2 DWPW + MBD.
    rounded(ax, 24.8, 11.0, 44.2, 8.1, "white", COLORS["mbd_edge"], radius=0.65, lw=0.9)
    ax.text(25.6, 18.35, "Quarter-resolution bottleneck", fontsize=7.3, weight="bold", va="top")
    ax.text(68.2, 18.35, "B×256×H/4×W/4", fontsize=5.4, color=COLORS["muted"], ha="right", va="top")
    x = 26.6
    previous_right = None
    for _ in range(12):
        poly_icon(ax, x, 14.55, 1.45, 1.18, "dwpw")
        if previous_right is not None:
            arrow(ax, (previous_right + 0.04, 15.14), (x - 0.08, 15.14), lw=0.5, ms=4.5, z=4)
        previous_right = x + 1.45
        x += 1.82
    ax.text(36.6, 13.65, "12 DWPW blocks", fontsize=5.6, color=COLORS["muted"], ha="center")
    arrow(ax, (previous_right + 0.1, 15.14), (49.0, 15.14), lw=0.6, ms=5)
    poly_icon(ax, 49.0, 14.35, 2.05, 1.55, "lrc")
    arrow(ax, (51.15, 15.14), (52.25, 15.14), lw=0.6, ms=5)
    poly_icon(ax, 52.25, 14.35, 2.05, 1.55, "mbd")
    arrow(ax, (54.4, 15.14), (55.55, 15.14), lw=0.6, ms=5)
    poly_icon(ax, 55.55, 14.55, 1.45, 1.18, "dwpw")
    arrow(ax, (57.1, 15.14), (57.85, 15.14), lw=0.5, ms=4.5)
    poly_icon(ax, 57.85, 14.55, 1.45, 1.18, "dwpw")
    arrow(ax, (59.4, 15.14), (60.55, 15.14), lw=0.6, ms=5)
    poly_icon(ax, 60.55, 14.35, 2.05, 1.55, "mbd")
    arrow(ax, (62.7, 15.14), (68.2, 15.14), lw=0.7, ms=6)
    for xx, label in ((50.0, "LRC"), (53.25, "MBD-1"), (57.4, "Refine"), (61.55, "MBD-2")):
        ax.text(xx, 13.25, label, fontsize=5.5, ha="center", color=COLORS["muted"])

    rounded(ax, 71.0, 17.7, 4.6, 5.5, COLORS["scale_fill"], COLORS["scale_edge"], radius=0.55, lw=0.9)
    ax.text(73.3, 22.15, "Up-1", ha="center", fontsize=6.8, weight="bold")
    ax.text(73.3, 20.7, "1×1: 256→512", ha="center", fontsize=5.1)
    ax.text(73.3, 19.55, "PixelShuffle(2)", ha="center", fontsize=5.1)
    ax.text(73.3, 18.45, "→128 channels", ha="center", fontsize=5.1, color=COLORS["muted"])
    add_node(ax, 78.0, 33.0)

    stage_group(ax, 80.0, 29.3, 18.0, 7.4, "Decoder D1", "B×128×H/2×W/2", [("dwpw", 1.9)] * 4 + [("mbd", 2.2)], COLORS["dwpw_edge"])
    rounded(ax, 100.0, 39.5, 4.6, 5.5, COLORS["scale_fill"], COLORS["scale_edge"], radius=0.55, lw=0.9)
    ax.text(102.3, 43.95, "Up-2", ha="center", fontsize=6.8, weight="bold")
    ax.text(102.3, 42.5, "1×1: 128→256", ha="center", fontsize=5.1)
    ax.text(102.3, 41.35, "PixelShuffle(2)", ha="center", fontsize=5.1)
    ax.text(102.3, 40.25, "→64 channels", ha="center", fontsize=5.1, color=COLORS["muted"])
    add_node(ax, 106.0, 47.3)

    stage_group(ax, 108.0, 44.0, 6.0, 6.7, "Decoder D0", "C=64", [("dwpw", 1.25)] * 3, COLORS["dwpw_edge"])
    rounded(ax, 115.0, 44.5, 3.0, 5.6, COLORS["stem_fill"], COLORS["stem_edge"], radius=0.55, lw=0.9)
    ax.text(116.5, 48.9, "Head", ha="center", fontsize=6.7, weight="bold")
    ax.text(116.5, 47.3, "3×3", ha="center", fontsize=5.7)
    ax.text(116.5, 45.85, "64 → 3", ha="center", fontsize=5.2, color=COLORS["muted"])
    add_node(ax, 119.1, 47.3)
    image_stack(ax, 119.3, 44.45, 2.1, 4.9, restored=True)
    ax.text(119.6, 43.75, "Restored", ha="center", fontsize=5.8, weight="bold")

    # Shared legend in the lower-right, matching the visual language of the reference figure.
    rounded(ax, 76.5, 1.1, 41.5, 11.0, "white", COLORS["muted"], radius=0.6, lw=0.8, z=3)
    ax.text(78.0, 11.15, "Legend", fontsize=7.2, weight="bold", va="top")
    legend_items = [
        ("dwpw", "DWPW generic restoration block", 78.0, 8.3),
        ("mbd", "MBD local motion-biased block", 78.0, 5.8),
        ("lrc", "LRC low-resolution context", 78.0, 3.3),
    ]
    for kind, label, xx, yy in legend_items:
        poly_icon(ax, xx, yy, 2.3, 1.55, kind)
        ax.text(xx + 3.3, yy + 0.75, label, fontsize=5.8, va="center")
    rounded(ax, 99.0, 8.25, 2.7, 1.6, COLORS["scale_fill"], COLORS["scale_edge"], radius=0.25, lw=0.75)
    ax.text(102.6, 9.05, "Downsample", fontsize=5.8, va="center")
    rounded(ax, 99.0, 5.75, 2.7, 1.6, COLORS["scale_fill"], COLORS["scale_edge"], radius=0.25, lw=0.75)
    ax.text(102.6, 6.55, "PixelShuffle upsample", fontsize=5.8, va="center")
    ax.add_patch(Circle((100.35, 4.0), 0.65, fc="white", ec=COLORS["ink"], lw=0.75))
    ax.text(100.35, 4.0, "+", ha="center", va="center", fontsize=7)
    ax.text(102.6, 4.0, "Element-wise addition", fontsize=5.8, va="center")
    arrow(ax, (99.0, 2.1), (101.6, 2.1), color=COLORS["skip"], lw=0.75, style=(0, (4, 3)), ms=6)
    ax.text(102.6, 2.1, "Cross-scale skip", fontsize=5.8, va="center")

    ax.text(2.0, 6.2, "Topology", fontsize=7.0, weight="bold", color=COLORS["ink"])
    ax.text(2.0, 4.5, "64 → 128 → 256 channels", fontsize=6.2, color=COLORS["muted"])
    ax.text(2.0, 2.9, "28 DWPW · 4 MBD · 1 LRC", fontsize=6.2, color=COLORS["muted"])
    ax.text(2.0, 1.3, "13.70 M parameters", fontsize=6.2, color=COLORS["muted"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(OUT_PREFIX) + ".alignment-layout.json",
        overlay_svg=str(OUT_PREFIX) + ".alignment-overlay.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(str(OUT_PREFIX) + ".svg", format="svg", bbox_inches=None)
    fig.savefig(str(OUT_PREFIX) + ".pdf", format="pdf", bbox_inches=None)
    fig.savefig(str(OUT_PREFIX) + ".png", format="png", dpi=300, bbox_inches=None)
    fig.savefig(str(OUT_PREFIX) + ".tiff", format="tiff", dpi=600, bbox_inches=None, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    build_figure()

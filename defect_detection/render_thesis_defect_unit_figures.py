"""Render thesis-ready figures for the trajectory self-reference defect unit.

All labels use the final paper terminology: MBDNet and ACHNet.  The script
creates a detailed vector structure diagram, a quantitative downstream-impact
figure, and four representative qualitative rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(r"H:\C3Netqing")
DATA = Path(r"H:\DATASET\quexian\dataset_96_320")
RESULT = ROOT / "results/defect_detection/tsr_defect_plus_final_normal1700_100t"
FIG_DIR = ROOT / "docs/figures/trajectory_self_reference_unit"

CLEAR_DIR = DATA / "images"
BLUR_DIR = DATA / "synblur_original48_k3to15"
RESTORED_DIR = ROOT / "results/deblurring_ablation/quexian_synblur48_mbdnetpplus_restored"
DIRECT_MASK_DIR = ROOT / "results/prediction/runs/m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260821_235214/results/segmentation/binary"
DEBLUR_MASK_DIR = ROOT / "results/prediction/runs/m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260822_000545/results/segmentation/binary"
LABEL_CSV = DATA / "defect_review_v2/defect_labels_template.csv"

DEFECTS = ("skip", "break", "skew", "loose")
ZH = {"skip": "跳线", "break": "断线", "skew": "线迹歪斜", "loose": "松线", "normal": "正常"}
COLORS = {
    "navy": "#17324D", "blue": "#2F6690", "cyan": "#4EA5B5",
    "green": "#398564", "gold": "#D7A83D", "orange": "#CC704B",
    "red": "#B84942", "paper": "#F6F8FA", "ink": "#1B2630",
    "muted": "#687681", "line": "#B8C4CC",
}


def configure_fonts() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    yellow = np.array([255, 213, 52], dtype=np.float32)
    out[mask] = 0.30 * out[mask] + 0.70 * yellow
    return out.astype(np.uint8)


def read_targets() -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    labels: dict[str, str] = {}
    focus: dict[str, tuple[int, int]] = {}
    with LABEL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = row["sample_id"].zfill(3)
            if int(sid) > 48 or row["augmentation"] != "original":
                continue
            for defect in DEFECTS:
                if row.get(defect, "0").strip() == "1":
                    labels[sid] = defect
                    if defect != "skew":
                        focus[sid] = (int(float(row["x0"])), int(float(row["x1"])))
    return labels, focus


def choose_examples(labels: dict[str, str], direct: dict, deblur: dict) -> dict[str, str]:
    holdout = set(load_json(RESULT / "report.json")["internal_holdout"]["source_ids"])
    selected: dict[str, str] = {}
    for defect in DEFECTS:
        candidates = [sid for sid in holdout if labels.get(sid) == defect]
        recovered = [sid for sid in candidates if direct["images"][sid]["final_type"] != defect and deblur["images"][sid]["final_type"] == defect]
        pool = recovered or candidates
        selected[defect] = max(
            pool,
            key=lambda sid: deblur["images"][sid]["score_over_threshold"][defect]
            - direct["images"][sid]["score_over_threshold"][defect],
        )
    return selected


def add_box(ax, xy, wh, text, face, *, edge=None, fs=10.2, lw=1.1, subtitle=None):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=face, edgecolor=edge or face, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.56 if subtitle else 0.50), text,
            ha="center", va="center", fontsize=fs, color="white" if face not in ("#FFFFFF", COLORS["paper"]) else COLORS["ink"], fontweight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.25, subtitle, ha="center", va="center",
                fontsize=fs - 1.8, color="#EAF2F6" if face not in ("#FFFFFF", COLORS["paper"]) else COLORS["muted"])
    return patch


def arrow(ax, start, end, *, color=None, style="-|>", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13,
                                linewidth=lw, color=color or COLORS["muted"],
                                connectionstyle=f"arc3,rad={rad}"))


def render_structure() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.04, 0.955, "轨迹自参照缺陷检测单元：详细结构流程", fontsize=21, fontweight="bold", color=COLORS["navy"], va="top")
    ax.text(0.04, 0.918, "输入为 ACHNet 的二值线迹掩码；输出为正常、跳线、断线、线迹歪斜或松线", fontsize=11.5, color=COLORS["muted"], va="top")

    # Online inference lane.
    ax.text(0.035, 0.845, "在线推理", fontsize=12, fontweight="bold", color=COLORS["navy"])
    add_box(ax, (0.035, 0.69), (0.115, 0.115), "ACHNet 掩码", COLORS["navy"], subtitle="320 × 320")
    add_box(ax, (0.18, 0.69), (0.125, 0.115), "掩码规范化", COLORS["blue"], subtitle="阈值化·去微小噪声")
    add_box(ax, (0.335, 0.69), (0.14, 0.115), "列质量与中心轨迹", COLORS["cyan"], subtitle="m(x) 与 y(x)")
    add_box(ax, (0.505, 0.69), (0.14, 0.115), "全局趋势估计", COLORS["green"], subtitle="稳健直线拟合·去趋势")
    add_box(ax, (0.675, 0.69), (0.14, 0.115), "自适应线迹分段", COLORS["gold"], subtitle="支撑谷值·六单元约束")
    add_box(ax, (0.845, 0.69), (0.12, 0.115), "边缘保护", COLORS["orange"], subtitle="首尾截断抑制")
    for x0, x1 in [(0.15, 0.18), (0.305, 0.335), (0.475, 0.505), (0.645, 0.675), (0.815, 0.845)]:
        arrow(ax, (x0, 0.748), (x1, 0.748))

    # Feature bank.
    ax.text(0.035, 0.62, "轨迹特征组", fontsize=12, fontweight="bold", color=COLORS["navy"])
    add_box(ax, (0.035, 0.47), (0.205, 0.11), "连续性特征", "#406E8E", subtitle="覆盖率·内部空缺率·末端缺失率")
    add_box(ax, (0.27, 0.47), (0.205, 0.11), "相对长度特征", "#4D8C84", subtitle="跨度比·弧长比·邻域长度比")
    add_box(ax, (0.505, 0.47), (0.205, 0.11), "形态偏离特征", "#80764B", subtitle="残差·弯曲能量·二次拟合增益")
    add_box(ax, (0.74, 0.47), (0.225, 0.11), "方向与全局特征", "#805B6D", subtitle="局部斜率·全局斜率·轨迹均方差")
    for cx in (0.1375, 0.3725, 0.6075, 0.8525):
        arrow(ax, (0.745, 0.69), (cx, 0.58), color=COLORS["line"], lw=1.0)

    # Class branches and decision.
    ax.text(0.035, 0.405, "类别判定与融合", fontsize=12, fontweight="bold", color=COLORS["navy"])
    add_box(ax, (0.035, 0.255), (0.155, 0.105), "跳线分支", COLORS["blue"], subtitle="空缺与覆盖异常")
    add_box(ax, (0.215, 0.255), (0.155, 0.105), "断线分支", COLORS["red"], subtitle="相对短缺与断裂")
    add_box(ax, (0.395, 0.255), (0.155, 0.105), "松线分支", COLORS["gold"], subtitle="弯曲与下垂异常")
    add_box(ax, (0.575, 0.255), (0.155, 0.105), "歪斜分支", "#805B6D", subtitle="全局斜率异常")
    add_box(ax, (0.765, 0.255), (0.20, 0.105), "阈值归一与互斥决策", COLORS["green"], subtitle="最大置信比·无报警回退正常")
    for cx in (0.1125, 0.2925, 0.4725, 0.6525):
        arrow(ax, (0.50, 0.47), (cx, 0.36), color=COLORS["line"], lw=1.0)
        arrow(ax, (cx + 0.0775, 0.3075), (0.765, 0.3075), color=COLORS["line"], lw=1.0)
    add_box(ax, (0.765, 0.105), (0.20, 0.09), "图像级输出", COLORS["navy"], subtitle="类别·置信度·缺陷单元位置")
    arrow(ax, (0.865, 0.255), (0.865, 0.195))

    # Offline lane.
    ax.add_patch(FancyBboxPatch((0.035, 0.07), 0.68, 0.125, boxstyle="round,pad=0.012,rounding_size=0.015",
                                facecolor="#F1F4F6", edgecolor=COLORS["line"], linewidth=1.0))
    ax.text(0.05, 0.176, "离线建模与固定参数", fontsize=11.5, fontweight="bold", color=COLORS["navy"])
    ax.text(0.055, 0.124, "标注掩码受控增强", fontsize=9.7, color=COLORS["ink"], ha="center")
    ax.text(0.235, 0.124, "正常轨迹统计参考", fontsize=9.7, color=COLORS["ink"], ha="center")
    ax.text(0.415, 0.124, "轻量分类器训练", fontsize=9.7, color=COLORS["ink"], ha="center")
    ax.text(0.595, 0.124, "验证集阈值校准", fontsize=9.7, color=COLORS["ink"], ha="center")
    arrow(ax, (0.13, 0.124), (0.185, 0.124), lw=1.1)
    arrow(ax, (0.31, 0.124), (0.365, 0.124), lw=1.1)
    arrow(ax, (0.49, 0.124), (0.545, 0.124), lw=1.1)
    arrow(ax, (0.68, 0.15), (0.83, 0.255), color=COLORS["green"], lw=1.3, rad=-0.18)

    ax.text(0.04, 0.025, "特点：无需显微镜针距标定；以线迹自身邻域作为长度与形态参考；首尾不完整线迹不直接作为局部短缺证据。",
            fontsize=10, color=COLORS["muted"])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "图1_轨迹自参照缺陷检测单元_详细结构流程图.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "图1_轨迹自参照缺陷检测单元_详细结构流程图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_metric_impact() -> None:
    seg_names = ["二值交并比（IoU）", "轨迹 F1@1"]
    seg_before = [70.76, 91.51]
    seg_after = [78.89, 97.34]
    det_names = ["四类缺陷宏平均 F1", "缺陷报警召回率"]
    det_before = [49.17, 66.67]
    det_after = [87.86, 100.00]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), dpi=250)
    for ax, names, before, after, title in zip(
        axes, (seg_names, det_names), (seg_before, det_before), (seg_after, det_after),
        ("(a) ACHNet 结构恢复质量", "(b) 缺陷检测结果"),
    ):
        y = np.arange(len(names))[::-1]
        for yy, b, a in zip(y, before, after):
            ax.plot([b, a], [yy, yy], color="#BBC5CB", lw=2.4, zorder=1)
            ax.scatter(b, yy, s=80, color=COLORS["red"], zorder=2)
            ax.scatter(a, yy, s=80, color=COLORS["green"], zorder=3)
            ax.text(b - 1.1, yy + 0.18, f"{b:.2f}", ha="right", va="center", fontsize=9.2, color=COLORS["red"])
            ax.text(a + 1.1, yy + 0.18, f"{a:.2f}", ha="left", va="center", fontsize=9.2, color=COLORS["green"])
        ax.set_yticks(y, names)
        ax.set_xlim(38 if ax is axes[1] else 65, 104)
        ax.set_xlabel("指标值（%）")
        ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["navy"], pad=12)
        ax.grid(axis="x", color="#E0E5E8", lw=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["red"], markersize=9, label="直接模糊输入"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["green"], markersize=9, label="经 MBDNet 恢复后"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10.5)
    fig.suptitle("MBDNet 对下游结构恢复与缺陷判定的影响", fontsize=17, fontweight="bold", color=COLORS["navy"], y=0.98)
    fig.text(0.5, 0.075, "正常误报警率：直接模糊输入与 MBDNet 恢复后均为 0.29%", ha="center", fontsize=9.5, color=COLORS["muted"])
    fig.subplots_adjust(left=0.17, right=0.985, top=0.82, bottom=0.23, wspace=0.38)
    fig.savefig(FIG_DIR / "图2_MBDNet对下游任务的影响.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "图2_MBDNet对下游任务的影响.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_focus(ax, defect: str, region: tuple[int, int] | None, shape: tuple[int, int, int]):
    h, w = shape[:2]
    if defect == "skew" or region is None:
        ax.add_patch(Rectangle((3, 3), w - 6, h - 6, fill=False, edgecolor="#E4007F", linewidth=2.2))
    else:
        x0, x1 = region
        ax.add_patch(Rectangle((x0, 3), max(2, x1 - x0), h - 6, fill=False, edgecolor="#E4007F", linewidth=2.2))


def render_qualitative() -> dict[str, str]:
    labels, focus = read_targets()
    direct = load_json(RESULT / "direct_blur48_inference.json")
    deblur = load_json(RESULT / "deblur48_inference.json")
    selected = choose_examples(labels, direct, deblur)
    fig, axes = plt.subplots(4, 5, figsize=(15.5, 12.2), dpi=250)
    headers = ["清晰参考图", "直接模糊图", "模糊输入的 ACHNet 掩码", "MBDNet 恢复图", "恢复后的 ACHNet 掩码"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=11, fontweight="bold", color=COLORS["navy"], pad=8)
    for row, defect in enumerate(DEFECTS):
        sid = selected[defect]
        clear = load_rgb(CLEAR_DIR / f"{sid}.png")
        blur = load_rgb(BLUR_DIR / f"{sid}.png")
        restored = load_rgb(RESTORED_DIR / f"{sid}.png")
        dmask = load_mask(DIRECT_MASK_DIR / f"{sid}.png")
        rmask = load_mask(DEBLUR_MASK_DIR / f"{sid}.png")
        panels = [clear, blur, overlay_mask(blur, dmask), restored, overlay_mask(restored, rmask)]
        for col, image in enumerate(panels):
            axes[row, col].imshow(image)
            add_focus(axes[row, col], defect, focus.get(sid), image.shape)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        dp = direct["images"][sid]["final_type"]
        rp = deblur["images"][sid]["final_type"]
        axes[row, 2].text(0.02, 0.98, f"判定：{ZH[dp]}", transform=axes[row, 2].transAxes,
                          va="top", ha="left", fontsize=9.5, color="white",
                          bbox={"boxstyle": "round,pad=0.25", "facecolor": COLORS["green"] if dp == defect else COLORS["red"], "edgecolor": "none", "alpha": .96})
        axes[row, 4].text(0.02, 0.98, f"判定：{ZH[rp]}", transform=axes[row, 4].transAxes,
                          va="top", ha="left", fontsize=9.5, color="white",
                          bbox={"boxstyle": "round,pad=0.25", "facecolor": COLORS["green"] if rp == defect else COLORS["red"], "edgecolor": "none", "alpha": .96})
        axes[row, 0].set_ylabel(f"{ZH[defect]}\n图像 {sid}", fontsize=11, rotation=0, labelpad=42, va="center", fontweight="bold", color=COLORS["navy"])
    fig.suptitle("模糊输入与 MBDNet 恢复输入的缺陷判别可视化", fontsize=17, fontweight="bold", color=COLORS["navy"], y=0.99)
    fig.text(0.5, 0.012, "黄色区域为 ACHNet 预测线迹，洋红框为标注缺陷单元（线迹歪斜使用整图框）；绿色表示类别正确，红色表示漏检或错分。",
             ha="center", fontsize=9.5, color=COLORS["muted"])
    fig.subplots_adjust(left=0.10, right=0.995, top=0.94, bottom=0.04, wspace=0.035, hspace=0.10)
    fig.savefig(FIG_DIR / "图3_MBDNet恢复前后多类别判别可视化.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "图3_MBDNet恢复前后多类别判别可视化.pdf", bbox_inches="tight")
    plt.close(fig)
    (FIG_DIR / "图3_样例选择.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def main() -> None:
    configure_fonts()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    render_metric_impact()
    selected = render_qualitative()
    print(json.dumps({"figure_dir": str(FIG_DIR), "selected": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

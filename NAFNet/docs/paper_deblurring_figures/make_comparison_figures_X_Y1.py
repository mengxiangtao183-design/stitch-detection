def draw_figure_y1(summary, per_image: list[dict], domain_means, win_rates) -> dict:
    domain, filename, db_median_delta, sample_delta = select_qualitative_case(per_image)
    gt_map = read_manifest(domain)
    blurred_path = DATA_ROOT / Path(
        next(
            row[1] for row in csv.reader((MANIFEST_DIR / f"test_{domain}.tsv").open("r", encoding="utf-8-sig"), delimiter="\t")
            if len(row) > 1 and Path(row[1]).name == filename
        ).replace("/", "\\")
    )
    target_path = gt_map[filename]
    roi = 96
    source = load_rgb(blurred_path)
    height, width = source.shape[:2]
    x0 = max(0, (width - roi) // 2)
    y0 = max(height // 3, (height - roi) // 2)
    y0 = min(y0, (2 * height) // 3 - roi)
    box = (x0, y0, x0 + roi, y0 + roi)
    models = ["loformer", "restormer", "mprnet", "nafnet", "OURS", "GT",
              "hinet", "mimo_unet_plus", "banet", "stripformer", "sfnet", "ufpnet"]
    width_cm, height_cm = 16.0, 6.2  # 压缩高度，移除标题和底部说明的留白
    fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54))
    gs = fig.add_gridspec(2, 7, width_ratios=[1.35, 1, 1, 1, 1, 1, 1],
                          left=0.035, right=0.99, top=0.94, bottom=0.08,
                          hspace=0.45, wspace=0.14)
    ax_input = fig.add_subplot(gs[:, 0])
    ax_input.imshow(source, interpolation="nearest")
    ax_input.add_patch(Rectangle((x0-.5, y0-.5), roi, roi, fill=False,
                                 edgecolor=PALETTE["ours"], linewidth=1.0))
    ax_input.set_title("Blurred input", fontsize=8, pad=3)
    ax_input.text(0.03, 0.03, f"{domain} · 320×320", transform=ax_input.transAxes,
                  ha="left", va="bottom", fontsize=8, color="white",
                  bbox={"facecolor": "#263746", "edgecolor": "none", "alpha": .78, "pad": 1.6})
    ax_input.set_xticks([]); ax_input.set_yticks([])
    for spine in ax_input.spines.values():
        spine.set_visible(False)

    metric_row = next(row for row in per_image if row["domain"] == domain and row["filename"] == filename)
    cell_axes = []
    row_groups = [[f"top-{i}" for i in range(6)], [f"bottom-{i}" for i in range(6)]]
    col_groups = [[f"top-{i}", f"bottom-{i}"] for i in range(6)]
    top_methods = ["loformer", "restormer", "mprnet", "nafnet", "OURS", "GT"]
    bottom_methods = ["hinet", "mimo_unet_plus", "banet", "stripformer", "sfnet", "ufpnet"]
    for row_index, row_methods in enumerate([top_methods, bottom_methods]):
        for col_index, model in enumerate(row_methods, start=1):
            ax = fig.add_subplot(gs[row_index, col_index])
            if model == "GT":
                path = target_path
                title = "Ground truth"
                metric_name = "GT"
            elif model == "OURS":
                path = prediction_path(OURS_ID, domain, filename)
                title = DISPLAY[OURS_ID]
                metric_name = OURS_ID
            else:
                path = prediction_path(model, domain, filename)
                title = DISPLAY[model]
                metric_name = model
            crop = load_rgb(path)[y0:y0+roi, x0:x0+roi]
            ax.imshow(crop, interpolation="lanczos")
            ax.set_title(title, fontsize=8, pad=2.1,
                         color=PALETTE["ours"] if model == "OURS" else PALETTE["dark"],
                         fontweight="bold" if model == "OURS" else "normal")
            if metric_name == "GT":
                ax.text(0.5, -0.055, "reference", transform=ax.transAxes, ha="center", va="top",
                        fontsize=8, color="#687984")
            else:
                ax.text(0.5, -0.055, f"{metric_row[metric_name]:.2f} dB", transform=ax.transAxes,
                        ha="center", va="top", fontsize=8,
                        color=PALETTE["ours"] if model == "OURS" else "#596A75",
                        fontweight="bold" if model == "OURS" else "normal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9 if model == "OURS" else 0.45)
                spine.set_edgecolor(PALETTE["ours"] if model == "OURS" else "#AAB6BD")
            cell_axes.append(ax)

    # 已移除顶部大标题、底部说明小字

    skill_scripts = Path(r"C:\Users\user\.codex\skills\nature-figure\scripts")
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(OUT / "FigureY1.alignment.json"),
        overlay_svg=str(OUT / "FigureY1.alignment.svg"),
        axes=cell_axes,
        panel_ids=[f"top-{i}" for i in range(6)] + [f"bottom-{i}" for i in range(6)],
        row_groups=row_groups,
        column_groups=col_groups,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )
    fig.savefig(OUT / "FigureY1.svg", bbox_inches="tight")
    fig.savefig(OUT / "FigureY1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "FigureY1.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "FigureY1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "domain": domain, "filename": filename, "roi_xyxy": list(box),
        "roi_width_height": [roi, roi], "db_median_comparator_mean_delta_psnr_db": db_median_delta,
        "selected_case_comparator_mean_delta_psnr_db": sample_delta,
        "image_adjustments": "RGB as saved; no local contrast, gamma or color changes",
    }

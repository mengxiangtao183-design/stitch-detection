from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "model_registry.json").read_text(encoding="utf-8"))
DOMAIN_ORDER = ["DB", "XJ", "SD", "RX", "RXHF", "TJML"]
EXTERNAL_SUMMARIES = {
    "mprnet": ROOT.parent / "Deblurring" / "results" / "MultiSourceV2_MPRNetOriginal_S30_S42_midroi" / "summary.json",
    "nafnet": ROOT.parent / "NAFNet" / "results" / "MultiSourceV2_A0_NAFNet_Fair_S30_S42_midroi" / "summary.json",
    "mbdnet_p_plus": ROOT.parent / "NAFNet" / "results" / "MultiSourceV2_MBDNetPPlusR12_S30_S42_midroi" / "summary.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate unified deblurring comparison summaries")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--output", default=str(ROOT / "results" / "comparison_summary.csv"))
    args = parser.parse_args()
    rows = []
    summary_sources = [
        (summary_path, "comparison_suite", None)
        for summary_path in sorted(Path(args.results_root).glob("*/summary.json"))
        if not summary_path.parent.name.startswith("_")
    ]
    for model, summary_path in EXTERNAL_SUMMARIES.items():
        if summary_path.is_file():
            summary_sources.append((summary_path, "verified_original_project", model))
    for summary_path, source, fallback_model in summary_sources:
        if summary_path.parent.name.startswith("_"):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model = summary.get("model", fallback_model)
        if model is None:
            raise KeyError(f"No model name in {summary_path}")
        meta = REGISTRY["models"].get(model, {})
        row = {
            "model": model,
            "label": meta.get("label", model),
            "year": meta.get("year", ""),
            "venue": meta.get("venue", ""),
            "result_source": source,
            "macro_full_psnr": summary["macro"]["psnr_full"],
            "macro_full_ssim": summary["macro"]["ssim_full"],
            "macro_mid_psnr": summary["macro"]["psnr_mid"],
            "macro_mid_ssim": summary["macro"]["ssim_mid"],
            "inference_ms": summary["macro"]["inference_ms"],
        }
        weighted = summary.get("weighted", {})
        for metric in ("psnr_full", "ssim_full", "psnr_mid", "ssim_mid", "inference_ms"):
            row[f"weighted_{metric}"] = weighted.get(metric, "")
        for domain in DOMAIN_ORDER:
            metrics = summary["domains"].get(domain, {})
            row[f"{domain}_psnr"] = metrics.get("psnr_full", "")
            row[f"{domain}_ssim"] = metrics.get("ssim_full", "")
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No summary.json under {args.results_root}")
    rows.sort(key=lambda row: row["macro_full_psnr"], reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()

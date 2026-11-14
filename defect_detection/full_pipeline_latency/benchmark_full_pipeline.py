"""Serial batch-one latency: decode -> frozen route -> restore -> segment -> BTSR.

No training or threshold calibration. Intermediate RGB/masks stay in memory.
The restored RGB is quantized as uint8 to match the existing PNG-based path.
Deployment-model predictions here are not independent accuracy estimates.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace


def args_and_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--policy", choices=("routed", "always_deblur", "bypass"), default="routed")
    parser.add_argument("--workload", choices=("mixed", "clear", "blurred"), default="mixed")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images; smoke testing only otherwise")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-images", action="store_true", help="Time PNG saving separately; not in decision latency")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and frozen parameters without importing CUDA libraries")
    args = parser.parse_args()
    args.config = args.config.resolve()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.repeats is not None:
        cfg["repeats"] = args.repeats
    if args.warmup is not None:
        cfg["warmup"] = args.warmup
    if cfg["batch_size"] != 1 or cfg["image_size"] != 320:
        raise ValueError("This protocol requires batch_size=1, image_size=320")
    if cfg["repeats"] < 1 or cfg["warmup"] < 2 or args.max_images < 0:
        raise ValueError("repeats >= 1, warmup >= 2, max-images >= 0 are required")
    for key in ("project_root", "router_summary", "deblur_checkpoint", "seg_checkpoint", "defect_model", "detector_config", "clear_dir", "blur_dir"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"{key}: {cfg[key]}")
    for key in ("clear_dir", "blur_dir"):
        if not Path(cfg[key]).is_dir():
            raise NotADirectoryError(cfg[key])
    files = {c: {p.name: p for p in Path(cfg[k]).glob("*.png")} for c, k in (("clear", "clear_dir"), ("blurred", "blur_dir"))}
    if set(files["clear"]) != set(files["blurred"]) or len(files["clear"]) != cfg["expected_sources"]:
        raise ValueError("Expected matching 112 clear/blur PNG filenames; input data are not paired")
    rows = [(c, files[c][name]) for name in sorted(files["clear"]) for c in ("clear", "blurred")
            if args.workload == "mixed" or args.workload == c]
    if args.max_images:
        rows = rows[:args.max_images]
    router = json.loads(Path(cfg["router_summary"]).read_text(encoding="utf-8"))["calibration"]
    if not 0 <= float(router["threshold"]) <= 1:
        raise ValueError("Invalid frozen routing threshold")
    return args, cfg, rows, router


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args, cfg, inputs, calibration = args_and_config()
    if args.check_only:
        print(json.dumps({"check": "PASS (paths, filenames, counts, frozen threshold)",
                          "sources": cfg["expected_sources"], "input_presentations": len(inputs),
                          "policy": args.policy, "threshold": calibration["threshold"],
                          "note": "CUDA, model weight compatibility and image decoding are tested by a smoke run."}, indent=2))
        return

    # Match the project's Windows runtime settings before importing numerical libraries.
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg["cpu_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg["cpu_threads"]))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import cv2
    import joblib
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    root = Path(cfg["project_root"])
    sys.path[:0] = [str(root), str(root / "defect_detection"), str(root / "NAFNet")]
    from evaluate_dynamic_deblur_skip import FEATURE_NAMES, quality_features_rgb
    from infer_quexian_nafnet_deblur import load_model as load_restorer, restore
    from com_models.registry import build_model
    from utils.checkpoint_utils import load_model_checkpoint, normalize_state_dict_keys
    from defect_detection.binary_topology_features import binary_topology_v2_feature_vector, FEATURE_NAMES_V2
    from defect_detection.train_trajectory_feature_classifier import load_detector, predict, LOCAL_CLASSES
    from defect_detection.run_nested_fourfold_tsr_defect import skew_score, DEFECT_CLASSES

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. CPU fallback is forbidden for the paper GPU timing protocol.")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(cfg["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    if tuple(calibration["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Frozen router feature order differs from implementation")
    mean = np.asarray(calibration["feature_mean"])
    scale = np.asarray(calibration["feature_scale"])
    coef = np.asarray(calibration["logistic_coefficients"])
    if mean.shape != (len(FEATURE_NAMES),) or coef.shape != mean.shape or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("Invalid frozen router coefficients")
    threshold = float(calibration["threshold"])
    intercept = float(calibration["logistic_intercept"])

    load_start = time.perf_counter()
    restoration = load_restorer(SimpleNamespace(checkpoint=Path(cfg["deblur_checkpoint"]),
                                                arch=cfg["deblur_arch"], width=cfg["deblur_width"]), device)
    checkpoint = load_model_checkpoint(cfg["seg_checkpoint"], map_location="cpu")
    model_cfg = checkpoint["config"]["model"]
    if model_cfg["image_processing"]["target_size"] != 320:
        raise ValueError("Segmentation checkpoint is not the 320-pixel protocol")
    segmentation = build_model(model_cfg)
    segmentation.load_state_dict(normalize_state_dict_keys(checkpoint["model_state_dict"]), strict=True)
    segmentation = segmentation.to(device).eval()
    del checkpoint
    package = joblib.load(cfg["defect_model"])
    if package.get("feature_set") != "binary_topology_v2" or tuple(package["feature_names"]) != tuple(FEATURE_NAMES_V2):
        raise ValueError("Expected the balanced binary-topology V2 deployment model")
    for model in package["models"].values():
        model.named_steps["extra_trees"].n_jobs = 1
    detector = load_detector(Path(cfg["detector_config"]))
    norm_mean = torch.tensor(model_cfg["image_processing"]["normalize_mean"]).view(1, 3, 1, 1)
    norm_std = torch.tensor(model_cfg["image_processing"]["normalize_std"]).view(1, 3, 1, 1)
    torch.cuda.synchronize(device)
    load_ms = (time.perf_counter() - load_start) * 1000

    @torch.inference_mode()
    def run_one(condition, path, policy):
        torch.cuda.synchronize(device)
        stamps = [time.perf_counter()]

        def tick():
            torch.cuda.synchronize(device)
            stamps.append(time.perf_counter())

        with Image.open(path) as source:
            rgb = np.array(source.convert("RGB"))
        if rgb.shape != (320, 320, 3):
            raise ValueError(f"Expected 320x320 RGB: {path} {rgb.shape}")
        tick()
        features = quality_features_rgb(rgb)
        logit = float(np.dot((features - mean) / scale, coef) + intercept)
        # Numerically stable sigmoid; no threshold tuning during timing.
        score = (1 / (1 + math.exp(-logit))) if logit >= 0 else (math.exp(logit) / (1 + math.exp(logit)))
        would_route = "bypass" if score >= threshold else "deblur"
        route = would_route if policy == "routed" else ("deblur" if policy == "always_deblur" else "bypass")
        tick()
        if route == "deblur":
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255)
        tick()
        if route == "deblur":
            restored = restore(restoration, tensor, (320, 320))
        tick()
        if route == "deblur":
            rgb = (restored[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        tick()
        # Preserve the established CPU bilinear resize + ImageNet normalization.
        seg_input = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255
        seg_input = F.interpolate(seg_input, size=(320, 320), mode="bilinear", align_corners=False, antialias=True)
        seg_input = ((seg_input - norm_mean) / norm_std).to(device)
        tick()
        outputs = segmentation(seg_input)
        tick()
        probability = F.interpolate(outputs["final_mask"].sigmoid(), size=(320, 320), mode="bilinear", align_corners=False)
        binary = (probability[0, 0].cpu().numpy() >= cfg["seg_threshold"]).astype(np.float32)
        tick()
        result = detector.detect(binary)
        units = [{"feature": binary_topology_v2_feature_vector(result, unit, binary)} for unit in result["units"]]
        if not units:
            raise RuntimeError("Detector produced no units")
        tick()
        probabilities = predict(package["models"], units)
        tick()
        ratios = {name: float(probabilities[:, col].max()) / max(float(package["thresholds"][name]), 1e-8)
                  for col, name in enumerate(LOCAL_CLASSES)}
        ratios["skew"] = float(np.mean([skew_score(row, detector) for row in units])) / max(float(package["skew_threshold"]), 1e-8)
        winner = max(DEFECT_CLASSES, key=lambda name: ratios[name])
        predicted_class = winner if ratios[winner] >= 1 else "normal"
        tick()
        keys = ("decode_ms", "router_ms", "deblur_prepare_ms", "deblur_forward_ms", "deblur_transfer_quantize_ms",
                "seg_preprocess_ms", "seg_forward_ms", "seg_binary_postprocess_ms", "tsr_features_ms", "tsr_classify_ms", "tsr_decision_ms")
        timing = {key: 1000 * (stamps[i + 1] - stamps[i]) for i, key in enumerate(keys)}
        # Attribute no GPU stage to skipped models; synchronization overhead remains in total.
        skipped_overhead = 0.0
        if route == "bypass":
            for key in ("deblur_prepare_ms", "deblur_forward_ms", "deblur_transfer_quantize_ms"):
                skipped_overhead += timing[key]
                timing[key] = 0.0
        timing["bypass_boundary_overhead_ms"] = skipped_overhead
        total = 1000 * (stamps[-1] - stamps[0])
        assert abs(sum(timing.values()) - total) < 1e-5
        timing.update({"total_decode_to_decision_ms": total, "total_compute_ms": total - timing["decode_ms"],
                       "deblur_stage_ms": sum(timing[k] for k in ("deblur_prepare_ms", "deblur_forward_ms", "deblur_transfer_quantize_ms")),
                       "seg_stage_ms": sum(timing[k] for k in ("seg_preprocess_ms", "seg_forward_ms", "seg_binary_postprocess_ms")),
                       "tsr_stage_ms": sum(timing[k] for k in ("tsr_features_ms", "tsr_classify_ms", "tsr_decision_ms"))})
        output = {"condition": condition, "filename": path.name, "policy": policy, "route": route,
                  "frozen_router_route": would_route, "clear_probability": score, "router_threshold": threshold,
                  "predicted_class": predicted_class, "defect_alarm": predicted_class != "normal", **timing}
        prediction = {"condition": condition, "filename": path.name, "route": route,
                      "predicted_class": predicted_class, "score_over_threshold": ratios,
                      "unit_probabilities": probabilities.tolist()}
        return output, prediction, rgb, binary

    # Exercise both branches, independently of the first image's router score.
    warm_start = time.perf_counter()
    for i in range(cfg["warmup"]):
        condition, path = inputs[i % len(inputs)]
        run_one(condition, path, "always_deblur" if i % 2 == 0 else "bypass")
    warmup_ms = (time.perf_counter() - warm_start) * 1000
    output_root = Path(cfg["output_root"])
    if not output_root.is_absolute():
        output_root = args.config.parent / output_root
    output = args.output or (output_root / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.policy}_{args.workload}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    timings, predictions = [], []
    for repeat in range(cfg["repeats"]):
        ordered = list(inputs)
        random.Random(cfg["seed"] + repeat).shuffle(ordered)
        for index, (condition, path) in enumerate(ordered):
            row, prediction, rgb, binary = run_one(condition, path, args.policy)
            row["repeat"] = repeat + 1
            row["save_ms"] = 0.0
            if args.save_images:
                save_start = time.perf_counter()
                directory = output / "images" / str(repeat + 1) / condition
                directory.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(directory / f"{path.stem}_routed.png")
                Image.fromarray((binary * 255).astype(np.uint8)).save(directory / f"{path.stem}_binary.png")
                row["save_ms"] = (time.perf_counter() - save_start) * 1000
            row["total_with_save_ms"] = row["total_decode_to_decision_ms"] + row["save_ms"]
            prediction["repeat"] = repeat + 1
            timings.append(row); predictions.append(prediction)
            if (index + 1) % 20 == 0 or index + 1 == len(ordered):
                print(f"repeat {repeat+1}/{cfg['repeats']}: {index+1}/{len(ordered)} images, latest={row['total_decode_to_decision_ms']:.2f} ms", flush=True)
    write_csv(output / "per_image_latency.csv", timings)
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    time_keys = [key for key in timings[0] if key.endswith("_ms")]
    def stats(rows):
        return {key: {"mean": float(np.mean(values)), "median": float(np.median(values)),
                      "p95": float(np.percentile(values, 95)), "min": float(np.min(values)), "max": float(np.max(values))}
                for key in time_keys for values in [np.asarray([r[key] for r in rows])]} if rows else None
    groups = {"all": timings}
    groups.update({f"condition_{c}": [r for r in timings if r["condition"] == c] for c in ("clear", "blurred")})
    groups.update({f"route_{c}": [r for r in timings if r["route"] == c] for c in ("bypass", "deblur")})
    groups.update({f"repeat_{i+1}": [r for r in timings if r["repeat"] == i + 1] for i in range(cfg["repeats"])})
    report = {
        "protocol": {"policy": args.policy, "workload": args.workload, "batch_size": 1,
                     "input_size": [320, 320], "precision": "FP32; AMP and TF32 disabled",
                     "gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
                     "cpu": platform.processor(), "cpu_threads": cfg["cpu_threads"],
                     "timing": "Serial wall-clock, synchronized at every GPU boundary; includes preprocess, transfers, quantization, feature extraction and final decision",
                     "exclusions": ["model loading", "warm-up", "logging", "summary output", "camera capture", "PLC/network communication"],
                     "saving": "Optional PNG saving timed separately; no intermediate file IO",
                     "caches": "Warm models; OS input cache not flushed. Repeat-level statistics retained.",
                     "accuracy_boundary": "Deployment model trained on 112 sources; outputs here are not independent accuracy evaluation",
                     "frozen_thresholds": True, "router_threshold": threshold,
                     "independent_source_count": len({p.stem for _, p in inputs}),
                     "input_presentations_per_repeat": len(inputs), "repeats": cfg["repeats"],
                     "warmup_calls": cfg["warmup"], "smoke_test": bool(args.max_images)},
        "config": cfg, "load_ms_excluded": load_ms, "warmup_ms_excluded": warmup_ms,
        "router_config_sha256": hashlib.sha256(Path(cfg["router_summary"]).read_bytes()).hexdigest(),
        "defect_thresholds": package["thresholds"], "skew_threshold": package["skew_threshold"],
        "route_counts": {c: sum(r["route"] == c for r in timings) for c in ("bypass", "deblur")},
        "groups": {k: {"measurements": len(v), "latency_ms": stats(v)} for k, v in groups.items()},
        "serial_images_per_second": 1000 / np.mean([r["total_decode_to_decision_ms"] for r in timings]),
    }
    (output / "latency_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = [{"group": k, "measurements": len(rows),
              **{metric: report["groups"][k]["latency_ms"][metric]["mean"] for metric in
                 ("decode_ms", "router_ms", "deblur_stage_ms", "seg_stage_ms", "tsr_stage_ms", "total_compute_ms", "total_decode_to_decision_ms")},
              "p95_total_ms": report["groups"][k]["latency_ms"]["total_decode_to_decision_ms"]["p95"]}
             for k, rows in groups.items() if rows]
    write_csv(output / "stage_summary.csv", table)
    print(f"RESULTS: {output}", flush=True)
    print(json.dumps({"all_mean_ms": report["groups"]["all"]["latency_ms"]["total_decode_to_decision_ms"]["mean"],
                      "all_p95_ms": report["groups"]["all"]["latency_ms"]["total_decode_to_decision_ms"]["p95"],
                      "route_counts": report["route_counts"]}, indent=2))


if __name__ == "__main__":
    main()

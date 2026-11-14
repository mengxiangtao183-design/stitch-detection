"""
合成退化图像，用于 AIO-IENet 增强网络训练。

从已有的清晰 COD 数据集生成退化版本，同时生成 Clear/ 参考目录。
退化类型：仅运动模糊（水平方向，核大小 5~11）。

用法:
    python methods/degrade_images.py \
        --input datasets/img3 \
        --output datasets/img3_degraded

输出结构:
    datasets/img3_degraded/
    ├── Imgs/      ← 退化后图像（模型输入）
    ├── GT/        ← 掩码（从原数据集复制）
    ├── Edges/     ← 边缘图（从原数据集复制）
    └── Clear/     ← 原始清晰图（AIO 监督目标）

Author: C3Net Team
Date  : 2026-06-16
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm


# ===========================================================================
# 退化操作
# ===========================================================================

def gaussian_blur(img: np.ndarray) -> np.ndarray:
    """高斯模糊：模拟失焦，k≤11 保证线迹结构可恢复"""
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img: np.ndarray, angle_min: int = -30, angle_max: int = 30) -> np.ndarray:
    """运动模糊：单向运动模糊，角度在 [angle_min, angle_max] 内随机"""
    length = random.randint(5, 11)
    angle = random.randint(angle_min, angle_max)
    # 生成运动模糊核
    kernel = np.zeros((length, length))
    center = length // 2
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    for i in range(length):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray) -> np.ndarray:
    """高斯噪声：轻度传感器噪声，避免伪纹理掩盖真实模糊"""
    sigma = random.uniform(3, 10)
    noise = np.random.randn(*img.shape) * sigma
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def low_light(img: np.ndarray) -> np.ndarray:
    """亮度降低：模拟光照不均，α 曲线分支可拉伸"""
    gamma = random.uniform(0.7, 0.85)
    img_float = img.astype(np.float32) / 255.0
    darkened = np.power(img_float, 1.0 / gamma) * 255.0
    return np.clip(darkened, 0, 255).astype(np.uint8)


def jpeg_compression(img: np.ndarray) -> np.ndarray:
    """JPEG 压缩：轻度传输失真，避免块效应混淆去模糊任务"""
    quality = random.randint(50, 75)
    _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# ===========================================================================
# 退化组合
# ===========================================================================

def degrade_image(img: np.ndarray) -> np.ndarray:
    """对单张图像施加退化——仅运动模糊，用于 AIO-IENet 增强网络训练。

    设计原则：
    - 仅施加水平方向运动模糊，核大小 5~11，角度 -15°~+15°，模拟传送带跑偏。
    - 不再叠加低光/噪声/压缩等辅助退化，保证模糊是唯一的退化因素。
    """
    degraded = img.copy()

    # ---- 仅运动模糊（传送带水平方向）----
    degraded = motion_blur(degraded, angle_min=-15, angle_max=15)

    return degraded


# ===========================================================================
# 主流程
# ===========================================================================

def process_dataset(input_dir: str, output_dir: str, copy_only: bool = False,
                    img_subdir: str = 'Imgs'):
    """
    处理整个数据集。

    Args:
        input_dir:  原始数据集路径（含 Imgs/、GT/、Edges/）
        output_dir: 输出数据集路径
        copy_only:  如果 True，只复制不退化（用于生成 Clear 参考）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 查找图片目录（优先 clear/，兼容 Imgs/ 及 train/test 子目录）
    img_sources = []
    for pattern in ['train/clear', 'test/clear', 'train/Imgs', 'test/Imgs', 'clear', 'Imgs']:
        src = input_path / pattern
        if src.is_dir():
            img_sources.append((src, pattern))

    if not img_sources:
        print(f"错误：在 {input_dir} 中找不到 clear/ 或 Imgs/ 目录")
        return

    for img_src, pattern in img_sources:
        # 确定对应的 GT / Edges 源目录
        base_dir = img_src.parent  # train/ 或 .
        gt_src = base_dir / 'GT'
        edges_src = base_dir / 'Edges'

        # 输出目录
        if base_dir.name in ('train', 'test'):
            out_img = output_path / base_dir.name / img_subdir
            out_clear = output_path / base_dir.name / 'Clear'
            out_gt = output_path / base_dir.name / 'GT'
            out_edges = output_path / base_dir.name / 'Edges'
        else:
            out_img = output_path / img_subdir
            out_clear = output_path / 'Clear'
            out_gt = output_path / 'GT'
            out_edges = output_path / 'Edges'

        for d in [out_img, out_clear, out_gt, out_edges]:
            d.mkdir(parents=True, exist_ok=True)

        # 只处理 .png .jpg（排除 Clear 目录自身避免嵌套）
        img_files = sorted([
            f for f in img_src.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ])

        if not img_files:
            print(f"  跳过 {img_src}：无图片")
            continue

        print(f"  处理 {len(img_files)} 张图片: {img_src} → {out_img.parent}")

        for img_file in tqdm(img_files, desc=f"  {'复制' if copy_only else '退化'}"):
            filename = img_file.name

            if copy_only:
                # 直接复制原图作为 Clear 参考（只在不存在时写入）
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"    警告：无法读取 {img_file}")
                    continue
                if not (out_clear / filename).exists():
                    cv2.imwrite(str(out_clear / filename), img)
            else:
                # 退化模式：始终从 Clear/ 读取原图，防止累积退化
                clear_file = out_clear / filename
                if clear_file.exists():
                    img = cv2.imread(str(clear_file))
                else:
                    img = cv2.imread(str(img_file))
                if img is None:
                    print(f"    警告：无法读取 {img_file}")
                    continue
                # 退化并保存
                degraded = degrade_image(img)
                cv2.imwrite(str(out_img / filename), degraded)
                # 复制清晰图到 Clear（只在不存在时写入，防止覆盖）
                if not (out_clear / filename).exists():
                    cv2.imwrite(str(out_clear / filename), img)

                # 复制 GT 掩码（跳过同文件）
                gt_file = gt_src / filename if gt_src.is_dir() else None
                if gt_file and gt_file.exists():
                    import shutil
                    dst = str(out_gt / filename)
                    if os.path.abspath(str(gt_file)) != os.path.abspath(dst):
                        shutil.copy(str(gt_file), dst)

                # 复制 Edges 边缘图（跳过同文件）
                edge_file = edges_src / filename if edges_src.is_dir() else None
                if edge_file and edge_file.exists():
                    import shutil
                    dst = str(out_edges / filename)
                    if os.path.abspath(str(edge_file)) != os.path.abspath(dst):
                        shutil.copy(str(edge_file), dst)

    print(f"\n完成！输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='合成退化图像数据集，用于 AIO-IENet 增强网络训练'
    )
    parser.add_argument(
        '--input', required=True,
        help='原始数据集路径（含 Imgs/, GT/, Edges/）'
    )
    parser.add_argument(
        '--output', required=True,
        help='输出数据集路径'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子 (默认 42)'
    )
    parser.add_argument(
        '--img_subdir', type=str, default='Imgs',
        help='退化图输出子目录名 (默认 Imgs，可设为 Imgs1 等避免覆盖)'
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("合成退化图像数据集")
    print("=" * 60)
    print(f"输入:  {args.input}")
    print(f"输出:  {args.output}")
    print(f"退化:  仅运动模糊 (核 5~11, 角度 -15°~+15°)")
    print(f"目录:  {args.img_subdir}")
    print("=" * 60)

    process_dataset(args.input, args.output, copy_only=False,
                    img_subdir=args.img_subdir)
    print("退化图像生成完毕。")


if __name__ == "__main__":
    main()

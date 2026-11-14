import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_edges_from_mask(
    mask_dir,        # 掩码文件夹路径（GT）
    edge_dir,        # 边缘图输出路径
    edge_width=1     # 边缘宽度（默认1，和原项目一致）
):
    # 创建边缘文件夹
    os.makedirs(edge_dir, exist_ok=True)
    
    # 获取所有掩码文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.json', '.jpeg'))]
    
    print(f"✅ 找到 {len(mask_files)} 张掩码图")
    print(f"📤 边缘图将保存到: {edge_dir}\n")
    
    for filename in tqdm(mask_files, desc="生成边缘图中"):
        # 读取掩码（灰度图）
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, 0)
        
        if mask is None:
            continue
        
        # Canny 边缘检测
        edges = cv2.Canny(mask, 10, 200)
        
        # 膨胀边缘（控制宽度）
        if edge_width > 1:
            kernel = np.ones((edge_width, edge_width), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 保存边缘图（文件名和掩码完全一致）
        edge_save_path = os.path.join(edge_dir, os.path.splitext(filename)[0] + '.png')
        cv2.imwrite(edge_save_path, edges)
    
    print("\n🎉 所有边缘图生成完成！")
    print(f"📂 生成路径: {edge_dir}")

# ====================== 【你只需要改这里！】 ======================
if __name__ == "__main__":
    """
    把下面路径改成你自己的数据集路径！
    """
    # 你的掩码（GT）文件夹
    MASK_DIR = r"E:/mxt/gt"
    
    # 边缘图要保存的文件夹（必须叫 Edges，和代码匹配）
    EDGE_DIR = r"E:/mxt/Edges"

    # 生成！
    generate_edges_from_mask(
        mask_dir=MASK_DIR,
        edge_dir=EDGE_DIR,
        edge_width=1
    )
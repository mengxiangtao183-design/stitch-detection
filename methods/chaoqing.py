from realesrgan import RealESRGANer
import cv2
import os

def upscale_image(img_path, save_path, scale=2):
    # 模型文件在项目根目录，直接用绝对路径
    model_path = r"E:\mxt\C3Net\RealESRGAN_x2plus.pth"
    
    # 初始化超分器
    upscaler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        tile=256,       # 分块处理，降低显存占用
        tile_pad=10,
        pre_pad=0,
        half=False      # 关闭半精度，兼容性更好
    )

    # 读取图片
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ 错误：无法读取图片文件 {img_path}")
        return

    # 执行超分辨率处理
    output, _ = upscaler.enhance(img, outscale=scale)

    # 保存结果
    cv2.imwrite(save_path, output)
    print(f"✅ 超清修复完成！结果已保存到：{save_path}")

if __name__ == "__main__":
    upscale_image(r"E:\mxt\1.jpeg", "output_hd.png", scale=2)
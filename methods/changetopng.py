from PIL import Image
import glob
import os

# JPG图像所在的目录
path = "H:/DATASET/TJML/*jpeg"

# 遍历所有的JPG图像，转换为PNG图像并保存
for file in glob.glob(path):
    im = Image.open(file)
    png_file = file[:-4] + ".png"
    im.save(png_file, "PNG")
    print(f"已将 {file} 转换为 {png_file}")

    # 删除原始JPG文件
    os.remove(file)
    print(f"已删除原始文件 {file}")

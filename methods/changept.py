

import os
from PIL import Image, ImageOps
import glob
import glob
#n
img_path = glob.glob("H:/DATASET/TJML/*.png")
path_save = "H:/DATASET/TJML/"
#

from PIL import Image
for file in img_path:
    img = Image.open(file)  # 读取原始图像
    new_size = (320,320)  # 指定新的分辨率
    img = img.resize(new_size,Image.LANCZOS)  # 拉伸图像到指定分辨率
    new_file = os.path.join(path_save, os.path.basename(file))

    img.save(new_file)  # 保存新的图像


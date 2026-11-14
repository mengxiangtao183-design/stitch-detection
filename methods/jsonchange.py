import argparse
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
import yaml
from labelme import utils

'''multiple json files'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', nargs='?', default=r"D:tao23\MVViewer\tjmllabel", help="json files directory")
    parser.add_argument('--label_out', default=r"D:tao23\MVViewer\GT")
    args = parser.parse_args()

    json_file = args.json_file
    label_out_dir = args.label_out

    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)

    file_list = os.listdir(json_file)

    for i in range(0, len(file_list)):
        path = os.path.join(json_file, file_list[i])
        filename = file_list[i][:-5]
        extension = file_list[i][-4:]
        if extension == 'json':
            if os.path.isfile(path):
                data = json.load(open(path))
                img = utils.image.img_b64_to_arr(data['imageData'])
                
                label_name_to_value = {"_background_": 0}
                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name not in label_name_to_value:
                        label_name_to_value[label_name] = len(label_name_to_value)

                lbl, _ = utils.shapes_to_label(
                    img.shape, data["shapes"], label_name_to_value
                )

                label_names = [None] * (max(label_name_to_value.values()) + 1)
                for name, value in label_name_to_value.items():
                    label_names[value] = name

                # 生成白色掩码（标注区域=白色，背景=黑色）
                white_mask = (lbl > 0).astype('uint8') * 255
                PIL.Image.fromarray(white_mask).save(osp.join(label_out_dir, f"{filename}.png"))

                with open(osp.join(label_out_dir, 'label_names.txt'), 'w') as f:
                    for lbl_name in label_names:
                        f.write(lbl_name + '\n')

                print(f'已保存: {filename}_label.png → {label_out_dir}')

if __name__ == "__main__":
    main()

# import argparse
# import base64
# import json
# import os
# import os.path as osp
# import numpy as np
# import PIL.Image
# from labelme import utils

# '''multiple json files，兼容多边形、矩形、画笔涂抹标注'''
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('json_file', nargs='?', default=r"E:\mxt\picture\dongbei", help="json files directory")
#     parser.add_argument('--label_out', default=r"E:\mxt\GT")
#     args = parser.parse_args()

#     json_file = args.json_file
#     label_out_dir = args.label_out

#     if not os.path.exists(label_out_dir):
#         os.makedirs(label_out_dir)

#     file_list = os.listdir(json_file)

#     for file in file_list:
#         # 只处理json文件
#         if not file.endswith('.json'):
#             continue
            
#         json_path = os.path.join(json_file, file)
#         filename = file[:-5]  # 去掉.json后缀
        
#         # 读取json
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # 解码图片
#         img = utils.image.img_b64_to_arr(data['imageData'])
#         h, w = img.shape[:2]
        
#         # 初始化标签掩码（全0=背景）
#         label_mask = np.zeros((h, w), dtype=np.uint8)
        
#         # 处理所有标注形状（兼容多边形 + mask画笔涂抹）
#         for shape in data['shapes']:
#             label = shape['label']
#             shape_type = shape.get('shape_type', 'polygon')
            
#             # 核心：处理画笔涂抹标注（mask）
#             if shape_type == 'mask':
#                 # 解码mask
#                 mask_data = shape['mask']
#                 mask = utils.image.img_b64_to_arr(mask_data)
#                 mask = mask[:, :, 0]  # 转单通道
#                 mask = (mask > 0).astype(np.uint8)
#                 label_mask[mask == 1] = 1  # 前景设为1
            
#             # 处理多边形/矩形标注
#             else:
#                 points = np.array(shape['points'], dtype=np.int32)
#                 cv2.fillPoly(label_mask, [points], 1)
        
#         # 生成黑白二值图：前景=白色(255)，背景=黑色(0)
#         white_mask = (label_mask > 0).astype(np.uint8) * 255
#         save_path = osp.join(label_out_dir, f"{filename}.png")
#         PIL.Image.fromarray(white_mask).save(save_path)
        
#         print(f"处理完成：{filename}.png")

# if __name__ == "__main__":
#     import cv2  # 在这里导入避免循环问题
#     main()
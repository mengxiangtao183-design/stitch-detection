import re

# 你的原始日志文本（可以替换为读取txt文件的代码）
log_text = """
正在处理第0张图，图像为：./Openresult\Blue\h026.png
浮点总数为5
the 3 point right neighbor distance is 55.68070583262892
the 0 point right neighbor distance is 53.31801763266082
the 4 point right neighbor distance is 50.59515865954921
the 2 point right neighbor distance is 55.03779761904762
耗时：0.22 秒
正在处理第1张图，图像为：./Openresult\Blue\h027.png
浮点总数为5
the 2 point right neighbor distance is 51.067046936114735
the 0 point right neighbor distance is 53.6149766899767
the 3 point right neighbor distance is 51.97267877641974
the 1 point right neighbor distance is 51.522341057635174
耗时：0.14 秒
正在处理第2张图，图像为：./Openresult\Blue\h028.png
浮点总数为5
the 2 point right neighbor distance is 48.65469443496073
the 0 point right neighbor distance is 53.268435013262604
the 4 point right neighbor distance is 54.155177345537766
the 1 point right neighbor distance is 58.34505824011072
耗时：0.15 秒
正在处理第3张图，图像为：./Openresult\Blue\h029.png
浮点总数为5
the 0 point right neighbor distance is 49.54970402525652
the 1 point right neighbor distance is 60.59798505037372
the 2 point right neighbor distance is 56.28018575851394
the 3 point right neighbor distance is 53.6547802597659
耗时：0.16 秒
正在处理第4张图，图像为：./Openresult\Blue\h030.png
浮点总数为5
the 0 point right neighbor distance is 51.172599637681145
the 3 point right neighbor distance is 56.637242268041234
the 1 point right neighbor distance is 59.30338273195878
the 2 point right neighbor distance is 50.41675311203319
"""

print("=== 方法一：提取所有距离值到一个列表中 ===")
# 使用正则表达式匹配 "distance is " 后面的所有数字和小数点
all_distances_str = re.findall(r"distance is ([\d\.]+)", log_text)
# 将字符串转换为浮点数
all_distances = [float(val) for val in all_distances_str]
print(f"共提取到 {len(all_distances)} 个值：")
print(all_distances)
print("\n")

print("=== 方法二：按图片名称结构化提取（推荐） ===")
structured_data = {}
current_image = None

# 逐行解析文本
for line in log_text.strip().split('\n'):
    # 查找图像名称
    img_match = re.search(r"图像为：(.+)", line)
    if img_match:
        current_image = img_match.group(1).strip()
        structured_data[current_image] = []  # 为新图片创建一个空列表

    # 查找距离值
    dist_match = re.search(r"distance is ([\d\.]+)", line)
    if dist_match and current_image:
        structured_data[current_image].append(float(dist_match.group(1)))

# 打印结构化结果
for img, dists in structured_data.items():
    print(f"[{img}] 包含的针距值:")
    for d in dists:
        print(f"  - {d}")
# import cv2
# import math
#
# # 全局变量，用于存储鼠标点击的坐标
# points = []
#
#
# def click_event(event, x, y, flags, param):
#     global points, img
#     # 当鼠标左键被按下时
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(points) < 2:
#             points.append((x, y))
#             # 在点击的位置画一个小圆点 (红色)
#             cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
#             cv2.imshow("Image", img)
#
#             # 当点了两个点之后，计算距离并画线
#             if len(points) == 2:
#                 # 画一条连接两点的线 (绿色)
#                 cv2.line(img, points[0], points[1], (0, 255, 0), 2)
#                 cv2.imshow("Image", img)
#
#                 # 计算两点之间的欧氏距离 (像素距离)
#                 x1, y1 = points[0]
#                 x2, y2 = points[1]
#                 pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#
#                 print("-" * 40)
#                 print(f"点 1 坐标: ({x1}, {y1})")
#                 print(f"点 2 坐标: ({x2}, {y2})")
#                 print(f"这两点之间的像素距离为: {pixel_distance:.4f} 像素")
#                 print("-" * 40)
#                 print("1. 如果这段距离在现实中是 1 cm (10 mm)")
#                 print(f"2. 比例尺 (Scale) = 10 / {pixel_distance:.4f} = {10 / pixel_distance:.4f} mm/像素")
#                 print("3. 按键盘任意键退出程序...")
#
#
# if __name__ == "__main__":
#     # ==========================================
#     # 请将这里的路径替换为你那张带有 1cm 标尺的图像路径
#     image_path ="E:/Line/TigerSewDataset(TSD)/T-shirt-Needlelength/bili256256/2026bilichi.png"
#     # ==========================================
#
#     img = cv2.imread(image_path)
#
#     if img is None:
#         print("错误：无法读取图像，请检查路径是否正确！")
#     else:
#         # 创建窗口并绑定鼠标回调函数
#         cv2.namedWindow("Image")
#         cv2.setMouseCallback("Image", click_event)
#
#         print("请在弹出的图像窗口中，用鼠标左键点击标尺的【起点】和【终点】...")
#
#         # 显示图像并等待操作
#         cv2.imshow("Image", img)
#         cv2.waitKey(0)  # 按任意键关闭窗口
#         cv2.destroyAllWindows()


import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==========================================
# 请将这里的路径替换为你那张带有 1cm 标尺的图像路径
image_path = "E:/Line/TigerSewDataset(TSD)/T-shirt-Needlelength/bili256256/2026bilichi.png"
# ==========================================

try:
    # 读取图像
    img = mpimg.imread(image_path)

    # 创建显示窗口
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title("请用鼠标左键在标尺的【起点】和【终点】各点击一下", fontsize=14)

    print("等待鼠标点击...")

    # ginput(2) 会自动等待用户在图像上点击两次
    # timeout=-1 表示无限等待
    points = plt.ginput(2, timeout=-1)
    plt.close()  # 点完两次后自动关闭窗口

    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]

        # 计算欧氏距离
        pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        print("-" * 40)
        print(f"点 1 坐标: ({x1:.1f}, {y1:.1f})")
        print(f"点 2 坐标: ({x2:.1f}, {y2:.1f})")
        print(f"这两点之间的像素距离为: {pixel_distance:.4f} 像素")
        print("-" * 40)
        print("1. 假设这段距离在现实中是 1 cm (10 mm)")
        print(f"2. 比例尺 (Scale) = 10 / {pixel_distance:.4f} = {10 / pixel_distance:.4f} mm/像素")

except FileNotFoundError:
    print("错误：找不到图像文件，请检查 image_path 路径中的斜杠是否正确（建议使用正斜杠 /）！")
except Exception as e:
    print(f"发生错误: {e}")
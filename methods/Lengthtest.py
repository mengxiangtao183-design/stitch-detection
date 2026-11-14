import asyncio
import platform
import pygame
import math
from pygame import Vector2

# 初始化Pygame
pygame.init()
# 设置默认窗口大小（可调整）
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1000, 1080
screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Measure Distance on Image")
FPS = 60

# 全局变量
image = None
points = []
distance = 0
font = pygame.font.Font(None, 36)
scale_factor = 1.0  # 缩放比例，1.0为原始大小
image_rect = None  # 图像的显示区域


def setup():
    global image, screen, image_rect, scale_factor
    try:
        # 替换为你的图片路径，使用绝对路径
        image_path = "E:/Line/photowhiteLiang/Detect/bilibili/yellow/fl001.png"  # 请修改为实际图片路径
        image = pygame.image.load(image_path)
        # 自适应窗口大小
        img_width, img_height = image.get_size()
        screen = pygame.display.set_mode((max(img_width, DEFAULT_WIDTH), max(img_height, DEFAULT_HEIGHT)),
                                         pygame.RESIZABLE)
        # 初始化缩放比例，使图像适配窗口
        scale_factor = min(screen.get_width() / img_width, screen.get_height() / img_height)
        image_rect = image.get_rect()
        image_rect.width = int(img_width * scale_factor)
        image_rect.height = int(img_height * scale_factor)
        print(f"图片加载成功，大小：{img_width}x{img_height}，初始缩放比例：{scale_factor:.2f}")
    except Exception as e:
        print(f"图片加载失败：{e}")
        image = pygame.Surface((DEFAULT_WIDTH, DEFAULT_HEIGHT))
        image.fill((255, 255, 255))
        screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)
        image_rect = image.get_rect()
        scale_factor = 1.0


def update_loop():
    global points, distance, screen, scale_factor, image_rect

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键点击
                # 将鼠标坐标转换为原始图像坐标
                mouse_pos = Vector2(event.pos)
                original_pos = Vector2(mouse_pos.x / scale_factor, mouse_pos.y / scale_factor)
                points.append(original_pos)
                if len(points) == 2:
                    # 计算原始图像上的像素距离
                    distance = points[0].distance_to(points[1])
                    points.append(points[0])  # 闭合线段
                elif len(points) > 2:
                    points = [original_pos]  # 重置为新起点
                    distance = 0
        elif event.type == pygame.VIDEORESIZE:
            # 处理窗口大小调整
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            # 重新计算缩放比例
            if image:
                img_width, img_height = image.get_size()
                scale_factor = min(screen.get_width() / img_width, screen.get_height() / img_height)
                image_rect.width = int(img_width * scale_factor)
                image_rect.height = int(img_height * scale_factor)
        elif event.type == pygame.KEYDOWN:
            # 按 +/- 键调整缩放
            if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                scale_factor *= 1.1  # 放大10%
            elif event.key == pygame.K_MINUS:
                scale_factor *= 0.9  # 缩小10%
            # 更新图像显示大小
            if image:
                img_width, img_height = image.get_size()
                image_rect.width = int(img_width * scale_factor)
                image_rect.height = int(img_height * scale_factor)

    # 绘制
    screen.fill((255, 255, 255))  # 清屏
    if image:
        # 缩放图像
        scaled_image = pygame.transform.scale(image, (image_rect.width, image_rect.height))
        screen.blit(scaled_image, (0, 0))

    # 绘制线段和点（基于缩放后的坐标）
    if len(points) >= 2:
        scaled_points = [Vector2(p.x * scale_factor, p.y * scale_factor) for p in points]
        pygame.draw.lines(screen, (255, 0, 0), False, scaled_points, 2)
        for point in scaled_points:
            pygame.draw.circle(screen, (0, 255, 0), (int(point.x), int(point.y)), 5)

    # 显示距离
    if distance > 0:
        text = font.render(f"距离: {distance:.2f} 像素", True, (0, 0, 0))
        screen.blit(text, (10, 10))

    pygame.display.flip()
    return True


async def main():
    setup()
    while True:
        if not update_loop():
            break
        await asyncio.sleep(1.0 / FPS)


if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        try:
            asyncio.run(main())
        finally:
            pygame.quit()
import pygame
import numpy as np

pygame.init()
pygame.font.init()

# =============================
# 基本参数
# =============================
screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("SSVEP 4-Frequency Stimulus")

Fs = 240
duration = 10
mean = 0.5
amp = 0.5
phi = 0

# 四个频率
Freqs = [8, 10, 12, 15]

# 四个刺激块位置（左右上下分布）
stim_rects = [
    pygame.Rect(810, 90, 300, 300),   # 中上
    pygame.Rect(510, 390, 300, 300),  # 左
    pygame.Rect(810, 690, 300, 300),   # 中下
    pygame.Rect(1110,390, 300, 300),  # 右
]

clock = pygame.time.Clock()
start = pygame.time.get_ticks()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # 当前时间（秒）
    t = (pygame.time.get_ticks() - start) / 1000.0
    if t >= duration:
        break

    screen.fill((0, 0, 0))

    # 四个刺激块分别计算亮度
    for i in range(4):
        f = Freqs[i]
        luminance = mean + amp * np.sin(2 * np.pi * f * t + phi)
        luminance = np.clip(luminance, 0.0, 1.0)
        gray = int(255 * luminance)

        pygame.draw.rect(
            screen,
            (gray, gray, gray),
            stim_rects[i]
        )

    pygame.display.flip()
    clock.tick(Fs)

pygame.quit()

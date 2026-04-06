import pygame
import numpy as np

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("SSVEP Stimulus")

stim_rect = pygame.Rect(760, 340, 400, 400)

Fs = 240
duration = 20
mean = 0.5
amp = 0.5
phi = 0
f = 10  # 闪烁频率(Hz)

clock = pygame.time.Clock()
start = pygame.time.get_ticks()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # 👉 按 ESC 立即退出
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    t = (pygame.time.get_ticks() - start) / 1000.0
    if t >= duration:
        break

    luminance = mean + amp * np.sin(2 * np.pi * f * t + phi)
    luminance = float(np.clip(luminance, 0.0, 1.0))
    gray = int(255 * luminance)

    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (gray, gray, gray), stim_rect)
    pygame.display.flip()

    clock.tick(Fs)  # 锁到 ~240 FPS

pygame.quit()

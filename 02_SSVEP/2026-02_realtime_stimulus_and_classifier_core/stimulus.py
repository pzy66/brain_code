# ==============================================================
#   stimulus.py
# ==============================================================

import pygame
import numpy as np
import time
import sys

from multiprocessing import Queue

REFRESH_HZ = 240.0

FREQ = {
    "U": 10.0,
    "D": 12.0,
    "L": 15.5,
    "R": 20.0,
}

BACKGROUND = (30,30,30)

HIGHLIGHT_COLOR = (255, 50, 50)

# =========================

def build_sine_luminance_series(f, hz, duration=3600, base=0.5, amp=0.40):
    n = int(hz * duration)
    i = np.arange(n)
    gray = 255*(base + amp*np.sin(2*np.pi*f*i/hz))
    return np.clip(gray, 0, 255).astype(np.uint8)

# =========================
def stimulus_loop(q: Queue):

    pygame.init()

    screen = pygame.display.set_mode(
        (0,0), pygame.FULLSCREEN | pygame.DOUBLEBUF, vsync=1
    )

    W,H = screen.get_size()

    gap  = int(min(W,H)*0.05)
    side = min((W-3*gap)//2, (H-3*gap)//2)

    rects = {
        "U": pygame.Rect((W-side)//2, gap, side, side),
        "D": pygame.Rect((W-side)//2, H-gap-side, side, side),
        "L": pygame.Rect(gap, (H-side)//2, side, side),
        "R": pygame.Rect(W-gap-side, (H-side)//2, side, side),
    }

    font = pygame.font.SysFont("simhei", 36)

    # 亮度序列
    series = {
        tag: build_sine_luminance_series(f, REFRESH_HZ)
        for tag,f in FREQ.items()
    }

    predicted_tag = None
    predicted_freq = None
    predicted_score = None

    clock = pygame.time.Clock()
    i = 0

    while True:

        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return

        # ------------------- 读取预测 ---------------------

        if not q.empty():
            msg = q.get()

            fre = msg["freq"]
            sc  = msg["score"]

            # 频率 -> tag 映射
            predicted_tag = min(
                FREQ,
                key=lambda t: abs(FREQ[t]-fre)
            )

            predicted_freq  = fre
            predicted_score = sc

        # ------------------- 绘制 --------------------------

        screen.fill(BACKGROUND)

        for tag,rect in rects.items():

            g = int(series[tag][i % len(series[tag])])
            col = (g,g,g)

            pygame.draw.rect(screen,col,rect)

            if tag == predicted_tag:
                pygame.draw.rect(
                    screen,
                    HIGHLIGHT_COLOR,
                    rect,
                    width=6
                )

        # 文本反馈
        if predicted_freq:

            txt = f"Pred: {predicted_freq:.2f} Hz  | rho={predicted_score:.3f}"
            img = font.render(txt, True, (255,255,255))
            screen.blit(img,(20,20))

        pygame.display.flip()

        i += 1
        clock.tick_busy_loop(0)



if __name__ == "__main__":
    stimulus_loop(None)

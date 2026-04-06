# ==============================================================
#  SSVEP sinusoidal-luminance flicker stimulus (frame-locked)
#  for 240 Hz monitor (no auto refresh detection)
# ==============================================================
#  - All targets same phase
#  - Frequencies distinguished only by frequency coding
#  - VSYNC driven (no sleep timing control)
# ==============================================================

import pygame
import numpy as np
import time
import sys

# =========================
# User parameters
# =========================
REFRESH_HZ = 240.0        # 使用你“标定”的刷新率
DURATION_SEC = 3600      # 最长持续时间（1 小时）

# 4 个 target 的频率
FREQ = {
    "U": 10.0,
    "D": 12.0,
    "L": 15.0,
    "R": 20.0
}

PHASE = 0.0              # 所有刺激同相位
L_BASE = 0.5             # 基线亮度（0~1）
AMPLITUDE = 0.40         # 亮度振幅（0~1），注意: L_BASE ± A 必须在 [0,1] 内

FONT_COLOR_LIGHT = (0,0,0)
FONT_COLOR_DARK  = (255,255,255)

BACKGROUND = (30,30,30)

# =========================
# Generate sine brightness sequence
# =========================

def build_sine_luminance_series(f, refresh_hz,
                                duration_sec,
                                base=L_BASE,
                                amp=AMPLITUDE,
                                phase=PHASE):
    """
    Generate sinusoidal brightness flicker sequence:
        L[i] = base + amp * sin(2*pi*f*i/F)
        L ∈ [0,1]  -> gray ∈ [0,255]
    """
    n_frames = int(refresh_hz * duration_sec)
    i = np.arange(n_frames)

    sine = np.sin(2 * np.pi * f * (i / refresh_hz) + phase)

    L = base + amp * sine
    L = np.clip(L, 0.0, 1.0)

    gray = (L * 255).astype(np.uint8)

    return gray

# =========================
# Drawing helpers
# =========================

def draw_text(surface, text, rect, color, font):
    img = font.render(text, True, color)
    surface.blit(img, img.get_rect(center=rect.center))


# =========================
# Main routine
# =========================

def main():

    pygame.init()

    # 全屏 & VSYNC
    screen = pygame.display.set_mode(
        (0,0),
        pygame.FULLSCREEN | pygame.DOUBLEBUF,
        vsync=1
    )

    pygame.display.set_caption("SSVEP Sinusoidal Stimulus (Frame-Locked)")
    pygame.mouse.set_visible(False)

    W, H = screen.get_size()

    gap  = int(min(W, H) * 0.05)
    side = min((W - 3*gap)//2, (H - 3*gap)//2)

    # 四个刺激区域布局
    rects = {
        "U": pygame.Rect((W-side)//2, gap, side, side),
        "D": pygame.Rect((W-side)//2, H-gap-side, side, side),
        "L": pygame.Rect(gap, (H-side)//2, side, side),
        "R": pygame.Rect(W-gap-side, (H-side)//2, side, side)
    }

    font_main  = pygame.font.SysFont(None, max(24, int(min(W,H)*0.04)))
    font_debug = pygame.font.SysFont(None, max(16, int(min(W,H)*0.02)))

    # =========================
    # Pre-generate brightness sequences
    # =========================

    print("\n==== Build SSVEP sinusoidal stimulus ====")
    print("Refresh rate:", REFRESH_HZ)

    series_map = {}

    for tag, f in FREQ.items():
        series = build_sine_luminance_series(
            f,
            REFRESH_HZ,
            duration_sec=DURATION_SEC,
        )
        series_map[tag] = series
        print(f"{tag}  ->  {f} Hz   frames/cycle ≈ {REFRESH_HZ/f:.2f}")

    # =========================
    # Frame-locked display loop
    # =========================

    clock     = pygame.time.Clock()
    frame_idx = 0

    timestamps = []

    print("\nPress ESC or Q to quit.\n")

    try:
        while True:

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        raise KeyboardInterrupt

            screen.fill(BACKGROUND)

            # 绘制每个刺激块
            for tag in ("U", "D", "L", "R"):

                gray = int(series_map[tag][frame_idx %
                                           len(series_map[tag])])

                color = (gray, gray, gray)

                pygame.draw.rect(screen, color, rects[tag])

                # 自动选择字颜色，确保可读
                txt_color = FONT_COLOR_LIGHT if gray > 128 else FONT_COLOR_DARK

                draw_text(
                    screen,
                    f"{FREQ[tag]:.2f} Hz",
                    rects[tag],
                    txt_color,
                    font_main
                )

            # 左上角调试显示
            info_lines = [
                f"Resolution: {W} x {H}",
                f"Refresh (fixed): {REFRESH_HZ:.2f} Hz",
                f"Frame idx: {frame_idx}"
            ]

            y = 10
            for s in info_lines:
                img = font_debug.render(s, True, (220,220,220))
                screen.blit(img, (10, y))
                y += img.get_height() + 3

            # 提交帧
            pygame.display.flip()

            timestamps.append(time.perf_counter())

            frame_idx += 1
            clock.tick_busy_loop(0)

    except KeyboardInterrupt:
        print("Exit.")

    finally:
        print("\n==== Timing summary ====")

        if len(timestamps) > 3:
            ts = np.array(timestamps)
            dt = np.diff(ts)

            real_F = 1.0 / dt.mean()

            print(f"Measured refresh ≈ {real_F:.3f} Hz")
            print(f"Frame jitter std ≈ {dt.std()*1000:.4f} ms")
            print(f"Total frames shown: {len(ts)}")

        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()

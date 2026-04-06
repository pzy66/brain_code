import cv2
import time
import threading

# 替换你的地址
STREAM_URL = "http://192.168.149.1:8080/stream?topic=/usb_cam/image_rect_color"


class CameraLoader:
    def __init__(self, url):
        self.stream = cv2.VideoCapture(url)
        if not self.stream.isOpened():
            raise ValueError("无法连接视频流")

        # 简单设置一下缓存（虽然对网络流不一定完全生效，但有好处）
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.frame_count = 0  # 记录接收到的总帧数

    def start(self):
        # 开启一个独立线程，专门负责“疯狂读取”
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            # 只要流还开着，就一直读，把缓冲区读空，保证 self.frame 是最新的
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break

            self.grabbed = grabbed
            self.frame = frame
            self.frame_count += 1

    def read(self):
        # 主程序调用这个，直接拿最新的帧，不需要等待解码
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


def test_real_fps():
    print(f"--- 正在连接视频流 (多线程模式) ---")
    try:
        cam = CameraLoader(STREAM_URL).start()
    except Exception as e:
        print(e)
        return

    print("连接成功！正在测试【真实网络传输帧率】...")

    # 等待一小会儿让流稳定
    time.sleep(2)

    start_time = time.time()
    start_frame_count = cam.frame_count

    last_print = start_time

    while True:
        # 这里模拟你的 YOLO 主程序
        frame = cam.read()

        # 显示画面
        if frame is not None:
            cv2.imshow("Real FPS Test", frame)

        # --- 核心统计逻辑 ---
        # 我们不统计 while 循环跑了多少次，而是统计 CameraLoader 到底接收了多少帧
        current_time = time.time()

        if current_time - last_print >= 1.0:
            # 计算过去 1 秒内，相机线程接收了多少张新图
            current_frame_count = cam.frame_count
            delta_frames = current_frame_count - start_frame_count

            real_fps = delta_frames / (current_time - start_time)

            print(f"真实网络FPS: {real_fps:.2f}")

            # 重置计数器，进行下一轮统计
            last_print = current_time
            start_time = current_time
            start_frame_count = current_frame_count

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_real_fps()
import requests

HOST = "http://192.168.149.1:8080"
TOPICS = [
    "/usb_cam/image_raw",
    "/usb_cam/image_color",
    "/usb_cam/image_rect",
    "/usb_cam/image_rect_color",
]

CANDIDATES = []
for t in TOPICS:
    CANDIDATES += [
        f"{HOST}/stream?topic={t}",
        f"{HOST}/snapshot?topic={t}",
        f"{HOST}{t}",
        f"{HOST}{t}?type=stream",
    ]

for url in CANDIDATES:
    try:
        r = requests.get(url, stream=True, timeout=2)
        ctype = r.headers.get("Content-Type", "")
        if "multipart/x-mixed-replace" in ctype.lower():
            print("✅ MJPEG:", url, "|", ctype)
        elif "image/" in ctype.lower():
            print("🖼️ Snapshot:", url, "|", ctype)
        else:
            # 其它类型一般不是图像
            pass
    except Exception:
        pass

import requests


CAMERA_URL = (
    "http://192.168.149.1:8080/stream?"
    "topic=/usb_cam/image_rect_color&type=mjpeg&width=640&height=480&quality=80"
)


def main() -> None:
    response = requests.get(CAMERA_URL, stream=True, timeout=5)
    content_type = response.headers.get("Content-Type", "")
    if "multipart/x-mixed-replace" not in content_type.lower():
        raise RuntimeError(f"Unexpected camera stream content type: {content_type}")
    print("official MJPEG stream:", CAMERA_URL, "|", content_type)


if __name__ == "__main__":
    main()

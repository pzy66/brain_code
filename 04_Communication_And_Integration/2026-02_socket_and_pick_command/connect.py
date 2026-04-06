import socket

JETSON_IP = "192.168.149.1"  # 改成你 Jetson IP
PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((JETSON_IP, PORT))
print("✅ connected. 输入 F/B/L/R/S, Q 退出")

try:
    while True:
        s = input("cmd> ").strip().upper()
        if not s:
            continue
        if s == "Q":
            break
        sock.sendall(s[0].encode("ascii"))
        print("📤 sent:", s[0])
finally:
    sock.close()

import socket
import threading

HOST = "127.0.0.1"
PORT = 10086


def recv_thread(sock):
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                print("\n[连接关闭]")
                break
            print("\n[收到]", data.decode("utf-8", errors="ignore"))
        except:
            break


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

print(f"已连接 {HOST}:{PORT}")
print("输入内容回车发送，Ctrl+C退出\n")

threading.Thread(target=recv_thread, args=(s,), daemon=True).start()

while True:
    try:
        msg = input("> ")
        s.send(msg.encode("utf-8"))
    except KeyboardInterrupt:
        print("\n退出")
        break

s.close()
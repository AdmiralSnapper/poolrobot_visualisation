import socket
import struct
import cv2
import numpy as np

def run_receiver(host='0.0.0.0', port=5000):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print(f"Listening on port {port}. Waiting for ROS node to connect...")
    conn, addr = server_sock.accept()
    print(f"Connected to ROS device at {addr}")

    data_buffer = b""
    header_size = struct.calcsize('>I')

    try:
        while True:
            # 1. Read message length header (4 bytes)
            while len(data_buffer) < header_size:
                packet = conn.recv(4096)
                if not packet:
                    return
                data_buffer += packet

            msg_size = struct.unpack('>I', data_buffer[:header_size])[0]
            data_buffer = data_buffer[header_size:]

            # 2. Read full JPEG payload based on msg_size
            while len(data_buffer) < msg_size:
                packet = conn.recv(4096)
                if not packet:
                    return
                data_buffer += packet

            frame_data = data_buffer[:msg_size]
            data_buffer = data_buffer[msg_size:]

            # 3. Decode JPEG buffer and show image window
            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                cv2.imshow('Live Stream from ROS2 Robot', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error receiving frame: {e}")
    finally:
        conn.close()
        server_sock.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run_receiver()

    
#!/usr/bin/env python3
# File: receiver_pc.py
import socket
import struct
import cv2
import numpy as np
import json

MSG_TYPE_IMAGE = 1
MSG_TYPE_POSE = 2

def run_receiver(host='0.0.0.0', port=5000):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print(f"Listening on port {port}. Waiting for ROS node to connect...")
    conn, addr = server_sock.accept()
    print(f"Connected to ROS device at {addr}")

    data_buffer = b""
    # Header size: 1 byte msg_type ('B') + 4 bytes uint32 payload_size ('I')
    header_size = struct.calcsize('>BI')

    try:
        while True:
            # 1. Read header (5 bytes)
            while len(data_buffer) < header_size:
                packet = conn.recv(4096)
                if not packet:
                    return
                data_buffer += packet

            msg_type, msg_size = struct.unpack('>BI', data_buffer[:header_size])
            data_buffer = data_buffer[header_size:]

            # 2. Read full payload
            while len(data_buffer) < msg_size:
                packet = conn.recv(4096)
                if not packet:
                    return
                data_buffer += packet

            payload = data_buffer[:msg_size]
            data_buffer = data_buffer[msg_size:]

            # 3. Handle message by type
            if msg_type == MSG_TYPE_IMAGE:
                np_arr = np.frombuffer(payload, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow('Live Stream from ROS2 Robot', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            elif msg_type == MSG_TYPE_POSE:
                pose_data = json.loads(payload.decode('utf-8'))
                pos = pose_data['position']
                orient = pose_data['orientation']
                print(f"[POSE] Pos: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f}) | Orient: ({orient['x']:.3f}, {orient['y']:.3f}, {orient['z']:.3f}, {orient['w']:.3f})")

    except Exception as e:
        print(f"Error receiving socket data: {e}")
    finally:
        conn.close()
        server_sock.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run_receiver()

    
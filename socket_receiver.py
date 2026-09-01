#!/usr/bin/env python3
"""Windows 11 receiver: live RGB window + live 3D camera/tag frames (Open3D)."""
import json
import socket
import struct
from collections import deque

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

MSG_TYPE_IMAGE = 1
MSG_TYPE_CAMERA_POSE = 2
MSG_TYPE_TAG_POSES = 3

HOST = '0.0.0.0'
PORT = 5000
CAMERA_FRAME_SIZE = 0.25
TAG_FRAME_SIZE = 0.15
TRAIL_LENGTH = 500


def inv_transform(T):
    """Closed-form inverse of a 4x4 homogeneous transform."""
    Tinv = np.eye(4)
    Tinv[:3, :3] = T[:3, :3].T
    Tinv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Tinv


class LiveViewer:
    def __init__(self):
        # --- Open3D scene ---
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Live 3D Frames', width=1024, height=768)

        # 1. Camera Frame (matches tag size)
        self.cam_geom = None
        self.cam_T = np.eye(4)

        # 2. Tag frames & path trail
        self.tag_frames = {}          # id -> [geometry, current_T]
        self.path_points = deque(maxlen=TRAIL_LENGTH)
        self.path_line = None
        self.path_added = False

    # ---------- geometry updates ----------
    def _set_pose(self, geom, T_new, T_old):
        """Update geometry to T_new without accumulation errors."""
        geom.transform(T_new @ inv_transform(T_old))
        self.vis.update_geometry(geom)

    def set_camera_pose(self, pos, quat):
        pos_arr = np.asarray(pos, dtype=float)

        # Ignore lost detection / zero fallback poses
        if np.allclose(pos_arr, [0.0, 0.0, 0.0]):
            return

        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T[:3, 3] = pos_arr

        if self.cam_geom is None:
            self.cam_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=CAMERA_FRAME_SIZE)
            self.cam_geom.transform(T)
            self.vis.add_geometry(self.cam_geom)
        else:
            self._set_pose(self.cam_geom, T, self.cam_T)

        self.cam_T = T

        # --- trail ---
        self.path_points.append(T[:3, 3])
        pts = np.asarray(self.path_points)
        if len(pts) < 2:
            return

        lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
        colors = np.tile([0.0, 0.8, 0.2], (len(lines), 1))

        if not self.path_added:
            self.path_line = o3d.geometry.LineSet()
            self.path_line.points = o3d.utility.Vector3dVector(pts)
            self.path_line.lines = o3d.utility.Vector2iVector(lines)
            self.path_line.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.path_line)
            self.path_added = True
        else:
            self.path_line.points = o3d.utility.Vector3dVector(pts)
            self.path_line.lines = o3d.utility.Vector2iVector(lines)
            self.path_line.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.path_line)


    def set_tag_pose(self, tag_id, pos, quat, rel_to_camera=True):
        T_rel = np.eye(4)
        T_rel[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T_rel[:3, 3] = np.asarray(pos, dtype=float)
        T_world = self.cam_T @ T_rel if rel_to_camera else T_rel

        entry = self.tag_frames.get(tag_id)
        if entry is None:
            geom = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=TAG_FRAME_SIZE)
            geom.transform(T_world)          # set pose BEFORE adding
            self.vis.add_geometry(geom)      # renderer picks it up on next pump
            self.tag_frames[tag_id] = [geom, T_world]
        else:
            geom, T_old = entry
            geom.transform(T_world @ inv_transform(T_old))
            self.vis.update_geometry(geom)   # only for already-visible geoms
            entry[1] = T_world


    # ---------- message handling ----------
    def handle(self, msg_type, payload):
        if msg_type == MSG_TYPE_IMAGE:
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow('Live RGB', img)
        elif msg_type == MSG_TYPE_CAMERA_POSE:
            data = json.loads(payload)
            self.set_camera_pose(data['position'], data['quaternion'])
        elif msg_type == MSG_TYPE_TAG_POSES:
            data = json.loads(payload)
            for p in data['poses']:
                self.set_tag_pose(p['id'], p['position'], p['quaternion'])

    def pump(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        cv2.waitKey(1)

    def close(self):
        self.vis.destroy_window()
        cv2.destroyAllWindows()


def recvall(conn, n):
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError('connection closed')
        data += chunk
    return data


def main():
    viewer = LiveViewer()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f'Listening on port {PORT}...')
    conn, addr = server.accept()
    conn.settimeout(0.1)
    print(f'Connected: {addr}')

    buf = b''
    try:
        while True:
            try:
                chunk = conn.recv(65536)
                if not chunk:
                    print('Connection lost.')
                    break
                buf += chunk
                while len(buf) >= 5:
                    msg_type, length = struct.unpack('>BI', buf[:5])
                    if len(buf) < 5 + length:
                        break
                    payload = buf[5:5 + length]
                    buf = buf[5 + length:]
                    viewer.handle(msg_type, payload)
            except socket.timeout:
                pass
            viewer.pump()
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        server.close()
        viewer.close()


if __name__ == '__main__':
    main()

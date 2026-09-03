#!/usr/bin/env python3
"""
Windows 11 receiver: live RGB window + live 3D camera/tag frames (Open3D).

Protocol (same framing as the Pi sender):
    frame = 1 byte type | 4 byte big-endian length | payload
        type 1 = image (JPEG bytes)
        type 2 = camera pose (JSON: position, quaternion, frame_id, stamp)
        type 3 = tag poses (JSON: poses: [{id, position, quaternion}, ...])

Run with Python 3.11:  py -3.11 socket_receiver.py
"""
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

# --- physical tag sizes (metres) ---
TAG_SIZE_DEFAULT = 0.223
TAG_SIZES = {
    # per-id override if your tags differ in size, e.g.:
    # 0: 0.1175, 1: 0.0840, 2: 0.04187
}


def inv_transform(T):
    """Closed-form inverse of a 4x4 homogeneous transform."""
    Tinv = np.eye(4)
    Tinv[:3, :3] = T[:3, :3].T
    Tinv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Tinv


def _make_tag_square(size):
    """Flat white tag face with black border, lying in the XY plane."""
    s = size / 2.0

    square = o3d.geometry.TriangleMesh.create_box(
        width=size, height=size, depth=0.0015)
    square.translate([-s, -s, -0.00075])       # center on the tag origin
    square.paint_uniform_color([0.95, 0.95, 0.95])

    pts = [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]]
    border = o3d.geometry.LineSet()
    border.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
    border.lines = o3d.utility.Vector2iVector(
        np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32))
    border.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.0, 0.0], (4, 1)))
    return square, border


class LiveViewer:
    def __init__(self):
        # --- Open3D scene ---
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Live 3D Frames', width=1024, height=768)

        # Camera frame is created lazily on the first valid pose
        self.cam_geom = None
        self.cam_T = np.eye(4)

        # Tag frames: id -> [frame_geometry, current_T]
        self.tag_frames = {}
        # Tag squares: id -> [square_mesh, border_lines, current_T]
        self.tag_squares = {}
        # Previous tag pose snapshot for change detection
        self._last_tag_poses = None

        # Camera trail
        self.path_points = deque(maxlen=TRAIL_LENGTH)
        self.path_line = None
        self.path_added = False

    def _set_pose(self, geom, T_new, T_old):
        """Update geometry in place from old transform to new."""
        delta = T_new @ inv_transform(T_old)
        geom.transform(delta)
        self.vis.update_geometry(geom)

    def set_camera_pose(self, pos, quat):
        pos_arr = np.asarray(pos, dtype=float)

        # Ignore lost-detection / zero fallback poses
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

        # --- coordinate frame ---
        entry = self.tag_frames.get(tag_id)
        if entry is None:
            geom = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=TAG_FRAME_SIZE)
            geom.transform(T_world)
            self.vis.add_geometry(geom)
            self.tag_frames[tag_id] = [geom, T_world]
        else:
            geom, T_old = entry
            self._set_pose(geom, T_world, T_old)
            entry[1] = T_world

        # --- physical square ---
        size = TAG_SIZES.get(tag_id, TAG_SIZE_DEFAULT)
        entry_sq = self.tag_squares.get(tag_id)
        if entry_sq is None:
            square, border = _make_tag_square(size)
            square.transform(T_world)
            border.transform(T_world)
            self.vis.add_geometry(square)
            self.vis.add_geometry(border)
            self.tag_squares[tag_id] = [square, border, T_world]
        else:
            square, border, T_old = entry_sq
            delta = T_world @ inv_transform(T_old)
            square.transform(delta)
            border.transform(delta)
            self.vis.update_geometry(square)
            self.vis.update_geometry(border)
            entry_sq[2] = T_world

    def handle(self, msg_type, payload):
        if msg_type == MSG_TYPE_IMAGE:
            frame = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Camera Feed', frame)
                cv2.waitKey(1)

        elif msg_type == MSG_TYPE_CAMERA_POSE:
            data = json.loads(payload)
            self.set_camera_pose(data['position'], data['quaternion'])

        elif msg_type == MSG_TYPE_TAG_POSES:
            data = json.loads(payload)

            # Skip if identical to the previous message
            snapshot = tuple(
                (p['id'], tuple(p['position']), tuple(p['quaternion']))
                for p in data['poses']
            )
            if snapshot == self._last_tag_poses:
                return

            self._last_tag_poses = snapshot

            for p in data['poses']:
                self.set_tag_pose(p['id'], p['position'], p['quaternion'])

    def run(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)

        print(f'Listening on port {PORT}. Waiting for the ROS node to connect...')
        conn, addr = server_sock.accept()
        print(f'Connected: {addr}')

        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Feed', 1280, 960)
        conn.settimeout(0.05)

        data_buffer = b''
        header_size = 5  # 1 byte type + 4 byte length

        try:
            while True:
                # Pump the Open3D viewer every iteration
                if not self.vis.poll_events():
                    print('3D window closed.')
                    break
                self.vis.update_renderer()

                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue          # no data this tick; keep rendering
                if not chunk:
                    print('Connection closed by sender.')
                    break
                data_buffer += chunk

                while len(data_buffer) >= header_size:
                    msg_type = data_buffer[0]
                    (length,) = struct.unpack('>I', data_buffer[1:5])

                    if len(data_buffer) < header_size + length:
                        break  # wait for the full payload

                    payload = data_buffer[header_size:header_size + length]
                    data_buffer = data_buffer[header_size + length:]

                    self.handle(msg_type, payload)
        except KeyboardInterrupt:
            print('\nStopped by user.')
        except Exception as e:
            print(f'Error: {e}')
        finally:
            conn.close()
            server_sock.close()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    LiveViewer().run()

#!/usr/bin/env python3
import re
import subprocess
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from mss import mss
from rclpy.node import Node
from sensor_msgs.msg import Image

RVIZ_WINDOW_CLASS = 'rviz'
CAPTURE_TOPIC = '/rviz/image'
CAPTURE_FPS = 12.0
REGION_REFRESH_INTERVAL = 2.0
MAX_CAPTURE_WIDTH = 400 #1280
MAX_CAPTURE_HEIGHT = 400 #720
RVIZ_WINDOW_NAME_RE = re.compile(r'\brviz2?\b', re.I)


def check_command_available(command):
    proc = subprocess.run(['which', command], capture_output=True, text=True)
    return proc.returncode == 0 and proc.stdout.strip()


def parse_geometry(stdout):
    data = {}
    for line in stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            try:
                data[key] = int(value)
            except ValueError:
                data[key] = value

    if not all(k in data for k in ('X', 'Y', 'WIDTH', 'HEIGHT')):
        return None

    return {
        'top': data['Y'],
        'left': data['X'],
        'width': data['WIDTH'],
        'height': data['HEIGHT'],
    }


def geometry_for_win(win_id):
    geom = subprocess.run(
        ['xdotool', 'getwindowgeometry', '--shell', win_id],
        capture_output=True,
        text=True
    )
    if geom.returncode != 0 or not geom.stdout.strip():
        return None
    return parse_geometry(geom.stdout)


def find_rviz_window():
    win_ids = []

    for cmd in [
        ['xdotool', 'search', '--onlyvisible', '--class', RVIZ_WINDOW_CLASS],
        ['xdotool', 'search', '--onlyvisible', '--class', 'rviz2'],
    ]:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            win_ids.extend(proc.stdout.strip().splitlines())

    unique_win_ids = list(dict.fromkeys(win_ids))
    candidates = []

    for win_id in unique_win_ids:
        geom = geometry_for_win(win_id)
        if not geom:
            continue

        if geom['width'] < 100 or geom['height'] < 100:
            continue

        name_proc = subprocess.run(
            ['xdotool', 'getwindowname', win_id],
            capture_output=True,
            text=True
        )
        win_name = name_proc.stdout.strip() if name_proc.returncode == 0 else ''
        if not win_name:
            continue

        if not RVIZ_WINDOW_NAME_RE.search(win_name):
            continue

        candidates.append((geom['width'] * geom['height'], win_id, geom, win_name))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    _, win_id, geom, win_name = candidates[0]
    return win_id, geom, win_name


def resize_frame(frame):
    height, width = frame.shape[:2]

    if width <= MAX_CAPTURE_WIDTH and height <= MAX_CAPTURE_HEIGHT:
        return frame

    scale = min(MAX_CAPTURE_WIDTH / width, MAX_CAPTURE_HEIGHT / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def sanitize_region(region):
    if region is None:
        return None

    width = int(region['width'])
    height = int(region['height'])
    left = int(region['left'])
    top = int(region['top'])

    if width <= 0 or height <= 0:
        return None

    return {
        'left': left,
        'top': top,
        'width': width,
        'height': height,
    }


class RvizScreenPublisher(Node):
    def __init__(self):
        super().__init__('rviz_screen_publisher')

        self.pub = self.create_publisher(Image, CAPTURE_TOPIC, 10)
        self.bridge = CvBridge()
        self.sct = mss()

        self.window_id = None
        self.window_name = None
        self.region = None
        self.last_region_check = 0.0
        self.last_window_log_time = 0.0
        self.last_not_found_log_time = 0.0

        self.timer = self.create_timer(1.0 / CAPTURE_FPS, self.publish_frame)

        self.get_logger().info(f'RViz screen publisher initialized: topic={CAPTURE_TOPIC}')

    def _log_window_info(self):
        now = time.time()
        if now - self.last_window_log_time > 5.0:
            self.get_logger().info(
                f'Using RViz window id={self.window_id} name={self.window_name} region={self.region}'
            )
            self.last_window_log_time = now

    def _log_not_found(self):
        now = time.time()
        if now - self.last_not_found_log_time > 2.0:
            self.get_logger().warn('RViz window not found.')
            self.last_not_found_log_time = now

    def _update_window_if_needed(self, force=False):
        now = time.time()

        if (not force) and self.window_id is not None and (now - self.last_region_check <= REGION_REFRESH_INTERVAL):
            return

        found = find_rviz_window()
        self.last_region_check = now

        if found is None:
            self.window_id = None
            self.window_name = None
            self.region = None
            self._log_not_found()
            return

        new_window_id, new_region, new_window_name = found
        new_region = sanitize_region(new_region)

        if new_region is None:
            self.window_id = None
            self.window_name = None
            self.region = None
            self.get_logger().warn('Found RViz window, but region is invalid.')
            return

        changed = (
            self.window_id != new_window_id or
            self.window_name != new_window_name or
            self.region != new_region
        )

        self.window_id = new_window_id
        self.window_name = new_window_name
        self.region = new_region

        if changed:
            self.get_logger().info(
                f'Using RViz window id={self.window_id} name={self.window_name} region={self.region}'
            )
            self.last_window_log_time = now
        else:
            self._log_window_info()

    def _capture_region(self):
        if self.region is None:
            return None

        try:
            raw = np.array(self.sct.grab(self.region))
            # mss 결과는 BGRA
            frame = raw[:, :, :3]   # BGR
            frame = resize_frame(frame)
            return frame
        except Exception as e:
            self.get_logger().warn(f'mss capture failed: {e}')
            return None

    def publish_frame(self):
        self._update_window_if_needed()

        if self.window_id is None or self.region is None:
            return

        frame = self._capture_region()

        if frame is None:
            # 캡처 실패 시 창을 즉시 버리지 않고, 다음 프레임에서 다시 시도
            # 단, region이 바뀌었을 수 있으니 즉시 한 번 재탐색만 수행
            self._update_window_if_needed(force=True)
            return

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(img_msg)


def main(args=None):
    if not check_command_available('xdotool'):
        print('xdotool이 설치되어 있지 않습니다. Ubuntu/Debian 기반 시스템에서는 `sudo apt install xdotool`로 설치하세요.')
        return

    rclpy.init(args=args)
    node = RvizScreenPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
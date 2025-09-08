# person_fusion_follower.py
# 카메라(보행자 검출로 각도) + 라이다(거리) 융합 → /cmd_vel 추종 제어

import math
import time
import numpy as np
import cv2 as cv

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32

from cv_bridge import CvBridge


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class PID:
    def __init__(self, kp, ki, kd, out_min=None, out_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0
        self.prev = None
        self.out_min = out_min
        self.out_max = out_max

    def reset(self):
        self.i = 0.0
        self.prev = None

    def __call__(self, ref, meas):
        e = ref - meas
        self.i += e
        d = 0.0 if self.prev is None else (e - self.prev)
        self.prev = e
        u = self.kp * e + self.ki * self.i + self.kd * d
        if self.out_min is not None:
            u = max(self.out_min, u)
        if self.out_max is not None:
            u = min(self.out_max, u)
        return u


class FusionFollower(Node):
    def __init__(self):
        super().__init__('fusion_follower')

        # ---------- Parameters ----------
        # Topics
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # Camera
        self.declare_parameter('camera_fov_deg', 70.0)        # 수평 FOV(대충 60~90 사이)
        self.declare_parameter('hog_scale', 1.05)             # HOG pyramid scale
        self.declare_parameter('hog_win_stride', 8)           # HOG winStride
        self.declare_parameter('frame_skip', 2)               # 매 N프레임만 검출

        # Fusion / Control
        self.declare_parameter('response_dist', 0.9)          # 목표 거리(m)
        self.declare_parameter('angle_window_deg', 3.0)       # 라이다 각도 윈도우(±)
        self.declare_parameter('front_stop_dist', 0.45)       # 정면 안전 정지 거리
        self.declare_parameter('front_cone_deg', 12.0)        # 정면 판단 각도(±)
        self.declare_parameter('conf_min_bbox_w', 40.0)       # 바운딩박스 최소 폭(픽셀)

        # Speed limits
        self.declare_parameter('lin_max', 0.6)
        self.declare_parameter('lin_min', -0.35)
        self.declare_parameter('ang_max', 1.8)

        # PID gains
        self.declare_parameter('lin_kp', 1.2)
        self.declare_parameter('lin_ki', 0.0)
        self.declare_parameter('lin_kd', 0.6)
        self.declare_parameter('ang_kp', 0.03)   # 각도 단위: deg
        self.declare_parameter('ang_ki', 0.0)
        self.declare_parameter('ang_kd', 0.02)

        # ---------- Resolve params ----------
        self.camera_topic = self.get_parameter('camera_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.fov_deg = float(self.get_parameter('camera_fov_deg').value)
        self.hog_scale = float(self.get_parameter('hog_scale').value)
        self.hog_win_stride = int(self.get_parameter('hog_win_stride').value)
        self.frame_skip = int(self.get_parameter('frame_skip').value)

        self.response_dist = float(self.get_parameter('response_dist').value)
        self.angle_window_deg = float(self.get_parameter('angle_window_deg').value)
        self.front_stop_dist = float(self.get_parameter('front_stop_dist').value)
        self.front_cone_deg = float(self.get_parameter('front_cone_deg').value)
        self.conf_min_bbox_w = float(self.get_parameter('conf_min_bbox_w').value)

        self.lin_max = float(self.get_parameter('lin_max').value)
        self.lin_min = float(self.get_parameter('lin_min').value)
        self.ang_max = float(self.get_parameter('ang_max').value)

        self.pid_lin = PID(
            float(self.get_parameter('lin_kp').value),
            float(self.get_parameter('lin_ki').value),
            float(self.get_parameter('lin_kd').value),
            out_min=self.lin_min, out_max=self.lin_max
        )
        self.pid_ang = PID(
            float(self.get_parameter('ang_kp').value),
            float(self.get_parameter('ang_ki').value),
            float(self.get_parameter('ang_kd').value),
            out_min=-self.ang_max, out_max=self.ang_max
        )

        # ---------- QoS ----------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ---------- Sub/Pub ----------
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, self.camera_topic, self.on_image, sensor_qos)
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, sensor_qos)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, sensor_qos)
        # 상태/디버그(선택)
        self.pub_state = self.create_publisher(String, 'fusion_follower/state', sensor_qos)
        self.pub_angle = self.create_publisher(Float32, 'fusion_follower/angle_deg', sensor_qos)
        self.pub_dist = self.create_publisher(Float32, 'fusion_follower/dist_m', sensor_qos)

        # ---------- HOG pedestrian detector ----------
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

        # ---------- State ----------
        self.last_scan = None
        self.last_angle_deg = None
        self.last_dist = None
        self.frame_count = 0
        self.img_w = None

        # 제어 루프 (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('FusionFollower started: camera=%s, scan=%s' %
                               (self.camera_topic, self.scan_topic))

    # ---------- Callbacks ----------
    def on_scan(self, msg: LaserScan):
        self.last_scan = msg

    def on_image(self, msg: Image):
        self.frame_count += 1

        # 프레임 스킵으로 CPU 절약
        if self.frame_skip > 0 and (self.frame_count % self.frame_skip != 0):
            return

        # OpenCV 이미지 변환
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'CvBridge error: {e}')
            return

        h, w = frame.shape[:2]
        self.img_w = w

        # HOG 보행자 검출
        # (winStride는 (s,s), padding은 (8,8) 정도가 무난)
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(self.hog_win_stride, self.hog_win_stride),
            padding=(8, 8),
            scale=self.hog_scale
        )

        if len(rects) == 0:
            # 검출 실패 → 각도 정보 제거 (control_loop에서 검색 상태)
            self.last_angle_deg = None
            return

        # 가장 "사람일 법한" 박스 고르기: 폭이 큰 것(=가까운/중앙 가능성↑)
        best_i = int(np.argmax([r[2] for r in rects]))  # r=(x,y,w,h)
        x, y, bw, bh = rects[best_i]

        # 바운딩박스 너무 작으면 (노이즈) 무시
        if bw < self.conf_min_bbox_w:
            self.last_angle_deg = None
            return

        # 화면 중심 대비 픽셀 오프셋 → 각도(deg)
        cx = x + bw / 2.0
        center_x = w / 2.0
        # 픽셀 오프셋 비율 * (수평 FOV/2)
        # 좌(+)/우(-)로 정의 (원하는 부호로 맞추세요)
        ratio = (cx - center_x) / (w / 2.0)
        angle_deg = ratio * (self.fov_deg / 2.0)

        self.last_angle_deg = float(angle_deg)
        self.pub_angle.publish(Float32(data=self.last_angle_deg))

    # ---------- Helpers ----------
    def lidar_distance_at_angle(self, scan: LaserScan, angle_deg: float, win_deg: float):
        if scan is None:
            return None
        n = len(scan.ranges)
        if n == 0:
            return None

        idx = np.arange(n, dtype=np.float32)
        ang = (scan.angle_min + scan.angle_increment * idx) * 180.0 / math.pi
        ang = (ang + 180.0) % 360.0 - 180.0

        m = np.abs(ang - angle_deg) <= win_deg
        if not np.any(m):
            return None

        r = np.asarray(scan.ranges, dtype=np.float32)[m]
        r = r[np.isfinite(r) & (r > 0.0)]
        if r.size == 0:
            return None

        return float(np.median(r))

    def front_obstacle(self, scan: LaserScan):
        if scan is None:
            return False
        n = len(scan.ranges)
        if n == 0:
            return False

        idx = np.arange(n, dtype=np.float32)
        ang = (scan.angle_min + scan.angle_increment * idx) * 180.0 / math.pi
        ang = (ang + 180.0) % 360.0 - 180.0

        m = np.abs(ang) <= self.front_cone_deg
        r = np.asarray(scan.ranges, dtype=np.float32)[m]
        r = r[np.isfinite(r) & (r > 0.0)]
        return (r.size > 0) and (np.min(r) < self.front_stop_dist)

    # ---------- Control Loop ----------
    def control_loop(self):
        cmd = Twist()

        # 1) 안전: 정면 근접 장애물 있으면 정지/회피
        if self.front_obstacle(self.last_scan):
            # 간단 회피: 최근 각도를 반대로 살짝 돌며 멈춤
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6 if (self.last_angle_deg is None or self.last_angle_deg >= 0.0) else -0.6
            self.pub_state.publish(String(data='obstacle_avoid'))
            self.pub_cmd.publish(cmd)
            return

        # 2) 타깃(사람) 각도가 없으면: 검색 모드(제자리 회전)
        if self.last_angle_deg is None:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # 탐색 회전
            self.pub_state.publish(String(data='searching'))
            self.pub_cmd.publish(cmd)
            return

        # 3) 라이다로 해당 각도 주변 거리 추정
        dist = self.lidar_distance_at_angle(self.last_scan, self.last_angle_deg, self.angle_window_deg)
        if dist is not None:
            self.last_dist = dist
            self.pub_dist.publish(Float32(data=self.last_dist))

        # 4) 각속도: 카메라 각도를 0으로 정렬
        # (ref=0, meas=angle_deg)
        ang_u = self.pid_ang(0.0, self.last_angle_deg)
        cmd.angular.z = clip(ang_u, -self.ang_max, self.ang_max)

        # 5) 선속도: 라이다 거리 → response_dist로
        if self.last_dist is not None:
            lin_u = self.pid_lin(self.response_dist, self.last_dist)
            # 너무 가까우면 후진은 약하게 제한
            if self.last_dist < self.response_dist:
                lin_u = max(lin_u, -0.2)
            cmd.linear.x = clip(lin_u, self.lin_min, self.lin_max)
            self.pub_state.publish(String(data='tracking'))
        else:
            # 거리 없으면 각만 맞추고 전진 보류
            cmd.linear.x = 0.0
            self.pub_state.publish(String(data='align_only'))

        self.pub_cmd.publish(cmd)


def main():
    rclpy.init()
    node = FusionFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

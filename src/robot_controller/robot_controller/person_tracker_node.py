#!/usr/bin/env python3
import rclpy, math
from rclpy.node import Node
from geometry_msgs.msg import Twist, PolygonStamped
from sensor_msgs.msg import LaserScan

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

class PersonTrackerNode(Node):
    """
    AI가 탐지한 사람의 좌표와 라이다 거리를 이용해 사람을 추적하는 노드
    - 정면 인덱스 계산: angle_min/angle_increment 기반(0rad 또는 오프셋)
    - 거리 대표값: 메디안(튀는 값에 강인)
    - 제어: 데드밴드 + 속도 제한 + 후진 허용 옵션
    - 바운딩박스 중심: 모든 꼭짓점 평균(점 순서 무관)
    """
    def __init__(self):
        super().__init__('person_tracker_node')

        # --- 파라미터 선언 ---
        self.declare_parameter('target_distance', 1.5)          # m
        self.declare_parameter('p_gain_angular', 0.005)          # rad/s per pixel
        self.declare_parameter('p_gain_linear', 0.4)             # m/s per m
        self.declare_parameter('camera_width', 320.0)            # px
        self.declare_parameter('lidar_angle_range', 10.0)        # deg (정면 ±lidar_angle_range/2만 사용)
        self.declare_parameter('front_yaw_offset_deg', 0.0)      # deg (라이다가 전방에서 몇 도 돌아갔는지)
        self.declare_parameter('px_deadband', 8.0)               # px (화면 중심 오차 허용)
        self.declare_parameter('d_deadband', 0.10)               # m  (거리 오차 허용)
        self.declare_parameter('w_max', 1.2)                     # rad/s (yaw 최대)
        self.declare_parameter('v_max', 0.40)                    # m/s   (전후 최대)
        self.declare_parameter('allow_backward', True)           # 후진 허용 여부

        # --- 발행자 ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- 구독자 ---
        self.detection_sub = self.create_subscription(
            PolygonStamped, '/person_detector/detection', self.detection_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # 최신 데이터
        self.latest_detection = None
        self.latest_scan = None

        # 10Hz 제어 루프
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f"'{self.get_name()}' 노드가 시작되었다.")

    # -------------------- 콜백 --------------------
    def detection_callback(self, msg: PolygonStamped):
        self.latest_detection = msg

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    # -------------------- 메인 제어 루프 --------------------
    def control_loop(self):
        if self.latest_detection is None or self.latest_scan is None:
            self.stop_robot()
            return

        twist = Twist()

        # ---------- 1) 회전 제어 ----------
        cam_w = self.get_parameter('camera_width').get_parameter_value().double_value
        screen_cx = cam_w * 0.5

        pts = self.latest_detection.polygon.points
        if not pts:
            self.stop_robot()
            return

        # 점 순서 가정하지 않고 평균으로 중심 X 계산
        person_cx = sum(p.x for p in pts) / len(pts)

        ang_err_px = (screen_cx - person_cx)  # +이면 좌로 돌 필요(카메라 좌표계 기준)
        p_gain_ang = self.get_parameter('p_gain_angular').get_parameter_value().double_value
        px_deadband = self.get_parameter('px_deadband').get_parameter_value().double_value
        w_max = self.get_parameter('w_max').get_parameter_value().double_value

        if abs(ang_err_px) <= px_deadband:
            twist.angular.z = 0.0
        else:
            twist.angular.z = clamp(p_gain_ang * ang_err_px, -w_max, w_max)

        # ---------- 2) 거리 제어 ----------
        dist = self.get_distance_from_scan()
        if dist is None:
            # 전방 유효 거리 없으면 멈춤
            self.stop_robot()
            return

        target = self.get_parameter('target_distance').get_parameter_value().double_value
        lin_err = dist - target  # +: 멀다→전진, -: 가깝다→후진
        p_gain_lin = self.get_parameter('p_gain_linear').get_parameter_value().double_value
        d_deadband = self.get_parameter('d_deadband').get_parameter_value().double_value
        v_max = self.get_parameter('v_max').get_parameter_value().double_value
        allow_backward = self.get_parameter('allow_backward').get_parameter_value().bool_value

        if abs(lin_err) <= d_deadband:
            twist.linear.x = 0.0
        else:
            v = clamp(p_gain_lin * lin_err, -v_max, v_max)
            if not allow_backward and v < 0.0:
                v = 0.0
            twist.linear.x = v

        # ---------- 3) 발행 ----------
        self.cmd_vel_pub.publish(twist)

    # -------------------- 라이다 전방 거리 산출 --------------------
    def get_distance_from_scan(self):
        """
        정면(0rad, 또는 front_yaw_offset_deg 반영) 주변 각도 범위에서 유효 거리들의 메디안을 반환한다.
        - angle_min/angle_increment 기반으로 인덱스를 계산한다.
        - 유효 범위(range_min~range_max) + finite 값만 사용한다.
        """
        scan = self.latest_scan
        if scan is None or not scan.ranges:
            return None

        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        n = len(scan.ranges)

        # 정면 기준 각도(라이다 물리 장착 오프셋 보정)
        front_yaw_deg = self.get_parameter('front_yaw_offset_deg').get_parameter_value().double_value
        front_yaw = math.radians(front_yaw_deg)

        # 0rad(또는 오프셋) → 인덱스
        front_idx = int(round((front_yaw - angle_min) / angle_inc))
        front_idx = max(0, min(n - 1, front_idx))

        # 전방 섹터 폭
        half_deg = self.get_parameter('lidar_angle_range').get_parameter_value().double_value / 2.0
        half_rad = math.radians(max(0.1, half_deg))  # 최소 0.1도
        idx_half = max(1, int(round(half_rad / angle_inc)))

        start = max(0, front_idx - idx_half)
        end = min(n, front_idx + idx_half + 1)

        seg = scan.ranges[start:end]

        # 유효값 필터링
        valid = [
            r for r in seg
            if (scan.range_min <= r <= scan.range_max) and (not math.isinf(r)) and (not math.isnan(r))
        ]
        if not valid:
            return None

        # 메디안
        valid.sort()
        m = len(valid) // 2
        return valid[m] if (len(valid) % 2 == 1) else 0.5 * (valid[m - 1] + valid[m])

    # -------------------- 정지 --------------------
    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())
        # 다음 루프에서 다시 정상 복귀하도록 탐지값은 유지해도 되나, 원 로직 유지 시 아래 라인 사용
        # self.latest_detection = None

def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

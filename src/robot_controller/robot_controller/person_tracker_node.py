#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, PolygonStamped
from sensor_msgs.msg import LaserScan

class PersonTrackerNode(Node):
    """
    AI가 탐지한 사람의 좌표와 라이다 거리를 이용해 사람을 추적하는 노드
    """
    def __init__(self):
        super().__init__('person_tracker_node')

        # --- 파라미터 선언 ---
        self.declare_parameter('target_distance', 1.5)      # 목표 추적 거리 (미터)
        self.declare_parameter('p_gain_angular', 0.005)     # 회전 제어를 위한 P 게인 값
        self.declare_parameter('p_gain_linear', 0.4)       # 거리 제어를 위한 P 게인 값
        self.declare_parameter('camera_width', 320.0)       # 카메라 영상의 너비 (픽셀)
        self.declare_parameter('lidar_angle_range', 10.0)   # 정면으로 인식할 라이다 각도 범위 (도)

        # --- 발행자(Publisher) 생성 ---
        # 로봇의 속도를 제어하기 위한 /cmd_vel 토픽 발행
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- 구독자(Subscriber) 생성 ---
        # AI 탐지 노드가 발행하는 바운딩 박스 좌표 구독
        self.detection_sub = self.create_subscription(
            PolygonStamped,
            '/person_detector/detection',
            self.detection_callback,
            10)
        # 라이다 센서 데이터 구독
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
            
        # 최신 데이터를 저장할 변수 초기화
        self.latest_detection = None
        self.latest_scan = None
        
        # 0.1초마다 제어 루프를 실행하는 타이머 생성 (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f"'{self.get_name()}' 노드가 시작되었습니다.")

    def detection_callback(self, msg):
        """AI 탐지 결과를 받으면 최신 정보로 업데이트"""
        self.latest_detection = msg

    def scan_callback(self, msg):
        """라이다 스캔 데이터를 받으면 최신 정보로 업데이트"""
        self.latest_scan = msg

    def control_loop(self):
        """주기적으로 실행되는 메인 제어 로직"""
        # AI가 사람을 탐지하지 못했거나, 라이다 데이터가 없으면 정지
        if self.latest_detection is None or self.latest_scan is None:
            self.stop_robot()
            return

        twist_msg = Twist()
        
        # --- 1. 회전 제어 로직 ---
        camera_width = self.get_parameter('camera_width').get_parameter_value().double_value
        screen_center_x = camera_width / 2.0
        
        # 바운딩 박스의 중심 X좌표 계산
        p1 = self.latest_detection.polygon.points[0]
        p2 = self.latest_detection.polygon.points[2]
        person_center_x = (p1.x + p2.x) / 2.0
        
        # 화면 중앙과 사람 중심의 오차 계산
        angular_error = screen_center_x - person_center_x
        
        p_gain_angular = self.get_parameter('p_gain_angular').get_parameter_value().double_value
        twist_msg.angular.z = p_gain_angular * angular_error
        
        # --- 2. 거리 제어 로직 ---
        # 라이다 데이터의 정면 부분만 사용하여 현재 거리 계산
        current_distance = self.get_distance_from_scan()
        
        if current_distance is None:
            self.stop_robot()
            return

        target_distance = self.get_parameter('target_distance').get_parameter_value().double_value
        
        # 목표 거리와 현재 거리의 오차 계산
        linear_error = current_distance - target_distance
        
        p_gain_linear = self.get_parameter('p_gain_linear').get_parameter_value().double_value
        twist_msg.linear.x = p_gain_linear * linear_error

        # 계산된 속도 명령을 로봇에게 발행
        self.cmd_vel_pub.publish(twist_msg)

    def get_distance_from_scan(self):
        """라이다 데이터의 정면 부분에서 유효한 최소 거리를 계산"""
        angle_range_deg = self.get_parameter('lidar_angle_range').get_parameter_value().double_value
        angle_increment = self.latest_scan.angle_increment
        
        # 정면으로 간주할 각도 범위에 해당하는 인덱스 계산
        center_index = len(self.latest_scan.ranges) // 2
        index_range = int(math.radians(angle_range_deg / 2.0) / angle_increment)
        
        # 정면 범위의 거리 값들 추출
        front_ranges = self.latest_scan.ranges[center_index - index_range : center_index + index_range]
        
        # 유효한(inf, nan이 아닌) 거리 값만 필터링
        valid_ranges = [r for r in front_ranges if not (math.isinf(r) or math.isnan(r))]
        
        if not valid_ranges:
            return None
        
        # 가장 가까운 거리를 현재 거리로 사용 (가장 보수적인 선택)
        return min(valid_ranges)

    def stop_robot(self):
        """로봇을 정지시키는 메시지 발행"""
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        # 탐지 실패 시, 다음 루프에서 다시 탐지할 수 있도록 최신 탐지 결과 초기화
        self.latest_detection = None

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
# obstacle_follower.py
import math, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

def clip(v, lo, hi): return lo if v < lo else hi if v > hi else v

class PID:
    def __init__(self, kp, ki, kd, out_min=None, out_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0; self.prev = None
        self.out_min = out_min; self.out_max = out_max
    def __call__(self, ref, meas):
        e = ref - meas
        self.i += e
        d = 0.0 if self.prev is None else (e - self.prev)
        self.prev = e
        u = self.kp*e + self.ki*self.i + self.kd*d
        if self.out_min is not None: u = max(self.out_min, u)
        if self.out_max is not None: u = min(self.out_max, u)
        return u

class ObstacleFollower(Node):
    def __init__(self):
        super().__init__('obstacle_follower')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.sub = self.create_subscription(LaserScan, '/scan', self.on_scan, qos)
        self.pub = self.create_publisher(Twist, '/cmd_vel', qos)

        # params
        self.declare_parameter('front_cone_deg', 10.0)      # 추종 후보 각도 범위(±)
        self.declare_parameter('standoff', 0.4)             # 유지 거리(m)
        self.declare_parameter('min_valid', 0.10)           # 너무 가까운 잡신호 컷(m)
        self.declare_parameter('max_valid', 6.0)            # 최대 유효 거리
        self.declare_parameter('lin_max', 0.5)
        self.declare_parameter('ang_max', 1.5)
        self.declare_parameter('cluster_dr', 0.10)          # 연속빔 거리 차 임계
        self.declare_parameter('cluster_dang', 2.0)         # 연속빔 각도 차 임계(°)

        # PID
        self.pid_lin = PID(1.2, 0.0, 0.6, out_min=-0.3, out_max=0.5)
        self.pid_ang = PID(0.03, 0.0, 0.02, out_min=-1.5, out_max=1.5)

        self.get_logger().info('ObstacleFollower started')

    def on_scan(self, scan: LaserScan):
        n = len(scan.ranges)
        if n == 0: return
        idx = np.arange(n, dtype=np.float32)
        ang = (scan.angle_min + scan.angle_increment*idx) * 180.0/math.pi
        ang = (ang + 180.0) % 360.0 - 180.0

        r = np.asarray(scan.ranges, dtype=np.float32)
        r = np.where(np.isfinite(r), r, np.inf)

        # 유효 & 정면 콘
        min_valid = max(self.get_parameter('min_valid').value, scan.range_min)
        max_valid = min(self.get_parameter('max_valid').value, scan.range_max)
        front = self.get_parameter('front_cone_deg').value
        mask = (np.abs(ang) <= front) & (r >= min_valid) & (r <= max_valid)
        if not np.any(mask):
            self.pub.publish(Twist()); return

        # 연속 빔 클러스터링
        cluster_dr = self.get_parameter('cluster_dr').value
        cluster_dang = self.get_parameter('cluster_dang').value
        inds = np.where(mask)[0]
        clusters = []
        cur = [inds[0]]
        for a, b in zip(inds[:-1], inds[1:]):
            if abs(float(r[b] - r[a])) <= cluster_dr and abs(float(ang[b] - ang[a])) <= cluster_dang:
                cur.append(b)
            else:
                if len(cur) >= 3: clusters.append(cur)
                cur = [b]
        if len(cur) >= 3: clusters.append(cur)
        if not clusters:
            self.pub.publish(Twist()); return

        # 각 클러스터의 평균 거리/각 → 가장 가까운 클러스터 선택
        stats = []
        for c in clusters:
            rc = float(np.median(r[c]))
            ac = float(np.mean(ang[c]))
            stats.append((rc, ac, c))
        stats.sort(key=lambda x: x[0])  # 거리 오름차순
        dist, angle_deg, _ = stats[0]

        # 제어
        cmd = Twist()
        cmd.linear.x  = self.pid_lin(self.get_parameter('standoff').value, dist)
        cmd.angular.z = self.pid_ang(0.0, angle_deg)

        # 속도 제한 및 아주 근접 시 정지
        cmd.linear.x  = clip(cmd.linear.x, -self.get_parameter('lin_max').value,  self.get_parameter('lin_max').value)
        cmd.angular.z = clip(cmd.angular.z, -self.get_parameter('ang_max').value, self.get_parameter('ang_max').value)
        if dist < (self.get_parameter('standoff').value * 0.6):
            cmd.linear.x = min(cmd.linear.x, 0.0)  # 너무 가까우면 전진 금지

        self.pub.publish(cmd)

def main():
    rclpy.init()
    rclpy.spin(ObstacleFollower())
    rclpy.shutdown()

if __name__ == '__main__':
    main()

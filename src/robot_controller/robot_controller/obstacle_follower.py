# obstacle_follower.py
import math, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32

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
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub = self.create_subscription(LaserScan, '/scan', self.on_scan, qos_sensor)
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )                
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', qos_cmd)

        # 상태/디버그 토픽
        self.pub_state = self.create_publisher(String, 'obstacle_follower/state', qos)
        self.pub_angle = self.create_publisher(Float32, 'obstacle_follower/angle_deg', qos)
        self.pub_dist  = self.create_publisher(Float32, 'obstacle_follower/dist_m', qos)

        # params
        self.declare_parameter('front_cone_deg', 45.0)   # 추종 후보 각도 범위(±)
        self.declare_parameter('standoff', 0.8)          # 유지 거리(m)
        self.declare_parameter('min_valid', 0.10)        # 너무 가까운 잡신호 컷(m)
        self.declare_parameter('max_valid', 6.0)         # 최대 유효 거리
        self.declare_parameter('lin_max', 0.5)
        self.declare_parameter('ang_max', 1.5)
        self.declare_parameter('cluster_dr', 0.10)       # 연속빔 거리 차 임계
        self.declare_parameter('cluster_dang', 2.0)      # 연속빔 각도 차 임계(°)
        self.declare_parameter('prefer', 'nearest')      # 'nearest' or 'largest'
        self.declare_parameter('search_when_none', True) # 후보 없으면 탐색 회전
        self.declare_parameter('search_omega', 0.5)      # 탐색 회전 속도(rad/s)
        self.declare_parameter('min_cluster_points', 3)  # 최소 포인트 수

        # PID
        self.pid_lin = PID(1.2, 0.0, 0.6, out_min=-0.3, out_max=0.5)
        self.pid_ang = PID(0.03, 0.0, 0.02, out_min=-1.5, out_max=1.5)

        self.get_logger().info('ObstacleFollower started')

    def on_scan(self, scan: LaserScan):
        cmd = Twist()
        n = len(scan.ranges)
        if n == 0:
            self.pub_state.publish(String(data='no_scan'))
            self.pub_cmd.publish(cmd); return

        idx = np.arange(n, dtype=np.float32)
        ang = (scan.angle_min + scan.angle_increment*idx) * 180.0/math.pi
        ang = (ang + 180.0) % 360.0 - 180.0

        r = np.asarray(scan.ranges, dtype=np.float32)
        # 유효 범위 필터 (스캐너 스펙 반영)
        min_valid = max(self.get_parameter('min_valid').value, scan.range_min)
        max_valid = min(self.get_parameter('max_valid').value, scan.range_max)
        valid = np.isfinite(r) & (r >= min_valid) & (r <= max_valid)

        # 정면 콘
        front = float(self.get_parameter('front_cone_deg').value)
        mask = valid & (np.abs(ang) <= front)
        if not np.any(mask):
            # 후보 없음: 탐색 모드 또는 정지
            if bool(self.get_parameter('search_when_none').value):
                cmd.angular.z = float(self.get_parameter('search_omega').value)
                self.pub_state.publish(String(data='searching'))
            else:
                self.pub_state.publish(String(data='no_points'))
            self.pub_cmd.publish(cmd); return

        # 연속 빔 클러스터링
        cluster_dr = float(self.get_parameter('cluster_dr').value)
        cluster_dang = float(self.get_parameter('cluster_dang').value)
        min_pts = int(self.get_parameter('min_cluster_points').value)

        inds = np.where(mask)[0]
        clusters = []
        cur = [inds[0]]
        for a, b in zip(inds[:-1], inds[1:]):
            if abs(float(r[b] - r[a])) <= cluster_dr and abs(float(ang[b] - ang[a])) <= cluster_dang:
                cur.append(b)
            else:
                if len(cur) >= min_pts: clusters.append(cur)
                cur = [b]
        if len(cur) >= min_pts: clusters.append(cur)

        if not clusters:
            idxs = np.where(mask)[0]
            if idxs.size > 0:
                i_min = idxs[int(np.argmin(r[idxs]))]
                dist = float(r[i_min])
                angle_deg = float(ang[i_min])

                # 너무 가까우면 전진 금지
                standoff = float(self.get_parameter('standoff').value)
                cmd.linear.x  = self.pid_lin(dist, standoff)
                cmd.angular.z = self.pid_ang(0.0, angle_deg)

                lin_max = float(self.get_parameter('lin_max').value)
                ang_max = float(self.get_parameter('ang_max').value)
                cmd.linear.x  = clip(cmd.linear.x, -lin_max, lin_max)
                cmd.angular.z = clip(cmd.angular.z, -ang_max, ang_max)
                if dist < (standoff * 0.6):
                    cmd.linear.x = min(cmd.linear.x, 0.0)
                    self.pub_state.publish(String(data='too_close(fallback_point)'))
                else:
                    self.pub_state.publish(String(data='tracking(fallback_point)'))

                self.pub_angle.publish(Float32(data=angle_deg))
                self.pub_dist.publish(Float32(data=dist))
                self.pub_cmd.publish(cmd)
                return
            if bool(self.get_parameter('search_when_none').value):
                cmd.angular.z = float(self.get_parameter('search_omega').value)
                self.pub_state.publish(String(data='searching'))
            else:
                self.pub_state.publish(String(data='no_clusters'))
            self.pub_cmd.publish(cmd); return

        # 클러스터 통계
        stats = []
        for c in clusters:
            rc = float(np.median(r[c]))      # 거리: 중앙값
            ac = float(np.mean(ang[c]))      # 각도: 평균
            stats.append((rc, ac, len(c), c))

        prefer = str(self.get_parameter('prefer').value).lower()
        if prefer == 'largest':
            # 포인트 수 가장 많은 클러스터 (사람/카트 같은 큰 대상 우선)
            stats.sort(key=lambda x: x[2], reverse=True)
        else:
            # 가장 가까운 클러스터
            stats.sort(key=lambda x: x[0])

        dist, angle_deg, pts, _ = stats[0]

        # 디버그 토픽
        self.pub_angle.publish(Float32(data=angle_deg))
        self.pub_dist.publish(Float32(data=dist))

        # 제어
        standoff = float(self.get_parameter('standoff').value)
        cmd.linear.x  = self.pid_lin(dist, standoff)
        cmd.angular.z = self.pid_ang(0.0, angle_deg)

        # 속도 제한
        lin_max = float(self.get_parameter('lin_max').value)
        ang_max = float(self.get_parameter('ang_max').value)
        cmd.linear.x  = clip(cmd.linear.x, -lin_max, lin_max)
        cmd.angular.z = clip(cmd.angular.z, -ang_max, ang_max)

        # 히스테리시스/안전: 너무 가까우면 전진 금지
        if dist < (standoff * 0.6):
            cmd.linear.x = min(cmd.linear.x, 0.0)
            self.pub_state.publish(String(data='too_close'))
        else:
            self.pub_state.publish(String(data='tracking'))

        self.pub_cmd.publish(cmd)

def main():
    rclpy.init()
    rclpy.spin(ObstacleFollower())
    rclpy.shutdown()

if __name__ == '__main__':
    main()

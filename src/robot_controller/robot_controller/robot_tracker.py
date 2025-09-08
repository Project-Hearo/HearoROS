# ros libs
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

import math
import numpy as np
from yahboomcar_laser.common import SinglePID

RAD2DEG = 180.0 / math.pi

class LaserTracker(Node):
    def __init__(self, name: str):
        super().__init__(name)

        
        self.declare_parameter("priorityAngle", 10.0)   
        self.declare_parameter("LaserAngle", 45.0)      
        self.declare_parameter("ResponseDist", 0.55)    
        self.declare_parameter("Switch", False)
        self.declare_parameter("offset", 0.5)           

        self.priorityAngle = float(self.get_parameter("priorityAngle").value)
        self.LaserAngle    = float(self.get_parameter("LaserAngle").value)
        self.ResponseDist  = float(self.get_parameter("ResponseDist").value)
        self.Switch        = bool(self.get_parameter("Switch").value)
        self.offset        = float(self.get_parameter("offset").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.sub_laser    = self.create_subscription(LaserScan, "/scan", self.on_scan, qos)
        self.sub_joy      = self.create_subscription(Bool, "/JoyState", self.on_joy, qos)
        self.pub_cmd_vel  = self.create_publisher(Twist, "/cmd_vel", qos)

        
        self.joy_active = False
        self.moving     = False

        self.lin_pid = SinglePID(2.0, 0.0, 2.0)
        self.ang_pid = SinglePID(3.0, 0.0, 5.0)

        self.create_timer(0.01, self._poll_params)

        self.get_logger().info("import done / laser tracker started")

    def _poll_params(self):
        self.Switch        = bool(self.get_parameter("Switch").value)
        self.priorityAngle = float(self.get_parameter("priorityAngle").value)
        self.LaserAngle    = float(self.get_parameter("LaserAngle").value)
        self.ResponseDist  = float(self.get_parameter("ResponseDist").value)
        self.offset        = float(self.get_parameter("offset").value)

    def on_joy(self, msg: Bool):
        self.joy_active = bool(msg.data)

    def on_scan(self, scan: LaserScan):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return

        valid = np.isfinite(ranges) & (ranges > 0.0)
        if not np.any(valid):
            if self.moving:
                self.pub_cmd_vel.publish(Twist())
                self.moving = False
            return

        idx = np.arange(ranges.size, dtype=np.float32)
        angles = (scan.angle_min + scan.angle_increment * idx) * RAD2DEG
        angles = (angles + 180.0) % 360.0 - 180.0
        abs_ang = np.abs(angles)

        m_front = valid & (abs_ang < self.priorityAngle) & (ranges < (self.ResponseDist + self.offset))
        m_main  = valid & (abs_ang < self.LaserAngle)

        cand_idx = np.where(m_front)[0]
        if cand_idx.size == 0:
            cand_idx = np.where(m_main)[0]
        if cand_idx.size == 0:
            if self.moving:
                self.pub_cmd_vel.publish(Twist())
                self.moving = False
            return

        i_min   = cand_idx[np.argmin(ranges[cand_idx])]
        minDist = float(ranges[i_min])
        minAng  = float(angles[i_min])  

        if self.joy_active or self.Switch:
            if self.moving:
                self.pub_cmd_vel.publish(Twist())
                self.moving = False
            return

        self.moving = True
        cmd = Twist()

        if abs(minDist - self.ResponseDist) < 0.1:
            minDist = self.ResponseDist

        cmd.linear.x = -self.lin_pid.pid_compute(self.ResponseDist, minDist)

        ang_u = self.ang_pid.pid_compute(minAng / 48.0, 0.0)
        cmd.angular.z = 0.0 if abs(ang_u) < 0.1 else ang_u

        self.pub_cmd_vel.publish(cmd)

    def exit_pro(self):
        self.pub_cmd_vel.publish(Twist())

def main():
    rclpy.init()
    node = LaserTracker("laser_Tracker")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.exit_pro()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

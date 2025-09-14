import rclpy, time, math
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class RobotMapLocation(Node):
    def __init__(self):
        super().__init__('robot_map_location')
        
        self.declare_parameter('pose_topic', '/amcl_pose')
        self.declare_parameter('pub_topic', '/robot/map/location')
        self.declare_parameter('rate_hz', 2.0)
        
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        pub_topic  = self.get_parameter('pub_topic').get_parameter_value().string_value
        rate_hz    = float(self.get_parameter('rate_hz').value)
        qos_loc_pub = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.pub = self.create_publisher(PointStamped, pub_topic, qos_loc_pub)
        self.sub = self.create_subscription(PoseWithCovarianceStamped, pose_topic, self._pose_cb, qos_profile_sensor_data)
        
        self.period = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._latest_pose = None
        
        if self.period > 0:
            self.timer = self.create_timer(self.period, self._tick)
        
    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        self._latest_pose = msg
        
    def _tick(self):
        if self._latest_pose is None:
            return
        pose_msg = self._latest_pose
        p = pose_msg.pose.pose.position
        
        out = PointStamped()
        out.header = pose_msg.header
        out.point.x = p.x
        out.point.y = p.y
        out.point.z = p.z
        
        self.pub.publish(out)
        
def main():
    rclpy.init()
    node = RobotMapLocation()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()
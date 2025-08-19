import copy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class CameraInfo(Node):
    def __init__(self):
        super().__init__('camera_info')
        
        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_topic', '/image_raw_refine')
        self.declare_parameter('rate_hz', 10.0)
        
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        rate_hz = float(self.get_parameter('rate_hz').get_parameter_value().double_value)

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.sub = self.create_subscription(Image, input_topic, self.callback, sensor_qos)
        self.pub = self.create_publisher(Image, output_topic, sensor_qos)

        self._period = 1.0 / max(0.1, rate_hz)
        self._next_pub = 0.0
        
    def _now_sec(self) -> float:
        t = self.get_clock().now().to_msg()
        return t.sec + t.nanosec * 1e-9
    
    def callback(self, msg:Image):
        
        
        now = self._now_sec()
        
        if now < self._next_pub:
            return
        out = Image()
        out.header = copy.copy(msg.header)
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id

        out.height       = msg.height
        out.width        = msg.width
        out.encoding     = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step         = msg.step
        out.data         = msg.data  
        
        self.pub.publish(out)
        self._next_pub = now + self._period
def main(args=None):
    rclpy.init(args=args)
    node = CameraInfo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
import rclpy
from geometry_msgs.msg import PointStamped
from .base import TelemetryHandler
from .registry import register_telemetry
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

@register_telemetry
class LocationTelemetry(TelemetryHandler):
    name = "location"
    
    def __init__(self, node, *, rate_hz: float = 2.0):
        super().__init__(node, rate_hz=rate_hz)
        self.sub = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.running = False
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.sub = self.node.create_subscription(PointStamped, '/robot/location', self.callback, 10)
        
    def stop(self):
        if not self.running: return
        try: 
            if self.sub: 
                self.node.destroy_subscription(self.sub)
        finally:
            self.sub = None
            self.running = False
            
    def callback(self, msg: PointStamped):
        if not self.tick(): return
        try: 
            tf = self.tf_buffer.lookup_transform(
                'map', msg.header.frame_id, rclpy.time.Time()
            )
            point_in_map = do_transform_point(msg, tf)
            self.node._publish_telemetry(
                "location", {
                "x":point_in_map.point.x, 
                "y": point_in_map.point.y, 
                "z":point_in_map.point.z,
                "frame_id":"map",
            })
        except Exception as e:
            self.node.get_logger().warn(f"Tranform failed: {e}")
        
    
import rclpy
from geometry_msgs.msg import PointStamped
from .base import TelemetryHandler
from .registry import register_telemetry
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from rclpy.duration import Duration
@register_telemetry
class LocationTelemetry(TelemetryHandler):
    name = "location"
    
    def __init__(self, node, *, rate_hz: float = 2.0):
        super().__init__(node, rate_hz=rate_hz)
        self.sub = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=False)
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
            
    def _normalize_source_frame(self, frame_id: str) -> str:
        if not frame_id:
            return 'base_link'
        aliases = {
            'odom_frame': 'odom',
            'odom_combined': 'odom',
            'map_frame': 'map',
            'basefootprint': 'base_footprint',
        }
        f = frame_id.strip()
        return aliases.get(f,f)
            
    def callback(self, msg: PointStamped):
        if not self.tick(): return
        try: 
            source = self._normalize_source_frame(msg.header.frame_id)
            frame_out = None
            point = None
            
            if self.tf_buffer.can_transform('map', source, Time(), timeout=Duration(seconds=0.2)):
                tf = self.tf_buffer.lookup_transform('map', source, Time(), timeout=Duration(seconds=0.2))
                pt = do_transform_point(msg, tf)
                frame_out = 'map'
                point = pt.point

            elif self.tf_buffer.can_transform('odom', source, Time(), timeout=Duration(seconds=0.2)):
                tf = self.tf_buffer.lookup_transform('odom', source, Time(), timeout=Duration(seconds=0.2))
                pt = do_transform_point(msg, tf)
                frame_out = 'odom'
                point = pt.point

            else:
                self.node.get_logger().warn(f"TF not ready: map/odom <- {source}, publish raw")
                frame_out = source
                point = msg.point
                
            self.node._publish_telemetry(
                "location",
                {"x": point.x, "y": point.y, "z": point.z, "frame_id": frame_out},
                qos=1, retain=False
            )
        except Exception as e:
            self.node.get_logger().warn(f"Transform failed: {e}")
        
    
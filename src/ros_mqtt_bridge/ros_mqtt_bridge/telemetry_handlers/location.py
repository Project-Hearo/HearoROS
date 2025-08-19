import json, time
from geometry_msgs.msg import PointStamped
from .base import TelemetryHandler
from .registry import register_telemetry

from tf2_ros import Buffer, TransformListener

@register_telemetry
class LocationTelemetry(TelemetryHandler):
    name = "location"
    
    def __init__(self, node, *, rate_hz: float = 2.0):
        super().__init__(node, rate_hz=rate_hz)
        self.sub = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
    
    def start(self):
        if self._running: return
        self._running = True
        self.sub = self.node.create_subscription(PointStamped, '/robot/location', self.callback, 10)
        
    def stop(self):
        if not self._running: return
        try: 
            if self.sub: self.node.destroy_subscription(self.sub)
        finally:
            self.sub = None
            self._running = False
    def callback(self, msg: PointStamped):
        if not self._tick(): return
        try: 
            point_in_map = self.tf_buffer.transform(msg, 'map')
            self.node._publish_telemetry(
                "location", {
                "x":point_in_map.point.x, 
                "y": point_in_map.point.y, 
                "z":point_in_map.point.z,
                "frame_id":"map",
            })
        except Exception as e:
            self.get_logger().warn(f"Tranform failed: {e}")
        
    
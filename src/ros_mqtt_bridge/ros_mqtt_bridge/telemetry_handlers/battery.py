from .base import TelemetryHandler
from .registry import register_telemetry
from std_msgs.msg import UInt16
class BatteryTelemetry(TelemetryHandler):
    name = "battery"
    
    def __init__(self, node, *, rate_hz: float = 2.0):
        super().__init__(node, rate_hz=rate_hz)
        self.sub = None
        self.running = False
        
    def start(self):
        if self.running:
            return
        self.sub = self.node.create_subscription(
            UInt16,
            '/battery',
            self.callback,
            10
        )
        self.running = True
    def stop(self):
        if not self.running: return
        try:
            if self.sub:
                self.node.destroy_subscription(self.sub)    
        finally:
            self.sub = None
            self.running = False
            
    def callback(self, msg:UInt16):
        if not self.tick(): return
        try:
            payload = {
                "raw": int(msg.data)
            }
            if 0 <= msg.data <= 100:
                payload["percent"] = int(msg.data)
                
            self.node._publish_telemetry(
                "battery",
                payload,
                qos=1,
                retain=False
            )
        except Exception as e:
            self.node.get_logger().warn(f"battery failed: {e}")
        
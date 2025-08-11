import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt16
from sensor_msgs.msg import Imu
from my_robot_interfaces.srv import RobotStatus  
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time

import json
import threading

QoS_SENSOR_LAST = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
class BatteryImuServiceNode(Node):
    def __init__(self):
        super().__init__('status_service_node')

        self._lock = threading.Lock()
        self.battery_value = None
        self.imu_data = None
        self._imu_json_cache = "{}"
        self.last_battery_time: Time | None = None
        self.last_imu_time: Time | None = None
        self.stale_sec = 2.0
        
        self.create_subscription(UInt16, '/battery', self.battery_callback, QoS_SENSOR_LAST)
        self.create_subscription(Imu, '/imu', self.imu_callback, QoS_SENSOR_LAST)

        self.srv = self.create_service(RobotStatus, 'status/get', self.handle_service_request)
    def _is_stale(self, t: Time | None) -> bool:
        if t is None: return True
        now = self.get_clock().now().to_msg()
        dt = (now.sec + now.nanosec*1e-9) - (t.sec + t.nanosec*1e-9)
        return dt > self.stale_sec
    
    def battery_callback(self, msg):
        with self._lock:
            self.battery_value = msg.data
            self.last_battery_time = self.get_clock().now().to_msg()
        self.get_logger().info(f'🔋 배터리 수신: {msg.data}')
        

    def imu_callback(self, msg):
        with self._lock:       
            self.imu_data = {
                'orientation': {
                    'x': msg.orientation.x,
                    'y': msg.orientation.y,
                    'z': msg.orientation.z,
                    'w': msg.orientation.w
                },
                'angular_velocity': {
                    'x': msg.angular_velocity.x,
                    'y': msg.angular_velocity.y,
                    'z': msg.angular_velocity.z
                },
                'linear_acceleration': {
                    'x': msg.linear_acceleration.x,
                    'y': msg.linear_acceleration.y,
                    'z': msg.linear_acceleration.z
                }
            }
            self._imu_json_cache = json.dumps(self.imu_data, separators=(',',':'))
            self.last_imu_time = self.get_clock().now().to_msg()
        self.get_logger().debug("Imu 수신")

    def handle_service_request(self, request, response):
        sensor_type = request.sensor_type.lower()
        self.get_logger().info(f'서비스 요청 수신: sensor_type = {sensor_type}')

        with self._lock:
            want_imu = sensor_type in ('imu', 'both')
            want_batt = sensor_type in ('battery', 'both')
            
            response.imu_json = self._imu_json_cache if (want_imu and not self._is_stale(self.last_imu_time)) else "{}"
            response.battery = int(self.battery_value) if (want_batt and not self._is_stale(self.last_battery_time)) else 0
        
        if sensor_type not in ('imu', 'battery', 'both'):
            self.get_logger().warn(f'잘못된 sensor_type 요초이 {sensor_type}')
        
        self.get_logger().info(
            f'응답 전송 완료: battery={response.battery},'
            f'imu_json_len={len(response.imu_json)}'
        )
        return response
    
def main(args=None):
    rclpy.init(args=args)
    node = BatteryImuServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt16
from sensor_msgs.msg import Imu
from my_robot_interfaces.srv import RobotStatus  

import json
import threading

class BatteryImuServiceNode(Node):
    def __init__(self):
        super().__init__('battery_imu_service_node')

        self._lock = threading.Lock()
        self.battery_value = None
        self.imu_data = None
        self._imu_json_cache = "{}"

        # 센서 토픽 구독
        self.create_subscription(UInt16, '/battery', self.battery_callback, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile_sensor_data)

        # 서비스 서버 생성
        self.srv = self.create_service(RobotStatus, 'get_robot_sensor', self.handle_service_request)

    def battery_callback(self, msg):
        with self._lock:
            self.battery_value = msg.data
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
            self._imu_json_cache = json.dumps(self.imu_data)
        self.get_logger().debug("Imu 수신")

    def handle_service_request(self, request, response):
        sensor_type = request.sensor_type.lower()
        self.get_logger().info(f'서비스 요청 수신: sensor_type = {sensor_type}')

        with self._lock:
            if sensor_type in ('imu', 'both') and self.imu_data:
                response.imu_json = self._imu_json_cache
            else:
                response.imu_json = "{}"
            
            if sensor_type in ('battery', 'both') and self.battery_value is not None:
                response.battery = self.battery_value
            else:
                response.battery = 0
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

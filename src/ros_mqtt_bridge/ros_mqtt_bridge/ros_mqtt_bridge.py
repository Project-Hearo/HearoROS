import rclpy
from rclpy.node import Node
import paho.mqtt.client as mqtt
import json
import json
from pathlib import Path
from threading import Thread
import time

config_path = Path(__file__).parent / 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
    
class RosMqttBridge(Node):
    def __init__(self):
        super().__init__('ros_mqtt_bridge')
        
        self.broker_host = config['broker_host']
        self.broker_port = config['broker_port']
        self.robot_id = config['robot_id']
        self.connected = False
        self.topic_online = f'robot/{self.robot_id}/status/online'
        
        self.mqtt_client = mqtt.Client(client_id=self.robot_id, clean_session=False)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        will_payload = json.dumps({"online": False})
        # 프로세스 강제 종료 또는 전원 꺼짐 등 비정상종료시 자동 발행된다.
        self.mqtt_client.will_set(
            self.topic_online,
            payload=will_payload,
            qos=1,
            retain=True            
        )
        
        self.mqtt_client.connect(
            self.broker_host,
            self.broker_port,
            keepalive=45
        )
        
        self.mqtt_client.loop_start()
        self.get_logger().info("ros_mqtt_bridge 초기화 및 MQTT 연결 시도 중...")
        
        Thread(target=self.post_connect_task, daemon=True).start()
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.get_logger().info("MQTT 브로커 연결 성공")
            payload = json.dumps({"online": True})
            result = client.publish(self.topic_online, payload=payload, qos=1, retain=True)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.connected = True
                self.get_logger().info(f"online:true 메시지 발행 완료 -> {self.topic_online}")
            else:
                self.get_logger().error(f"메시지 발행 실패 (result.rc={result.rc})")
        else:   
            self.get_logger().error(f"MQTT 연결 실패 rc={rc}")
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.get_logger().warn(f"MQTT 연결 끊김 (rc={rc})")
        
    def post_connect_task(self):
        while not self.connected:
            time.sleep(0.5)
        self.get_logger().info("연결 이후 작업 시작")
            
def main(args=None):
    rclpy.init(args=args)
    node = RosMqttBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass    
    finally:
        node.mqtt_client.loop_stop()
        node.mqtt_client.disconnect()
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()


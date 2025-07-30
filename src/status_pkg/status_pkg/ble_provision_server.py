# net_status_pkg/ble_provision_server.py
import json, subprocess
from bluezero import peripheral
import rclpy
from rclpy.node import Node
# 하나의 서비스 아래 4개의 캐릭터리스틱을 정의한다.
WIFI_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e'
SSID_CHAR_UUID    = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'
PASS_CHAR_UUID    = '6e400003-b5a3-f393-e0a9-e50e24dcca9e'
APPLY_CHAR_UUID   = '6e400004-b5a3-f393-e0a9-e50e24dcca9e'
STATUS_CHAR_UUID  = '6e400005-b5a3-f393-e0a9-e50e24dcca9e'

class BLEProvNode(Node):
    def __init__(self):
        super().__init__('ble_provision_server')
        self._ssid = ''
        self._pass = ''
        self.pub = self.create_publisher(String, '/wifi/provision_status', 10)

    def publish_status(self, d: dict):
        from std_msgs.msg import String
        msg = String()
        msg.data = json.dumps(d)
        self.pub.publish(msg)
        self.get_logger().info(f"[BLE] {msg.data}")

def main(args=None):
    #ROS 컨텍스트/노드 생성한다
    rclpy.init(args=args)
    node = BLEProvNode()

    #STATUS characteristic의 Read요청에 기본 상태를 돌려준다.
    def read_status():
        return json.dumps({"state": "idle"}).encode()

    def on_ssid_write(value, options):
        node._ssid = value.decode('utf-8').strip()
        node.get_logger().info(f"SSID set: {node._ssid}")

    def on_pass_write(value, options):
        node._pass = value.decode('utf-8')
        node.get_logger().info("Password received")

    #APPLT characteristic에 값이 쓰이면 실행된다.
    #apply_wifi()로 실제 접속을 시도하고, 결과를 STATUS characteristic과 ROS토픽으로 둘다 알린다.
    def on_apply_write(value, options):
        # value == b'\x01'이면 적용 시도
        node.get_logger().info("Apply requested")
        ok, out = apply_wifi(node._ssid, node._pass)
        status = {"state": "applied" if ok else "failed", "detail": out}
        status_char.set_value(json.dumps(status).encode())
        node.publish_status(status)

    def apply_wifi(ssid, passwd):
        try:
            if not ssid:
                return False, "empty ssid"
            # NetworkManager 사용 시:
            cmd = ["nmcli", "dev", "wifi", "connect", ssid, "password", passwd]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            ok = (cp.returncode == 0)
            return ok, (cp.stdout if ok else cp.stderr)
        except Exception as e:
            return False, str(e)

    # GATT 구성
    ssid_char = peripheral.Characteristic(SSID_CHAR_UUID,
                                          ['write'], ['encrypt-write'],
                                          write_callback=on_ssid_write)
    pass_char = peripheral.Characteristic(PASS_CHAR_UUID,
                                          ['write'], ['encrypt-write'],
                                          write_callback=on_pass_write)
    apply_char = peripheral.Characteristic(APPLY_CHAR_UUID,
                                           ['write'], [],
                                           write_callback=on_apply_write)
    status_char = peripheral.Characteristic(STATUS_CHAR_UUID,
                                            ['read', 'notify'], [],
                                            read_callback=lambda options: read_status())

    wifi_service = peripheral.Service(WIFI_SERVICE_UUID, True)
    wifi_service.add_characteristic(ssid_char)
    wifi_service.add_characteristic(pass_char)
    wifi_service.add_characteristic(apply_char)
    wifi_service.add_characteristic(status_char)

    # 광고 이름: HEARO-SETUP-XXXX
    p = peripheral.Peripheral(adapter_addr=None, local_name='HEARO-SETUP', services=[wifi_service])

    # 한 명만 연결되게(단순락)
    p.add_characteristic_status_callback(lambda *a, **kw: None)  # 필요 시 연결 상태 콜백 구현

    node.get_logger().info("BLE Provisioning server started")
    try:
        # bluezero는 내부 메인루프를 돈다(블로킹).
        # 간단히 별 프로세스로 띄우거나, 별 스레드에서 p.run() 돌리는 방식도 가능.
        p.run()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

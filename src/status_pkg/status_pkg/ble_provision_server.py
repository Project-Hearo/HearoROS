# net_status_pkg/ble_provision_server.py
import json, subprocess, os, hmac, hashlib, base64, socket, re   # ★추가
from bluezero import peripheral
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# 하나의 Wi-Fi Provisioning 서비스 아래 6개의 캐릭터리스틱을 둔다.

WIFI_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e'
SSID_CHAR_UUID    = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'
PASS_CHAR_UUID    = '6e400003-b5a3-f393-e0a9-e50e24dcca9e'
APPLY_CHAR_UUID   = '6e400004-b5a3-f393-e0a9-e50e24dcca9e'
STATUS_CHAR_UUID  = '6e400005-b5a3-f393-e0a9-e50e24dcca9e'
HELLO_CHAR_UUID = '6e400006-b5a3-f393-e0a9-e50e24dcca9e'
HELLO_RESP_UUID   = '6e400007-b5a3-f393-e0a9-e50e24dcca9e'  



# 앱과 디바이스가 공유하는 HMAC 비밀키. 개발 중엔 env/하드코드, 실제는 장치별 키 권장
SECRET_KEY = os.environ.get('HEARO_BLE_SECRET', 'dev-debug-secret').encode('utf-8')

#HMAC 바이트를 base64url로 안전하게 문자열화
def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode('ascii')

# 앱과 장치가 동일 규칙으로 계산해야 하는 서명
# 메시지 포멧 "{nonce}.{device_id}"
def hmac_sign(key: bytes, nonce: str, device_id: str) -> str:
    msg = f'{nonce}.{device_id}'.encode('uft-8')
    return b64u(hmac.new(key, msg, hashlib.sha256).digest())

# 앱 화면/로그에 보여줄 장치 식별자. 인증 서명에도 들어감
def get_device_id() -> str:
    try:
        with open('/proc/cpuinfo', 'r') as f:
            m = re.search(r'^Serial\s*:\s*([0-9a-fA-F]+)', f.read(), re.M)
            if m: return m.group(1)
    except Exception:
        pass
    return socket.gethostname()

# 인증 상태, 세션 nonce, 디바이스 ID를 보관한다.
# /wifi/provision_status 토픽으로 상태 JSON을 퍼블리시할 퍼블리셔를 준비한다.
class BLEProvNode(Node):
    def __init__(self):
        super().__init__('ble_provision_server')
        self._ssid = ''
        self._pass = ''
        self._authed = False
        self._device_id = get_device_id()
        self._nonce = b64u(os.urandom(12))
        self.pub = self.create_publisher(String, '/wifi/provision_status', 10)
        # BLE 처리 자체는 bluezeor가 담당하고,상태 메시지는 ROS 토픽으로 내보냄
        
    # STATUS 특성 업데이트와 같은 내용을 ROS 토픽으로도 내보낸다.
    def publish_status(self, d: dict):
        msg = String()
        msg.data = json.dumps(d)
        self.pub.publish(msg)
        self.get_logger().info(f"[BLE] {msg.data}")

def main(args=None):
    # 내부 콟백/헬퍼 정의
    rclpy.init(args=args)
    node = BLEProvNode()
    # STATUS특성 값을 갱신하고, ROS 토픽도 동시에 발행
    def set_status(state: str,detail: str = None):
        payload = {'state': state}
        if detail: payload['detail'] = detail
        status_char.set_value(json.dumps(payload).encode())
        node.publish_status(payload)
    
    # 앱이 STATUS를 Read하면 기본 상태는 idle.
    def read_status():
        return json.dumps({"state": "idle"}).encode()

    # 앱이 HELLO를 Read하면 우리만의 정보를 받는다.
    # 앱이 이걸로 HMAC 서명을 계산한다.
    def read_hello(_opts=None):
        hello = {
            "vendor": "HEARO",
            "model":  "PI5-BOT",
            "proto":  1,
            "device_id": node._device_id,
            "nonce": node._nonce
        }
        return json.dumps(hello).encode()
    
    # 앱이 계산한 서명과 장치가 계산한 서명을 비교한다 -> 같으면 인증이 완료된다.
    def on_hello_resp_write(value, options):
        try:
            client_sig = value.decode('utf-8').strip()
            expected = hmac_sign(SECRET_KEY, node._nonce, node._device_id)
            if hmac.compare_digest(client_sig, expected):
                node._authed = True
                set_status('connected', 'auth_ok')     # 앱은 이걸 듣고 화면 전환
            else:
                set_status('auth_failed', 'bad_signature') #만약에 실패하면 이걸 발행
        except Exception as e:
            set_status('auth_failed', f'exception:{e}')
    
    # 인증 게이트 + 실제 Wi-Fi 연결
    # 인증 전엔 모두 거부
    # APPLY에서 nmcli dev wifi connect <ssid> password <pass>실행 -> 결과
    # STATUS/ROS로 통지
    def on_ssid_write(value, options):
        if not node._authed:
            set_status('unauthorized', 'ssid_before_auth')
            return 
        node._ssid = value.decode('utf-8').strip()
        node.get_logger().info(f"SSID set: {node._ssid}")

    def on_pass_write(value, options):
        if not node._authed:
            set_status('unauthorized', 'pass_before_auth')
            return
        node._pass = value.decode('utf-8')
        node.get_logger().info("Password received")

    #APPLT characteristic에 값이 쓰이면 실행된다.
    #apply_wifi()로 실제 접속을 시도하고, 결과를 STATUS characteristic과 ROS토픽으로 둘다 알린다.
    def on_apply_write(value, options):
        # value == b'\x01'이면 적용 시도
        if not node._authed:
            set_status('unauthorized', 'apply_before_auth')
            return
        node.get_logger().info("Apply requested")
        ok, out = apply_wifi(node._ssid, node._pass)
        
        node._pass = ''
        set_status('applied' if ok else 'failed', out)
        

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

    # GATT 오브젝트 구성
    # flags: 각 특성이 제공하는 권한을 나타낸다.
    # encrypt-write -> 암호화된 연결에서만 쓰기 허용하지만 개발 초반이기때문에 허용한다.
    ssid_char = peripheral.Characteristic(SSID_CHAR_UUID,
                                          ['write'], [],
                                          write_callback=on_ssid_write)
    pass_char = peripheral.Characteristic(PASS_CHAR_UUID,
                                          ['write'], [],
                                          write_callback=on_pass_write)
    apply_char = peripheral.Characteristic(APPLY_CHAR_UUID,
                                           ['write'], [],
                                           write_callback=on_apply_write)
    status_char = peripheral.Characteristic(STATUS_CHAR_UUID,
                                            ['read', 'notify'], [],
                                            read_callback=lambda options: read_status())

    hello_char      = peripheral.Characteristic(HELLO_CHAR_UUID,     ['read','notify'], [], read_callback=lambda _o: read_hello())
    hello_resp_char = peripheral.Characteristic(HELLO_RESP_UUID,     ['write'], [],     write_callback=on_hello_resp_write)
    
    wifi_service = peripheral.Service(WIFI_SERVICE_UUID, True)
    for ch in (ssid_char, pass_char, apply_char, status_char, hello_char, hello_resp_char):
        wifi_service.add_characteristic(ch)
    # 광고 이름: HEARO-SETUP-XXXX
    # 서비스에 특성들을 추가한다 -> 광고 이름은 HEARO-SETUP이다.
    
    name_suffix = node._device_id[-4:] if len(node._device_id) >= 4 else node._device_id
    p = peripheral.Peripheral(adapter_addr=None, local_name=f'HEARO-SETUP-{name_suffix}', services=[wifi_service])


    set_status('idle')
    
    node.get_logger().info("BLE Provisioning server started")
    try:
        # bluezero는 내부 메인루프를 돈다(블로킹).
        # 간단히 별 프로세스로 띄우거나, 별 스레드에서 p.run() 돌리는 방식도 가능.
        # p.run()은 bluezero의 GLib 메인루프 진입(블로킹). 여기서 GATT서버로 동작
        p.run()
    finally:
        # 마지막 finally: rclpy.shutdown()으로 종료 시 ROS 정리
        rclpy.shutdown()

if __name__ == '__main__':
    main()

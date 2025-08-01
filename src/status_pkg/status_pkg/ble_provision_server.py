# net_status_pkg/ble_provision_server.py
import json, subprocess, os, hmac, hashlib, base64, socket, re  
from bluezero.peripheral import Peripheral                      
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# 하나의 Wi-Fi Provisioning 서비스 아래 6개의 캐릭터리스틱을 둔다.

WIFI_SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e'
SSID_CHAR_UUID    = '6e400002-b5a3-f393-e0a9-e50e24dcca9e'
PASS_CHAR_UUID    = '6e400003-b5a3-f393-e0a9-e50e24dcca9e'
APPLY_CHAR_UUID   = '6e400004-b5a3-f393-e0a9-e50e24dcca9e'
STATUS_CHAR_UUID  = '6e400005-b5a3-f393-e0a9-e50e24dcca9e'
HELLO_CHAR_UUID   = '6e400006-b5a3-f393-e0a9-e50e24dcca9e'
HELLO_RESP_UUID   = '6e400007-b5a3-f393-e0a9-e50e24dcca9e'  

# 앱과 디바이스가 공유하는 HMAC 비밀키. 개발 중엔 env/하드코드, 실제는 장치별 키 권장
SECRET_KEY = os.environ.get('HEARO_BLE_SECRET', 'dev-debug-secret').encode('utf-8')

# HMAC 바이트를 base64url로 안전하게 문자열화
def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode('ascii')

# 앱과 장치가 동일 규칙으로 계산해야 하는 서명
# 메시지 포멧 "{nonce}.{device_id}"
def hmac_sign(key: bytes, nonce: str, device_id: str) -> str:
    msg = f'{nonce}.{device_id}'.encode('utf-8')
    return b64u(hmac.new(key, msg, hashlib.sha256).digest())

# 앱 화면/로그에 보여줄 장치 식별자. 인증 서명에도 들어감
def get_device_id() -> str:
    try:
        with open('/proc/cpuinfo', 'r') as f:
            m = re.search(r'^Serial\s*:\s*([0-9a-fA-F]+)', f.read(), re.M)
            if m:
                return m.group(1)
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
        # BLE 처리 자체는 bluezero가 담당하고, 상태 메시지는 ROS 토픽으로 내보냄
        
    # STATUS 특성 업데이트와 같은 내용을 ROS 토픽으로도 내보낸다.
    def publish_status(self, d: dict):
        msg = String()
        msg.data = json.dumps(d)
        self.pub.publish(msg)
        self.get_logger().info(f"[BLE] {msg.data}")

def main(args=None):
    # 내부 콜백/헬퍼 정의
    rclpy.init(args=args)
    node = BLEProvNode()

    # ---------------- Peripheral/GATT 구성 (bluezero 스타일) ----------------
    svc_id = 'wifi'  # 서비스 식별자(문자열 임의)
    name_suffix = node._device_id[-4:] if len(node._device_id) >= 4 else node._device_id
    # 광고 이름: HEARO-SETUP-XXXX
    # 서비스에 특성들을 추가한다 -> 광고 이름은 HEARO-SETUP이다.
    name = f'HEARO-SETUP-{name_suffix}'
    try:
        # 1순위
        p = Peripheral(adapter_address=None, local_name=name)
    except TypeError:
        try:
            #2순위
            p = Peripheral(adapter_addr=None, local_name=name)
        except TypeError:
            #3순위
            p = Peripheral(local_name = name)
            
    p.add_service(svc_id=svc_id, uuid=WIFI_SERVICE_UUID, primary=True)

    # 앱이 STATUS를 Read하면 기본 상태는 idle.
    def read_status(_opts=None):
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

    # STATUS특성 값을 갱신하고, ROS 토픽도 동시에 발행
    def set_status(state: str, detail: str = None):
        payload = {'state': state}
        if detail:
            payload['detail'] = detail
        data = json.dumps(payload).encode()
        # value 업데이트 + notify
        p.update_characteristic_value(svc_id, 'status', data)
        try:
            p.notify(svc_id, 'status')   # 클라이언트가 CCCD 구독했을 때 전송
        except Exception:
            pass
        node.publish_status(payload)
    
    # 앱이 계산한 서명과 장치가 계산한 서명을 비교한다 -> 같으면 인증이 완료된다.
    def on_hello_resp_write(value, options=None):
        try:
            client_sig = value.decode('utf-8').strip()
            expected = hmac_sign(SECRET_KEY, node._nonce, node._device_id)
            if hmac.compare_digest(client_sig, expected):
                node._authed = True
                set_status('connected', 'auth_ok')     # 앱은 이걸 듣고 화면 전환
            else:
                set_status('auth_failed', 'bad_signature') # 만약에 실패하면 이걸 발행
        except Exception as e:
            set_status('auth_failed', f'exception:{e}')
    
    # 인증 게이트 + 실제 Wi-Fi 연결
    # 인증 전엔 모두 거부
    # APPLY에서 nmcli dev wifi connect <ssid> password <pass> 실행 -> 결과
    # STATUS/ROS로 통지
    def on_ssid_write(value, options=None):
        if not node._authed:
            set_status('unauthorized', 'ssid_before_auth')
            return 
        node._ssid = value.decode('utf-8').strip()
        node.get_logger().info(f"SSID set: {node._ssid}")

    def on_pass_write(value, options=None):
        if not node._authed:
            set_status('unauthorized', 'pass_before_auth')
            return
        node._pass = value.decode('utf-8')
        node.get_logger().info("Password received")

    # APPLy characteristic에 값이 쓰이면 실행된다.
    # apply_wifi()로 실제 접속을 시도하고, 결과를 STATUS characteristic과 ROS토픽으로 둘 다 알린다.
    def on_apply_write(value, options=None):
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

    # ---------------- 특성 추가 (flags로 권한 지정) ----------------
    # encrypt-write -> 암호화된 연결에서만 쓰기 허용하지만 개발 초반이기때문에 허용한다.
    # STATUS (read, notify)
    p.add_characteristic(
        svc_id=svc_id, chr_id='status', uuid=STATUS_CHAR_UUID,
        value=read_status(), notifying=True, flags=['read', 'notify'],
        read_callback=read_status
    )
    # HELLO (read, notify)
    p.add_characteristic(
        svc_id=svc_id, chr_id='hello', uuid=HELLO_CHAR_UUID,
        value=read_hello(), notifying=False, flags=['read', 'notify'],
        read_callback=read_hello
    )
    # HELLO_RESP (write)
    p.add_characteristic(
        svc_id=svc_id, chr_id='hello_resp', uuid=HELLO_RESP_UUID,
        value=b'', notifying=False, flags=['write'],
        write_callback=on_hello_resp_write
    )
    # SSID (write)
    p.add_characteristic(
        svc_id=svc_id, chr_id='ssid', uuid=SSID_CHAR_UUID,
        value=b'', notifying=False, flags=['write'],  # 출시 시 ['encrypt-write']
        write_callback=on_ssid_write
    )
    # PASS (write)
    p.add_characteristic(
        svc_id=svc_id, chr_id='pass', uuid=PASS_CHAR_UUID,
        value=b'', notifying=False, flags=['write'],  # 출시 시 ['encrypt-write']
        write_callback=on_pass_write
    )
    # APPLY (write)
    p.add_characteristic(
        svc_id=svc_id, chr_id='apply', uuid=APPLY_CHAR_UUID,
        value=b'', notifying=False, flags=['write'],
        write_callback=on_apply_write
    )

    # 시작 상태(초기 STATUS 값 반영)
    set_status('idle')

    node.get_logger().info("BLE Provisioning server started (Peripheral API)")
    try:
        # bluezero는 환경에 따라 publish() 또는 run()을 제공
        try:
            p.publish()   # 권장
        except AttributeError:
            p.run()       # 없으면 run() 사용
    finally:
        # 마지막 finally: rclpy.shutdown()으로 종료 시 ROS 정리
        rclpy.shutdown()

if __name__ == '__main__':
    main()

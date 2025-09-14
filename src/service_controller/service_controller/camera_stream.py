import os, time, threading, subprocess, cv2, rclpy, base64
from queue import Queue, Empty
from rclpy.node import Node
from std_srvs.srv import SetBool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dotenv import load_dotenv
import numpy as np
from collections import deque

# 추가
import paho.mqtt.client as mqtt

load_dotenv()

class RtspStream(Node):
    def __init__(self):
        super().__init__('rtsp_stream')
        self.declare_parameters('', [
            # Camera
            ('device', '0'),
            ('width', 320),
            ('height', 240),
            ('fps', 60),
            ('camera_fourcc', 'MJPG'),

            # ROS
            ('publish_topic', '/stream_image_raw'),

            # ===== MQTT 발행 옵션 =====
            ('mqtt_enable', True),
            ('mqtt_broker', os.getenv('MQTT_SERVER', 'localhost')),
            ('mqtt_port', int(os.getenv('MQTT_PORT', '1883'))),
            ('mqtt_username', os.getenv('MQTT_USERNAME', '')),
            ('mqtt_password', os.getenv('MQTT_PASSWORD', '')),
            ('mqtt_topic', os.getenv('MQTT_TOPIC', 'robot/cam/jpg')),
            ('mqtt_qos', 0),                      # 0 권장(저지연, 프레임드랍 허용)
            ('mqtt_base64', True),                # True면 base64 텍스트로, False면 바이너리 payload
            ('mqtt_fps', 10),                     # MQTT 발행 FPS(카메라 FPS와 별도)
            ('jpeg_quality', 70),                 # MQTT로 보낼 JPEG 품질(50~80 추천)
            ('mqtt_resize_width', 0),             # 0이면 리사이즈 안함
            ('mqtt_resize_height', 0),

            # ===== (기존 ffmpeg 관련은 쓰지 않음) =====
            ('ffmpeg_debug_log', False),

            # Optional v4l2loopback (유지: 필요시 켜서 /dev/videoX로도 뿌릴 수 있음)
            ('enable_loopback', False),
            ('loopback_device', '/dev/video10'),

            # Runtime
            ('reconnect_delay_s', 3),
            ('autostart', False),
        ])

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            Image, self.get_parameter('publish_topic').value, qos
        )

        self.lock = threading.Lock()
        self.running = False

        self.cap = None
        self.capture_thread = None

        # 최신 1장만 유지
        self.frame_q = Queue(maxsize=1)

        # (선택) loopback
        self.lb_proc = None
        self.lb_writer_thread = None

        # MQTT
        self.mqtt_client = None
        self.mqtt_thread = None
        self._mqtt_connected = False

        self.timer = self.create_timer(2.0, self._watchdog)
        self.srv = self.create_service(SetBool, 'control', self._on_control)

        if self.get_parameter('autostart').value:
            ok, msg = self.start_stream()
            self.get_logger().info(f'autostart: {ok}, {msg}')

    # ───────── controls ───────── #
    def _on_control(self, req, resp):
        try:
            if req.data:
                ok, msg = self.start_stream()
            else:
                ok, msg = self.stop_stream()
            resp.success, resp.message = ok, msg
        except Exception as e:
            resp.success, resp.message = False, f'error: {e}'
        return resp

    # ───────── lifecycle ───────── #
    def start_stream(self):
        with self.lock:
            if self.running:
                return True, 'already running'

            dev_param = str(self.get_parameter('device').value)
            dev = int(dev_param) if dev_param.isdigit() else dev_param
            w   = int(self.get_parameter('width').value)
            h   = int(self.get_parameter('height').value)
            fps = int(self.get_parameter('fps').value)
            fourcc = self.get_parameter('camera_fourcc').value

            self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS,          fps)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                return False, f'failed to open camera: {dev}'

            # warm-up
            probe_ok = False
            for _ in range(10):
                ok, _ = self.cap.read()
                if ok:
                    probe_ok = True
                    break
                time.sleep(0.01)
            if not probe_ok:
                self.cap.release(); self.cap = None
                return False, 'camera opened but failed to read frames (check width/height/fps/fourcc)'

            aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps = self.cap.get(cv2.CAP_PROP_FPS)
            fc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fcs = "".join([chr((fc >> 8*i) & 0xFF) for i in range(4)])
            self.get_logger().info(f"camera ready: {aw}x{ah}@{afps:.2f}, FOURCC={fcs}")

            self.running = True

            # capture loop
            if not self.capture_thread or not self.capture_thread.is_alive():
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()

            # MQTT publisher
            if bool(self.get_parameter('mqtt_enable').value):
                self._start_mqtt_publisher()

            # (선택) loopback
            if self.get_parameter('enable_loopback').value:
                self._start_loopback_writer(aw, ah, afps if afps > 0 else fps)

            return True, 'started'

    def stop_stream(self):
        with self.lock:
            self.running = False

            # MQTT stop
            self._stop_mqtt_publisher()

            # loopback stop
            self._stop_loopback_writer()

            # capture stop
            try:
                if self.capture_thread and self.capture_thread.is_alive():
                    self.capture_thread.join(timeout=1.0)
            except Exception:
                pass
            self.capture_thread = None

            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None

            return True, 'stopped'

    # ───────── capture loop ───────── #
    def _capture_loop(self):
        while self.running and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.003)
                continue

            # ROS publish (BGR8)
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.pub.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'publish err: {e}')

            # 최신 1장만 유지
            try:
                if self.frame_q.full():
                    _ = self.frame_q.get_nowait()
                self.frame_q.put_nowait(frame)
            except Exception:
                pass

    # ───────── MQTT publisher ───────── #
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        self._mqtt_connected = (rc == 0)
        if self._mqtt_connected:
            self.get_logger().info(f"MQTT connected rc={rc}")
        else:
            self.get_logger().warn(f"MQTT connect failed rc={rc}")

    def _start_mqtt_publisher(self):
        broker = self.get_parameter('mqtt_broker').value
        port   = int(self.get_parameter('mqtt_port').value)
        user   = self.get_parameter('mqtt_username').value
        pwd    = self.get_parameter('mqtt_password').value

        self.mqtt_client = mqtt.Client(client_id=f"rtsp_stream_pub_{int(time.time())}", clean_session=True, transport="tcp")
        if user:
            self.mqtt_client.username_pw_set(user, pwd or None)
        self.mqtt_client.on_connect = self._on_mqtt_connect

        try:
            self.mqtt_client.connect(broker, port, 30)
        except Exception as e:
            self.get_logger().warn(f"MQTT connect error: {e}")

        # network loop in background
        self.mqtt_client.loop_start()

        # start publisher thread
        if not self.mqtt_thread or not self.mqtt_thread.is_alive():
            self.mqtt_thread = threading.Thread(target=self._mqtt_loop, daemon=True)
            self.mqtt_thread.start()

    def _stop_mqtt_publisher(self):
        try:
            if self.mqtt_thread and self.mqtt_thread.is_alive():
                # thread will exit when self.running=False
                self.mqtt_thread.join(timeout=1.0)
        except Exception:
            pass
        self.mqtt_thread = None

        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        except Exception:
            pass
        self.mqtt_client = None
        self._mqtt_connected = False

    def _mqtt_loop(self):
        topic   = self.get_parameter('mqtt_topic').value
        qos     = int(self.get_parameter('mqtt_qos').value)
        use_b64 = bool(self.get_parameter('mqtt_base64').value)
        mfps    = int(self.get_parameter('mqtt_fps').value)
        q       = int(self.get_parameter('jpeg_quality').value)
        rw      = int(self.get_parameter('mqtt_resize_width').value)
        rh      = int(self.get_parameter('mqtt_resize_height').value)

        period = 1.0 / max(1, mfps)
        next_t = time.time()

        while self.running and self.mqtt_client:
            # FPS 레이트 리밋
            next_t += period
            try:
                frame = self.frame_q.get(timeout=0.1)
            except Empty:
                time.sleep(max(0.0, next_t - time.time()))
                continue

            try:
                # (선택) 리사이즈
                if rw > 0 and rh > 0 and (frame.shape[1] != rw or frame.shape[0] != rh):
                    frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)

                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                if not ok:
                    continue

                if use_b64:
                    payload = base64.b64encode(enc).decode('ascii')
                else:
                    payload = enc.tobytes()

                # QoS 0 권장(저지연)
                res = self.mqtt_client.publish(topic, payload, qos=qos, retain=False)
                # (옵션) 결과 확인: res.rc == mqtt.MQTT_ERR_SUCCESS

            except Exception as e:
                self.get_logger().warn(f"mqtt publish err: {e}")

            # 슬립(다음 타임슬롯까지)
            dt = next_t - time.time()
            if dt > 0:
                time.sleep(dt)
            else:
                # 밀렸으면 다음 주기로 보정
                next_t = time.time()

    # ───────── loopback (optional) ───────── #
    def _start_loopback_writer(self, w, h, fps):
        dev = self.get_parameter('loopback_device').value
        args = [
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'info', '-nostats',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
            '-f', 'v4l2', dev
        ]
        self.get_logger().info(f"starting ffmpeg (loopback->{dev}): {' '.join(args)}")
        self.lb_proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        self.lb_writer_thread = threading.Thread(target=self._writer_loop_loopback, daemon=True)
        self.lb_writer_thread.start()

    def _stop_loopback_writer(self):
        try:
            if self.lb_proc:
                self.lb_proc.kill()
        except Exception:
            pass
        self.lb_proc = None
        self.lb_writer_thread = None

    def _writer_loop_loopback(self):
        while self.running and self.lb_proc and self.lb_proc.poll() is None:
            try:
                frame = self.frame_q.get(timeout=0.1)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                self.lb_proc.stdin.write(frame.tobytes())
            except Empty:
                continue
            except Exception:
                break

    # ───────── misc ───────── #
    def _watchdog(self):
        pass

    def destroy_node(self):
        self.stop_stream()
        super().destroy_node()

def main():
    rclpy.init()
    node = RtspStream()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

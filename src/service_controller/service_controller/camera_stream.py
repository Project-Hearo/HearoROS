import rclpy, subprocess, shlex, signal, time, threading, cv2
from rclpy.node import Node
from std_srvs.srv import SetBool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dotenv import load_dotenv
load_dotenv()
import os

class RtspStream(Node):
    def __init__(self):
        super().__init__('rtsp_stream')
        self.declare_parameters('', [
            ('device', '/dev/video0'),
            ('width', 320),
            ('height', 240),
            ('fps', 120),
            ('codec', 'libx264'),             
            ('bitrate', '1M'),
            ('gop', 120),
            ('preset', 'veryfast'),
            ('tune', 'zerolatency'),
            ('rtsp_transport', 'tcp'),
            ('reconnect_delay_s', 3),
            ('autostart', False),
            ('publish_topic', '/stream_image_raw'),
            ('camera_fourcc', 'MJPG'),         
            ('rtsp_url', os.getenv('RTSP_URL', 'rtsp://localhost:8554/robot')),
        ]) 
  
        qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST
        )
        
        self.proc = None
        self.cap = None
        self.writer_thread = None
        self.running = False

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.get_parameter('publish_topic').value, qos)

        self.lock = threading.Lock()
        self.timer = self.create_timer(2.0, self._watchdog)
        self.srv = self.create_service(SetBool, 'control', self._on_control)

        if self.get_parameter('autostart').value:
            self.start_stream()

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

    def _build_cmd(self, w, h, fps):
        codec = self.get_parameter('codec').value
        br    = self.get_parameter('bitrate').value
        gop   = int(self.get_parameter('gop').value)
        preset= self.get_parameter('preset').value
        tune  = self.get_parameter('tune').value
        url   = self.get_parameter('rtsp_url').value
        rtspt = self.get_parameter('rtsp_transport').value

        cmd = (
            f"ffmpeg -nostdin -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - "
            f"-vf format=yuv420p -c:v {codec} -preset {preset} -tune {tune} "
            f"-b:v {br} -maxrate {br} -bufsize {br} -g {gop} "
            f"-f rtsp -rtsp_transport {rtspt} {url}"
        )
        return cmd

    # ----- lifecycle
    def start_stream(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return True, 'already running'

            dev = self.get_parameter('device').value
            w   = int(self.get_parameter('width').value)
            h   = int(self.get_parameter('height').value)
            fps = int(self.get_parameter('fps').value)
            fourcc = self.get_parameter('camera_fourcc').value

            # Open camera once
            self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)   
            self.cap.set(cv2.CAP_PROP_FPS,          fps)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

            if not self.cap.isOpened():
                return False, f'failed to open camera: {dev}'

            cmd = self._build_cmd(w, h, fps)
            self.get_logger().info(f"starting ffmpeg: {cmd}")
            self.proc = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,          
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0
            )
            self.running = True
            self.writer_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.writer_thread.start()
            return True, 'started'

    def stop_stream(self):
        with self.lock:
            self.running = False
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None

            try:
                if self.proc and self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass

            if self.proc and self.proc.poll() is None:
                self.get_logger().info("stopping ffmpeg...")
                try:
                    self.proc.send_signal(signal.SIGINT)
                    try:
                        self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            return True, 'stopped'

    def _capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.pub.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'publish err: {e}')

            try:
                if self.proc and self.proc.stdin:
                    self.proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass
            except Exception as e:
                self.get_logger().warn(f'pipe err: {e}')

    def _watchdog(self):
        p = self.proc
        if not p:
            return
        if p.poll() is None:
            return

        try:
            if p.stdout:
                tail = []
                for _ in range(40):
                    line = p.stdout.readline()
                    if not line: break
                    if isinstance(line, bytes):
                        line = line.decode(errors='ignore')
                    tail.append(line.strip())
                if tail:
                    self.get_logger().warn("ffmpeg exited. tail:\n" + "\n".join(tail[-10:]))
        except Exception:
            pass

        delay = int(self.get_parameter('reconnect_delay_s').value)
        self.get_logger().warn(f"Restarting ffmpeg in {delay}s...")
        time.sleep(delay)
        self.start_stream()

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

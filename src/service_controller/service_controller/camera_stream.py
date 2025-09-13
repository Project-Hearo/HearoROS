import os, time, shlex, signal, threading, subprocess, cv2, rclpy
from queue import Queue, Empty
from rclpy.node import Node
from std_srvs.srv import SetBool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dotenv import load_dotenv
import numpy as np
load_dotenv()

class RtspStream(Node):
    def __init__(self):
        super().__init__('rtsp_stream')
        self.declare_parameters('', [
            ('device', '0'),                 
            ('width', 320),
            ('height', 240),
            ('fps', 120),
            ('camera_fourcc', 'MJPG'),       
            ('publish_topic', '/stream_image_raw'),

            ('codec', 'libx264'),       
            ('bitrate', '1M'),
            ('gop', 120),
            ('preset', 'veryfast'),
            ('tune', 'zerolatency'),
            ('rtsp_transport', 'tcp'),
            ('rtsp_url', os.getenv('RTSP_URL', 'rtsp://localhost:8554/robot')),

            ('enable_loopback', False),
            ('loopback_device', '/dev/video10'),

            ('reconnect_delay_s', 3),
            ('autostart', False),
        ])

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.get_parameter('publish_topic').value, qos)

        self.lock = threading.Lock()
        self.running = False

        self.cap = None
        self.capture_thread = None

        self.frame_q = Queue(maxsize=1)  

        self.rtsp_proc = None
        self.rtsp_writer_thread = None

        self.lb_proc = None
        self.lb_writer_thread = None
        self.rtsp_log_thread = None

        self.timer = self.create_timer(2.0, self._watchdog)  
        self.srv = self.create_service(SetBool, 'control', self._on_control)

        if self.get_parameter('autostart').value:
            ok, msg = self.start_stream()
            self.get_logger().info(f'autostart: {ok}, {msg}')

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

    def _build_rtsp_cmd(self, w, h, fps):
        codec = self.get_parameter('codec').value
        br    = self.get_parameter('bitrate').value
        gop   = int(self.get_parameter('gop').value)
        preset= self.get_parameter('preset').value
        tune  = self.get_parameter('tune').value
        url   = self.get_parameter('rtsp_url').value
        rtspt = self.get_parameter('rtsp_transport').value

        return (
            f"ffmpeg -nostdin -hide_banner -loglevel info -nostats "
            f"-f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - "
            f"-vf format=yuv420p -an "
            f"-c:v {codec} -preset {preset} -tune {tune} -bf 0 "
            f"-b:v {br} -maxrate {br} -bufsize {br} -g {gop} "
            f"-f rtsp -rtsp_transport {rtspt} {url}"
        )

    def _build_loopback_cmd(self, w, h, fps, dev):
        return (
            f"ffmpeg -nostdin -hide_banner -loglevel error -nostats "
            f"-f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - "
            f"-f v4l2 {dev}"
        )

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

            probe_ok = False
            for _ in range(10):
                ok, _ = self.cap.read()
                if ok:
                    probe_ok = True
                    break
                time.sleep(0.01)
            if not probe_ok:
                self.cap.release()
                self.cap = None
                return False, 'camera opened but failed to read frames (check width/height/fps/fourcc)'

            aw  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps= self.cap.get(cv2.CAP_PROP_FPS)
            fc  = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fcs = "".join([chr((fc >> 8*i) & 0xFF) for i in range(4)])
            self.get_logger().info(f"camera ready: {aw}x{ah}@{afps:.2f}, FOURCC={fcs}")

            self.running = True

            if not self.capture_thread or not self.capture_thread.is_alive():
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()

            self._start_rtsp_writer(w, h, fps)

            if self.get_parameter('enable_loopback').value:
                self._start_loopback_writer(w, h, fps)

            return True, 'started'

    def stop_stream(self):
        with self.lock:
            self.running = False

            # stop writers first
            self._stop_rtsp_writer()
            self._stop_loopback_writer()

            # stop capture
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

    def _capture_loop(self):
        while self.running and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.003)
                continue

            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.pub.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'publish err: {e}')

            try:
                if self.frame_q.full():
                    _ = self.frame_q.get_nowait()
                self.frame_q.put_nowait(frame)
            except Exception:
                pass

    def _start_rtsp_writer(self, w, h, fps):
        cmd = self._build_rtsp_cmd(w, h, fps)
        self.get_logger().info(f"starting ffmpeg (rtsp): {cmd}")
        self.rtsp_proc = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,   
            stderr=subprocess.STDOUT,
            bufsize=0
        
        )
        self.rtsp_log_thread = threading.Thread(target=self._ffmpeg_log_drain, daemon=True)
        self.rtsp_log_thread.start()
        
        self.rtsp_writer_thread = threading.Thread(target=self._writer_loop_rtsp, daemon=True)
        self.rtsp_writer_thread.start()

    def _stop_rtsp_writer(self):
        try:
            if self.rtsp_proc and self.rtsp_proc.stdin:
                self.rtsp_proc.stdin.close()
        except Exception:
            pass
        try:
            if self.rtsp_proc and self.rtsp_proc.poll() is None:
                self.rtsp_proc.terminate()
                try:
                    self.rtsp_proc.wait(timeout=1.0)
                except Exception:
                    self.rtsp_proc.kill()
        except Exception:
            pass
        self.rtsp_proc = None
        self.rtsp_writer_thread = None
        self.rtsp_log_thread = None

    def _writer_loop_rtsp(self):
        try:
            while self.running and self.rtsp_proc and self.rtsp_proc.poll() is None:
                try:
                    frame = self.frame_q.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    self.rtsp_proc.stdin.write(frame.tobytes())
                except Exception as e:
                    self.get_logger().warn(f'ffmpeg(rtsp) pipe err: {e}')
                    break
        finally:
            if self.running:
                delay = int(self.get_parameter('reconnect_delay_s').value)
                self.get_logger().warn(f"ffmpeg(rtsp) down, restarting in {delay}s...")
                time.sleep(delay)
                w = int(self.get_parameter('width').value)
                h = int(self.get_parameter('height').value)
                fps = int(self.get_parameter('fps').value)
                self._stop_rtsp_writer()
                self._start_rtsp_writer(w, h, fps)

    def _start_loopback_writer(self, w, h, fps):
        dev = self.get_parameter('loopback_device').value
        cmd = self._build_loopback_cmd(w, h, fps, dev)
        self.get_logger().info(f"starting ffmpeg (loopback->{dev}): {cmd}")
        self.lb_proc = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            bufsize=0
        )
        self.lb_writer_thread = threading.Thread(target=self._writer_loop_loopback, daemon=True)
        self.lb_writer_thread.start()

    def _stop_loopback_writer(self):
        try:
            if self.lb_proc and self.lb_proc.stdin:
                self.lb_proc.stdin.close()
        except Exception:
            pass
        try:
            if self.lb_proc and self.lb_proc.poll() is None:
                self.lb_proc.kill()
        except Exception:
            pass
        self.lb_proc = None
        self.lb_writer_thread = None

    def _writer_loop_loopback(self):
        try:
            while self.running and self.lb_proc and self.lb_proc.poll() is None:
                try:
                    frame = self.frame_q.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    self.lb_proc.stdin.write(frame.tobytes())
                except Exception as e:
                    self.get_logger().warn(f'ffmpeg(loopback) pipe err: {e}')
                    break
        finally:
            if self.running:
                delay = int(self.get_parameter('reconnect_delay_s').value)
                dev = self.get_parameter('loopback_device').value
                self.get_logger().warn(f"ffmpeg(loopback->{dev}) down, restarting in {delay}s...")
                time.sleep(delay)
                w = int(self.get_parameter('width').value)
                h = int(self.get_parameter('height').value)
                fps = int(self.get_parameter('fps').value)
                self._stop_loopback_writer()
                self._start_loopback_writer(w, h, fps)

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

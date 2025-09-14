import os, time, shlex, threading, subprocess, cv2, rclpy
from queue import Queue, Empty
from rclpy.node import Node
from std_srvs.srv import SetBool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dotenv import load_dotenv
import numpy as np
from collections import deque

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

            # FFmpeg encode
            ('codec', 'libx264'),
            ## [수정] 기본 비트레이트를 현실적인 1M으로 변경
            ('bitrate', '1M'),
            ('gop', 120),
            ('preset', 'veryfast'),
            ('tune', 'zerolatency'),
            ('rtsp_transport', 'tcp'),
            ('rtsp_url', os.getenv('RTSP_URL', 'rtsp://localhost:8554/robot')),

            # Pipe/Input tuning
            ('pipe_format', 'mjpeg'),
            ('jpeg_quality', 80),
            ('pix_fmt', 'yuv420p'),
            ('ffmpeg_debug_log', True),
            ('dry_run_null', False),

            # Optional v4l2loopback
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
        self.frame_q = Queue(maxsize=1)
        self.rtsp_proc = None
        self.rtsp_writer_thread = None
        self.rtsp_log_thread = None
        self._ffbuf = deque(maxlen=200)
        self.lb_proc = None
        self.lb_writer_thread = None
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

    ## [수정] 함수명을 바꾸고 문자열 대신 인수 '리스트'를 반환하도록 변경
    def _build_rtsp_args(self, w, h, fps):
        codec = self.get_parameter('codec').value
        br    = self.get_parameter('bitrate').value
        gop   = int(self.get_parameter('gop').value)
        preset= self.get_parameter('preset').value
        tune  = self.get_parameter('tune').value
        url   = self.get_parameter('rtsp_url').value
        rtspt = self.get_parameter('rtsp_transport').value
        pipef = self.get_parameter('pipe_format').value
        dry   = bool(self.get_parameter('dry_run_null').value)
        pixfmt= self.get_parameter('pix_fmt').value

        # 기본 ffmpeg 명령어 리스트
        args = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'info', '-nostats']

        # 입력 포맷 설정
        if pipef.lower() == 'mjpeg':
            args.extend(['-f', 'mjpeg', '-r', str(fps), '-i', '-'])
        else:
            args.extend(['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-'])

        # 비디오 필터 (따옴표 문제 없이 안전하게 추가)
        args.extend(['-vf', f'format={pixfmt},setsar=1'])
        args.extend(['-an']) # 오디오 없음

        # 인코더 및 주요 옵션
        args.extend(['-c:v', codec, '-preset', preset, '-tune', tune, '-bf', '0'])
        args.extend(['-b:v', br, '-maxrate', br, '-bufsize', br, '-g', str(gop)])

        # 저지연 옵션
        args.extend(['-fflags', '+genpts+nobuffer', '-use_wallclock_as_timestamps', '1', '-fps_mode', 'passthrough'])
        args.extend(['-analyzeduration', '500000', '-probesize', '50000'])
        args.extend(['-muxpreload', '0', '-muxdelay', '0', '-flush_packets', '1'])

        # 출력 설정
        if dry:
            args.extend(['-f', 'null', '-'])
        else:
            args.extend(['-f', 'rtsp', '-rtsp_transport', rtspt, url])
            
        return args

    def _build_loopback_args(self, w, h, fps, dev):
        return [
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'info', '-nostats',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
            '-f', 'v4l2', dev
        ]

    def start_stream(self):
        with self.lock:
            if self.running:
                return True, 'already running'

            dev_param = str(self.get_parameter('device').value)
            dev = int(dev_param) if dev_param.isdigit() else dev_param
            w, h, fps = [int(self.get_parameter(p).value) for p in ('width', 'height', 'fps')]
            fourcc = self.get_parameter('camera_fourcc').value

            self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                return False, f'failed to open camera: {dev}'

            probe_ok = False
            for _ in range(10):
                ok, _ = self.cap.read()
                if ok: probe_ok = True; break
                time.sleep(0.01)
            if not probe_ok:
                self.cap.release(); self.cap = None
                return False, 'camera opened but failed to read frames (check fourcc/params)'

            aw, ah = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps, fc = self.cap.get(cv2.CAP_PROP_FPS), int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fcs = "".join([chr((fc >> 8*i) & 0xFF) for i in range(4)])
            self.get_logger().info(f"camera ready: {aw}x{ah}@{afps:.2f}, FOURCC={fcs}")

            self.running = True

            if not self.capture_thread or not self.capture_thread.is_alive():
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()

            self._start_rtsp_writer(aw, ah, afps if afps > 0 else fps)

            if self.get_parameter('enable_loopback').value:
                self._start_loopback_writer(aw, ah, afps if afps > 0 else fps)

            return True, 'started'

    def stop_stream(self):
        with self.lock:
            self.running = False
            self._stop_rtsp_writer()
            self._stop_loopback_writer()
            if self.capture_thread: self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
            if self.cap: self.cap.release()
            self.cap = None
            return True, 'stopped'

    def _capture_loop(self):
        while self.running and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.003)
                continue
            # self.get_logger().info("<<<<< Frame captured successfully from OpenCV! >>>>>")

            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.pub.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'publish err: {e}')
            try:
                if self.frame_q.full(): self.frame_q.get_nowait()
                self.frame_q.put_nowait(frame)
            except Exception: pass

    def _start_rtsp_writer(self, w, h, fps):
        ## [수정] 새 함수를 호출하고 shlex.split 없이 리스트를 사용
        args = self._build_rtsp_args(w, h, fps)
        self.get_logger().info(f"starting ffmpeg (rtsp): {' '.join(args)}")

        self._ffbuf.clear()
        DEBUG_LOG = bool(self.get_parameter('ffmpeg_debug_log').value)
        
        self.rtsp_proc = subprocess.Popen(
            args, # shlex.split() 제거!
            stdin=subprocess.PIPE,
            stdout=(subprocess.PIPE if DEBUG_LOG else subprocess.DEVNULL),
            stderr=subprocess.STDOUT,
            bufsize=0
        )

        if DEBUG_LOG:
            self.rtsp_log_thread = threading.Thread(target=self._ffmpeg_log_drain, daemon=True)
            self.rtsp_log_thread.start()

        self.rtsp_writer_thread = threading.Thread(
            target=self._writer_loop_rtsp, daemon=True
        )
        self.rtsp_writer_thread.start()

    def _ffmpeg_log_drain(self):
        p = self.rtsp_proc
        if not p or not p.stdout: return
        while self.running and p.poll() is None:
            line = p.stdout.readline()
            if not line: break
            text = line.decode(errors='ignore').strip()
            self._ffbuf.append(text)
            self.get_logger().info(f"[ffmpeg] {text}")

    def _stop_rtsp_writer(self):
        if self.rtsp_proc and self.rtsp_proc.poll() is None:
            try:
                if self.rtsp_proc.stdin: self.rtsp_proc.stdin.close()
            except Exception: pass
            self.rtsp_proc.terminate()
            try:
                rc = self.rtsp_proc.wait(timeout=1.0)
                # 정상 종료가 아닐 때만 로그 출력
                if rc != 0:
                    tail = "\n".join(list(self._ffbuf)[-15:])
                    self.get_logger().warn(f"ffmpeg exited rc={rc}. tail logs:\n{tail}")
            except Exception:
                self.rtsp_proc.kill()
        
        self.rtsp_proc = None
        self.rtsp_writer_thread = None
        self.rtsp_log_thread = None

    def _writer_loop_rtsp(self):
        # writer 루프는 재시작 로직을 담당
        pipef = self.get_parameter('pipe_format').value.lower()
        w = int(self.get_parameter('width').value)
        h = int(self.get_parameter('height').value)
        frame_bytes = w * h * 3

        try:
            time.sleep(0.05) # ffmpeg 프로세스가 완전히 준비될 시간
            while self.running and self.rtsp_proc and self.rtsp_proc.poll() is None:
                try:
                    frame = self.frame_q.get(timeout=0.1)
                    if pipef == 'mjpeg':
                        q = int(self.get_parameter('jpeg_quality').value)
                        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                        if ok: self.rtsp_proc.stdin.write(enc.tobytes())
                    else: # raw
                        if frame.nbytes == frame_bytes: self.rtsp_proc.stdin.write(frame.tobytes())
                except Empty:
                    continue
                except Exception as e:
                    self.get_logger().warn(f'ffmpeg(rtsp) pipe err: {e}')
                    break
        finally:
            if self.running: # 사용자가 stop을 호출한게 아니라면 재시작
                delay = int(self.get_parameter('reconnect_delay_s').value)
                self.get_logger().warn(f"ffmpeg(rtsp) process down, restarting in {delay}s...")
                self._stop_rtsp_writer() # 현재 프로세스 정리
                time.sleep(delay)
                # 최신 파라미터로 재시작
                w2, h2, fps2 = [int(self.get_parameter(p).value) for p in ('width', 'height', 'fps')]
                self._start_rtsp_writer(w2, h2, fps2)

    def _start_loopback_writer(self, w, h, fps):
        dev = self.get_parameter('loopback_device').value
        args = self._build_loopback_args(w, h, fps, dev)
        self.get_logger().info(f"starting ffmpeg (loopback->{dev}): {' '.join(args)}")
        self.lb_proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        self.lb_writer_thread = threading.Thread(target=self._writer_loop_loopback, daemon=True)
        self.lb_writer_thread.start()

    def _stop_loopback_writer(self):
        if self.lb_proc: self.lb_proc.kill()
        self.lb_proc = None
        self.lb_writer_thread = None

    def _writer_loop_loopback(self):
        while self.running and self.lb_proc and self.lb_proc.poll() is None:
            try:
                frame = self.frame_q.get(timeout=0.1)
                self.lb_proc.stdin.write(frame.tobytes())
            except Empty:
                continue
            except Exception:
                break

    def _watchdog(self):
        # 재시작 로직은 _writer_loop_rtsp의 finally 블록에서 처리하므로 watchdog은 비워둠
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
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist

from ros_mqtt_bridge.telemetry_handlers.telemetry_manager import TelemetryManager

from ros_mqtt_bridge.cmd_handlers.dispatcher import Dispatcher

import paho.mqtt.client as mqtt
import time, json, threading
from collections import deque
import importlib.resources
from ros_mqtt_bridge import config
QOS_KEEP_LAST_1 = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

class RosMqttBridge(Node):
    def __init__(self):
        # ros_mqtt_bridge 노드 발행
        super().__init__('ros_mqtt_bridge')
        # config.json파일에 있는 정보를 이용해서 mqtt서버에 연결을 시도한다.

        self.robot_id    = config.ROBOT_ID
        self.broker_host = config.MQTT_SERVER
        self.broker_port = int(config.MQTT_PORT)
        self.socket_port = int(config.MQTT_SOCKET_PORT)
        self.keepalive   = int(config.KEEPALIVE)

        # ---- mqtt topics ----
        # 명령은 robot/<id>/cmd/<리소스>/<액션>
        # 응답은 robot/<id>/resp/<리소스>/<액션>/<req_id>
        self.topic_cmd_base    = f'robot/{self.robot_id}/cmd' # 명령 입력
        self.topic_resp_base   = f'robot/{self.robot_id}/resp' #결과/에러 응답
        
        self.topic_online = f'robot/{self.robot_id}/status/online' #LWT 포함 올라인 표시
        self.topic_bridge = f'robot/{self.robot_id}/status/bridge'
        
        self.topic_telemetry = f'robot/{self.robot_id}/telemetry'
        
        #req_id -> cmd_path('status/get') 매핑 저장
        self._req_cmd = {}

        # ---- mqtt client ----
        self.connected = False
        self.mqtt = mqtt.Client(client_id=self.robot_id, clean_session=False)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_disconnect = self._on_disconnect
        self.mqtt.on_message = self._on_message
        #콜백 등록, LWT로 비정상 종료 시 online:false 자동 발행, 백그라운드 네트워크 루프시작
        self.mqtt.will_set(self.topic_online, json.dumps({"online": False}), qos=1, retain=True)
        self.mqtt.connect(self.broker_host, self.broker_port, keepalive=self.keepalive)
        self.mqtt.loop_start() #여기를 이용해서 백그라운드 스레드를 돌린다.

        # ---- ROS interfaces ----
        # status/get을 처리할 서비스 클라이언트 준비
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', QOS_KEEP_LAST_1)

        # ---- command queue & timer (lightweight) ----
        # MQTT 콜백 스레드에서 받은 요청을 큐에 넣고, 타이머가 50Hz로 꺼내 디스패치
        self._q = deque()
        self._qlock = threading.Lock()
        self.create_timer(0.02, self._process_queue) #50Hz

        # 5초마다 backlog/q_max 등을 로그+MQTT로 송신한다.
        self.get_logger().info("RosMqttBridge up")
        self.metrics = {"recv": 0, "ack": 0, "errors": 0, "q_max": 0}
        self.topic_bridge = f'robot/{self.robot_id}/status/bridge'
        self.create_timer(5.0, self._report_health)
        self._stop_timer = None # 모션 정지용 타이머 핸들 저장(핸들러에서 쓴다)
        
        #핸들러 플러그인 자동 드로
        self.dispatcher = Dispatcher(self, threaded=True, max_workers=2)
        self.dispatcher.load_plugins() # handlers/*.py를 스캔해 @register된 핸들러들을 인스턴스화 한다.
        
        self.telemetry_manager = TelemetryManager(self)
        self.telemetry_manager.load_plugins()
        self.telemetry_manager.start_all()
    # ------------- MQTT ------------
    # 모든 명령(.../cmd/#)구독, 온라인 true알름
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            client.publish(self.topic_online, json.dumps({"online": True}), qos=1, retain=True)
            client.subscribe(self.topic_cmd_base + '/#', qos=1)
            self.get_logger().info(f"MQTT connected; subscribed {self.topic_cmd_base}")
        else:
            self.get_logger().error(f"MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.get_logger().warn(f"MQTT disconnected rc={rc}")
    
    def _extract_topic(self ,topic: str, *, base_prefix: str = None, keyword: str = None) -> str:
        if base_prefix:
            prefix = base_prefix.rstrip('/') + '/'
            return topic[len(prefix):] if topic.startswith(prefix) else ""
        if keyword:
            parts = topic.split('/')
            return '/'.join(parts[parts.index(keyword)+1:]) if keyword in parts else ""
        return ""
        
    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8', errors='strict'))
            req_id = payload.get('req_id')
            request = payload.get('request', {})
            cmd_path = self._extract_topic(msg.topic, base_prefix=self.topic_cmd_base)
            if not isinstance(req_id, str) or not req_id:
                self.get_logger().warn("bad request: missing req_id")
                return
        except Exception as e:
            self._publish_resp(None, ok=False, error={"code":"bad_request","message":str(e)})
            return

        with self._qlock:
            self._q.append((req_id, cmd_path, request))
            self._req_cmd[req_id] = cmd_path
            self.metrics["recv"] += 1
            self.metrics["q_max"] = max(self.metrics["q_max"], len(self._q))

    def _process_queue(self):
        if not self._q:
            return
        with self._qlock:
            req_id, cmd_path, args = self._q.popleft()
        try:
            self.dispatcher.dispatch(req_id, cmd_path, args)
        except Exception as e:
            self.get_logger().exception("handler dispatch error")
            self._publish_resp(req_id, ok=False, error={"code":"handler_error","message":str(e)})

    def _publish_ack(self, req_id, extra=None):
        self.metrics["ack"] += 1
        self._publish_resp(req_id, ok=True, data={"state":"accepted", **(extra or {})})

    def _publish_result(self, req_id, result: dict):
        self._publish_resp(req_id, ok=True, data={"state":"result", **result}, finalize=True)

    def _publish_feedback(self, req_id, feedback: dict):
        self._publish_resp(req_id, ok=True, data={"state":"feedback", **feedback})

    def _publish_resp(self, req_id, ok=True, data=None, error=None, *, finalize=False):
        if not ok:
            self.metrics["errors"] += 1
        msg = {
            "req_id": req_id,
            "ok": ok,
            "data": data if ok else None,
            "error": None if ok else error,
            "ts": int(time.time()*1000),
        }
        if not self.connected:
            self.get_logger().warn("skip publish: mqtt disconnected")
            return
        
        cmd_path = self._req_cmd.get(req_id)
        if not req_id or not cmd_path:
            self.get_logger().warn(f"skip resp: missing req_id or cmd_path (req_id={req_id}, cmd_path={cmd_path})")
            return
        
        topic = f"{self.topic_resp_base}/{cmd_path}"
        
        s = json.dumps(msg, separators=(',',':'))
        r = self.mqtt.publish(topic, s, qos=1, retain=False)
        if r.rc != mqtt.MQTT_ERR_SUCCESS:
            self.get_logger().error(f"publish failed rc={r.rc}")
        if finalize and req_id:
            self._req_cmd.pop(req_id, None)
            
    def _publish_telemetry(
        self,
        path: str,
        payload: dict,
        *, qos: int = 1, retain:bool = False):
        
        norm = path.replace('.','/').strip('/').lower()
        topic = f"{self.topic_telemetry}/{norm}"
        
        if not self.connected:
            self.get_logger().warn("skip publish: mqtt disconnected")
            return
        
        try:
            s = json.dumps(payload, separators=(',', ':'))
            response = self.mqtt.publish(topic, s, qos=qos, retain=retain)
            if response.rc != mqtt.MQTT_ERR_SUCCESS:
                self.get_logger().error(f"telemetry publish failed rc={response.rc} topic={topic}")
        except Exception as e:
            self.metrics["errors"] += 1
            self.get_logger().exception(f"telemetry publish error: {e}")
        
        
    def _arm_timeout(self, req_id, future, timeout_sec: float):
        start = time.time()
        def _check_timeout():
            if future.done():
                t.cancel()
                return
            if time.time() - start >= timeout_sec:
                try:
                    # future가 끝나도 결과는 버림(이미 타임아웃 보냈으니)
                    self._publish_resp(req_id, ok=False, error={"code":"timeout","message":"operation timed out"})
                finally:
                    t.cancel()
        t = self.create_timer(0.05, _check_timeout)  # 20Hz로 상태 체크
        
    def _report_health(self):
        backlog = self.metrics["recv"] - self.metrics["ack"]
        if backlog > 100 or self.metrics["q_max"] > 200:
            self.get_logger().warn(
                f"bridge load: backlog={backlog} q_max={self.metrics['q_max']} recv={self.metrics['recv']} err={self.metrics['errors']}"
                )
        else:
            self.get_logger().info(
                f"bridge ok: backlog={backlog} q_max={self.metrics['q_max']} recv={self.metrics['recv']} err={self.metrics['errors']}"
                )
        
        if self.connected:
            payload = {
                "online": True,
                "backlog": backlog,
                "q_max": self.metrics["q_max"],
                "recv": self.metrics["recv"],
                "errors": self.metrics["errors"],
                "ts": int(time.time()*1000)
            }
            self.mqtt.publish(self.topic_bridge, json.dumps(payload), qos=0, retain=False)

    def _wait_service(self, client, name, timeout):
        waited = 0.0
        while not client.wait_for_service(timeout_sec=0.5):
            waited += 0.5
            if waited >= timeout:
                self.get_logger().warn(f"service {name} not ready; keep trying")
                break
    
def main(args=None):
    rclpy.init(args=args)
    node = RosMqttBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("shutdown...")
        node.telemetry_manager.stop_all()
        node.mqtt.loop_stop()
        node.mqtt.disconnect()
        node.destroy_node()
        node.dispatcher.shutdown()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

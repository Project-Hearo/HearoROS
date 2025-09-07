from typing import Optional, Tuple

import shutil, time
from pathlib import Path
from slam_toolbox.srv import SaveMap
import os, re
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from my_robot_interfaces.action import SlamSession

from geometry_msgs.msg import PoseArray, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration

from std_msgs.msg import String as RosString
from visualization_msgs.msg import MarkerArray


class SlamAction(Node):
    def __init__(self):
        super().__init__('slam_action_server')
        
        self.declare_parameter('frontiers_topic', 'explore/frontiers')
        self.declare_parameter('zero_hold_sec', 5.0)
        self.declare_parameter('feedback_period', 0.2)
        self.declare_parameter('simulate_progress', False)
        self.declare_parameter('no_msg_timeout_sec', 20.0)
        self.declare_parameter('startup_grace_sec', 60.0)   
        self.declare_parameter('no_msg_abort_sec', 60.0)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        
        self.declare_parameter('map_dir_pgm', '/root/maps/pgm')
        self.declare_parameter('map_dir_yaml', '/root/maps/yaml')
        self.declare_parameter('map_stage_dir', '/root/maps/stage')

        self.map_dir_pgm   = Path(self.get_parameter('map_dir_pgm').value)
        self.map_dir_yaml  = Path(self.get_parameter('map_dir_yaml').value)
        self.map_stage_dir = Path(self.get_parameter('map_stage_dir').value)
        
        self.save_map_cli = self.create_client(SaveMap, '/slam_toolbox/save_map')
        for d in (self.map_dir_pgm, self.map_dir_yaml, self.map_stage_dir):
            d.mkdir(parents=True, exist_ok=True)
            
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.last_pose: Optional[PoseStamped] = None
        self.cb_group = ReentrantCallbackGroup()
        
        self.frontier_count: Optional[int] = None
        self.zero_since: Optional[float] = None
        self.last_msg_time: float = time.monotonic()
        self.started_at: float = 0.0
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        topic = self.get_parameter('frontiers_topic').get_parameter_value().string_value

        def _subscribe_pose():
            self.sub_pose = self.create_subscription(
                PoseArray, topic, self.on_frontiers, qos, callback_group=self.cb_group
            )
            self.get_logger().info(f"Subscribing frontiers (PoseArray) on: {self.resolve_topic_name(topic)}")

        def _subscribe_marker():
            self.sub_marker = self.create_subscription(
                MarkerArray, topic, self.on_frontiers_marker, qos, callback_group=self.cb_group
            )
            self.get_logger().info(f"Subscribing frontiers (MarkerArray) on: {self.resolve_topic_name(topic)}")


        resolved = self.resolve_topic_name(topic)
        types_map = dict(self.get_topic_names_and_types())
        types = types_map.get(resolved, [])

        if 'geometry_msgs/msg/PoseArray' in types and 'visualization_msgs/msg/MarkerArray' in types:
    
            _subscribe_pose()
        elif 'geometry_msgs/msg/PoseArray' in types:
            _subscribe_pose()
        elif 'visualization_msgs/msg/MarkerArray' in types:
            _subscribe_marker()
        else:
    
            self.get_logger().warning(f"No publishers yet on {resolved}; will auto-detect...")
            def _retry_auto():
                types_map = dict(self.get_topic_names_and_types())
                types = types_map.get(resolved, [])
                if 'geometry_msgs/msg/PoseArray' in types:
                    _subscribe_pose()
                    self.auto_retry_timer.cancel()
                elif 'visualization_msgs/msg/MarkerArray' in types:
                    _subscribe_marker()
                    self.auto_retry_timer.cancel()
            self.auto_retry_timer = self.create_timer(2.0, _retry_auto, callback_group=self.cb_group)

        self.server = ActionServer(
            self,
            SlamSession,
            'slam/session',
            execute_callback=self.execute_callback,
            goal_callback = self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )
        self.get_logger().info(f"SlamAction ready. frontiers: {topic}")
        
    def _resolve_map_paths(self, result, pgm_dir: Path, yaml_dir: Path, name: str):
        rpath = getattr(result, "map_path", "") or ""
        stem = Path(rpath).stem if rpath else name
        pgm_path = pgm_dir / f"{stem}.pgm"
        yaml_path = yaml_dir / f"{stem}.yaml"
        return pgm_path, yaml_path

    def _patch_yaml_image(self, yaml_path: Path, pgm_path: Path):
        try:
            txt = yaml_path.read_text(encoding='utf-8')
        
            rel = os.path.relpath(str(pgm_path), start=str(yaml_path.parent))
        
            # new = re.sub(r'(^\s*image\s*:\s*).*$',
            #          rf'\1{rel}',
            #          txt, flags=re.MULTILINE)
            # if new != txt:
            #     yaml_path.write_text(new, encoding='utf-8')
            #     self.get_logger().info(f"Patched YAML image -> {rel}")
        except Exception as e:
            self.get_logger().warn(f"YAML image patch failed: {e}")
    def _call_save_map(self, base_path: str, fmt: str = "pgm", timeout_sec: float = 10.0):
        if not self.save_map_cli.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("save_map service not available")
            return None

        req = SaveMap.Request()

        def _assign_field(msg, field: str, value: str) -> bool:
            if not hasattr(msg, field):
                return False
            attr = getattr(msg, field)
            try:
                setattr(msg, field, value)
                return True
            except Exception:
                try:
                    if hasattr(attr, "data"):
                        attr.data = value
                        return True
                except Exception:
                    pass
            return False

        set_name = (
            _assign_field(req, "name", base_path) or
            _assign_field(req, "filename", base_path) or
            _assign_field(req, "map_file_name", base_path)
        )
        if not set_name:
            self.get_logger().error("SaveMap request has no known 'name' field (name/filename/map_file_name)")
            return False

        _assign_field(req, "format", fmt)

        fut = self.save_map_cli.call_async(req)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline and not fut.done():
            time.sleep(0.01)

        if not fut.done():
            self.get_logger().error("save_map service call timeout")
            return False

        try:
            res = fut.result()
        except Exception as e:
            self.get_logger().error(f"save_map service raised exception: {e}")
            return False

        if res is None:
            self.get_logger().error("save_map service returned None")
            return False

        ok = bool(getattr(res, "success", True))
        msg = getattr(res, "message", "")
        if msg:
            (self.get_logger().info if ok else self.get_logger().error)(f"save_map response: {msg}")
        return ok

    
    def _save_map_to_split_dirs(self, map_name: str) -> Optional[Tuple[str, str]]:
        
        stage_base = self.map_stage_dir / map_name
        ok = self._call_save_map(str(stage_base), fmt="pgm", timeout_sec=15.0)
        if not ok:
            return None

        src_pgm  = stage_base.with_suffix(".pgm")
        src_yaml = stage_base.with_suffix(".yaml")

        if not src_pgm.exists() or not src_yaml.exists():
            self.get_logger().error(f"saved files missing: {src_pgm} or {src_yaml}")
            return None

        dst_pgm  = self.map_dir_pgm  / f"{map_name}.pgm"
        dst_yaml = self.map_dir_yaml / f"{map_name}.yaml"

        try:
            if dst_pgm.exists():  dst_pgm.unlink()
            if dst_yaml.exists(): dst_yaml.unlink()

            shutil.move(str(src_pgm),  str(dst_pgm))
            shutil.move(str(src_yaml), str(dst_yaml))
            self._patch_yaml_image(Path(dst_yaml), Path(dst_pgm))
            
        except Exception as e:
            self.get_logger().error(f"moving files failed: {e}")
            for p in (src_pgm, src_yaml):
                if p.exists():
                    p.unlink(missing_ok=True)
            return None

        return str(dst_pgm), str(dst_yaml)
        
        
        
    def _get_robot_pose(self) -> Optional[PoseStamped]:

        global_frame = self.get_parameter('global_frame').value
        base_frame   = self.get_parameter('base_frame').value

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=global_frame,
                source_frame=base_frame,
                time=Time(),
                timeout=Duration(seconds=0.2)
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed ({global_frame}->{base_frame}): {e}")
            return None

        p = PoseStamped()
        p.header = tf.header
        p.header.frame_id = global_frame
        p.pose.position.x = tf.transform.translation.x
        p.pose.position.y = tf.transform.translation.y
        p.pose.position.z = tf.transform.translation.z
        p.pose.orientation = tf.transform.rotation
        return p
    
    
    def on_frontiers(self, msg):
        cnt = len(msg.poses)
        self.frontier_count = cnt
        self.last_msg_time = time.monotonic()
        
        if cnt == 0:
            if self.zero_since is None:
                self.zero_since = time.monotonic()
        else:
            self.zero_since = None
    def on_frontiers_marker(self, msg: MarkerArray):
        cnt = len(getattr(msg, "markers", []))
        self.frontier_count = cnt
        self.last_msg_time = time.monotonic()
        if cnt == 0:
            if self.zero_since is None:
                self.zero_since = time.monotonic()
        else:
            self.zero_since = None
                
    def goal_callback(self, requests: SlamSession.Goal):
        self.get_logger().info(f"SlamAction요청 탐지!")
        if not getattr(requests, 'map_name', None):
            self.get_logger().warn('SlamAction : Empty map_name')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        self.get_logger().warn("SlamAction: cancel요청 발생!")
        return CancelResponse.ACCEPT
        
    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        save_map = goal.save_map
        map_name = goal.map_name
        feedback = SlamSession.Feedback()
        result = SlamSession.Result()

        zero_hold_sec    = float(self.get_parameter('zero_hold_sec').value)
        feedback_period  = float(self.get_parameter('feedback_period').value)
        simulate_progress= bool(self.get_parameter('simulate_progress').value)
        no_msg_timeout   = float(self.get_parameter('no_msg_timeout_sec').value)

        try:
            startup_grace = float(self.get_parameter('startup_grace_sec').value)
        except Exception:
            startup_grace = 60.0
        try:
            no_msg_abort = float(self.get_parameter('no_msg_abort_sec').value)
        except Exception:
            no_msg_abort = max(180.0, no_msg_timeout)

        start_t = time.monotonic()
        self.last_msg_time = 0.0  

        progress = 0.0

        while rclpy.ok():
            now = time.monotonic()

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("SlamAction: 작업 취소됨")
                result.success = False
                result.map_path = ""
                result.message = "canceled"
                return result

        
            if now - start_t < startup_grace:
                pass  
            elif self.last_msg_time == 0.0 and (now - start_t) > no_msg_abort:
                self.get_logger().warning("no frontiers first message timeout")
                
                if save_map:
                    pair = self._save_map_to_split_dirs(map_name)
                    if pair is None:
                        goal_handle.abort()
                        result.success = False
                        result.map_path = ""
                        result.message = "timeout: save_map failed"
                        return result
                    final_pgm, final_yaml = pair
                    goal_handle.succeed()
                    result.success = True
                    result.map_path = final_pgm
                    result.message = "timeout: no frontiers first msg → map saved"
                    return result
                else:
                    goal_handle.succeed()
                    result.success = True
                    result.map_path = ""
                    result.message = "timeout: no frontiers first msg"
                    return result
            elif self.last_msg_time > 0.0 and (now - self.last_msg_time) > no_msg_abort:
                self.get_logger().warning("frontiers stalled timeout")
                if save_map:
                    pair = self._save_map_to_split_dirs(map_name)
                    if pair is None:
                        goal_handle.abort()
                        result.success = False
                        result.map_path = ""
                        result.message = "timeout(stalled): save_map failed"
                        return result
                    final_pgm, final_yaml = pair
                    goal_handle.succeed()
                    result.success = True
                    result.map_path = final_pgm
                    result.message = "timeout(stalled): map saved"
                    return result
                else:
                    goal_handle.succeed()
                    result.success = True
                    result.map_path = ""
                    result.message = "timeout(stalled)"
                    return result


            done, left = self._is_mapping_done(zero_hold_sec)

            if simulate_progress and progress < 1.0:
                progress = min(1.0, progress + 0.01)

            pose = self._get_robot_pose()
            if pose is not None:
                self.last_pose = pose
            feedback.pose = self.last_pose if self.last_pose is not None else PoseStamped()
            feedback.progress = progress
            feedback.quality = 0.0

        
            status_bits = []
            if now - start_t < startup_grace:
                status_bits.append("grace")
            status_bits.append(f"frontiers={self.frontier_count if self.frontier_count is not None else -1}")
            status_bits.append(f"zero_hold_left={max(0.0, left):.2f}s")
            feedback.status = " | ".join(status_bits)

            goal_handle.publish_feedback(feedback)

            if done:
                break

            time.sleep(feedback_period)

        if save_map:
            pair = self._save_map_to_split_dirs(map_name)
            if pair is None:
                goal_handle.abort()
                result.success = False
                result.message = "save_map failed"
                result.map_path = ""
                return result
            final_pgm, final_yaml = pair
            goal_handle.succeed()
            result.success = True
            result.message = "mapping done and map saved"
            result.map_path = final_pgm
            return result
        else:
            goal_handle.succeed()
            result.success = True
            result.message = "mapping done (save_map=false)"
            result.map_path = ""
            return result

            
    def _is_mapping_done(self, hold_sec: float):
        if self.frontier_count is None:
            return False, hold_sec
        if self.frontier_count != 0:
            return False, hold_sec
        if self.zero_since is None:
            return False, hold_sec
        elapsed = time.monotonic() - self.zero_since
        left = max(0.0, hold_sec - elapsed)
        return (elapsed >= hold_sec), left
        
def main():
    rclpy.init()
    node = SlamAction()
        
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        executor.add_node(node)
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

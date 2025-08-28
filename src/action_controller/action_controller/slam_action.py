import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from my_robot_interfaces.action import SlamSession

from geometry_msgs.msg import PoseArray, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration



class SlamAction(Node):
    def __init__(self):
        super().__init__('slam_action_server')
        
        self.declare_parameter('frontiers_topic', '/explore/frontiers')
        self.declare_parameter('zero_hold_sec', 5.0)
        self.declare_parameter('feedback_period', 0.2)
        self.declare_parameter('simulate_progress', False)
        self.declare_parameter('no_msg_timeout_sec', 10.0)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.last_pose: Optional[PoseStamped] = None
        self.cb_group = ReentrantCallbackGroup()
        
        self.frontier_count: Optional[int] = None
        self.zero_since: Optional[float] = None
        self.last_msg_time: float = time.monotonic()
    
        topic = self.get_parameter('frontiers_topic').get_parameter_value().string_value
        self.sub = self.create_subscription(
            PoseArray,              # MarkerArray로 바꿀 경우 여기 타입 변경
            topic,
            self.on_frontiers,
            10,
            callback_group=self.cb_group
        )
        
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
        
    def _get_robot_pose(self) -> Optional[PoseStamped]:

        global_frame = self.get_parameter('global_frame').value
        base_frame   = self.get_parameter('base_frame').value

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=global_frame,
                source_frame=base_frame,
                time=rclpy.time.Time())
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
                
    def goal_callback(self, requests: SlamSession.Goal()):
        self.get_logger().info(f"SlamAction요청 탐지! {requests.session_id}")
        if not getattr(requests, 'session_id', None):
            self.get_logger().warn('SlamAction : Empty session_id')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, requests):
        self.get_logger().warn("SlamAction: cancel요청 발생!")
        return CancelResponse.ACCEPT
        
    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        feedback = SlamSession.Feedback()
        result = SlamSession.Result()
        
        session_id = goal.session_id or 'unknown'
        zero_hold_sec = float(self.get_parameter('zero_hold_sec').value)
        feedback_period = float(self.get_parameter('feedback_period').value)
        simulate_progress = bool(self.get_parameter('simulate_progress').value)
        no_msg_timeout = float(self.get_parameter('no_msg_timeout_sec').value)
        
        self.get_logger().info(f"SlamAction: 실행 시작 (session_id={session_id}, zero_hold={zero_hold_sec}s)")

        progress = 0.0
        
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("SlamAction: 작업 취소됨")
                result.success = False
                result.map_path = ""
                result.message = "canceled"
                return result
            if time.monotonic() - self.last_msg_time > no_msg_timeout:
                self.get_logger().warn("종료로 판정")
                goal_handle.abort()
                result.success = False
                result.map_path = ""
                result.message = "no frontiers message timeout"
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
            feedback.status = f"frontiers={self.frontier_count if self.frontier_count is not None else -1}, zero_hold_left={max(0.0, left):.2f}s"
            goal_handle.publish_feedback(feedback)
            
            if done:
                break
            
            time.sleep(feedback_period)
            
        goal_handle.succeed()
        self.get_logger().info("SlamAction: frontiers 0 지속 -> 맵핑 완료로 판정")
        
        result.success =True
        result.map_path = f"/data/maps/{session_id}.pgm"
        
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
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

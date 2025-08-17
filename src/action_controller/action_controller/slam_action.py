import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from my_robot_interfaces.action import SlamSession

from geometry_msgs.msg import PoseArray


class SlamAction(Node):
    def __init__(self):
        super().__init__('slam_action_server')
        
        self.declare_parameter('frontiers_topic', '/explore/frontiers')
        self.declare_parameter('zero_hold_sec', 5.0)
        self.declare_parameter('feedback_period', 0.2)
        self.declare_parameter('simulate_progress', False)
        
        self.dc_group = ReentrantCallbackGroup()
        
        self.frontier_count: Optional[int] = None
        self.zero_since: Optional[float] = None
        
        topic = self.get_parameter('frontiers_topic').get_parameter_value().string_value
        self.sub = self.create_subscription(
            PoseArray,              # MarkerArray로 바꿀 경우 여기 타입 변경
            topic,
            self._on_frontiers,
            10,
            callback_group=self.cb_group
        )
        
        self.server = ActionServer(
            self,
            SlamSession,
            'slam/session',
            execute_callback=self.execute_callback,
            goal_callback = self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        self.get_logger().info(f"SlamAction ready. frontiers: {topic}")
        
    def on_frontiers(self, msg):
        cnt = len(msg.poses)
        
        self.frontier_count = cnt
        
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
    
    def calcel_callback(self, requests):
        self.get_logger().warn("SlamAction: cancel요청 발생!")
        return CancelResponse.ACCEPT
        
    def execute_callback(self, goal_handle):
        goal = goal_handle.requests
        feedback = SlamSession.Feedback()
        result = SlamSession.Result()
        
        session_id = goal.session_id or 'unknown'
        zero_hold_sec = float(self.get_parameter('zero_hold_sec').value)
        feedback_period = float(self.get_parameter('feedback_period').value)
        simulate_progress = bool(self.get_parameter('simulate_progress').value)
        
        self.get_logger().info(f"SlamAction: 실행 시작 (session_id={session_id}, zero_hold={zero_hold_sec}s)")

        progress = 0.0
        
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("SlamAction: 작업 취소됨")
                result.success = False
                result.map_path = ""
                return result
            
            done, left = self._is_mapping_done(zero_hold_sec)
            
            if simulate_progress and progress < 1.0:
                progress = main(1.0, progress + 0.01)
            
            feedback.progress = progress
            fc = self.frontier_count if self.frontier_count is not None else -1
            feedback.status = f"frontiers={fc},zero_hold_left={max(0.0, left):.2f}s"
            goal_handle.publish_feedback(feedback);
            
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
        # 멀티스레드로 돌려야 액션 실행 중에도 서브스크립션 콜백이 잘 돈다
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

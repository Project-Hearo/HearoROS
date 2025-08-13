import rclpy, time, math
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
class RobotLocation(Node):
    def __init__(self):
        super().__init__('robot_location')
        self.publisher = self.create_publisher(PointStamped, '/robot/location', 10)
        self.publish_period = 0.5
        self.last = 0.0
        
        self.subscriber = self.create_subscription(Odometry, "/odom_raw", self.callback, 10)
    def callback(self,msg: Odometry):
        now = time.time()
        if now - self.last < self.publish_period:
            return
        self.last = now
        
        point = msg.pose.pose.position
        
        response = PointStamped()
        response.header = msg.header
        response.point.x = point.x
        response.point.y = point.y
        response.point.z = point.z
        self.publisher.publish(response)
        
def main():
    rclpy.init()
    node = RobotLocation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
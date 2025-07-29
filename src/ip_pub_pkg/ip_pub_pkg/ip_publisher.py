import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess

class IPPublisher(Node):
    def __init__(self):
        super().__init__('ip_publisher')
        self.pub = self.create_publiser(String, '/rpi5_ip', 10)
        self.timer = self.create_timer(2.0, self.timer_cb)
        
    def pick_ip_from_hostnameI(self):
        try:
            out = subprocess.check_output("hostname -I", shell=True, text=True).strip().split()
            ignore_prefixes = ('127.', '172.17.')
            cand = [ip for ip in out if ip and not ip.startswith(ignore_prefixes)]
            pref = (lamda ip:
                    (0 if ip.startswith('10.42.') else
                     1 if ip.startswith('192.168.') else
                     2 if ip.startswith('10.')else
                     3))
            cand.sort(key=pref)
            return cand[0] if cand else "unknown"
        except Exception as e:
            self.get_logger().warn(f"hostname -I failed: {e}")
            return "unknows"
        
    def timer_cb(self):
        ip = self.pick_ip_from_hostnameI()
        self.pub.publish(String(data=ip))
        self.get_logger().info(f"? IP publish: {ip}")
        
def main(args=None):
    rclpy.init(args=args)
    node = IPPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
            
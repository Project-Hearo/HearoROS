from .base import CommandHandler
from .registry import register

from rclpy.action import ActionClient
from my_robot_interfaces.action import SlamSession
from geometry_msgs.msg import Pose2D

import requests
from pathlib import Path
import time

@register
class SlamStartHandler(CommandHandler):
    commands = ('slam/start',)
    
    def __init__(self, node):
        super().__init__(node)
        self.ac = ActionClient(node, SlamSession, 'slam/session')
        
        self.upload_url = getattr(node, "upload_url", "https://<주소>/<메서드>")
        self.upload_token = getattr(node, "upload_token", None)   # Bearer token 등
        self.map_dir = Path(getattr(node, "map_dir", "/data/maps"))  # SLAM 저장 경로
        
        self._goals = {}
        
    def handle(self, req_id, args):
        
        if not self.ac.wait_for_server(timeout_sec=2.0):
            self.node._publish_resp(req_id, ok=False,
                error={"code":"no_action","message":"slam/session not available"})
            return
        
        goal = SlamSession.Goal()
        goal.session_id = str(args.get('session_id', "default"))
        goal.save_map = bool(args.get('save_map', True))
        goal.map_name = str(args.get('map_name', goal.session_id))
        goal.duration_sec = float(args.get('duration_sec', 0.0))
        
        def fb_cb(fb):
            f = fb.feedback
            self.node._publish_feedback(req_id, {
                "pose": {"x": f.pose.x, "y": f.pose.y, "theta": f.pose.theta},
                "progress": float(f.progress),
                "quality": float(f.quality),
            })
            
        send_future = self.ac.send_goal_async(goal, feedback_callback=fb_cb)
        
        def on_goal_sent(fut):
            try:
                gh = fut.result()
            except Exception as e:
                self.node._publish_resp(req_id, ok=False,
                    error={"code":"send_goal_failed","message":str(e)})
                return

            if not gh.accepted:
                self.node._publish_resp(req_id, ok=False,
                    error={"code":"rejected","message":"goal rejected"})
                return
            res_future = gh.get_result_async()
            res_future.add_done_callback(lambda rf: self._on_result(req_id, goal, rf))
            
        send_future.add_done_callback(on_goal_sent)
        
    def _on_result(self, req_id, goal, res_future):
        try:
            res = res_future.result()
            if not res.success:
                self.node._publish_resp(req_id, ok=False,
                error={"code":"slam_failed","message":res.message})
                return
                
            pgm = self.map_dir / f"{goal.map_name}.pgm"
            yaml = self.map_dir / f"{goal.map_name}.yaml"
                
            self.node._publish_feedback(req_id, {
                "phase": "uploading",
                "files": [str(pgm), str(yaml)]
            })
                
            url = self.upload_url
            headers = {}
            if self.upload_token:
                headers["Authorization"] = f"Bearer {self.upload_token}"                
                
            meta = {
                "map_id": f"{goal.session_id}-{int(time.time())}",
                "robot_id": self.node.robot_id,
                "map_name": goal.map_name,
                "session_id": goal.session_id,
            }
                
            files = {}
            if pgm.exists():
                files["map_pgm"] = (pgm.name, open(pgm, "rb"), "image/x-portable-graymap")
            if yaml.exists():
                files["map_yaml"] = (yaml.name, open(yaml, "rb"), "application/x-yaml")
                    
            if not files:
                self.node._publish_resp(req_id, ok=False,
                error={"code":"missing_files","message":"pgm/yaml not found"})
                return

            r = requests.post(url, data=meta, files=files, headers=headers, timeout=120)
            for f in files.values():
                try: f[1].close()
                except Exception: pass
                
            r.raise_for_status()
            resp_json = {}
            try:
                resp_json = {}
            except Exception:
                pass
                
            self.node._publish_result(req_id, {
                "success": True,
                "message": "SLAM finished and uploaded",
                "map_path": getattr(res, "map_path", ""),
                "upload_response": resp_json
            })
        except Exception as e:
            self.node._publish_resp(req_id, ok=False,
                error={"code":"upload_failed","message":str(e)})




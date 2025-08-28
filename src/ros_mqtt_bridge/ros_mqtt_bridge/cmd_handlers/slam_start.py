from .base import CommandHandler
from .registry import register

from rclpy.action import ActionClient
from my_robot_interfaces.action import SlamSession
from geometry_msgs.msg import Pose2D
from my_robot_interfaces.srv import MapUpload

import requests
from pathlib import Path
import time

@register
class SlamStartHandler(CommandHandler):
    commands = ('slam/start',)
    
    def __init__(self, node):
        super().__init__(node)
        self.ac = ActionClient(node, SlamSession, 'slam/session')
        
        self.upload_url = getattr(node, "upload_url", "https://<주소>/<메서드>") #map을 어디에 올릴지 url이 필요하다.
        self.map_dir = Path(getattr(node, "map_dir", "/data/maps"))  #저장 경로 설정이 필요하다.
        self.upload_token = getattr(node ,"upload_token", None) #토큰은 아마 없어도 될 거 같긴하다
        
        self.upload_cli = self.node.create_client(MapUpload, '/map_uploader/upload')
        self._goals = {}
        
    def handle(self, req_id, args):
        args = args or {}
        
        if not self.ac.wait_for_server(timeout_sec=2.0):
            self.node._publish_resp(req_id, ok=False,
                error={"code":"no_action","message":"slam/session not available"})
            return
        
        goal = SlamSession.Goal()
        goal.session_id = str(args.get('session_id', "default"))
        goal.save_map = bool(args.get('save_map', True))
        goal.map_name = str(args.get('map_name', goal.session_id))
        goal.duration_sec = float(args.get('duration_sec', 0.0))
        
        def feedback_callback(fb):
            f = getattr(fb, "feedback", fb)
            self.node._publish_feedback(req_id, {
                "pose": {"x": f.pose.x, "y": f.pose.y, "theta": f.pose.theta},
                "progress": float(getattr(f, "progress", 0.0)),
                "quality": float(getattr(f, "quality", 0.0)),
            })
            
        send_future = self.ac.send_goal_async(goal, feedback_callback=feedback_callback)
        
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
            
            self.node._publish_ack(req_id, {"goal_accepted": True})
            
            res_future = gh.get_result_async()
            res_future.add_done_callback(lambda rf: self._on_result(req_id, goal, rf))
            
        send_future.add_done_callback(on_goal_sent)
        
    def _on_result(self, req_id, goal, res_future):
        try:
            response = res_future.result()
            result = getattr(response, "result", response)
            success = getattr(result, "success", False)
            message = getattr(result, "message", "")
            map_path = getattr(result, "map_path", "")
            
            if not success:
                self.node._publish_resp(req_id, ok=False,
                    error={"code":"slam_failed","message": message})
                return
            pgm = self.map_dir / f"{goal.map_name}.pgm"
            yaml = self.map_dir / f"{goal.map_name}.yaml"
            
            self.node._publish_feedback(req_id, {
                "phase": "uploading",
                "files": [str(pgm), str(yaml)]
            })
            
            if not self.upload_cli.wait_for_service(timeout_sec=2.0):
                self.node._publish_resp(req_id, ok=False,
                    error={"code":"upload_service_unavailable","message":"MapUploader not available"})
                return
            
            
            request = MapUpload.Request()
            request.upload_url = self.upload_url
            request.token = self.upload_token or ""
            request.map_name = goal.map_name
            request.pgm_path = str(pgm)
            request.yaml_path = str(yaml)
            
            future = self.upload_cli.call_async(request)
            def on_uploaded(_fut):
                try:
                    srv_res = _fut.result()  # MapUpload.Response
                except Exception as e:
                    self.node._publish_resp(req_id, ok=False,
                        error={"code":"upload_exception","message":str(e)})
                    return

                if not getattr(srv_res, "ok", False):
                    self.node._publish_resp(req_id, ok=False, error={
                        "code":    getattr(srv_res, "code", "UPLOAD_FAILED"),
                        "message": getattr(srv_res, "message", "upload failed"),
                        "detail":  getattr(srv_res, "upload_json", "")[:512],
                    })
                    return

                self.node._publish_result(req_id, data={
                    "success": True,
                    "message": getattr(srv_res, "message", "SLAM finished and uploaded"),
                    "map_path": map_path,
                    "upload_code": getattr(srv_res, "code", ""),
                    "upload_response": getattr(srv_res, "upload_json", ""),
                })

            future.add_done_callback(on_uploaded)

        except Exception as e:
            self.node._publish_resp(req_id, ok=False,
                error={"code":"result_error","message":str(e)})
            
            
           



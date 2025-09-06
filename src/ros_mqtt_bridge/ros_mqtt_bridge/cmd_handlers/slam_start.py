from .base import CommandHandler
from .registry import register

from rclpy.action import ActionClient
from my_robot_interfaces.action import SlamSession
from my_robot_interfaces.srv import MapUpload
from action_msgs.msg import GoalStatus
from ros_mqtt_bridge import config 
from pathlib import Path
import uuid, time

import os, asyncio
from launch import LaunchService
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


@register
class SlamStartHandler(CommandHandler):
    commands = ('slam/start',)

    def __init__(self, node):
        super().__init__(node)
        self.launch_service = None
        self.launch_task = None
        self.ac = ActionClient(node, SlamSession, 'slam/session')
        
        self.pgm_upload_url = config.pgm_upload_url or getattr(node, "pgm_upload_url", "")
        self.yaml_upload_url = config.yaml_upload_url or getattr(node, "yaml_upload_url", "")
        self.post_url = config.post_url or getattr(node, "post_url", "")
        self.upload_token = config.upload_token or getattr(node, "upload_token", "")

        self.map_dir_pgm = Path(getattr(node, "map_dir_pgm", "/root/maps/pgm"))
        self.map_dir_yaml = Path(getattr(node, "map_dir_yaml", "/root/maps/yaml"))
    
        self.upload_cli = self.node.create_client(MapUpload, '/map_uploader/upload')

        self._goals = {}

    def _resolve_map_paths(self, result, pgm_dir: Path, yaml_dir: Path, name: str):
        rpath = getattr(result, "map_path", "") or ""
        if rpath:
            p = Path(rpath)
            stem = p.stem
        else:
            stem = name

        pgm_path = pgm_dir / f"{stem}.pgm"
        yaml_path = yaml_dir / f"{stem}.yaml"
        return pgm_path, yaml_path

    def _wait_service_with_retries(self, client, retries=10, per_wait_sec=1.0, req_id=None):
        for _ in range(retries):
            if client.wait_for_service(timeout_sec=per_wait_sec):
                return True
            if req_id is not None:
                self.node._publish_feedback(req_id, {"phase": "waiting_upload_service"})
        return False
    async def _launch_stack(self):
        slam = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('slam_toolbox'),
                    'launch',
                    'online_async_launch.py'
                )
            ),
            launch_arguments={'use_sim_time': 'false'}.items()
        )

        nav2 = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('nav2_bringup'),
                    'launch',
                    'bringup_launch.py'
                )
            ),
            launch_arguments={'use_sim_time': 'false'}.items()
        )

        explore = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('explore_lite'),
                    'launch',
                    'explore.launch.py'
                )
            )
        )

        self.launch_service = LaunchService()
        for desc in [slam, nav2, explore]:
            self.launch_service.include_launch_description(desc)

        await self.launch_service.run_async()

    def handle(self, req_id, args):
        args = args or {}
        loop = asyncio.run_coroutine_threadsafe(self._launch_stack(), loop)
        self.launch_task = loop.create_task(self._launch_stack())

        if not self.ac.wait_for_server(timeout_sec=8.0):
            self.node._publish_resp(req_id, ok=False,
                error={"code": "no_action", "message": "slam/session not available"})
            return

        goal = SlamSession.Goal()
        goal.save_map = bool(args.get('save_map', True))
        sid  = str(uuid.uuid4())[:8]
        base = str(args.get('map_name', f"HearoMap-{time.strftime('%Y%m%d-%H%M%S')}-{sid}"))
        goal.map_name = base

        def feedback_callback(fb):
            f = getattr(fb, "feedback", fb)
            self.node._publish_feedback(req_id, {
                "pose": {"x": f.pose.x, "y": f.pose.y, "theta": f.pose.theta},
                "progress": float(getattr(f, "progress", 0.0)),
                "quality":  float(getattr(f, "quality",  0.0)),
            })

        send_future = self.ac.send_goal_async(goal, feedback_callback=feedback_callback)

        def on_goal_sent(fut):
            try:
                gh = fut.result()
            except Exception as e:
                self.node._publish_resp(req_id, ok=False,
                    error={"code": "send_goal_failed", "message": str(e)})
                return

            if not gh.accepted:
                self.node._publish_resp(req_id, ok=False,
                    error={"code": "rejected", "message": "goal rejected"})
                return

            self._goals[req_id] = gh
            self.node._publish_ack(req_id, {"goal_accepted": True})

            res_future = gh.get_result_async()
            res_future.add_done_callback(lambda rf: self._on_result(req_id, goal, rf))

        send_future.add_done_callback(on_goal_sent)

    def _on_result(self, req_id, goal, res_future):
        try:
            response = res_future.result()  
            status = getattr(response, "status", GoalStatus.STATUS_UNKNOWN)
            result = getattr(response, "result", None)
            
            success_flag = bool(getattr(result, "success", False)) if result is not None else False
            if status != GoalStatus.STATUS_SUCCEEDED or not success_flag:
                msg = getattr(result, "message", f"status={status}")
                self.node._publish_resp(req_id, ok=False,
                    error={"code": "slam_failed", "message": msg})
                return

            pgm, yaml = self._resolve_map_paths(result, self.map_dir_pgm, self.map_dir_yaml, goal.map_name)
            if not pgm.exists() or not yaml.exists():
                self.node._publish_resp(req_id, ok=False,
                    error={"code": "map_files_missing",
                           "message": f"missing map files: {pgm} or {yaml}"})
                return

            if not goal.save_map:
                self.node._publish_result(req_id, data={
                    "success": True,
                    "message": "SLAM finished (no upload)",
                    "map_files": [str(pgm), str(yaml)],
                })
                return

            self.node._publish_feedback(req_id, {
                "phase": "uploading",
                "files": [str(pgm), str(yaml)]
            })

            if not self._wait_service_with_retries(self.upload_cli, retries=10, per_wait_sec=1.0, req_id=req_id):
                self.node._publish_resp(req_id, ok=False,
                    error={"code": "upload_service_unavailable", "message": "MapUploader not available"})
                return

            request = MapUpload.Request()
            request.pgm_upload_url = self.pgm_upload_url
            request.yaml_upload_url = self.yaml_upload_url
            request.token      = self.upload_token
            request.map_name   = goal.map_name
            request.pgm_path   = str(pgm)
            request.yaml_path  = str(yaml)
            request.post_url = self.post_url

            future = self.upload_cli.call_async(request)

            def on_uploaded(_fut):
                try:
                    srv_res = _fut.result() 
                except Exception as e:
                    self.node._publish_resp(req_id, ok=False,
                        error={"code": "upload_exception", "message": str(e)})
                    return
                finally:
                    self._goals.pop(req_id, None)

                if not getattr(srv_res, "ok", False):
                    self.node._publish_resp(req_id, ok=False, error={
                        "code":    getattr(srv_res, "code", "UPLOAD_FAILED"),
                        "message": getattr(srv_res, "message", "upload failed"),
                        "detail":  (getattr(srv_res, "upload_json", "") or "")[:512],
                    })
                    return

                self.node._publish_result(req_id, data={
                    "success": True,
                    "message": getattr(srv_res, "message", "SLAM finished and uploaded"),
                    "map_files": [str(pgm), str(yaml)],
                    "upload_code": getattr(srv_res, "code", ""),
                    "upload_response": getattr(srv_res, "upload_json", ""),
                })
                if self.launch_service:
                    self.node.get_logger().info("Shutting down slam/nav2/explore...")
                    self.launch_service.shutdown()
                    self.launch_service = None

            future.add_done_callback(on_uploaded)


        except Exception as e:
            self.node._publish_resp(req_id, ok=False,
                error={"code": "result_error", "message": str(e)})
        finally:
            # 예외 경로에서도 GoalHandle 정리 시도
            self._goals.pop(req_id, None)

from .base import CommandHandler
from .registry import register

from rclpy.action import ActionClient
from my_robot_interfaces.action import SlamSession
from my_robot_interfaces.srv import MapUpload
from action_msgs.msg import GoalStatus
from ros_mqtt_bridge import config
from pathlib import Path
import uuid, time
import threading
import multiprocessing as mp
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

        # 런치 서비스: 매핑 스택 / 네비 스택 분리
        self.ls_mapping = None    # slam_toolbox + explore_lite + nav2(slam:=True)
        self.ls_nav = None        # nav2(map:=..., slam:=False)

        # 상태 플래그 (중복 기동 방지)
        self.is_mapping_running = False
        self.is_nav_running = False

        # SLAM 액션 클라이언트
        self.ac = ActionClient(node, SlamSession, 'slam/session')

        # 업로드·경로 설정
        self.pgm_upload_url = config.PGM_UPLOAD_URL or getattr(node, "pgm_upload_url", "")
        self.yaml_upload_url = config.YAML_UPLOAD_URL or getattr(node, "yaml_upload_url", "")
        self.post_url = config.POST_URL or getattr(node, "post_url", "")
        self.upload_token = config.MAP_UPLOAD_TOKEN or getattr(node, "upload_token", "")

        self.map_dir_pgm = Path(getattr(node, "map_dir_pgm", "/root/maps/pgm"))
        self.map_dir_yaml = Path(getattr(node, "map_dir_yaml", "/root/maps/yaml"))

        self.upload_cli = self.node.create_client(MapUpload, '/map_uploader/upload')
        self._goals = {}

    # -------------------------
    # 내부 유틸
    # -------------------------
    def _resolve_map_paths(self, result, pgm_dir: Path, yaml_dir: Path, name: str):
        rpath = getattr(result, "map_path", "") or ""
        stem = Path(rpath).stem if rpath else name
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

    def _run_async_in_process(self, coro_fn, *args, name: str = None):
        def _runner():
            asyncio.run(coro_fn(*args))
        proc = mp.Process(target=_runner, deamon=True, name=name or "launch-proc")
        
        proc.start()
        return proc

    # -------------------------
    # 런치 스택: 매핑(탐색)용
    #   - slam_toolbox + nav2(slam:=True) + explore_lite
    # -------------------------
    async def _launch_mapping_stack(self):
        if self.is_mapping_running:
            return
        self.is_mapping_running = True
        try:
            slam = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('slam_toolbox'),
                        'launch', 'online_async_launch.py'
                    )
                ),
                launch_arguments={'use_sim_time': 'false'}.items()
            )

            nav2_slam = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('nav2_bringup'),
                        'launch', 'bringup_launch.py'
                    )
                ),
                # SLAM 모드로 기동 (map 인자 없이, slam:=True)
                launch_arguments={'use_sim_time': 'false', 'slam': 'True'}.items()
            )

            explore = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('explore_lite'),
                        'launch', 'explore.launch.py'
                    )
                )
            )

            self.ls_mapping = LaunchService()
            for desc in (slam, nav2_slam, explore):
                self.ls_mapping.include_launch_description(desc)

            await self.ls_mapping.run_async()
        finally:
            # run_async()가 리턴되면 내려간 상태
            self.is_mapping_running = False
            self.ls_mapping = None

    # -------------------------
    # 런치 스택: 운용(로컬라이즈)용
    #   - nav2(map:=<yaml>, slam:=False)
    # -------------------------
    async def _launch_nav_with_map(self, yaml_path: str):
        if self.is_nav_running:
            return
        self.is_nav_running = True
        try:
            nav2_localize = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('nav2_bringup'),
                        'launch', 'bringup_launch.py'
                    )
                ),
                launch_arguments={
                    'use_sim_time': 'false',
                    'slam': 'False',
                    'map': yaml_path
                }.items()
            )

            self.ls_nav = LaunchService()
            self.ls_nav.include_launch_description(nav2_localize)
            await self.ls_nav.run_async()
        finally:
            self.is_nav_running = False
            self.ls_nav = None

    # -------------------------
    # 명령 처리
    # -------------------------
    def handle(self, req_id, args):
        args = args or {}

        # 매핑 스택이 안 떠 있으면 기동
        if not self.is_mapping_running and self.ls_mapping is None:
            self.node.get_logger().info("Starting MAPPING stack: slam_toolbox + nav2(slam) + explore ...")
            self._run_async_in_process(self._launch_mapping_stack, name="mapping-launch")

        # SLAM 액션 서버 체크
        if not self.ac.wait_for_server(timeout_sec=8.0):
            self.node._publish_resp(req_id, ok=False,
                error={"code": "no_action", "message": "slam/session not available"})
            return

        # Goal 준비
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

    # -------------------------
    # SLAM 결과 처리
    # -------------------------
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

            # 맵 파일 경로 결정 + 존재 확인
            pgm, yaml = self._resolve_map_paths(result, self.map_dir_pgm, self.map_dir_yaml, goal.map_name)
            if not pgm.exists() or not yaml.exists():
                self.node._publish_resp(req_id, ok=False, error={
                    "code": "map_files_missing",
                    "message": f"missing map files: {pgm} or {yaml}"
                })
                return

            if not goal.save_map:
                # 업로드 스킵 케이스: 결과만 알리고 매핑 스택은 유지(원하면 외부에서 stop)
                self.node._publish_result(req_id, data={
                    "success": True,
                    "message": "SLAM finished (no upload)",
                    "map_files": [str(pgm), str(yaml)],
                })
                return

            # 업로드 단계
            self.node._publish_feedback(req_id, {"phase": "uploading", "files": [str(pgm), str(yaml)]})

            if not self._wait_service_with_retries(self.upload_cli, retries=10, per_wait_sec=1.0, req_id=req_id):
                self.node._publish_resp(req_id, ok=False,
                    error={"code": "upload_service_unavailable", "message": "MapUploader not available"})
                return

            request = MapUpload.Request()
            request.pgm_upload_url = self.pgm_upload_url
            request.yaml_upload_url = self.yaml_upload_url
            request.token = self.upload_token
            request.map_name = goal.map_name
            request.pgm_path = str(pgm)
            request.yaml_path = str(yaml)
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
                        "code": getattr(srv_res, "code", "UPLOAD_FAILED"),
                        "message": getattr(srv_res, "message", "upload failed"),
                        "detail": (getattr(srv_res, "upload_json", "") or "")[:512],
                    })
                    return

                # 업로드 성공 응답
                self.node._publish_result(req_id, data={
                    "success": True,
                    "message": getattr(srv_res, "message", "SLAM finished and uploaded"),
                    "map_files": [str(pgm), str(yaml)],
                    "upload_code": getattr(srv_res, "code", ""),
                    "upload_response": getattr(srv_res, "upload_json", ""),
                })

                # === 전환: 매핑 스택 종료 → Nav2(맵기반) 기동 ===
                # 1) Nav2(맵) 기동
                self.node.get_logger().info(f"Starting NAV stack with map: {yaml}")
                self._run_async_in_process(self._launch_nav_with_map, str(yaml), name="nav-launch")

                # 2) 매핑 스택 종료 (slam/explore/nav2_slam)
                if self.ls_mapping:
                    self.node.get_logger().info("Shutting down MAPPING stack (slam/explore/nav2_slam)...")
                    try:
                        self.ls_mapping.shutdown()
                    except Exception:
                        pass  # 이미 내려갔거나 종료 중일 수 있음

            future.add_done_callback(on_uploaded)

        except Exception as e:
            self.node._publish_resp(req_id, ok=False,
                error={"code": "result_error", "message": str(e)})
        finally:
            self._goals.pop(req_id, None)

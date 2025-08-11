from .base import CommandHandler
from .base import CommandHandler
from .registry import register
from my_robot_interfaces.srv import RobotStatus

@register
class StatusGetHandler(CommandHandler):
    commands = ('status/get',)
    def __init__(self, node):
        super().__init__(node)
        self._cli = None
        
    def _ensure_client(self):
        if self._cli is None:
            self._cli = self.node.create_client(RobotStatus, 'status/get')
        if not self._cli.wait_for_service(Timeout_sec =0.2):
            return False
        return True
    
    def handle(self, req_id, args):
        if not self._ensure_client():
            self.node._publish_resp(req_id, ok=False,
                error={"code":"service_unavailable","message":"status/get not ready"})
            return
        
        sensor_type = args.get('sensor_type', 'both')
        if sensor_type not in ('battery','imu','both'):
            self.node._publish_resp(req_id, ok=False,
                                    error={"code":"bad_arg","message":"sensor_type"})
            return
        
        req = RobotStatus.Request(); 
        req.sensor_type = sensor_type
        fut = self.node.cli_status.call_async(req)

        def done(f):
            try:
                r = f.result()
                self.node._publish_result(req_id, {
                    "battery": int(r.battery), "imu_json": r.imu_json
                })
            except Exception as e:
                self.node._publish_resp(req_id, ok=False,
                                        error={"code":"service_failed","message":str(e)})
        fut.add_done_callback(done)
        to = float(args.get('timeout', 0))
        if to > 0:
            self.node._arm_timeout(req_id, fut, to)
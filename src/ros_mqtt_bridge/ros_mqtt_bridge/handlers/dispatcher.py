from importlib import import_module
import pkgutil
from ros_mqtt_bridge.handlers.registry import build_handlers

class Dispatcher:
    def __init__(self, node, *, threaded: bool = False, max_workers: int = 0):
        self.node = node
        self.handlers = {}
        self.pool = None
        if threaded and max_workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=max_workers)
    def _normalize(self, cmd: str) -> str:
        return cmd.replace('.', '/').strip('/').lower()
    
    def _iter_modules(self, pkg_name: str):
        pkg = import_module(pkg_name)
        for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + '.'):
            leaf = m.name.rsplit('.', 1)[-1]
            if not leaf.startswith('_'):
                yield m.name


    def load_plugins(self, pkg='ros_mqtt_bridge.handlers'):
        for mod in self._iter_modules(pkg):
            import_module(mod)
        raw = build_handlers(self.node)
        
        self.handlers = {self._normalize(k): v for k, v in raw.items()}
        self.node.get_logger().info(f"handlers loaded: {list(self.handlers.keys())}")

    def dispatch(self, req_id: str, cmd: str, args: dict):
        key = self._normalize(cmd)
        h = self.handlers.get(key)
        if not h:
            try:
                self.node.metrics["errors"] += 1  # 선택
            except Exception:
                pass
            self.node._publish_resp(req_id, ok=False,
                            error={"code":"unknown_cmd","message":key})
            return
        self.node._publish_ack(req_id)
        def run():
            try:
                h.handle(req_id, args)
            except Exception as e:
                self.node.get_logger().exception(f"handler '{key}' crashed")
                self.node._publish_resp(
                    req_id, ok=False,
                    error={"code": "handler_exception", "message": str(e)},
                    finalize=True
                )

        if self.pool:
            self.pool.submit(run)
        else:
            run()

    def shutdown(self, wait: bool = True):
        if self.pool:
            self.pool.shutdown(wait=wait)
            
            

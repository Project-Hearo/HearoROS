from importlib import import_module
import pkgutil
from ros_mqtt_bridge.telemetry_handlers.registry import build_telemetries
class TelemetryManager:
    def __init__(self, node):
        self.node = node
        self.telemetries = []
    
    def _iter_modules(self, pkg_name: str):
        pkg = import_module(pkg_name)
        for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + '.'):
            leaf = m.name.rsplit('.', 1)[-1]
            if not leaf.startswith('_'):
                yield m.name
                
    def load_plugins(self, pkg='ros_mqtt_bridge.telemetry_handlers'):
        for mod in self._iter_modules(pkg):
            import_module(mod)
        
        self.telemetries = build_telemetries(self.node)
        self.node.get_logger().info(
            f"telemetries loaded: {[getattr(t, 'name', t.__class__.__name__) for t in self.telemetries]}"
        )
        
    def start_all(self):
        for t in self.telemetries:
            t.start()
            
    def stop_all(self):
        for t in self.telemetries:
            t.stop()

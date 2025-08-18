from abc import ABC, abstractmethod
import time

class TelemetryHandler(ABC):
    def __init__(self, node, *, rate_hz: float = 2.0):
        self.node = node
        self.rate_sec = 1.0 /rate_hz if rate_hz > 0 else 0.0
        self.last = 0.0
        self.sub = None
        self.running = False
        
    def tick(self) -> bool:
        if self.rate_sec <= 0: return True
        now = time.time()
        if now - self.last >= self.rate_sec:
            self.last = now
            return True
        return False
    
    @abstractmethod
    def start(self):...
    @abstractmethod
    def stop(self):...
    
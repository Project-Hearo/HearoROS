
from .registry import register
from .base import CommandHandler

@register
class SlamStartHandler(CommandHandler):
    commands = ('location',)
    
    def __init__(self, node, rate_hz: float = 5.0):
        super().__init__(node)
        self.rate_sec = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._last_timestamp = 0.0
        self.sub = None
        self.running=False
        
        


import time
from .base import CommandHandler
from .registry import register

@register
class PingHandler(CommandHandler):
    commands = ('ping',)
    def handle(self, req_id, args):
        self.node._publish_result(req_id, {"pong": True, "t": time.time()})
        
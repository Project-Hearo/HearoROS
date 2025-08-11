from abc import ABC, abstractmethod

class CommandHandler(ABC):
    commands: tuple[str, ...] = ()
    
    def __init__(self, node):
        self.node = node
    
    @abstractmethod
    def handle(self, req_id: str, args: dict) -> None:
        ...
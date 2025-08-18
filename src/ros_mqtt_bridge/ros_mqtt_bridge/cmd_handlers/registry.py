from .base import CommandHandler

_registry: dict[str, type[CommandHandler]] = {}

def register(cls: type[CommandHandler]):
    for cmd in cls.commands:
        if cmd in _registry:
            raise RuntimeError(f"duplicate handler for {cmd}")
        _registry[cmd] = cls
    return cls

def build_handlers(node):
    return {cmd: cls(node) for cmd, cls in _registry.items()}


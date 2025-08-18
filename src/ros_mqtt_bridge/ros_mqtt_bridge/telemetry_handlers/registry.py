from .base import TelemetryHandler

_classes: list[type[TelemetryHandler]] = []

def register_telemetry(cls: type[TelemetryHandler]):
    _classes.append(cls)
    return cls

def build_telemetries(node):
    return [cls(node) for cls in _classes]
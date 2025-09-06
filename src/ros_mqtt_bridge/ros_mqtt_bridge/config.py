import os
from dotenv import load_dotenv
from pathlib import Path

_here = Path(__file__).resolve()
for p in [_here, *_here.parents]:
    cand = p / ".env"
    if cand.exists():
        load_dotenv(cand)
        break


ROBOT_ID = os.getenv("ROBOT_ID","")
MQTT_SERVER = os.getenv("MQTT_SERVER", "")
MQTT_PORT = os.getenv("MQTT_PORT", "")
KEEPALIVE = os.getenv("KEEPALIVE", "")
MQTT_SOCKET_PORT = os.getenv("MQTT_SOCKET_PORT", "")
PGM_UPLOAD_URL   = os.getenv("PGM_UPLOAD_URL", "")
YAML_UPLOAD_URL  = os.getenv("YAML_UPLOAD_URL", "")
POST_URL         = os.getenv("POST_URL", "")
MAP_UPLOAD_TOKEN = os.getenv("MAP_UPLOAD_TOKEN", "")

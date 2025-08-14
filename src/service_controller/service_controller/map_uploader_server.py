import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
import requests

from my_robot_interfaces.srv import MapUpload  


class MapUploader(Node):
    def __init__(self):
        super().__init__('map_uploader')

        self.declare_parameter('connect_timeout', 10.0)
        self.declare_parameter('read_timeout', 120.0)
        self.declare_parameter('verify_tls', True)
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('backoff_initial', 1.0)

        self.srv = self.create_service(MapUpload, '/map_uploader/upload', self.on_request)
        self.get_logger().info("MapUploader ready: /map_uploader/upload")

    # ---- helpers ----
    def _timeouts(self):
        return (
            float(self.get_parameter('connect_timeout').value),
            float(self.get_parameter('read_timeout').value),
        )

    def _verify(self):
        return bool(self.get_parameter('verify_tls').value)

    def _post_with_retries(self, url, headers, files, data):
        max_retries = int(self.get_parameter('max_retries').value)
        backoff = float(self.get_parameter('backoff_initial').value)
        connect_timeout, read_timeout = self._timeouts()
        verify = self._verify()

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, files=files, data=data,
                    timeout=(connect_timeout, read_timeout), verify=verify
                )
                return resp
            except requests.RequestException as e:
                last_exc = e
                self.get_logger().warn(
                    f"[{attempt}/{max_retries}] upload failed: {e} -> retry in {backoff:.1f}s"
                )
                time.sleep(backoff)
                backoff *= 2
        if last_exc:
            raise last_exc
    def on_request(self, req: MapUpload.Request, res: MapUpload.Response):
        res.ok = False
        res.code = ""
        res.message = ""
        res.upload_json = ""

        if not req.upload_url:
            res.code = "INVALID_URL"
            res.message = "upload_url이 비어있다"
            self.get_logger().error(res.message)
            return res

        pgm_path = Path(req.pgm_path) if req.pgm_path else None
        yaml_path = Path(req.yaml_path) if req.yaml_path else None

        if not (pgm_path and pgm_path.exists()):
            res.code = "FILE_NOT_FOUND"
            res.message = f"pgm_path를 찾을 수 없다: {pgm_path}"
            self.get_logger().error(res.message)
            return res

        if not (yaml_path and yaml_path.exists()):
            res.code = "FILE_NOT_FOUND"
            res.message = f"yaml_path를 찾을 수 없다: {yaml_path}"
            self.get_logger().error(res.message)
            return res

        headers = {}
        if req.token:
            headers["Authorization"] = f"Bearer {req.token}"

        meta = {
            "map_name": req.map_name or "",
            "timestamp": int(time.time() * 1000),
        }

        files = {
            "map_image": (pgm_path.name, open(pgm_path, "rb"),
                          "image/x-portable-graymap" if pgm_path.suffix.lower() == ".pgm" else "image/png"),
            "map_yaml": (yaml_path.name, open(yaml_path, "rb"), "text/yaml"),
        }
        data = {
            "meta": json.dumps(meta, ensure_ascii=False),
        }

        try:
            resp = self._post_with_retries(req.upload_url, headers, files, data)
        except Exception as e:
            for f in files.values():
                try: f[1].close()
                except: pass
            res.code = "UPLOAD_EXCEPTION"
            res.message = f"예외 발생: {e}"
            self.get_logger().error(res.message)
            return res
        finally:
            for f in files.values():
                try: f[1].close()
                except: pass
        res.code = str(resp.status_code)
        res.ok = 200 <= resp.status_code < 300
        try:
            body = resp.json()
            res.upload_json = json.dumps(body, ensure_ascii=False)
            if isinstance(body, dict):
                res.message = str(body.get("message", ""))
        except Exception:
            res.upload_json = resp.text[:4096] 
        if not res.message:
            res.message = f"HTTP {resp.status_code}"

        log_fn = self.get_logger().info if res.ok else self.get_logger().error
        log_fn(f"upload result ok={res.ok} code={res.code} msg={res.message}")

        return res


def main():
    rclpy.init()
    node = MapUploader()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

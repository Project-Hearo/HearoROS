import json, time, re
from pathlib import Path
from typing import Tuple, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import requests
import rclpy
from rclpy.node import Node

from my_robot_interfaces.srv import MapUpload


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    m = re.match(r'^s3://([^/]+)/?(.*)$', s3_uri.strip())
    if not m:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = m.group(1)
    prefix = m.group(2) or ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


class MapUploader(Node):
    def __init__(self):
        super().__init__('map_uploader')

        self.declare_parameter('connect_timeout', 10.0)
        self.declare_parameter('read_timeout', 120.0)
        self.declare_parameter('verify_tls', True)
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('backoff_initial', 1.0)
        self.declare_parameter('use_map_name_as_filename', True)

        self.srv = self.create_service(MapUpload, '/map_uploader/upload', self.on_request)
        self.get_logger().info("MapUploader ready: /map_uploader/upload")

    # ---- helpers ----
    def _timeouts(self) -> Tuple[float, float]:
        return (
            float(self.get_parameter('connect_timeout').value),
            float(self.get_parameter('read_timeout').value),
        )

    def _verify(self) -> bool:
        return bool(self.get_parameter('verify_tls').value)

    def _guess_content_type(self, p: Path) -> str:
        ext = p.suffix.lower()
        if ext == ".pgm":
            return "image/x-portable-graymap"
        if ext in (".yaml", ".yml"):
            return "text/yaml"
        return "application/octet-stream"

    def _put_with_retries(self, url: str, data_stream, content_type: str):
        max_retries = int(self.get_parameter('max_retries').value)
        backoff = float(self.get_parameter('backoff_initial').value)
        connect_timeout, read_timeout = self._timeouts()
        verify = self._verify()
        headers = {'Content-Type': content_type} if content_type else {}
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.put(
                    url,
                    data=data_stream,
                    headers=headers,
                    timeout=(connect_timeout, read_timeout),
                    verify=verify
                )
                return resp
            except requests.RequestException as e:
                last_exc = e
                self.get_logger().warn(
                    f"[{attempt}/{max_retries}] PUT failed: {e} -> retry in {backoff:.1f}s"
                )
                sleep_s = backoff * (1.0 + 0.1 * (attempt % 3))
                time.sleep(sleep_s)
                backoff *= 2
        if last_exc:
            raise last_exc

    def _post_multipart_with_retries(
        self,
        url: str,
        pgm_path: Path,
        yaml_path: Path,
        map_name: Optional[str],
        token: Optional[str]
    ):
        max_retries = int(self.get_parameter('max_retries').value)
        backoff = float(self.get_parameter('backoff_initial').value)
        connect_timeout, read_timeout = self._timeouts()
        verify = self._verify()

        headers = {}
        if token:
            headers['Authorization'] = f"Bearer {token}"

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                with open(pgm_path, 'rb') as f_pgm, open(yaml_path, 'rb') as f_yaml:
                    files = {
                        'pgm': (pgm_path.name, f_pgm, self._guess_content_type(pgm_path)),
                        'yaml': (yaml_path.name, f_yaml, self._guess_content_type(yaml_path)),
                    }
                    data = {'map_name': map_name or ''}
                    resp = requests.post(
                        url,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=(connect_timeout, read_timeout),
                        verify=verify
                    )
                    return resp
            except requests.RequestException as e:
                last_exc = e
                self.get_logger().warn(f"[{attempt}/{max_retries}] POST failed: {e} -> retry {backoff:.1f}s")
                time.sleep(backoff)
                backoff *= 2
        if last_exc:
            raise last_exc

    def _upload_to_s3(self, bucket, key, local_path: Path, content_type: str):
        s3 = boto3.client('s3')
        extra = {"ContentType": content_type} if content_type else None
        try:
            if extra:
                s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra)
            else:
                s3.upload_file(str(local_path), bucket, key)
            self.get_logger().info(f"S3 uploaded s3://{bucket}/{key}")
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"S3 upload failed for s3://{bucket}/{key}: {e}")

    def _build_s3_keys(self, prefix: str, pgm_path: Path, yaml_path: Path, map_name: Optional[str], use_map_name: bool):
        if use_map_name and map_name:
            pgm_key = f"{prefix}{map_name}.pgm"
            yaml_key = f"{prefix}{map_name}.yaml"
        else:
            pgm_key = f"{prefix}{pgm_path.name}"
            yaml_key = f"{prefix}{yaml_path.name}"
        return pgm_key, yaml_key

    def on_request(self, req: MapUpload.Request, res: MapUpload.Response):
        res.ok = False
        res.code = ""
        res.message = ""
        res.upload_json = ""

        pgm_path = Path(req.pgm_path) if req.pgm_path else None
        yaml_path = Path(req.yaml_path) if req.yaml_path else None
        if not (pgm_path and pgm_path.exists()):
            res.code, res.message = "FILE_NOT_FOUND", f"pgm_path를 찾을 수 없음: {pgm_path}"
            self.get_logger().error(res.message); return res
        if not (yaml_path and yaml_path.exists()):
            res.code, res.message = "FILE_NOT_FOUND", f"yaml_path를 찾을 수 없음: {yaml_path}"
            self.get_logger().error(res.message); return res

    
        pgm_url = getattr(req, 'pgm_upload_url', '') or ''
        yaml_url = getattr(req, 'yaml_upload_url', '') or ''
        pgm_is_s3 = pgm_url.startswith("s3://")
        yaml_is_s3 = yaml_url.startswith("s3://")

        try:
            if pgm_is_s3 and yaml_is_s3:
            # s3 직접 업로드
                bucket_pgm, prefix_pgm = _parse_s3_uri(pgm_url)
                bucket_yaml, prefix_yaml = _parse_s3_uri(yaml_url)
                use_map_name = bool(self.get_parameter('use_map_name_as_filename').value)
                pgm_key, _ = self._build_s3_keys(prefix_pgm, pgm_path, yaml_path, req.map_name, use_map_name)
                _, yaml_key = self._build_s3_keys(prefix_yaml, pgm_path, yaml_path, req.map_name, use_map_name)

                self._upload_to_s3(bucket_pgm, pgm_key, pgm_path, self._guess_content_type(pgm_path))
                self._upload_to_s3(bucket_yaml, yaml_key, yaml_path, self._guess_content_type(yaml_path))

                resp_pgm_status, resp_yaml_status = 200, 200
                resp_pgm_reason, resp_yaml_reason = "OK(S3)", "OK(S3)"

            else:
                # presigned PUT
                with open(pgm_path, "rb") as f_pgm:
                    resp_pgm = self._put_with_retries(pgm_url, f_pgm, self._guess_content_type(pgm_path))
                with open(yaml_path, "rb") as f_yaml:
                    resp_yaml = self._put_with_retries(yaml_url, f_yaml, self._guess_content_type(yaml_path))

                resp_pgm_status, resp_yaml_status = resp_pgm.status_code, resp_yaml.status_code
                resp_pgm_reason, resp_yaml_reason = resp_pgm.reason, resp_yaml.reason

        except Exception as e:
            res.code, res.message = "UPLOAD_EXCEPTION", f"S3 업로드 예외: {e}"
            self.get_logger().error(res.message); return res

        ok_pgm = 200 <= resp_pgm_status < 300
        ok_yaml = 200 <= resp_yaml_status < 300
        if not (ok_pgm and ok_yaml):
            res.code = f"PGM:{resp_pgm_status},YAML:{resp_yaml_status}"
            res.message = "S3 업로드 실패"
            return res

    # === 2단계: post_url 전송 ===
        post_url = getattr(req, 'post_url', '') or getattr(req, 'pose_url', '') or ''
        post_status, post_reason, post_preview = None, None, None
        if post_url.strip():
            try:
                resp = self._post_multipart_with_retries(
                    post_url.strip(), pgm_path, yaml_path, req.map_name, req.token
                )
                post_status, post_reason = resp.status_code, resp.reason
                post_preview = resp.text[:500] if resp.text else ""
            except Exception as e:
                post_status, post_reason = 500, f"POST 예외: {e}"
                self.get_logger().error(f"[multipart_post] 실패: {e}")

    # === 응답 구성 ===
        res.ok = (200 <= resp_pgm_status < 300) and (200 <= resp_yaml_status < 300) and \
             (post_status is None or 200 <= post_status < 300)
        res.code = f"PGM:{resp_pgm_status},YAML:{resp_yaml_status},POST:{post_status or '-'}"
        res.message = "OK" if res.ok else "Upload or POST failed"

        body = {
            "map_name": req.map_name or "",
            "pgm": {"status": resp_pgm_status, "reason": resp_pgm_reason},
            "yaml": {"status": resp_yaml_status, "reason": resp_yaml_reason},
            "post": {"status": post_status, "reason": post_reason, "preview": post_preview} if post_status else None,
            "ts": int(time.time() * 1000)
        }
        res.upload_json = json.dumps(body, ensure_ascii=False)

        log_fn = self.get_logger().info if res.ok else self.get_logger().error
        log_fn(f"최종 결과 ok={res.ok} code={res.code} msg={res.message}")
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

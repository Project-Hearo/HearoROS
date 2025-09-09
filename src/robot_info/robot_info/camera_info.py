#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2 as cv
import rclpy
from rclpy.node import Node

def fourcc_to_str(v: float) -> str:
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

class CameraInfo(Node):
    def __init__(self,
                 device_index: int = 0,       # /dev/video0
                 want_mjpg: bool = True,
                 window_name: str = "MJPG Preview"):
        super().__init__('camera_info')

        self.window_name = window_name
        self.cap = cv.VideoCapture(device_index, cv.CAP_V4L2)  # V4L2 백엔드 권장

        # 1) MJPG 강제 (가능하면)
        if want_mjpg:
            self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # 2) “최고 FPS 우선” 모드 시도 (v4l2-ctl 결과 기반)
        #    원하는 순서로 시도하고, 적용 여부는 get()으로 검증
        candidate_modes = [
            (640,  480, 120),
            (320,  240, 120),
            (1280, 720,  60),
            (800,  600,  60),
            (1920,1080,  30),
            (1024, 768,  30),
            (1280,1024,  30),
        ]

        applied = False
        for w, h, fps in candidate_modes:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH,  w)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv.CAP_PROP_FPS,          fps)

            # 적용값 읽어서 매치 여부 확인
            rw  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            rh  = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            rfs = round(self.cap.get(cv.CAP_PROP_FPS))  # 일부 드라이버는 0/엉뚱값 반환하기도 함
            fcc = fourcc_to_str(self.cap.get(cv.CAP_PROP_FOURCC))

            if rw == w and rh == h and (rfs == fps or rfs == 0):
                # FPS get()이 0으로 나오는 드라이버도 있어 해상도 매치면 통과로 봄
                applied = True
                self.get_logger().info(f"Applied mode: {rw}x{rh}@{rfs if rfs>0 else fps} FOURCC={fcc}")
                break

        if not applied:
            # 마지막 시도값 출력
            rw  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            rh  = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            rfs = round(self.cap.get(cv.CAP_PROP_FPS))
            fcc = fourcc_to_str(self.cap.get(cv.CAP_PROP_FOURCC))
            self.get_logger().warn(f"Could not force preferred modes. Current: {rw}x{rh}@{rfs} FOURCC={fcc}")

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera device.")
            raise SystemExit(1)

        # rclpy 타이머로 폴링(소형 주기). GUI 루프와 호환 위해 0.0 보다는 짧은 주기 사용.
        self.prev_t = time.time()
        self.timer = self.create_timer(0.0, self.loop_once)  # 가능한 한 빠르게

    def loop_once(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Failed to read frame")
            return

        # FPS 계산
        now = time.time()
        fps = 1.0 / max(1e-6, (now - self.prev_t))
        self.prev_t = now

        # 오버레이
        cv.putText(frame, f"FPS: {int(fps)}", (16, 32),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv.LINE_AA)

        # 미리보기
        cv.imshow(self.window_name, frame)
        # GUI 갱신 & 종료키
        if cv.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        # 자원 정리
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        finally:
            cv.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = CameraInfo(device_index=0, want_mjpg=True)
    try:
        # spin 대신 수동 루프: OpenCV GUI와 충돌 줄이기
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.destroy_node()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge

import os
import time
import cv2
import numpy as np
import tensorflow as tf


class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector_node')

        # ───────────────────── Params ─────────────────────
        pkg_share = get_package_share_directory('robot_info')
        default_model_path = os.path.join(pkg_share, 'models', 'Yolo-X_w8a8.tflite')

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('confidence_threshold', 0.5)   # 권장: 0.5~0.6
        self.declare_parameter('nms_threshold', 0.5)
        self.declare_parameter('input_topic', '/stream_image_raw')
        self.declare_parameter('detection_topic', '/person_detector/detection')
        self.declare_parameter('result_image_topic', '/person_detector/image_result')
        self.declare_parameter('min_box_area', 300.0)         # 너무 작은 잡음 박스 제거(px^2)

        self.add_on_set_parameters_callback(self._on_params)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_th = float(self.get_parameter('confidence_threshold').value)
        self.nms_th = float(self.get_parameter('nms_threshold').value)
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.det_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.res_topic = self.get_parameter('result_image_topic').get_parameter_value().string_value
        self.min_box_area = float(self.get_parameter('min_box_area').value)

        # ───────────────────── I/O ─────────────────────
        self.bridge = CvBridge()
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.image_sub = self.create_subscription(Image, input_topic, self.image_callback, sensor_qos)
        self.det_pub = self.create_publisher(PolygonStamped, self.det_topic, 10)
        self.img_pub = self.create_publisher(Image, self.res_topic, 10)

        # ───────────────────── Model ─────────────────────
        if not os.path.exists(model_path):
            self.get_logger().error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            rclpy.shutdown()
            return

        # 스레드 튜닝 가능: num_threads=4 등
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.in_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()

        # 입력 텐서 크기/타입
        self.in_h = int(self.in_details[0]['shape'][1])
        self.in_w = int(self.in_details[0]['shape'][2])
        self.in_dtype = self.in_details[0]['dtype']  # np.uint8 or np.float32

        self.get_logger().info(f"모델 로드 완료: {model_path}")
        self.get_logger().info(f"입력: {self.in_w}x{self.in_h}, dtype={self.in_dtype}")
        self.get_logger().info(f"출력 텐서: {self.out_details}")

        self.color = (0, 255, 0)
        self.prev_time = 0.0

    # 파라미터 런타임 변경
    def _on_params(self, params):
        for p in params:
            if p.name == 'confidence_threshold':
                self.conf_th = float(p.value)
                self.get_logger().info(f"confidence_threshold = {self.conf_th:.2f}")
            elif p.name == 'nms_threshold':
                self.nms_th = float(p.value)
                self.get_logger().info(f"nms_threshold = {self.nms_th:.2f}")
            elif p.name == 'min_box_area':
                self.min_box_area = float(p.value)
                self.get_logger().info(f"min_box_area = {self.min_box_area:.1f}")
        return SetParametersResult(successful=True)

    # ================ Letterbox helpers ================
    def _letterbox(self, img, new_shape, color=(114, 114, 114), scaleup=True):
        """비율 유지 리사이즈 + 패딩. 반환: letterboxed_img, r, (pad_left, pad_top), (h0, w0)"""
        h0, w0 = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        new_h, new_w = new_shape

        r = min(new_w / w0, new_h / h0)
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw = new_w - new_unpad[0]
        dh = new_h - new_unpad[1]
        dw *= 0.5
        dh *= 0.5

        if (w0, h0) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return img, r, (left, top), (h0, w0)

    @staticmethod
    def _clip_box(x1, y1, x2, y2, w, h):
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return x1, y1, x2, y2

    # ================ ROS callback ================
    def image_callback(self, msg: Image):
        # 1) ROS Image → OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV-Bridge 변환 실패: {e}")
            return

        # 2) 전처리 (LETTERBOX)
        inp, meta = self._preprocess(frame)

        # 3) 추론
        try:
            self.interpreter.set_tensor(self.in_details[0]['index'], inp)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(d['index']) for d in self.out_details]
        except Exception as e:
            self.get_logger().warn(f"TFLite invoke 실패: {e}")
            return

        # 4) 후처리 (+ 원본 좌표 복원)
        boxes, scores, class_ids = self._postprocess(outputs, meta)

        # 5) 첫 번째 박스만 Polygon으로 퍼블리시(원하면 for 루프 돌리면 됨)
        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            poly = PolygonStamped()
            poly.header = msg.header
            poly.polygon.points = [
                Point32(x=float(x1), y=float(y1), z=0.0),
                Point32(x=float(x2), y=float(y1), z=0.0),
                Point32(x=float(x2), y=float(y2), z=0.0),
                Point32(x=float(x1), y=float(y2), z=0.0),
            ]
            self.det_pub.publish(poly)

        # 6) 결과 이미지 퍼블리시
        vis = frame.copy()
        self._draw(vis, boxes, scores, class_ids)
        t = time.time()
        if self.prev_time > 0:
            fps = 1.0 / max(1e-6, (t - self.prev_time))
            cv2.putText(vis, f"INF: {fps:.1f} fps", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.color, 2)
        self.prev_time = t

        try:
            out_msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            out_msg.header = msg.header
            self.img_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"결과 이미지 발행 실패: {e}")

    # ================ Inference utils ================
    def _preprocess(self, bgr):
        # 320x240 → (in_w,in_h)로 letterbox (비율 유지 + 패딩 기록)
        letter, r, (dl, dt), (h0, w0) = self._letterbox(
            bgr, (self.in_h, self.in_w), color=(114, 114, 114), scaleup=True
        )
        rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)

        if self.in_dtype == np.uint8:
            inp = np.expand_dims(rgb.astype(np.uint8), axis=0)
        else:
            inp = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

        meta = {
            'r': r, 'dl': dl, 'dt': dt,
            'h0': h0, 'w0': w0,
            'ih': self.in_h, 'iw': self.in_w
        }
        return inp, meta

    def _postprocess(self, outputs, meta):
        try:
            if len(outputs) < 3:
                return [], [], []

            boxes = np.squeeze(outputs[0])   # (N,4) [x1,y1,x2,y2] on model input
            scores = np.squeeze(outputs[1])  # (N,)
            classes = np.squeeze(outputs[2]) # (N,)

            if boxes.ndim == 1:
                boxes = boxes.reshape(-1, 4)
            if scores.ndim == 0:
                scores = scores.reshape(-1)
            if classes.ndim == 0:
                classes = classes.reshape(-1)

            # 좌표 정규화(0~1) 모델 대응: 최댓값이 1 근처면 입력 크기로 스케일
            if boxes.size and float(np.max(boxes)) <= 1.5:
                boxes[:, [0, 2]] *= meta['iw']
                boxes[:, [1, 3]] *= meta['ih']

            # confidence 필터
            m = scores >= self.conf_th
            boxes, scores, classes = boxes[m], scores[m], classes[m]
            if boxes.size == 0:
                return [], [], []

            # 사람(0)만
            m2 = classes.astype(np.int32) == 0
            boxes, scores, classes = boxes[m2], scores[m2], classes[m2]
            if boxes.size == 0:
                return [], [], []

            # NMS (letterboxed 좌표계)
            yxyx = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
            keep = tf.image.non_max_suppression(yxyx, scores,
                                                max_output_size=20,
                                                iou_threshold=self.nms_th)
            keep = keep.numpy().tolist()
            if not keep:
                return [], [], []

            # 언패드/언스케일 → 원본 좌표
            r, dl, dt, h0, w0 = meta['r'], meta['dl'], meta['dt'], meta['h0'], meta['w0']
            final_boxes, final_scores, final_cls = [], [], []

            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                X1 = (x1 - dl) / r
                Y1 = (y1 - dt) / r
                X2 = (x2 - dl) / r
                Y2 = (y2 - dt) / r
                X1, Y1, X2, Y2 = self._clip_box(X1, Y1, X2, Y2, w0, h0)

                # 너무 작은 박스 제거
                if (X2 - X1) * (Y2 - Y1) < self.min_box_area:
                    continue

                final_boxes.append([X1, Y1, X2, Y2])
                final_scores.append(float(scores[i]))
                final_cls.append(0)

            return final_boxes, final_scores, final_cls

        except Exception as e:
            self.get_logger().warn(f"후처리 오류: {e}")
            return [], [], []

    def _draw(self, img, boxes, scores, class_ids):
        for (x1, y1, x2, y2), sc in zip(boxes, scores):
            cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)
            label = f'person: {sc:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 2, y1), self.color, -1)
            cv2.putText(img, label, (x1 + 1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    rclpy.init()
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

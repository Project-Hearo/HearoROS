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

    # ================ Helper functions ================
    @staticmethod
    def _clip_box(x1, y1, x2, y2, w, h):
        """바운딩 박스 좌표를 이미지 경계 내로 클리핑"""
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
        """detection.py와 동일한 전처리 방식 사용"""
        # 단순 리사이즈 (letterbox 대신)
        resized_image = cv2.resize(bgr, (self.in_w, self.in_h))
        # BGR -> RGB 변환, 차원 확장(batch), uint8 유지
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_image, axis=0).astype(np.uint8)
        
        # 메타데이터 (원본 크기 정보)
        h0, w0 = bgr.shape[:2]
        meta = {
            'h0': h0, 'w0': w0,
            'ih': self.in_h, 'iw': self.in_w
        }
        return input_data, meta

    def _postprocess(self, outputs, meta):
        
        try:
            if len(outputs) < 3:
                return [], [], []

            # 양자화된 YOLO-X 출력 파싱 (detection.py와 동일)
            boxes_quantized = outputs[0][0]  # [8400, 4] UINT8
            scores_quantized = outputs[1][0]  # [8400] UINT8  
            classes_quantized = outputs[2][0]  # [8400] UINT8
            
            # 양자화 파라미터로 디코딩 (모델에서 직접 가져오기)
            boxes_quantization = self.out_details[0]['quantization_parameters']
            scores_quantization = self.out_details[1]['quantization_parameters']
            
            boxes_scale = boxes_quantization['scales'][0]
            boxes_zero_point = boxes_quantization['zero_points'][0]
            scores_scale = scores_quantization['scales'][0]
            scores_zero_point = scores_quantization['zero_points'][0]
            
            # 디코딩
            boxes = (boxes_quantized.astype(np.float32) - boxes_zero_point) * boxes_scale
            scores = (scores_quantized.astype(np.float32) - scores_zero_point) * scores_scale
            classes = classes_quantized.astype(np.int32)
            
            # 사람 클래스만 필터링 (COCO dataset에서 person = 0)
            person_indices = np.where(classes == 0)[0]
            
            if len(person_indices) == 0:
                return [], [], []
            
            # 사람 클래스 탐지 결과만 추출
            person_boxes = boxes[person_indices]
            person_scores = scores[person_indices]
            person_classes = classes[person_indices]
            
            # 신뢰도 임계값 필터링
            score_mask = person_scores >= self.conf_th
            person_boxes = person_boxes[score_mask]
            person_scores = person_scores[score_mask]
            person_classes = person_classes[score_mask]
            
            if len(person_boxes) == 0:
                return [], [], []
            
            # NMS 적용하여 중복 제거 (detection.py와 동일)
            boxes_for_nms = person_boxes.copy()
            # YOLO 출력은 [x1, y1, x2, y2] 형식이므로 그대로 사용
            selected_indices = tf.image.non_max_suppression(
                boxes_for_nms, person_scores, max_output_size=1, iou_threshold=self.nms_th
            )
            
            # NMS 결과를 원본 이미지 크기로 변환
            h_orig, w_orig = meta['h0'], meta['w0']
            h_res, w_res = meta['ih'], meta['iw']
            
            final_boxes, final_scores, final_class_ids = [], [], []
            
            for index in selected_indices:
                x1, y1, x2, y2 = person_boxes[index]
                
                # 좌표가 이미 정규화되어 있는지 확인하고 적절히 변환 (detection.py와 동일)
                if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
                    # 이미 픽셀 좌표인 경우 (모델 입력 크기 기준)
                    # 원본 이미지 크기로 스케일링
                    scale_x = w_orig / meta['iw']
                    scale_y = h_orig / meta['ih']
                    X1 = max(0, int(x1 * scale_x))
                    Y1 = max(0, int(y1 * scale_y))
                    X2 = min(w_orig, int(x2 * scale_x))
                    Y2 = min(h_orig, int(y2 * scale_y))
                else:
                    # 정규화된 좌표인 경우
                    X1 = max(0, int(x1 * w_orig))
                    Y1 = max(0, int(y1 * h_orig))
                    X2 = min(w_orig, int(x2 * w_orig))
                    Y2 = min(h_orig, int(y2 * h_orig))
                
                # 유효한 바운딩 박스인지 확인
                if X2 > X1 and Y2 > Y1 and X2 <= w_orig and Y2 <= h_orig:
                    # 너무 작은 박스 제거
                    if (X2 - X1) * (Y2 - Y1) < self.min_box_area:
                        continue
                    
                    final_boxes.append([X1, Y1, X2, Y2])
                    final_scores.append(float(person_scores[index]))
                    final_class_ids.append(0)
                    
                    self.get_logger().info(f"탐지된 사람: confidence={person_scores[index]:.3f}, bbox=[{X1}, {Y1}, {X2}, {Y2}]")

            return final_boxes, final_scores, final_class_ids

        except Exception as e:
            self.get_logger().warn(f"후처리 오류: {e}")
            import traceback
            self.get_logger().error(f"후처리 상세 오류: {traceback.format_exc()}")
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

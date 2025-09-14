#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import SetParametersResult

import cv2
import numpy as np
import tensorflow as tf
import os
import time

class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector_node')
        
        package_share_path = get_package_share_directory('robot_info')
        default_model_path = os.path.join(package_share_path, 'models', 'Yolo-X_w8a8.tflite')

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_topic', '/stream_image_raw')
        self.declare_parameter('detection_topic', '/person_detector/detection')
        self.declare_parameter('result_image_topic', '/person_detector/image_result')
        
        self.add_on_set_parameters_callback(self.parameters_callback)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.nms_threshold = self.get_parameter('nms_threshold').get_parameter_value().double_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        result_image_topic = self.get_parameter('result_image_topic').get_parameter_value().string_value
        
        self.bridge = CvBridge()

        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(Image, input_topic, self.image_callback, sensor_qos_profile)
        self.detection_pub = self.create_publisher(PolygonStamped, detection_topic, 10)
        self.result_image_pub = self.create_publisher(Image, result_image_topic, 10)
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            rclpy.shutdown()
            return
            
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
        self.color = (0, 255, 0)
        self.prev_time = 0

        self.get_logger().info(f"'{self.get_name()}' 노드가 시작되었습니다. 모델: {model_path}")
        self.get_logger().info(f"모델 출력 상세 정보: {self.output_details}")

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'confidence_threshold':
                self.confidence_threshold = param.value
                self.get_logger().info(f"신뢰도 임계값이 {self.confidence_threshold:.2f} (으)로 변경되었습니다.")
        return SetParametersResult(successful=True)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV-Bridge 변환 실패: {e}")
            return

        input_data, resized_frame = self.preprocess_image(frame)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        outputs = [self.interpreter.get_tensor(detail['index']) for detail in self.output_details]

        boxes, scores, class_ids = self.postprocess_output(
            outputs, frame.shape[:2], resized_frame.shape[:2]
        )

        if len(boxes) > 0:
            self.get_logger().info(f">>> 탐지된 사람 수: {len(boxes)}")
            box = boxes[0] 
            x1, y1, x2, y2 = box
            
            detection_msg = PolygonStamped()
            detection_msg.header = msg.header
            detection_msg.polygon.points = [
                Point32(x=float(x1), y=float(y1), z=0.0), Point32(x=float(x2), y=float(y1), z=0.0),
                Point32(x=float(x2), y=float(y2), z=0.0), Point32(x=float(x1), y=float(y2), z=0.0),
            ]
            self.detection_pub.publish(detection_msg)
        
        self.draw_detections(frame, boxes, scores, class_ids)
        
        curr_time = time.time()
        # prev_time이 0이면 첫 프레임이므로 FPS 계산 생략
        if self.prev_time > 0:
            fps = 1 / (curr_time - self.prev_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
        self.prev_time = curr_time
        
        try:
            result_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            result_msg.header = msg.header
            self.result_image_pub.publish(result_msg)
        except Exception as e:
            self.get_logger().error(f"결과 이미지 발행 실패: {e}")

    def preprocess_image(self, image):
        resized_image = cv2.resize(image, (self.input_width, self.input_height))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_image, axis=0).astype(np.uint8)
        return input_data, resized_image

    # --- [수정] 후처리 함수 전체를 안정적인 새 로직으로 교체 ---
    def postprocess_output(self, outputs, original_shape, resized_shape):
        try:
            # 모델 출력이 여러 개일 수 있으나, 보통 첫 번째 텐서에 모든 정보가 담겨있음
            # (1, N, 85) -> 1=배치, N=탐지후보수, 85=cx,cy,w,h,conf,80개클래스점수
            detections = np.squeeze(outputs[0])

            # 신뢰도(Confidence)가 일정 값 이상인 후보만 필터링
            conf_mask = detections[:, 4] > self.confidence_threshold
            detections = detections[conf_mask]
            if len(detections) == 0:
                return [], [], []

            # 클래스 점수 계산 및 필터링
            class_scores = detections[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = detections[:, 4] * class_scores[np.arange(len(detections)), class_ids]
            score_mask = scores > self.confidence_threshold
            
            detections = detections[score_mask]
            scores = scores[score_mask]
            class_ids = class_ids[score_mask]
            if len(detections) == 0:
                return [], [], []

            # '사람' 클래스(ID: 0)만 필터링
            person_mask = class_ids == 0
            boxes_data = detections[person_mask][:, :4]
            person_scores = scores[person_mask]
            person_class_ids = class_ids[person_mask]
            if len(boxes_data) == 0:
                return [], [], []

            # 바운딩 박스 좌표 변환 (cx,cy,w,h -> y1,x1,y2,x2 for NMS)
            cx, cy, w, h = boxes_data[:, 0], boxes_data[:, 1], boxes_data[:, 2], boxes_data[:, 3]
            y1 = cy - h / 2
            x1 = cx - w / 2
            y2 = cy + h / 2
            x2 = cx + w / 2
            boxes_for_nms = np.stack([y1, x1, y2, x2], axis=1)
            
            # NMS(Non-Max Suppression) 실행
            selected_indices = tf.image.non_max_suppression(
                boxes_for_nms, person_scores, max_output_size=10, iou_threshold=self.nms_threshold
            )
            
            # 최종 결과 정리 및 원본 이미지 크기로 스케일링
            final_boxes, final_scores, final_class_ids = [], [], []
            h_orig, w_orig = original_shape
            h_res, w_res = resized_shape
            
            for index in selected_indices:
                x1_res, y1_res, x2_res, y2_res = boxes_data[index][:4]
                # 중심점과 너비/높이를 원본 이미지 기준으로 변환
                cx_orig = x1_res * (w_orig / w_res)
                cy_orig = y1_res * (h_orig / h_res)
                w_orig_s = x2_res * (w_orig / w_res)
                h_orig_s = y2_res * (h_orig / h_res)
                
                x1_final = int(cx_orig - w_orig_s / 2)
                y1_final = int(cy_orig - h_orig_s / 2)
                x2_final = int(cx_orig + w_orig_s / 2)
                y2_final = int(cy_orig + h_orig_s / 2)

                final_boxes.append([x1_final, y1_final, x2_final, y2_final])
                final_scores.append(person_scores[index])
                final_class_ids.append(person_class_ids[index])
            
            return final_boxes, final_scores, final_class_ids

        except Exception as e:
            self.get_logger().error(f"후처리 중 오류 발생: {e}", throttle_duration_sec=2)
            # import traceback
            # traceback.print_exc()
        return [], [], []

    def draw_detections(self, frame, boxes, scores, class_ids):
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            label_text = f'person: {score:.2f}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), self.color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    rclpy.init()
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
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
        self.declare_parameter('confidence_threshold', 0.25) # 초기 임계값을 약간 낮춤
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

    # --- [수정] 후처리 함수 전체를 다중 텐서 출력에 맞게 재작성 ---
    def postprocess_output(self, outputs, original_shape, resized_shape):
        try:
            # 모델이 '좌표', '점수', '클래스'를 별도 텐서로 출력한다고 가정
            # 모델 출력 구조 로그를 보고 순서가 다르면 이 부분을 조절해야 합니다.
            # 예: outputs[0] = boxes, outputs[1] = scores, outputs[2] = classes
            if len(outputs) < 3:
                # 모델이 예상과 다른 출력을 할 경우, 빈 결과를 반환
                if len(outputs) > 0 and outputs[0].shape[2] > 4:
                     # 단일 텐서 출력에 대한 이전 로직을 여기에 예비로 넣을 수 있음
                     pass # 지금은 생략
                return [], [], []

            # 배치 차원 제거 (1, N, 4) -> (N, 4)
            boxes_data = np.squeeze(outputs[0])
            scores_data = np.squeeze(outputs[1])
            classes_data = np.squeeze(outputs[2])

            # 신뢰도(Confidence)가 임계값 이상인 후보만 1차 필터링
            score_mask = scores_data > self.confidence_threshold
            
            boxes_filtered = boxes_data[score_mask]
            scores_filtered = scores_data[score_mask]
            classes_filtered = classes_data[score_mask]

            if len(scores_filtered) == 0:
                return [], [], []

            # '사람' 클래스(ID: 0)만 2차 필터링
            person_mask = classes_filtered == 0
            
            person_boxes = boxes_filtered[person_mask]
            person_scores = scores_filtered[person_mask]
            person_class_ids = classes_filtered[person_mask]

            if len(person_scores) == 0:
                return [], [], []
            
            # NMS(Non-Max Suppression) 실행
            # tf.image.non_max_suppression은 [y1, x1, y2, x2] 형식을 기대하지만
            # 많은 모델은 [x1, y1, x2, y2]를 출력하므로 그대로 사용해봄
            selected_indices = tf.image.non_max_suppression(
                person_boxes, person_scores, max_output_size=10, iou_threshold=self.nms_threshold
            )
            
            # 최종 결과 정리 및 원본 이미지 크기로 스케일링
            final_boxes, final_scores, final_class_ids = [], [], []
            h_orig, w_orig = original_shape
            
            for index in selected_indices:
                box = person_boxes[index]
                # 좌표가 0~1 사이로 정규화되어 있다고 가정하고 스케일링
                x1 = int(box[0] * w_orig)
                y1 = int(box[1] * h_orig)
                x2 = int(box[2] * w_orig)
                y2 = int(box[3] * h_orig)
                
                final_boxes.append([x1, y1, x2, y2])
                final_scores.append(person_scores[index])
                final_class_ids.append(int(person_class_ids[index]))
            
            return final_boxes, final_scores, final_class_ids

        except Exception as e:
            self.get_logger().error(f"후처리 중 오류 발생: {e}", throttle_duration_sec=2)
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
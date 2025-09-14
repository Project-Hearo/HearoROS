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
        
        # [통합] 패키지 경로 설정
        package_share_path = get_package_share_directory('robot_info')
        default_model_path = os.path.join(package_share_path, 'models', 'Yolo-X_w8a8.tflite')

        # [유지] ROS 2 파라미터 선언
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_topic', '/stream_image_raw')
        self.declare_parameter('detection_topic', '/person_detector/detection')
        self.declare_parameter('result_image_topic', '/person_detector/image_result')
        
        # [개선] 파라미터 변경 시 호출될 콜백 등록
        self.add_on_set_parameters_callback(self.parameters_callback)

        # 파라미터 값 가져오기
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.nms_threshold = self.get_parameter('nms_threshold').get_parameter_value().double_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        result_image_topic = self.get_parameter('result_image_topic').get_parameter_value().string_value
        
        self.bridge = CvBridge()

        # QoS 프로파일 생성
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 구독자 및 발행자 생성
        self.image_sub = self.create_subscription(Image, input_topic, self.image_callback, sensor_qos_profile)
        self.detection_pub = self.create_publisher(PolygonStamped, detection_topic, 10)
        self.result_image_pub = self.create_publisher(Image, result_image_topic, 10)
        
        # TFLite 모델 로드
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
        
        # [통합] 사람 탐지용 색상 (녹색)
        self.color = (0, 255, 0)

        # [통합] FPS 계산을 위한 변수
        self.prev_time = 0

        self.get_logger().info(f"'{self.get_name()}' 노드가 시작되었습니다. 모델: {model_path}")
        self.get_logger().info(f"모델 출력 상세 정보: {self.output_details}")

    # [개선] 파라미터 동적 변경을 위한 콜백 함수
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
        
        # [통합] FPS 계산 및 표시
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
        
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

    # [통합] 독립 실행 파일의 상세한 후처리 로직으로 교체
    def postprocess_output(self, outputs, original_shape, resized_shape):
        try:
            # 양자화된 모델 출력 파싱 (출력 텐서가 3개 이상일 경우)
            if len(outputs) >= 3:
                boxes_quantized = outputs[0][0]
                scores_quantized = outputs[1][0]
                classes_quantized = outputs[2][0]
                
                # 양자화 파라미터로 디코딩
                boxes_scale = self.output_details[0]['quantization_parameters']['scales'][0]
                boxes_zero_point = self.output_details[0]['quantization_parameters']['zero_points'][0]
                scores_scale = self.output_details[1]['quantization_parameters']['scales'][0]
                scores_zero_point = self.output_details[1]['quantization_parameters']['zero_points'][0]
                
                boxes = (boxes_quantized.astype(np.float32) - boxes_zero_point) * boxes_scale
                scores = (scores_quantized.astype(np.float32) - scores_zero_point) * scores_scale
                classes = classes_quantized.astype(np.int32)
                
                # 사람 클래스(id=0)만 필터링
                person_indices = np.where(classes == 0)[0]
                if len(person_indices) == 0: return [], [], []
                
                person_boxes = boxes[person_indices]
                person_scores = scores[person_indices]
                person_classes = classes[person_indices]
                
                # 신뢰도 임계값 필터링
                score_mask = person_scores >= self.confidence_threshold
                person_boxes = person_boxes[score_mask]
                person_scores = person_scores[score_mask]
                person_classes = person_classes[score_mask]
                
                if len(person_boxes) == 0: return [], [], []
                
                # NMS 적용
                selected_indices = tf.image.non_max_suppression(
                    person_boxes, person_scores, max_output_size=10, iou_threshold=self.nms_threshold
                )
                
                # 최종 결과 정리
                final_boxes, final_scores, final_class_ids = [], [], []
                h_orig, w_orig = original_shape
                
                for index in selected_indices:
                    x1_norm, y1_norm, x2_norm, y2_norm = person_boxes[index]
                    
                    x1 = int(x1_norm * w_orig)
                    y1 = int(y1_norm * h_orig)
                    x2 = int(x2_norm * w_orig)
                    y2 = int(y2_norm * h_orig)

                    # 바운딩 박스 좌표가 이미지 경계를 벗어나지 않도록 조정
                    x1_orig, y1_orig = max(0, x1), max(0, y1)
                    x2_orig, y2_orig = min(w_orig, x2), min(h_orig, y2)
                    
                    if x2_orig > x1_orig and y2_orig > y1_orig:
                        final_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                        final_scores.append(person_scores[index])
                        final_class_ids.append(person_classes[index])
                        
                return final_boxes, final_scores, final_class_ids

        except Exception as e:
            self.get_logger().error(f"후처리 중 오류 발생: {e}", throttle_duration_sec=2)
            import traceback
            traceback.print_exc()

        return [], [], []

    # [통합] 사람만 그리도록 간소화된 그리기 함수
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
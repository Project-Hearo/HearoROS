#!/usr/bin/env python3

# [추가] ROS 2 관련 라이브러리
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge

# [기존] 필요한 라이브러리들
import cv2
import numpy as np
import tensorflow as tf
import os
import logging
import time

# COCO 클래스 이름 (기존과 동일)
COCO_CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

# 로깅 설정 (기존과 동일)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# [변경] 클래스를 ROS 2 노드로 변경
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

        # 파라미터 값 가져오기
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.nms_threshold = self.get_parameter('nms_threshold').get_parameter_value().double_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        result_image_topic = self.get_parameter('result_image_topic').get_parameter_value().string_value
        
        # [추가] CvBridge 초기화
        self.bridge = CvBridge()

        # [추가] 구독자 및 발행자 생성
        self.image_sub = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.detection_pub = self.create_publisher(PolygonStamped, detection_topic, 10)
        self.result_image_pub = self.create_publisher(Image, result_image_topic, 10)
        
        # TFLite 모델 로드 (기존 __init__ 로직)
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
        self.colors = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3))

        self.get_logger().info(f"'{self.get_name()}' 노드가 시작되었습니다. 모델: {model_path}")

    # [변경] 기존 run() 메서드가 콜백 함수로 변경됨
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV-Bridge 변환 실패: {e}")
            return

        input_data, resized_frame = self.preprocess_image(frame)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        boxes, scores, class_ids = self.postprocess_output(
            output_data, frame.shape[:2], resized_frame.shape[:2]
        )

        # [추가] 탐지된 객체 좌표 발행
        if len(boxes) > 0:
            # 가장 신뢰도 높은 첫 번째 사람만 발행 (필요시 반복문으로 모두 발행 가능)
            box = boxes[0] 
            x1, y1, x2, y2 = box
            
            detection_msg = PolygonStamped()
            detection_msg.header = msg.header # 원본 이미지의 타임스탬프와 frame_id 사용
            
            # 바운딩 박스의 네 꼭짓점 좌표
            detection_msg.polygon.points = [
                Point32(x=float(x1), y=float(y1), z=0.0),
                Point32(x=float(x2), y=float(y1), z=0.0),
                Point32(x=float(x2), y=float(y2), z=0.0),
                Point32(x=float(x1), y=float(y2), z=0.0),
            ]
            self.detection_pub.publish(detection_msg)

        # [추가] 결과 영상 발행 (디버깅용)
        self.draw_detections(frame, boxes, scores, class_ids)
        try:
            result_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            result_msg.header = msg.header
            self.result_image_pub.publish(result_msg)
        except Exception as e:
            self.get_logger().error(f"결과 이미지 발행 실패: {e}")

    # --- 아래 함수들은 기존 코드와 거의 동일 (클래스 멤버 변수 사용하도록 수정) ---
    def preprocess_image(self, image):
        resized_image = cv2.resize(image, (self.input_width, self.input_height))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_image, axis=0).astype(np.float32)
        return input_data, resized_image

    def postprocess_output(self, output_data, original_shape, resized_shape):
        detections = output_data[0]
        conf_mask = detections[:, 4] > self.confidence_threshold
        detections = detections[conf_mask]
        if len(detections) == 0: return [], [], []
        class_scores = detections[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = detections[:, 4] * class_scores[np.arange(len(detections)), class_ids]
        score_mask = scores > self.confidence_threshold
        detections, scores, class_ids = detections[score_mask], scores[score_mask], class_ids[score_mask]
        person_mask = class_ids == 0
        detections, scores, class_ids = detections[person_mask], scores[person_mask], class_ids[person_mask]
        if len(detections) == 0: return [], [], []
        boxes_cxcywh = detections[:, :4]
        x1, y1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2, boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2, y2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2, boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes = np.stack([y1, x1, y2, x2], axis=1)
        selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=self.nms_threshold)
        final_boxes, final_scores, final_class_ids = [], [], []
        h_orig, w_orig = original_shape
        h_res, w_res = resized_shape
        for index in selected_indices:
            box = boxes[index]
            y1_orig, x1_orig = int(box[0] * (h_orig / h_res)), int(box[1] * (w_orig / w_res))
            y2_orig, x2_orig = int(box[2] * (h_orig / h_res)), int(box[3] * (w_orig / w_res))
            final_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
            final_scores.append(scores[index])
            final_class_ids.append(class_ids[index])
        return final_boxes, final_scores, final_class_ids

    def draw_detections(self, frame, boxes, scores, class_ids):
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            score, class_id = scores[i], class_ids[i]
            label, color = COCO_CLASSES[class_id], self.colors[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f'{label}: {score:.2f}'
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# [변경] ROS 2 노드를 실행하기 위한 main 함수
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
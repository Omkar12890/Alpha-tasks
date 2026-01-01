"""
Object Detection using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """YOLOv8-based object detector"""
    
    def __init__(self, model_name='yolov8n.pt', confidence=0.5):
        """
        Initialize detector with YOLOv8 model.
        model_name: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
        confidence: Detection confidence threshold
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.class_names = self.model.names
        
    def detect(self, frame):
        """
        Detect objects in frame.
        Returns: array of shape (N, 4) with format [x1, y1, x2, y2]
        """
        # Run inference
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id]
                })
        
        bboxes = np.array([d['bbox'] for d in detections]) if detections else np.empty((0, 4))
        return bboxes, detections
    
    def get_class_names(self):
        """Get list of all class names"""
        return list(self.class_names.values())

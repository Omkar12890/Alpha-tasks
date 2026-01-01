"""
SORT (Simple Online and Realtime Tracking) implementation
Based on the original SORT paper: https://arxiv.org/abs/1602.00763
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class KalmanFilterTracker:
    """Simple Kalman Filter for object tracking"""
    
    def __init__(self, bbox):
        """
        Initialize Kalman Filter tracker with initial bounding box.
        bbox: [x1, y1, x2, y2] in pixel coordinates
        """
        self.bbox = bbox
        self.state = self._bbox_to_state(bbox)
        self.covariance = np.eye(4) * 10
        self.age = 1
        self.hits = 1
        self.id = None
        
    def _bbox_to_state(self, bbox):
        """Convert bbox [x1, y1, x2, y2] to state [cx, cy, w, h]"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h])
    
    def _state_to_bbox(self, state):
        """Convert state [cx, cy, w, h] to bbox [x1, y1, x2, y2]"""
        cx, cy, w, h = state
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])
    
    def predict(self):
        """Predict new state using simple velocity model"""
        # Simple constant velocity model
        self.state = self.state  # Keep same state (can be enhanced with velocity)
        self.age += 1
        return self._state_to_bbox(self.state)
    
    def update(self, bbox):
        """Update state with new measurement"""
        new_state = self._bbox_to_state(bbox)
        # Simple average update
        self.state = 0.7 * self.state + 0.3 * new_state
        self.hits += 1
        self.bbox = self._state_to_bbox(self.state)
    
    def get_bbox(self):
        """Get current bounding box as [x1, y1, x2, y2]"""
        return self._state_to_bbox(self.state)


def iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) between two bboxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union area
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


class SORTTracker:
    """SORT (Simple Online and Realtime Tracking) tracker"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker.
        max_age: Maximum frames to keep alive a track without detections
        min_hits: Minimum detections to start tracking
        iou_threshold: IoU threshold for matching detections
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update tracks with new detections.
        detections: array of shape (N, 4) with format [x1, y1, x2, y2]
        Returns: array of shape (M, 5) with format [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        
        # Predict
        predictions = []
        for tracker in self.trackers:
            predictions.append(tracker.predict())
        predictions = np.array(predictions) if predictions else np.empty((0, 4))
        
        # Associate detections with predictions
        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, predictions
        )
        
        # Update matched trackers
        for d, t in matched:
            self.trackers[t].update(detections[d])
        
        # Create new trackers for unmatched detections
        for d in unmatched_dets:
            tracker = KalmanFilterTracker(detections[d])
            tracker.id = self.next_id
            self.next_id += 1
            self.trackers.append(tracker)
        
        # Remove dead trackers
        self.trackers = [
            t for t in self.trackers 
            if (t.age - t.hits) < self.max_age
        ]
        
        # Output
        ret = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = tracker.get_bbox()
                ret.append([*bbox, tracker.id])
        
        return np.array(ret) if ret else np.empty((0, 5))
    
    def _associate(self, detections, predictions):
        """
        Associate detections with predictions using IoU.
        Returns: matched pairs, unmatched detections, unmatched predictions
        """
        if len(predictions) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(predictions)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(predictions)))
        for d in range(len(detections)):
            for p in range(len(predictions)):
                iou_matrix[d, p] = iou(detections[d], predictions[p])
        
        # Hungarian algorithm
        det_indices, pred_indices = linear_sum_assignment(-iou_matrix)
        
        matched = []
        for d, p in zip(det_indices, pred_indices):
            if iou_matrix[d, p] > self.iou_threshold:
                matched.append([d, p])
        
        matched = np.array(matched)
        unmatched_dets = [d for d in range(len(detections)) 
                         if d not in matched[:, 0]]
        unmatched_trks = [p for p in range(len(predictions)) 
                         if p not in matched[:, 1]]
        
        return matched, unmatched_dets, unmatched_trks

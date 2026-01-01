"""
Visualization utilities for drawing bounding boxes and tracking IDs
"""

import cv2
import numpy as np


class Visualizer:
    """Visualization helper for detection and tracking results"""
    
    COLORS = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (128, 128, 0),    # Teal
        (128, 0, 0),      # Dark Red
        (0, 128, 128),    # Dark Cyan
    ]
    
    @staticmethod
    def draw_detections(frame, detections, thickness=2, font_scale=0.6):
        """
        Draw detection bounding boxes with class labels.
        detections: list of detection dicts with keys: bbox, class_name, confidence
        """
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label with confidence
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            label_y = max(y1, label_size[1] + 10)
            
            # Draw background rectangle for label
            cv2.rectangle(
                frame, 
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 5, label_y + 5),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1
            )
        
        return frame
    
    @staticmethod
    def draw_tracks(frame, tracks, thickness=2, font_scale=0.7):
        """
        Draw tracking bounding boxes with tracking IDs.
        tracks: array of shape (N, 5) with format [x1, y1, x2, y2, track_id]
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            
            # Get color based on track ID
            color_idx = track_id % len(Visualizer.COLORS)
            color = Visualizer.COLORS[color_idx]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw tracking ID
            label = f"ID: {track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            label_y = max(y1, label_size[1] + 10)
            
            # Draw background rectangle for label
            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 5, label_y + 5),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
        return frame
    
    @staticmethod
    def draw_combined(frame, detections, tracks, thickness=2):
        """
        Draw both detections and tracks on frame.
        """
        frame = Visualizer.draw_detections(frame, detections, thickness=thickness)
        frame = Visualizer.draw_tracks(frame, tracks, thickness=thickness)
        return frame
    
    @staticmethod
    def add_frame_info(frame, frame_num, fps, num_detections, num_tracks):
        """Add frame information (frame number, FPS, counts) to frame"""
        info_text = [
            f"Frame: {frame_num}",
            f"FPS: {fps:.1f}",
            f"Detections: {num_detections}",
            f"Tracks: {num_tracks}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y_offset += 30
        
        return frame

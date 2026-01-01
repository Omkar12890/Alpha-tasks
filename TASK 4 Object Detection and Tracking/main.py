"""
Main Object Detection and Tracking Pipeline
Combines YOLOv8 detection with SORT tracking for real-time video processing
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path

from detector import ObjectDetector
from sort_tracker import SORTTracker
from visualizer import Visualizer


class DetectionTrackingPipeline:
    """Main pipeline for detection and tracking"""
    
    def __init__(self, 
                 model_name='yolov8n.pt',
                 confidence=0.5,
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3):
        """
        Initialize the pipeline.
        
        Args:
            model_name: YOLOv8 model variant
            confidence: Detection confidence threshold
            max_age: Max frames to keep track without detections
            min_hits: Min detections to start tracking
            iou_threshold: IoU threshold for track association
        """
        self.detector = ObjectDetector(model_name, confidence)
        self.tracker = SORTTracker(max_age, min_hits, iou_threshold)
        self.visualizer = Visualizer()
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
    def process_frame(self, frame, draw_detections=True, draw_tracks=True):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (BGR image)
            draw_detections: Whether to draw detection boxes
            draw_tracks: Whether to draw tracking boxes
            
        Returns:
            processed_frame: Frame with visualizations
            detections: List of detection dicts
            tracks: Array of track info
        """
        self.frame_count += 1
        
        # Detect objects
        bboxes, detections = self.detector.detect(frame)
        
        # Track objects
        tracks = self.tracker.update(bboxes)
        
        # Draw on frame
        output_frame = frame.copy()
        if draw_detections:
            output_frame = self.visualizer.draw_detections(output_frame, detections)
        if draw_tracks:
            output_frame = self.visualizer.draw_tracks(output_frame, tracks)
        
        # Add frame info
        output_frame = self.visualizer.add_frame_info(
            output_frame,
            self.frame_count,
            self.fps,
            len(detections),
            len(tracks)
        )
        
        # Update FPS
        current_time = time.time()
        if current_time - self.prev_time > 0.1:  # Update every 0.1 seconds
            self.fps = self.frame_count / (current_time - self.prev_time)
            self.frame_count = 0
            self.prev_time = current_time
        
        return output_frame, detections, tracks
    
    def process_webcam(self, display=True, save_output=None):
        """
        Process video stream from webcam.
        
        Args:
            display: Whether to display output
            save_output: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
        print(f"Processing webcam stream at {width}x{height}@{fps}fps")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output_frame, detections, tracks = self.process_frame(frame)
                
                # Display
                if display:
                    cv2.imshow('Detection & Tracking', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save
                if writer:
                    writer.write(output_frame)
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
    
    def process_video_file(self, video_path, display=True, save_output=None):
        """
        Process video from file.
        
        Args:
            video_path: Path to input video file
            display: Whether to display output
            save_output: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
        print(f"Processing {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        print("Press 'q' to quit")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                output_frame, detections, tracks = self.process_frame(frame)
                
                # Display
                if display:
                    # Add progress info
                    progress_text = f"{frame_count}/{total_frames}"
                    cv2.putText(output_frame, progress_text, (10, height - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Detection & Tracking', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save
                if writer:
                    writer.write(output_frame)
                
                # Progress
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            print(f"Completed! Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description='Object Detection and Tracking Pipeline')
    parser.add_argument('--source', type=str, default='webcam',
                       help='Video source: "webcam" or path to video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model variant')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing detection and tracking pipeline...")
    pipeline = DetectionTrackingPipeline(
        model_name=args.model,
        confidence=args.confidence
    )
    
    # Process video
    if args.source.lower() == 'webcam':
        pipeline.process_webcam(
            display=not args.no_display,
            save_output=args.output
        )
    else:
        pipeline.process_video_file(
            args.source,
            display=not args.no_display,
            save_output=args.output
        )


if __name__ == '__main__':
    main()

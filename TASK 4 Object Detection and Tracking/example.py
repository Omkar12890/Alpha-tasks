"""
Example usage script for the Object Detection and Tracking Pipeline
"""

from main import DetectionTrackingPipeline


def example_webcam():
    """Example: Process webcam stream with real-time display"""
    print("Starting webcam detection and tracking...")
    
    pipeline = DetectionTrackingPipeline(
        model_name='yolov8n.pt',  # Nano model for speed
        confidence=0.5
    )
    
    # Process webcam
    pipeline.process_webcam(
        display=True,
        save_output=None  # Set to 'output.mp4' to save
    )


def example_video_file():
    """Example: Process video file"""
    video_path = 'path/to/your/video.mp4'
    
    print(f"Processing video: {video_path}")
    
    pipeline = DetectionTrackingPipeline(
        model_name='yolov8s.pt',  # Small model for balance
        confidence=0.5
    )
    
    # Process video file
    pipeline.process_video_file(
        video_path,
        display=True,
        save_output='output.mp4'
    )


def example_webcam_with_save():
    """Example: Process webcam and save output"""
    pipeline = DetectionTrackingPipeline(
        model_name='yolov8n.pt',
        confidence=0.45
    )
    
    pipeline.process_webcam(
        display=True,
        save_output='webcam_output.mp4'
    )


if __name__ == '__main__':
    # Run webcam example
    example_webcam()
    
    # Uncomment to run video file example instead
    # example_video_file()
    
    # Uncomment to run webcam with save example
    # example_webcam_with_save()

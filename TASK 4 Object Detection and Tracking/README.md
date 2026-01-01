# Object Detection and Tracking System

A complete real-time object detection and tracking pipeline using YOLOv8 and SORT algorithm.

## Features

- **Real-time Object Detection**: Uses YOLOv8 for fast and accurate object detection
- **Object Tracking**: Implements SORT (Simple Online and Realtime Tracking) algorithm
- **Webcam Support**: Process live webcam feeds
- **Video File Support**: Process video files
- **Real-time Visualization**: Display bounding boxes with labels and tracking IDs
- **Video Output**: Save processed videos with detections and tracks
- **Performance Monitoring**: FPS counter and detection statistics

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **First run** (downloads YOLOv8 pretrained weights):
```bash
python main.py --source webcam
```

## Usage

### Webcam Processing

**Basic usage (display only)**:
```bash
python main.py --source webcam
```

**With output video**:
```bash
python main.py --source webcam --output output.mp4
```

**With different model**:
```bash
python main.py --source webcam --model yolov8s.pt --confidence 0.4
```

### Video File Processing

**Process video file**:
```bash
python main.py --source path/to/video.mp4
```

**Process and save output**:
```bash
python main.py --source path/to/video.mp4 --output output.mp4
```

### Command-line Arguments

- `--source`: Video source ("webcam" or path to video file) [default: webcam]
- `--model`: YOLOv8 model variant [default: yolov8n.pt]
  - `yolov8n.pt`: Nano (fastest, least accurate)
  - `yolov8s.pt`: Small
  - `yolov8m.pt`: Medium
  - `yolov8l.pt`: Large
  - `yolov8x.pt`: Extra Large (slowest, most accurate)
- `--confidence`: Detection confidence threshold [default: 0.5]
- `--output`: Path to save output video (optional)
- `--no-display`: Disable display window (for headless processing)

## Architecture

### Components

1. **detector.py** - YOLOv8 Object Detection
   - Loads pretrained YOLO models
   - Detects objects in frames
   - Returns bounding boxes and class information

2. **sort_tracker.py** - SORT Tracking Algorithm
   - Kalman filter-based tracker
   - IoU-based association
   - Persistent tracking IDs across frames

3. **visualizer.py** - Visualization Utilities
   - Draws detection bounding boxes with labels
   - Draws tracking boxes with IDs
   - Displays FPS and statistics

4. **main.py** - Main Pipeline
   - Integrates detection and tracking
   - Handles webcam and video file input
   - Manages output video writing

### Pipeline Flow

```
Input (Webcam/Video)
    ↓
Frame Reading (OpenCV)
    ↓
Object Detection (YOLOv8)
    ↓
Object Tracking (SORT)
    ↓
Visualization
    ↓
Output (Display + Optional Save)
```

## How It Works

### Detection
- **YOLOv8**: State-of-the-art CNN-based detector
- Outputs bounding boxes with class predictions and confidence scores
- Runs on each frame independently

### Tracking
- **SORT Algorithm**:
  1. **Prediction**: Uses Kalman filter to predict object positions
  2. **Association**: Matches detections with predictions using IoU
  3. **Update**: Updates track states with new detections
  4. **Lifecycle**: Maintains tracks with configurable age and hit criteria

### Visualization
- Green bounding boxes: Detection results
- Colored bounding boxes: Track results (color varies by track ID)
- Labels: Class names with confidence scores
- Tracking IDs: Persistent across frames
- Center points: Track location indicators

## Performance Tips

1. **Model Selection**:
   - Use `yolov8n.pt` for speed (mobile/webcam)
   - Use `yolov8s.pt` or `yolov8m.pt` for balance
   - Use `yolov8l.pt` or `yolov8x.pt` for accuracy

2. **Confidence Threshold**:
   - Lower (0.3): More detections, more false positives
   - Higher (0.7): Fewer detections, higher confidence

3. **Resolution**:
   - YOLOv8 automatically resizes to optimal input size
   - Lower resolution → faster processing
   - Higher resolution → better accuracy

## Troubleshooting

### Slow Performance
- Use smaller model (yolov8n.pt)
- Increase confidence threshold
- Use GPU if available (torch with CUDA)

### Missing Detections
- Lower confidence threshold
- Use larger model
- Check lighting conditions

### Tracking Issues
- Adjust `max_age` parameter (more frames to keep dead tracks)
- Adjust `min_hits` parameter (more detections to start tracking)
- Adjust `iou_threshold` parameter (stricter/looser matching)

### CUDA/GPU Support
To use GPU acceleration (requires NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [SORT Algorithm Paper](https://arxiv.org/abs/1602.00763)
- [OpenCV Documentation](https://docs.opencv.org/)

## License

This project is provided as-is for educational purposes.

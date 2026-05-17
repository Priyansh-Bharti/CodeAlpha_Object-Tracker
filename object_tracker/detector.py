from ultralytics import YOLO


class ObjectDetector:
    """Detects objects in frames using YOLOv8 neural network."""
    
    def __init__(self, model_size='s', conf_threshold=0.45):
        """
        Initialize the ObjectDetector by loading a YOLOv8 model.
        
        YOLOv8 models available:
        - 'n' (nano): Fastest, lowest accuracy (~3M params)
        - 's' (small): Balanced speed/accuracy (~11M params) - RECOMMENDED
        - 'm' (medium): More accurate, slower (~25M params)
        - 'l' (large): High accuracy, slower (~53M params)
        
        Args:
            model_size (str): Size of the model ('n', 's', 'm', 'l')
            conf_threshold (float): Confidence threshold for detections (0-1)
        """
        # Load the YOLOv8 model with specified size
        # Small (s) provides better balance between speed and accuracy
        model_name = f'yolov8{model_size}.pt'
        print(f"Loading YOLOv8{model_size.upper()} model...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
    
    # Run object detection on a video frame
    def detect(self, frame):
        """
        Detect objects in a video frame using YOLO inference with NMS.
        
        YOLO (You Only Look Once) is a real-time object detection algorithm that:
        - Takes a frame as input
        - Outputs bounding boxes, confidence scores, and class labels
        - Runs inference in one forward pass (hence "only looks once")
        - Applies NMS (Non-Maximum Suppression) to remove duplicate detections
        
        Args:
            frame: Input video frame as a numpy array (height, width, channels)
        
        Returns:
            list: List of detections with confidence >= threshold. Each detection is a dict with:
                - x1, y1: Top-left corner coordinates of bounding box
                - x2, y2: Bottom-right corner coordinates of bounding box
                - confidence: Detection confidence score (0-1, higher is better)
                - class_name: Name of the detected object class (e.g., "person", "car")
        """
        # Run YOLO inference with NMS (Non-Maximum Suppression)
        # - conf: Confidence threshold for initial detections
        # - iou: IoU threshold for NMS (0.45 removes overlapping duplicate boxes)
        # - verbose: Disable per-frame console spam for cleaner runtime logs
        results = self.model(frame, conf=self.conf_threshold, iou=0.45, verbose=False)
        
        detections = []
        
        # Process each result (usually just one for a single frame)
        for result in results:
            # result.boxes contains all detected bounding boxes for this frame
            for box in result.boxes:
                # Extract bounding box coordinates in (x1, y1, x2, y2) format
                # x1, y1 = top-left corner
                # x2, y2 = bottom-right corner
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Extract confidence score (0 to 1)
                confidence = box.conf[0]
                
                # Confidence filtering already done by YOLO (conf parameter)
                # but we double-check here for safety
                if confidence < self.conf_threshold:
                    continue
                
                # Get the class ID (numeric) and convert to class name
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Create a detection dictionary with all information
                detection = {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'confidence': float(confidence),
                    'class_name': class_name
                }
                
                detections.append(detection)
        
        return detections

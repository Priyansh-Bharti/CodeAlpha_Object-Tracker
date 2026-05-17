"""
Object Detection and Tracking Application
==========================================

This program demonstrates real-time object detection and tracking in video streams.
It combines four core modules to create a complete tracking system:

1. utils.py (VideoStream class)
   - Handles video input from webcam or video files
   - Manages video stream lifecycle (open, read frames, release)

2. detector.py (ObjectDetector class)
   - Uses YOLOv8s neural network for real-time object detection
   - Returns bounding boxes and class labels for detected objects
   - Applies NMS (Non-Maximum Suppression) to remove duplicate detections
   - Filters low-confidence detections (default: < 0.45)

3. tracker.py (ObjectTracker class)
   - Implements SORT algorithm for multi-object tracking
   - Maintains consistent track IDs across frames
   - Matches detections to existing tracks using IoU (Intersection over Union)
   - Removes stale tracks that haven't been seen recently

4. utils.py (draw_tracks function)
    - Visualizes tracked results on video frames
    - Supports both classic and HUD-style overlays
    - Labels each object with class name and track ID

Flow:
  Video Frame → Detector (YOLO) → Tracker (SORT) → Visualization → Display

ENHANCEMENTS:
  ✓ YOLOv8s (small) model for better accuracy
  ✓ NMS with 0.45 IoU threshold to reduce duplicates
  ✓ IoU-based matching in tracker (better than distance-based)
    ✓ Theme-based visualization (HUD and classic)
  ✓ Real-time FPS display
  ✓ Frame resizing for faster processing
  ✓ Support for both webcam and video files
    ✓ Optional output video recording
"""

import cv2
import time
import argparse
import ctypes
import numpy as np
from utils import VideoStream, draw_tracks, resize_frame
from detector import ObjectDetector
from tracker import ObjectTracker


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Object Detection and Tracking')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source: 0 for webcam, or path to video file (default: 0)')
    parser.add_argument('--model', type=str, default='s', 
                       help='YOLO model size: n (nano), s (small), m (medium), l (large) (default: s)')
    parser.add_argument('--confidence', type=float, default=0.45, 
                       help='Confidence threshold for detections (default: 0.45)')
    parser.add_argument('--no-resize', action='store_true', 
                       help='Disable frame resizing (process at full resolution)')
    parser.add_argument('--theme', type=str, default='hud', choices=['hud', 'classic'],
                       help='Visualization theme: hud or classic (default: hud)')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Start the display in fullscreen mode (press F to toggle)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save output video to this path (e.g., output.mp4)')
    
    args = parser.parse_args()
    
    # Convert source string to int if it's a webcam index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source  # It's a file path
    
    # Create an ObjectDetector instance
    try:
        detector = ObjectDetector(model_size=args.model, conf_threshold=args.confidence)
    except Exception as e:
        print(f"Error loading detector: {e}")
        return
    
    # Create an ObjectTracker instance
    tracker = ObjectTracker()
    
    # Open the video stream
    video = VideoStream(source=source)
    
    if not video.is_open():
        print(f"Error: Could not open video source: {source}")
        return
    
    # Initialize video writer if output path is specified
    video_writer = None
    writer_width = None
    writer_height = None
    if args.output:
        # Get frame properties
        frame_width = int(video.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # If resizing is enabled, adjust dimensions
        if not args.no_resize:
            frame_width = min(frame_width, 1280)
            frame_height = min(frame_height, 720)

        writer_width = frame_width
        writer_height = frame_height
        
        # Create video writer
        # Using mp4v codec for MP4 files, or XVID for AVI
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (writer_width, writer_height))
        
        if not video_writer.isOpened():
            print(f"Warning: Could not create video writer for {args.output}")
            video_writer = None
        else:
            print(f"✓ Video will be saved to: {args.output}")
    
    # Determine source name for display
    source_name = "Webcam" if isinstance(source, int) else source
    print(f"✓ Video source opened: {source_name}")
    print(f"✓ YOLO model: yolov8{args.model.upper()}")
    print(f"✓ Confidence threshold: {args.confidence}")
    print(f"✓ Frame resizing: {'Disabled' if args.no_resize else 'Enabled'}")
    print(f"✓ Theme: {args.theme}")
    print("\nPress Q to quit.\n")
    
    # Initialize tracking variables
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0
    total_detections = 0
    total_tracks = 0
    
    # Main processing loop
    try:
        # Create display window (use WINDOW_NORMAL so we can toggle/resize)
        window_name = "Object Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Determine screen resolution (Windows)
        try:
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        except Exception:
            screen_w, screen_h = None, None

        fullscreen_active = False
        if args.fullscreen:
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                fullscreen_active = True
            except Exception:
                # Some OpenCV builds or platforms may not support fullscreen programmatically
                fullscreen_active = False
        while video.is_open():
            ret, frame = video.get_frame()
            
            if not ret:
                print("\n⚠ End of video stream")
                break
            
            # Resize frame if enabled (faster processing)
            if not args.no_resize:
                frame, _ = resize_frame(frame, max_width=1280, max_height=720)
            
            # Run object detection
            detections = detector.detect(frame)
            total_detections += len(detections)
            
            # Update tracker with detections
            tracked_objects = tracker.update(detections)
            total_tracks += len(tracked_objects)
            
            # Draw bounding boxes and labels
            annotated_frame = draw_tracks(
                frame,
                tracked_objects,
                theme=args.theme,
                frame_index=frame_count,
            )
            
            # Calculate FPS every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed_time = time.time() - fps_start_time
                current_fps = 10 / elapsed_time
                fps_start_time = time.time()
            
                    # Display performance information on frame
            info_text = f"FPS {current_fps:.1f}   DET {len(detections)}   TRK {len(tracked_objects)}"
            if args.theme == 'hud':
                        # HUD mode uses a tinted status panel to match the theme.
                cv2.rectangle(annotated_frame, (8, 8), (520, 44), (20, 34, 70), -1)
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 220, 255),
                    2,
                )
            else:
                # Classic mode keeps a plain OpenCV-style text overlay.
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            
            # Write frame to video file if output is enabled
            if video_writer:
                # Ensure frame has correct dimensions
                if annotated_frame.shape[1] != writer_width or annotated_frame.shape[0] != writer_height:
                    annotated_frame = cv2.resize(annotated_frame, (writer_width, writer_height))
                video_writer.write(annotated_frame)
            
            # Display the frame
            display_frame = annotated_frame
            # If fullscreen is active and we know the screen size, scale & letterbox to preserve aspect
            if fullscreen_active and screen_w and screen_h:
                h, w = annotated_frame.shape[:2]
                scale = min(screen_w / w, screen_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(annotated_frame, (new_w, new_h))
                # Letterbox onto a black background matching the screen
                bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                x_off = (screen_w - new_w) // 2
                y_off = (screen_h - new_h) // 2
                bg[y_off:y_off + new_h, x_off:x_off + new_w] = resized
                display_frame = bg

            cv2.imshow(window_name, display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n\n✓ User pressed Q - quitting...")
                break
            # Toggle fullscreen with 'f' key
            if key == ord('f') or key == ord('F'):
                fullscreen_active = not fullscreen_active
                try:
                    if fullscreen_active:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                except Exception:
                    # Ignore if unsupported
                    pass
    
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    
    except Exception as e:
        print(f"\n\n✗ Error during processing: {e}")
    
    finally:
        # Cleanup
        video.release()
        if video_writer:
            video_writer.release()
            print(f"\n✓ Video saved successfully to: {args.output}")
        cv2.destroyAllWindows()
        
        # Print statistics
        print(f"\n--- Session Statistics ---")
        print(f"Total frames processed: {frame_count}")
        print(f"Average detections per frame: {total_detections / frame_count:.1f}" if frame_count > 0 else "0")
        print(f"Average tracked objects: {total_tracks / frame_count:.1f}" if frame_count > 0 else "0")
        print(f"✓ Program ended successfully")


if __name__ == "__main__":
    main()

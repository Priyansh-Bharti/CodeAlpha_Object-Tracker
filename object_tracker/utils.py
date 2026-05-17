import cv2


def resize_frame(frame, max_width=1280, max_height=720):
    """
    Resize frame to fit within max dimensions while maintaining aspect ratio.
    
    This helps with:
    - Faster processing (smaller frames process faster)
    - Lower memory usage
    - Better display on smaller screens
    
    Args:
        frame: Input video frame
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
    
    Returns:
        tuple: (resized_frame, scale_factor)
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor to fit within max dimensions
    scale = min(max_width / width, max_height / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    
    return frame, 1.0


def _track_color(track_id):
    """Generate a stable color per track ID for consistent visuals."""
    palette = [
        (0, 180, 255),
        (0, 220, 255),
        (0, 255, 180),
        (80, 230, 255),
        (20, 200, 255),
        (0, 255, 230),
    ]
    return palette[track_id % len(palette)]


def _draw_hud_shell(frame, frame_index):
    """Draw the HUD shell (bars, reticle, and animated scan line)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top and bottom HUD bars for cockpit-like framing.
    cv2.rectangle(overlay, (0, 0), (w, 64), (12, 22, 52), -1)
    cv2.rectangle(overlay, (0, h - 46), (w, h), (10, 18, 40), -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    # Soft center reticle.
    cx, cy = w // 2, h // 2
    cv2.circle(frame, (cx, cy), 26, (0, 200, 255), 1)
    cv2.circle(frame, (cx, cy), 48, (0, 115, 185), 1)
    cv2.line(frame, (cx - 60, cy), (cx - 20, cy), (0, 170, 255), 1)
    cv2.line(frame, (cx + 20, cy), (cx + 60, cy), (0, 170, 255), 1)
    cv2.line(frame, (cx, cy - 60), (cx, cy - 20), (0, 170, 255), 1)
    cv2.line(frame, (cx, cy + 20), (cx, cy + 60), (0, 170, 255), 1)

    # Animated scan line.
    scan_y = int((frame_index * 6) % max(1, h))
    cv2.line(frame, (0, scan_y), (w, scan_y), (0, 140, 255), 1)


def _draw_hud_brackets(frame, x1, y1, x2, y2, color):
    """Draw HUD-style angular brackets around a tracked object."""
    lx = max(10, int((x2 - x1) * 0.22))
    ly = max(10, int((y2 - y1) * 0.22))
    t = 2

    cv2.line(frame, (x1, y1), (x1 + lx, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + ly), color, t)

    cv2.line(frame, (x2, y1), (x2 - lx, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + ly), color, t)

    cv2.line(frame, (x1, y2), (x1 + lx, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - ly), color, t)

    cv2.line(frame, (x2, y2), (x2 - lx, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - ly), color, t)


def draw_tracks(frame, tracks, theme="hud", frame_index=0):
    """
    Draw tracked objects in either classic mode or HUD mode.

    Args:
        frame: Input frame.
        tracks: List of tracked objects with keys x1, y1, x2, y2, track_id, class_name.
        theme: "hud" for themed overlays, "classic" for standard rectangles.
        frame_index: Frame counter used by HUD animations.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()

    if theme == "hud":
        _draw_hud_shell(annotated, frame_index)

    for track in tracks:
        x1 = int(track["x1"])
        y1 = int(track["y1"])
        x2 = int(track["x2"])
        y2 = int(track["y2"])
        track_id = int(track["track_id"])
        class_name = str(track["class_name"])

        color = _track_color(track_id)
        label = f"{class_name} #{track_id}"

        if theme == "classic":
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 60, 90), 1)
            _draw_hud_brackets(annotated, x1, y1, x2, y2, color)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        text_y1 = max(0, y1 - th - 12)
        text_y2 = y1
        cv2.rectangle(annotated, (x1, text_y1), (x1 + tw + 10, text_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (15, 20, 35),
            2,
        )

        # Small center marker makes object association easier to read.
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 3, color, -1)

    return annotated


class VideoStream:
    """Handles video input from either a webcam or video file."""
    
    def __init__(self, source=0):
        """
        Initialize the video stream.
        
        Args:
            source (int or str): Camera index (default 0 for webcam) or path to video file.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
    
    # Get the current frame from the video stream
    def get_frame(self):
        """
        Capture and return the current frame from the video stream.
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the video frame.
        """
        ret, frame = self.cap.read()
        return ret, frame
    
    # Check if the video stream is still open and running
    def is_open(self):
        """
        Check if the video stream is currently open and running.
        
        Returns:
            bool: True if the stream is open, False otherwise.
        """
        return self.cap.isOpened()
    
    # Clean up and close the video stream
    def release(self):
        """
        Release the video stream and clean up resources.
        """
        self.cap.release()

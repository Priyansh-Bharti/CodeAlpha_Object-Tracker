# Object Detection and Tracking Application

A real-time object detection and tracking app built with OpenCV, YOLOv8, and SORT.

## What It Does

- Opens a webcam stream or a video file
- Detects objects on every frame using YOLOv8
- Tracks objects across frames with stable IDs using SORT
- Draws labels and tracking IDs in real time
- Supports two visual styles:
  - `hud`: futuristic helmet-like overlay (default)
  - `classic`: clean standard box view
- Optionally saves the processed output video

## Features

- Real-time detection and tracking
- Webcam and video-file input support
- Configurable YOLO model size (`n`, `s`, `m`, `l`)
- Configurable confidence threshold
- IoU-based detection-to-track matching
- HUD theme with reticle, scanline, and angular brackets
- Optional frame resizing for performance
- Optional output video recording (`.mp4`)
- Session summary printed at exit

## Requirements

- Python 3.8+
- Webcam (optional if using a video file)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

### Basic (webcam)

```bash
python main.py
```

### Use a video file

```bash
python main.py --source path/to/video.mp4
```

### Change model size

```bash
python main.py --model n
python main.py --model s
python main.py --model m
python main.py --model l
```

### Change confidence threshold

```bash
python main.py --confidence 0.5
```

### Toggle theme

```bash
python main.py --theme hud
python main.py --theme classic
```

### Save output video

```bash
python main.py --output result.mp4
```

### Example combined command

```bash
python main.py --source input.mp4 --model m --confidence 0.5 --theme hud --output result.mp4
```

### Fullscreen

Start the display in fullscreen mode (press `F` to toggle while running):

```bash
python main.py --source path/to/video.mp4 --fullscreen
```

## CLI Options

- `--source`: `0` for webcam (default) or video file path
- `--model`: YOLO model size (`n`, `s`, `m`, `l`), default `s`
- `--confidence`: detection confidence threshold, default `0.45`
- `--no-resize`: disable frame resizing
- `--theme`: `hud` or `classic`, default `hud`
- `--output`: output video path (for example `result.mp4`)
 - `--fullscreen`: start the display in fullscreen mode; press `F` to toggle during runtime

## Controls

- Press `Q` to quit
 - Press `F` to toggle fullscreen (when supported by your OpenCV build)

## Project Structure

```text
object_tracker/
  main.py          # app entry, CLI, pipeline loop, display, output writing
  detector.py      # YOLOv8 model loading + frame detection
  tracker.py       # SORT-style tracker with IoU matching + Hungarian assignment
  utils.py         # frame resize, drawing styles (hud/classic), video stream wrapper
  requirements.txt # dependencies
  README.md
```

## Pipeline

```text
Video Frame
  -> (optional) resize
  -> YOLO detection (boxes/classes/confidence)
  -> SORT tracking (assign track IDs)
  -> render overlays (hud/classic)
  -> display and optional output write
```

## Notes

- First run may download YOLO weights automatically.
- If performance is slow, try:
  - `--model n` or `--model s`
  - keep resize enabled (do not use `--no-resize`)
  - avoid output writing when testing latency
 - Fullscreen behavior is controlled by OpenCV and the OS window manager; some OpenCV builds or platforms may not support programmatic fullscreen. If `--fullscreen` or `F` doesn't work, manually maximize the window or consider a GUI toolkit (PySide/PyQt) for a more robust interface.
 - Do not commit large binary artifacts (model weights, exported videos) to the repository. Use the provided `.gitignore` to exclude `*.pt`, `*.mp4`, and other generated files.


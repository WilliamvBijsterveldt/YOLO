import cv2
import threading
import queue
from roboflow import Roboflow
import pyrealsense2 as rs
import numpy as np

# Initialize Roboflow model
rf = Roboflow(api_key="IuXGCojGHoUDyiR9rAgr")
project = rf.workspace().project("object-detection-1-onzrn")
model = project.version("8").model

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the color stream from RealSense camera
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Variables for smoother performance
frame_queue = queue.Queue(maxsize=5)  # Limit queue size to avoid memory issues
latest_frame = None  # Raw frame (for display)
processed_frame = None  # Frame with detections
lock = threading.Lock()

def video_capture():
    """Continuously captures frames from the RealSense camera and performs object detection."""
    global latest_frame, processed_frame

    while True:
        # Wait for a frame from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue  # Skip if no frame is received

        # Convert RealSense frame to NumPy array
        frame = np.asanyarray(color_frame.get_data())

        # Perform object detection on the captured frame
        processed_frame = detect_objects(frame)

        # Update latest_frame for display
        with lock:
            latest_frame = frame

def detect_objects(frame):
    """Performs object detection on a given frame."""
    # Resize frame (smaller = faster inference)
    resized_frame = cv2.resize(frame, (320, 240))

    # Perform object detection (API call)
    predictions = model.predict(resized_frame, confidence=40, overlap=30).json()

    # Copy frame and draw detections
    detected_frame = frame.copy()
    for pred in predictions.get("predictions", []):
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        label = pred["class"]

        # Draw bounding box and label
        cv2.rectangle(detected_frame, (x - w // 2, y - h // 2),
                      (x + w // 2, y + h // 2), (0, 255, 0), 2)
        cv2.putText(detected_frame, label, (x - w // 2, y - h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detected_frame

# Start background threads
capture_thread = threading.Thread(target=video_capture, daemon=True)
capture_thread.start()

while True:
    with lock:
        display_frame = latest_frame
        detection_frame = processed_frame

    if display_frame is not None:
        # Display the live feed
        cv2.imshow("RealSense Webcam Feed", display_frame)

    if detection_frame is not None:
        # Display the detected frame
        cv2.imshow("Detected Objects", detection_frame)

    # Exit on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
pipeline.stop()
cv2.destroyAllWindows()

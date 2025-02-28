import cv2
import threading
import queue
from roboflow import Roboflow

# Initialize Roboflow model
rf = Roboflow(api_key="IuXGCojGHoUDyiR9rAgr")
project = rf.workspace().project("object-detection-1-onzrn")
model = project.version("8").model

# Open webcam
capture = cv2.VideoCapture(0)

# Variables for smoother performance
frame_queue = queue.Queue(maxsize=5)  # Limit queue size to avoid memory issues
latest_frame = None  # Raw frame (for display)
processed_frame = None  # Frame with detections
lock = threading.Lock()

# Flag to control screenshot capture
screenshot_taken = False
screenshot_frame = None

def video_capture():
    """Continuously captures frames from the webcam."""
    global latest_frame

    while True:
        ret, frame = capture.read()
        if not ret:
            continue  # Skip if frame capture fails

        with lock:
            latest_frame = frame  # Update latest frame (avoids lag)

def detect_objects(frame):
    """Performs object detection on a given frame."""
    # Resize frame (smaller = faster inference)
    resized_frame = cv2.resize(frame, (320, 240))
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, resized_frame)

    # Perform object detection (API call)
    predictions = model.predict(temp_image_path, confidence=40, overlap=30).json()

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

    if display_frame is not None:
        cv2.imshow("Webcam Feed", display_frame)

    # Check for spacebar press to capture a screenshot and detect objects
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Spacebar pressed
        with lock:
            if latest_frame is not None:
                screenshot_frame = latest_frame.copy()
                screenshot_taken = True

        # Detect objects in the screenshot
        if screenshot_taken and screenshot_frame is not None:
            processed_frame = detect_objects(screenshot_frame)
            screenshot_taken = False  # Reset flag after detection

    # Show the processed screenshot with detections
    if processed_frame is not None:
        cv2.imshow("Detected Objects", processed_frame)

    # Exit on pressing 'q'
    if key == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()

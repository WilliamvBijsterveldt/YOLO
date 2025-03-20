import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("runs/detect/train13/weights/best.pt")  # Replace with your trained model path if different

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start streaming
pipeline.start(config)

# Align depth to color stream
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize ROI coordinates
roi_selected = False
roi = (0, 0, 0, 0)  # (x, y, width, height)

# Function to select ROI using the mouse
def select_roi(frame):
    global roi_selected, roi
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    roi_selected = True

try:
    while True:
        # Wait for frames and align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to NumPy arrays
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Select ROI if not selected yet
        if not roi_selected:
            select_roi(frame)
        
        # Draw ROI rectangle on the frame
        if roi_selected and roi != (0, 0, 0, 0):
            x1, y1, w, h = roi
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        
        # Run YOLOv8 detection on the entire frame
        results = model(frame)

        # Process detection results
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0])  # Class index
                label = f"{model.names[cls]} {conf:.2f}"

                # Check if the detection is inside the selected ROI
                if roi_selected and (x1 >= roi[0] and y1 >= roi[1] and x2 <= roi[0] + roi[2] and y2 <= roi[1] + roi[3]):
                    # Compute the center of the bounding box
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2

                    # Get depth value at center (in millimeters)
                    depth_value = depth_frame.get_distance(x_center, y_center)

                    # Convert depth and 2D coordinates to real-world 3D coordinates
                    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    real_world_coords = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_center, y_center], depth_value)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Display object location and distance in absolute coordinates (in meters)
                    position_text = f"X: {real_world_coords[0]:.2f}, Y: {real_world_coords[1]:.2f}, Z: {real_world_coords[2]:.2f}m"
                    cv2.putText(frame, position_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("YOLOv8 RealSense Detection with ROI and Absolute Coordinates", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()

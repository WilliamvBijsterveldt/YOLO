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

        # Run YOLOv8 detection
        results = model(frame)

        # Process detection results
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0])  # Class index
                label = f"{model.names[cls]} {conf:.2f}"

                # Compute the center of the bounding box
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                # Get depth value at center (in millimeters)
                depth_value = depth_frame.get_distance(x_center, y_center)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display object location and distance
                position_text = f"X: {x_center}, Y: {y_center}, Distance: {depth_value:.2f}m"
                cv2.putText(frame, position_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("YOLOv8 RealSense Detection with Distance ", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()

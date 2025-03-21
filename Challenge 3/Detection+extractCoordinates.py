import pyrealsense2 as rs
import numpy as np
import cv2
import csv
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("runs/detect/train13/weights/best.pt")  # Update path if needed

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream
pipeline.start(config)

# Align depth to color stream
align_to = rs.stream.color
align = rs.align(align_to)

# ROI Parameters
roi_selected = False
roi = (0, 0, 0, 0)  # (x, y, width, height)
physical_width_cm = 40.0  # Longer side of ROI
physical_height_cm = 30.0  # Shorter side of ROI
camera_height_cm = 65.0  # Camera is 65 cm above the floor

# Function to select and scale ROI
def select_roi(frame):
    global roi_selected, roi
    raw_roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    
    # Ensure ROI matches fixed 30x40 cm ratio
    if raw_roi[2] > raw_roi[3]:  
        roi = (raw_roi[0], raw_roi[1], raw_roi[2], int(raw_roi[2] * (physical_height_cm / physical_width_cm)))
    else:
        roi = (raw_roi[0], raw_roi[1], int(raw_roi[3] * (physical_width_cm / physical_height_cm)), raw_roi[3])
    
    roi_selected = True

# Function to save detected object coordinates
def save_coordinates(objects):
    with open("object_coords.csv", mode="w", newline="") as file:  # Overwrite file
        writer = csv.writer(file)
        writer.writerow(["Object Name", "X (cm)", "Y (cm)", "Z (cm)"])  # Column headers
        
        for obj in objects:
            writer.writerow(obj)

# Function to calculate the average of a list of coordinates
def average_coordinates(objects):
    avg_objects = []
    for obj_name in set([obj[0] for obj in objects]):
        obj_coords = [obj for obj in objects if obj[0] == obj_name]
        avg_x = np.mean([obj[1] for obj in obj_coords])
        avg_y = np.mean([obj[2] for obj in obj_coords])
        avg_z = np.mean([obj[3] for obj in obj_coords])
        avg_objects.append([obj_name, round(avg_x, 2), round(avg_y, 2), round(avg_z, 2)])
    return avg_objects

try:
    while True:
        # Wait for frames and align
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert to NumPy arrays
        frame = np.asanyarray(color_frame.get_data())

        # Select ROI if not selected
        if not roi_selected:
            select_roi(frame)

        x1, y1, w, h = roi
        pixel_to_cm_x = physical_width_cm / w
        pixel_to_cm_y = physical_height_cm / h

        # Draw ROI
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        # Run YOLO detection
        results = model(frame)
        detected_objects = []

        for r in results:
            for box in r.boxes:
                x1_obj, y1_obj, x2_obj, y2_obj = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                object_name = model.names[cls]  # Get class name
                label = f"{object_name} {conf:.2f}"

                # Check if object is inside ROI
                if roi_selected and (x1_obj >= x1 and y1_obj >= y1 and x2_obj <= x1 + w and y2_obj <= y1 + h):
                    # Compute center in pixel coordinates
                    x_center = (x1_obj + x2_obj) // 2
                    y_center = (y1_obj + y2_obj) // 2
                    
                    # Convert to ROI grid (bottom-left as (0,0))
                    x_cm = (x_center - x1) * pixel_to_cm_x
                    y_cm = (h - (y_center - y1)) * pixel_to_cm_y  # Invert y-axis for bottom-left origin

                    # Get depth value (in meters) and convert to cm
                    depth_value_m = depth_frame.get_distance(x_center, y_center)
                    depth_value_cm = depth_value_m * 100  # Convert to cm

                    # Height relative to the floor
                    height_from_floor = camera_height_cm - depth_value_cm

                    # Save coordinates in ROI grid
                    detected_objects.append([object_name, round(x_cm, 2), round(y_cm, 2), round(height_from_floor, 2)])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1_obj, y1_obj), (x2_obj, y2_obj), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1_obj, y1_obj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw center point
                    cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

                    # Display X, Y, Z coordinates below bounding box
                    text_x = f"X = {x_cm:.2f} cm"
                    text_y = f"Y = {y_cm:.2f} cm"
                    text_z = f"Z = {height_from_floor:.2f} cm"

                    text_y_pos = y2_obj + 20  # Start position for text
                    cv2.putText(frame, text_x, (x1_obj, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, text_y, (x1_obj, text_y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, text_z, (x1_obj, text_y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show frame
        cv2.imshow("YOLOv8 RealSense Detection", frame)

        # Key press handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and detected_objects:
            # Measure coordinates over 5 seconds
            start_time = time.time()
            all_objects = []

            while time.time() - start_time < 5:
                # Capture another 5 seconds of data
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Run YOLO detection
                results = model(np.asanyarray(color_frame.get_data()))
                for r in results:
                    for box in r.boxes:
                        x1_obj, y1_obj, x2_obj, y2_obj = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cls = int(box.cls[0])
                        object_name = model.names[cls]  # Get class name

                        if roi_selected and (x1_obj >= x1 and y1_obj >= y1 and x2_obj <= x1 + w and y2_obj <= y1 + h):
                            # Compute center in pixel coordinates
                            x_center = (x1_obj + x2_obj) // 2
                            y_center = (y1_obj + y2_obj) // 2
                            
                            # Convert to ROI grid (bottom-left as (0,0))
                            x_cm = (x_center - x1) * pixel_to_cm_x
                            y_cm = (h - (y_center - y1)) * pixel_to_cm_y  # Invert y-axis for bottom-left origin

                            # Get depth value (in meters) and convert to cm
                            depth_value_m = depth_frame.get_distance(x_center, y_center)
                            depth_value_cm = depth_value_m * 100  # Convert to cm

                            # Height relative to the floor
                            height_from_floor = camera_height_cm - depth_value_cm

                            # Save coordinates in ROI grid
                            all_objects.append([object_name, round(x_cm, 2), round(y_cm, 2), round(height_from_floor, 2)])

            # Calculate average coordinates and save
            averaged_objects = average_coordinates(all_objects)
            save_coordinates(averaged_objects)
            print("Saved averaged object coordinates to object_coords.csv")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

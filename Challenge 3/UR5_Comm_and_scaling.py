import socket
import sys
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time
import pickle

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

# Socket setup for UR5 communication
HOST_IP_ADDRESS = "192.168.0.3"  # Your PC's IP
PORT = 30002  # Ensure UR5 is connecting to this port

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    server_socket.bind((HOST_IP_ADDRESS, PORT))
except socket.error as e:
    print(f"Error binding the socket: {e}")
    sys.exit(1)

server_socket.listen()
print(f"Listening on {HOST_IP_ADDRESS}:{PORT}")

client_socket, client_address = server_socket.accept()
print(f"Accepted connection from {client_address}")

# Load the transformation matrix (from .pkl file)
def load_transformation_matrix(filename="transformation_matrix.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

transformation_matrix = load_transformation_matrix()

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

# Function to transform coordinates using the transformation matrix
def transform_coordinates(camera_coords):
    # Convert the camera coordinates to homogeneous coordinates (4D vector)
    camera_point_homogeneous = np.array([camera_coords[0], camera_coords[1], camera_coords[2], 1])
    
    # Apply the transformation matrix
    robot_point_homogeneous = np.dot(transformation_matrix, camera_point_homogeneous)
    
    # Return the transformed robot coordinates (ignoring the last homogeneous component)
    return robot_point_homogeneous[:3]

# Function to handle user input for object selection
def get_user_input_for_object():
    return input("Enter the name of the object to go to: ").strip().lower()

try:
    detected_objects = []  # List to hold detected objects
    
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
        results = model(frame, verbose=False)
        detected_objects.clear()  # Clear previous objects

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

                    # Save object information including size (width and height in pixels)
                    detected_objects.append([object_name, x_cm, y_cm, height_from_floor, x2_obj - x1_obj, y2_obj - y1_obj])

                    # Draw bounding box and labels
                    cv2.rectangle(frame, (x1_obj, y1_obj), (x2_obj, y2_obj), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1_obj, y1_obj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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

        # Wait for user input for object selection
        user_input = get_user_input_for_object()

        # Check if the entered object name exists in the detected objects
        selected_object = None
        for obj in detected_objects:
            if obj[0].lower() == user_input:
                selected_object = obj
                break

        if selected_object is not None:
            # Transform the coordinates of the selected object
            transformed_coords = transform_coordinates([selected_object[1], selected_object[2], selected_object[3]])

            # Send the transformed coordinates to UR5 (converted to meters)
            message = f"({transformed_coords[0]/100}, {transformed_coords[1]/100}, {transformed_coords[2]/100})\n"
            client_socket.send(message.encode())

            print(f"Sent coordinates ({transformed_coords[0]/100}, {transformed_coords[1]/100}, {transformed_coords[2]/100}) to UR5.")

        # If the user presses 'q', break the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    client_socket.close()
    server_socket.close()

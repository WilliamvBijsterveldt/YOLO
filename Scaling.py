import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import socket

class CameraToRobotTransformer:
    def __init__(self, robot_ip='192.168.0.5', robot_port=30002):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.transformation_matrix = np.eye(4)
        self.calibration_points = []
        self.model = YOLO("runs/detect/train13/weights/best.pt")
        self.align = rs.align(rs.stream.color)
        
        # Robot connection
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.robot_ip, self.robot_port))

    def collect_calibration_points(self, num_points=5):
        self.calibration_points = []

        for i in range(num_points):
            input(f"Move to calibration point {i+1} and press Enter...")
            
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            cv2.imshow("Select Calibration Point", np.asanyarray(color_frame.get_data()))
            pixel = cv2.selectROI("Select Calibration Point", np.asanyarray(color_frame.get_data()))
            cv2.destroyAllWindows()

            x, y = int(pixel[0] + pixel[2]/2), int(pixel[1] + pixel[3]/2)
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_value = depth_frame.get_distance(x, y)
            camera_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)

            robot_x = float(input("Enter robot X coordinate: "))
            robot_y = float(input("Enter robot Y coordinate: "))
            robot_z = float(input("Enter robot Z coordinate: "))
            robot_point = [robot_x, robot_y, robot_z]
            
            self.calibration_points.append({'camera_point': camera_point, 'robot_point': robot_point})
        
        return self.calibration_points

    def compute_transformation_matrix(self):
        camera_points = np.array([p['camera_point'] for p in self.calibration_points])
        robot_points = np.array([p['robot_point'] for p in self.calibration_points])
        
        camera_centroid = np.mean(camera_points, axis=0)
        robot_centroid = np.mean(robot_points, axis=0)
        
        camera_centered = camera_points - camera_centroid
        robot_centered = robot_points - robot_centroid
        
        H = camera_centered.T @ robot_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        t = robot_centroid - R @ camera_centroid
        
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = R
        self.transformation_matrix[:3, 3] = t
        
        return self.transformation_matrix

    def transform_camera_to_robot(self, camera_point):
        camera_point_homogeneous = np.array(list(camera_point) + [1])
        robot_point_homogeneous = self.transformation_matrix @ camera_point_homogeneous
        return robot_point_homogeneous[:3]
    
    def send_coordinates_to_robot(self, robot_coords):
        command = f"{robot_coords[0]}, {robot_coords[1]}, {robot_coords[2]}\n"
        self.sock.sendall(command.encode('utf-8'))

    def detect_and_transform_objects(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            results = self.model(frame)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    depth_value = depth_frame.get_distance(x_center, y_center)
                    camera_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_center, y_center], depth_value)
                    robot_point = self.transform_camera_to_robot(camera_point)
                    
                    label = f"Robot Coords: {robot_point}"
                    cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Send only coordinates to robot
                    self.send_coordinates_to_robot(robot_point)
                    
            cv2.imshow("Object Detection with Robot Coordinates", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.sock.close()

if __name__ == "__main__":
    transformer = CameraToRobotTransformer()
    print("Collecting calibration points...")
    transformer.collect_calibration_points()
    transformation_matrix = transformer.compute_transformation_matrix()
    print("Transformation Matrix:", transformation_matrix)
    print("Starting object detection...")
    transformer.detect_and_transform_objects()

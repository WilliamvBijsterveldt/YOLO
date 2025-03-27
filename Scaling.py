import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import pickle
from typing import List, Dict, Tuple

class CameraToRobotTransformer:
    def __init__(self, model_path: str = "runs/detect/train13/weights/best.pt"):
        """
        Initialize the camera-to-robot transformation pipeline.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        
        self.transformation_matrix = np.eye(4)
        self.calibration_points: List[Dict[str, List[float]]] = []
        self.model = YOLO(model_path)
        self.align = rs.align(rs.stream.color)
        self.selected_point: Tuple[int, int] | None = None
        self.roi = None  # Store ROI for consistency

    def select_roi(self):
        """
        Allow user to select the Region of Interest (ROI) for both calibration and object detection.
        """
        print("Select the same ROI for calibration and detection.")
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame:
            print("Error: No color frame available.")
            return None

        frame = np.asanyarray(color_frame.get_data())
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        if w == 0 or h == 0:
            print("Error: Invalid ROI selected.")
            return None

        print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")
        self.roi = roi
        return roi

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.roi:
                roi_x, roi_y, roi_w, roi_h = self.roi
                if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
                    self.selected_point = (x, y)
                else:
                    print("Click inside the ROI only!")

    def collect_calibration_points(self, num_points: int = 4) -> None:
        """
        Collects at least 4 calibration points for a better transformation matrix.
        """
        print("Collecting calibration points... Minimum required: 4")
        self.calibration_points = []
        self.roi = self.select_roi()
        if not self.roi:
            print("ROI selection failed. Exiting calibration.")
            return

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)

        while len(self.calibration_points) < num_points:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            roi_x, roi_y, roi_w, roi_h = self.roi
            roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            results = self.model(roi_frame)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Calibration", frame)
            if self.selected_point:
                x, y = self.selected_point
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                depth_value = depth_frame.get_distance(x, y)
                camera_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
                print(f"Clicked at camera point: {camera_point}")
                
                robot_point = [
                    float(input(f"Enter robot X (meters) for point {len(self.calibration_points) + 1}: ")),
                    float(input(f"Enter robot Y (meters) for point {len(self.calibration_points) + 1}: ")),
                    float(input(f"Enter robot Z (meters) for point {len(self.calibration_points) + 1}: "))
                ]
                
                self.calibration_points.append({
                    'camera_point': camera_point, 
                    'robot_point': robot_point
                })
                self.selected_point = None
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def compute_transformation_matrix(self) -> np.ndarray:
        if len(self.calibration_points) < 4:
            raise ValueError("At least 4 calibration points are required")
        
        camera_points = np.array([p['camera_point'] for p in self.calibration_points])
        robot_points = np.array([p['robot_point'] for p in self.calibration_points])

        camera_centroid = np.mean(camera_points, axis=0)
        robot_centroid = np.mean(robot_points, axis=0)
        camera_centered = camera_points - camera_centroid
        robot_centered = robot_points - robot_centroid

        H = camera_centered.T @ robot_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = robot_centroid - R @ camera_centroid
        
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = R
        self.transformation_matrix[:3, 3] = t
        
        print("Computed Rotation Matrix:")
        print(R)
        print("Computed Translation Vector:")
        print(t)

        self.save_transformation_matrix()
        return self.transformation_matrix

    def transform_point(self, camera_point: List[float]) -> List[float]:
        camera_point_homogeneous = np.array(camera_point + [1])
        robot_point_homogeneous = self.transformation_matrix @ camera_point_homogeneous
        return robot_point_homogeneous[:3].tolist()

    def save_transformation_matrix(self, filename_pkl: str = "transformation_matrix.pkl", filename_csv: str = "transformation_matrix.csv") -> None:
        with open(filename_pkl, "wb") as f:
            pickle.dump(self.transformation_matrix, f)
        np.savetxt(filename_csv, self.transformation_matrix, delimiter=",")
        print(f"Transformation matrix saved to {filename_pkl} and {filename_csv}")

    def __del__(self):
        self.pipeline.stop()

if __name__ == "__main__":
    transformer = CameraToRobotTransformer()
    try:
        transformer.collect_calibration_points(num_points=4)
        transformation_matrix = transformer.compute_transformation_matrix()
        print("Final Transformation Matrix:\n", transformation_matrix)
        test_camera_point = [0.1, 0.2, 0.3]
        print("Transformed Test Point:", transformer.transform_point(test_camera_point))
    except Exception as e:
        print(f"An error occurred: {e}")

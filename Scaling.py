import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
from typing import List, Dict, Tuple

class CameraToRobotTransformer2D:
    def __init__(self):
        """
        Initialize the camera-to-robot 2D transformation pipeline.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        
        self.transformation_matrix = np.eye(3)  # 3x3 transformation matrix for 2D
        self.calibration_points: List[Dict[str, List[float]]] = []
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
                    self.selected_point = (x - roi_x, roi_h - (y - roi_y))  # Make (0,0) bottom-left
                    self.selected_point = (self.selected_point[0] * (0.4 / roi_w), self.selected_point[1] * (0.3 / roi_h))  # Scale to 30x40 cm
                    print(f"Selected point (ROI frame, scaled): {self.selected_point}")
                    self.test_transform_point()
                else:
                    print("Click inside the ROI only!")

    def collect_calibration_points(self, num_points: int = 6) -> None:
        """
        Collects at least 4 calibration points for computing the 2D transformation matrix.
        """
        print("Collecting calibration points... Click on the image to select points.")
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
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            cv2.imshow("Calibration", frame)

            if self.selected_point:
                x, y = self.selected_point
                camera_point = [x, y]  # 2D point within ROI
                print(f"Camera 2D point: {camera_point}")

                robot_point = [
                    float(input(f"Enter robot X (meters) for point {len(self.calibration_points) + 1}: ")),
                    float(input(f"Enter robot Y (meters) for point {len(self.calibration_points) + 1}: "))
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
        if len(self.calibration_points) < 6:
            raise ValueError("At least 6 calibration points are required for a 2D transformation.")
        
        camera_points = np.array([p['camera_point'] for p in self.calibration_points])
        robot_points = np.array([p['robot_point'] for p in self.calibration_points])

        transformation_matrix, _ = cv2.estimateAffine2D(camera_points, robot_points)
        transformation_matrix = np.vstack([transformation_matrix, [0, 0, 1]])  # Convert to 3x3
        
        self.transformation_matrix = transformation_matrix
        print("Computed 2D Transformation Matrix:")
        print(transformation_matrix)
        
        self.save_transformation_matrix()
        return transformation_matrix

    def transform_point(self, camera_point: List[float]) -> List[float]:
        camera_point_homogeneous = np.array(camera_point + [1])
        robot_point_homogeneous = self.transformation_matrix @ camera_point_homogeneous
        return robot_point_homogeneous[:2].tolist()
    
    def test_transform_point(self) -> None:
        if self.selected_point:
            robot_point = self.transform_point(self.selected_point)
            print(f"Camera ROI Point: {self.selected_point} -> Robot Point: {robot_point}")

    def save_transformation_matrix(self, filename_pkl: str = "transformation_matrix_2d.pkl", filename_csv: str = "transformation_matrix_2d.csv") -> None:
        with open(filename_pkl, "wb") as f:
            pickle.dump(self.transformation_matrix, f)
        np.savetxt(filename_csv, self.transformation_matrix, delimiter=",")
        print(f"Transformation matrix saved to {filename_pkl} and {filename_csv}")

    def __del__(self):
        self.pipeline.stop()

if __name__ == "__main__":
    transformer = CameraToRobotTransformer2D()
    try:
        transformer.collect_calibration_points(num_points=6)
        transformation_matrix = transformer.compute_transformation_matrix()
        print("Final 2D Transformation Matrix:\n", transformation_matrix)
    except Exception as e:
        print(f"An error occurred: {e}")
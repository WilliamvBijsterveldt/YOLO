import numpy as np
import cv2
import pickle
from typing import List, Dict, Tuple
import random

class MockCameraToRobotTransformer:
    def __init__(self):
        """
        Initialize a mock camera-to-robot transformation pipeline.
        """
        # Initialize transformation matrix and calibration points
        self.transformation_matrix = np.eye(4)
        self.calibration_points: List[Dict[str, List[float]]] = []
        
        # Simulated random seed for reproducibility
        np.random.seed(42)

    def collect_calibration_points(self, num_points: int = 2) -> None:
        """
        Simulate collecting calibration points.
        
        :param num_points: Number of calibration points to collect (default: 2)
        """
        print("Simulating calibration point collection...")
        self.calibration_points = []

        for i in range(num_points):
            # Generate mock camera point with some random noise
            camera_point = [
                random.uniform(0.1, 0.5),  # x
                random.uniform(0.1, 0.5),  # y
                random.uniform(0.5, 1.0)   # z
            ]
            
            # Generate corresponding robot point 
            # (with a predictable but slightly offset relationship)
            robot_point = [
                camera_point[0] * 1.1 + random.uniform(-0.05, 0.05),
                camera_point[1] * 1.2 + random.uniform(-0.05, 0.05),
                camera_point[2] * 0.9 + random.uniform(-0.1, 0.1)
            ]

            print(f"Point {i + 1}:")
            print(f"  Camera Point: {camera_point}")
            print(f"  Robot Point:  {robot_point}")

            self.calibration_points.append({
                'camera_point': camera_point, 
                'robot_point': robot_point
            })

        print("Calibration points collected.")

    def compute_transformation_matrix(self) -> np.ndarray:
        """
        Compute the transformation matrix using Procrustes analysis.
        
        :return: 4x4 transformation matrix
        """
        if len(self.calibration_points) < 2:
            raise ValueError("At least 2 calibration points are required")

        camera_points = np.array([p['camera_point'] for p in self.calibration_points])
        robot_points = np.array([p['robot_point'] for p in self.calibration_points])
        
        # Compute centroids
        camera_centroid = np.mean(camera_points, axis=0)
        robot_centroid = np.mean(robot_points, axis=0)
        
        # Center the points
        camera_centered = camera_points - camera_centroid
        robot_centered = robot_points - robot_centroid
        
        # Compute the optimal rotation matrix using SVD
        H = camera_centered.T @ robot_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Compute translation
        t = robot_centroid - R @ camera_centroid
        
        # Construct transformation matrix
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[:3, :3] = R
        self.transformation_matrix[:3, 3] = t
        
        self.save_transformation_matrix()
        
        return self.transformation_matrix

    def transform_point(self, camera_point: List[float]) -> List[float]:
        """
        Transform a point from camera coordinates to robot coordinates.
        
        :param camera_point: 3D point in camera coordinate system
        :return: Transformed point in robot coordinate system
        """
        # Add homogeneous coordinate
        camera_point_homogeneous = np.array(camera_point + [1])
        
        # Apply transformation
        robot_point_homogeneous = self.transformation_matrix @ camera_point_homogeneous
        
        return robot_point_homogeneous[:3].tolist()

    def save_transformation_matrix(self, 
                                   filename_pkl: str = "mock_transformation_matrix.pkl", 
                                   filename_csv: str = "mock_transformation_matrix.csv") -> None:
        """
        Save the transformation matrix to pickle and CSV files.
        
        :param filename_pkl: Filename for pickle serialization
        :param filename_csv: Filename for CSV export
        """
        # Save as pickle
        with open(filename_pkl, "wb") as f:
            pickle.dump(self.transformation_matrix, f)
        print(f"Transformation matrix saved to {filename_pkl}")

        # Save as CSV
        np.savetxt(filename_csv, self.transformation_matrix, delimiter=",")
        print(f"Transformation matrix saved to {filename_csv}")

def main():
    transformer = MockCameraToRobotTransformer()
    
    try:
        print("Collecting mock calibration points...")
        transformer.collect_calibration_points(num_points=2)
        
        transformation_matrix = transformer.compute_transformation_matrix()
        print("\nTransformation Matrix:\n", transformation_matrix)
        
        # Demonstrate point transformation
        print("\nPoint Transformation Examples:")
        test_camera_points = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        for camera_point in test_camera_points:
            robot_point = transformer.transform_point(camera_point)
            print(f"Camera Point: {camera_point} → Robot Point: {robot_point}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
import pickle
import numpy as np

def load_transformation_matrix(filename_pkl: str = "transformation_matrix_2d.pkl") -> np.ndarray:
    """
    Load the saved 2D transformation matrix from a file.
    """
    with open(filename_pkl, "rb") as f:
        transformation_matrix = pickle.load(f)
    print("Loaded Transformation Matrix:")
    print(transformation_matrix)
    return transformation_matrix

def transform_point(camera_point: list, transformation_matrix: np.ndarray) -> list:
    """
    Transform a given 2D camera point to robot coordinates using the transformation matrix.
    """
    camera_point_homogeneous = np.array(camera_point + [1])  # Convert to homogeneous coordinates
    robot_point_homogeneous = transformation_matrix @ camera_point_homogeneous
    return robot_point_homogeneous[:2].tolist()

def main():
    transformation_matrix = load_transformation_matrix()
    
    while True:
        try:
            x = float(input("Enter camera X coordinate: "))
            y = float(input("Enter camera Y coordinate: "))
            robot_point = transform_point([x, y], transformation_matrix)
            print(f"Robot coordinates: {robot_point}")
        except ValueError:
            print("Invalid input. Please enter numerical values.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
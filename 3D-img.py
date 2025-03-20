import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams: RGB and Depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Align depth to color stream
align_to = rs.stream.color
align = rs.align(align_to)

# Create an Open3D PointCloud object
point_cloud = o3d.geometry.PointCloud()

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
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Create point cloud from depth and color
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        # Get the RGB data and depth data
        points = []
        colors = []

        for y in range(0, depth_image.shape[0], 2):  # Downsample for performance
            for x in range(0, depth_image.shape[1], 2):
                depth_value = depth_image[y, x]
                if depth_value == 0:
                    continue  # Skip invalid depth

                # Convert depth pixel to 3D world coordinates (X, Y, Z)
                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
                points.append(point)

                # Get the color corresponding to the point
                color = color_image[y, x] / 255.0  # Normalize color
                colors.append(color)

        # Convert the lists to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Add points and colors to Open3D PointCloud object
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals for the point cloud (required for Poisson reconstruction)
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Perform surface reconstruction using Poisson Surface Reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9, width=0, scale=1.1, linear_fit=False)

        bbox = mesh.get_axis_aligned_bounding_box()
        p_mesh_crop = mesh.crop(bbox)


        # Visualize the mesh
        o3d.visualization.draw_geometries([p_mesh_crop])

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()

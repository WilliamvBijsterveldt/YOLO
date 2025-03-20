import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import keyboard

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align depth to color stream
align_to = rs.stream.color
align = rs.align(align_to)

# Apply spatial filtering to smooth depth
spatial = rs.spatial_filter()

# Initialize global point cloud
global_pcd = o3d.geometry.PointCloud()
seen_points = set()  # Keep track of seen points (as tuples for uniqueness)

# Setup non-blocking Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(global_pcd)

try:
    while True:
        # Wait for frames and align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("⚠️ WARNING: No frames captured!")
            continue

        # Apply spatial filter for noise reduction
        depth_frame = spatial.process(depth_frame)

        # Convert frames to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Show depth and color images
        cv2.imshow("Depth Image", cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))
        cv2.imshow("Color Image", color_image)
        cv2.waitKey(1)

        # Get intrinsics
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        new_points = []
        new_colors = []

        # Process every 2nd pixel for performance boost
        for y in range(0, depth_image.shape[0], 4):  
            for x in range(0, depth_image.shape[1], 4):  
                depth_value = depth_image[y, x] * depth_scale  # Convert to meters
                if depth_value <= 0.1 or depth_value > 5.0:  # Ignore anything too close or too far
                    continue

                # Convert to 3D point
                point = tuple(rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value))

                # Only add if the point is not already seen
                if point not in seen_points:
                    seen_points.add(point)
                    new_points.append(point)
                    new_colors.append(color_image[y, x] / 255.0)  # Normalize color

        if len(new_points) == 0:
            print("⚠️ WARNING: No new unique 3D points found in this frame!")
            continue
        else:
            print(f"✅ Captured {len(new_points)} new points.")

        # Convert to Open3D format
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.array(new_points))
        new_pcd.colors = o3d.utility.Vector3dVector(np.array(new_colors))

        # Merge new points into global point cloud
        global_pcd += new_pcd  # This now **adds** points instead of replacing them

        print(f"Total accumulated points: {len(global_pcd.points)}")  # Debug output

        # Downsample to keep it lightweight (optional)
        global_pcd = global_pcd.voxel_down_sample(voxel_size=0.005)

        # Update Open3D visualizer (FAST REAL-TIME UPDATE)
        vis.clear_geometries()
        vis.add_geometry(global_pcd)
        vis.update_geometry(global_pcd)
        vis.poll_events()
        vis.update_renderer()

        if keyboard.is_pressed('q'):
            o3d.io.write_point_cloud("point_cloud.ply", global_pcd)


            

finally:
    # Stop RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()

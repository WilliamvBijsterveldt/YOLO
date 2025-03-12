import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        plt.imshow(color_image)
        plt.axis("off")  # Hide axes
        plt.pause(0.01)  # Small delay to refresh the image
        plt.clf()  # Clear the previous frame

finally:
    pipeline.stop()
    plt.close()

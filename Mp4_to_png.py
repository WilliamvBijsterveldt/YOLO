import cv2
import os

def capture_screenshots_from_videos(video_folder, output_folder, interval):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.MOV', '.mkv'))]
    video_files.sort()  # Ensure order if needed
    
    frame_count = 0
    screenshot_count = 0
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open {video_file}")
            continue
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        frame_interval = int(fps * interval)  # Convert time interval to frame count
        
        print(f"Processing {video_file}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit if video ends
            
            if frame_count % frame_interval == 0:
                screenshot_filename = os.path.join(output_folder, f"screenshot_{screenshot_count:06d}.jpg")
                cv2.imwrite(screenshot_filename, frame)
                print(f"Saved: {screenshot_filename}")
                screenshot_count += 1
            
            frame_count += 1
        
        cap.release()
    
    print("Done!")

# Example usage
video_folder = "Data/videos"  # Folder containing video files
output_folder = "Data/images_raw"  # Folder where screenshots will be saved
interval = 0.5  # Interval in seconds

capture_screenshots_from_videos(video_folder, output_folder, interval)

import os
import sys
import rosbag
import numpy as np
import cv2
from dvs_msgs.msg import EventArray

# Function to create a grayscale image from events
def create_grayscale_image(events, target_width, target_height, original_width=640, original_height=480):
    intensity_image = np.zeros((target_height, target_width), dtype=np.float32)
    
    # Scale factor
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    for event in events:
        # Scale the coordinates to match the target resolution
        x = int(event.x * scale_x)
        y = int(event.y * scale_y)
        polarity = event.polarity
        
        if x >= target_width or y >= target_height or x < 0 or y < 0:
            continue
        
        intensity_image[y, x] += 1 if polarity else -1
    
    min_intensity = np.min(intensity_image)
    max_intensity = np.max(intensity_image)
    
    if max_intensity > min_intensity:
        normalized_image = 255 * (intensity_image - min_intensity) / (max_intensity - min_intensity)
    else:
        normalized_image = intensity_image
    
    grayscale_image = normalized_image.astype(np.uint8)
    return grayscale_image

# Ensure the script receives the required arguments
if len(sys.argv) != 3:
    print("Usage: python3 nombre.py <rosbag_path> <output_dir>")
    sys.exit(1)

# Get the arguments from the command line
rosbag_path = sys.argv[1]
output_dir = os.path.join(sys.argv[2], "rosbag")

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

print(f"Output directory is set to: {output_dir}")

# Initialize parameters
width, height = 304, 240  # Example resolution, adjust as needed

# Open the rosbag file
bag = rosbag.Bag(rosbag_path)
print(f"Opened rosbag file: {rosbag_path}")

# Process events from the /dvxplorer_left/events topic
for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dvxplorer_left/events'])):
    if topic == '/dvxplorer_left/events':
        print(f"Processing message {i+1} at timestamp {t.to_nsec()} (ns)")

        
        # Create a grayscale image from the events
        grayscale_image = create_grayscale_image(msg.events, width, height)
        
        # Convert the timestamp to microseconds
        timestamp_ns = t.to_nsec()
        timestamp_us = timestamp_ns // 1000  # Convert nanoseconds to microseconds
        print(f"Converted timestamp to {timestamp_ns} (us)")
        
        # Generate the filename using the full timestamp in microseconds
        new_filename = f"{timestamp_ns}.png"
        print(f"Generated filename: {new_filename}")
        
        # Construct the full path to save the image
        image_filename = os.path.join(output_dir, new_filename)
        print(f"Saving image to: {image_filename}")
        
        # Save the image with the new filename
        cv2.imwrite(image_filename, grayscale_image)
        print(f"Image saved successfully.\n")

bag.close()
print("Finished processing all messages.")


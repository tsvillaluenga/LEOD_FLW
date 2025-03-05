import os
import sys
import rosbag
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
# Ensure the script receives the required arguments
if len(sys.argv) != 3:
    print("Usage: python3 nombre.py <rosbag_path> <output_dir>")
    sys.exit(1)

# Get the arguments from the command line
rosbag_path = sys.argv[1]
output_dir = os.path.join(sys.argv[2], "rgb")

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

print(f"Output directory is set to: {output_dir}")

# Open the rosbag file
bag = rosbag.Bag(rosbag_path)
print(f"Opened rosbag file: {rosbag_path}")

# Process events from the /dvxplorer_left/events topic
for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['/rgb/image_raw'])):
    if topic == '/rgb/image_raw':
        
        print(f"Processing message {i+1} at timestamp {t.to_nsec()} (ns)")
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  # Convertir a imagen OpenCV
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
        cv2.imwrite(image_filename, cv_image)
        print(f"Image saved successfully.\n")

bag.close()
print("Finished processing all messages.")


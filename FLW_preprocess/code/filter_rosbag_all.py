#!/usr/bin/env python3

import os
import rosbag
import shutil

# Path to the folder containing the rosbags to process (modify this with your specific path)
rosbag_directory = "/root/bags"

# Define the topics to keep
topics_to_keep = {"/dvxplorer_left/events"}  # , "/rgb/image_raw"}

def filterRosbag(input_bag_path, topics_to_keep):
    # Define the name for the temporary filtered rosbag
    temp_bag_path = input_bag_path + ".filtered"

    # Open the original rosbag in read mode and create a new rosbag for writing
    with rosbag.Bag(input_bag_path, 'r') as in_bag, rosbag.Bag(temp_bag_path, 'w') as out_bag:
        for topic, msg, t in in_bag.read_messages():
            if topic in topics_to_keep:
                out_bag.write(topic, msg, t)

    # Replace the original rosbag with the filtered one
    os.remove(input_bag_path)  # Remove the original
    shutil.move(temp_bag_path, input_bag_path)  # Rename the filtered bag with the original name
    print(f"Filtered rosbag created: {input_bag_path}")

# Iterate over all .bag files in the specified folder
for filename in os.listdir(rosbag_directory):
    if filename.endswith(".bag"):
        rosbag_path = os.path.join(rosbag_directory, filename)
        print(f"Processing: {rosbag_path}")
        filterRosbag(rosbag_path, topics_to_keep)
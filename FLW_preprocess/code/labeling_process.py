#!/usr/bin/env python3
import os
import subprocess
import shutil

def main():
    # Define the path of the parent folder that contains the subfolders to process
    parent_path = "/home/admin_flw/TUD_Thesis/flw_dataset/zivid"
    
    # Verify that the path exists
    if not os.path.isdir(parent_path):
        print(f"Error: The path '{parent_path}' does not exist or is not a directory.")
        return
    
    # Iterate over each item in the parent folder
    for item in os.listdir(parent_path):
        # Build the full path of the subfolder
        X_PATH = os.path.join(parent_path, item)
        
        # Verify that it is a directory
        if os.path.isdir(X_PATH):
            # Build the paths of the required subfolders
            images_dir = os.path.join(X_PATH, "segmentation")
            rosbag_dir = os.path.join(X_PATH, "rosbag")
            rgb_dir    = os.path.join(X_PATH, "rgb")
            utils_dir  = os.path.join(X_PATH, "utils")
            output_masks_dir = os.path.join(X_PATH, "output_masks")
            
            # Check if the required subfolders exist
            if all(os.path.isdir(d) for d in [images_dir, rosbag_dir, rgb_dir]):
                print(f"Processing folder: {X_PATH}")
                
                # Execute the first script with X_PATH as argument
                try:
                    subprocess.run([
                        "python3",
                        "/home/admin_flw/TUD_Thesis/RVT_FLW_Dataset_Github-main/FLW_Dataset/update/events_to_grayscaleV2.py",
                        X_PATH
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error executing events_to_grayscaleV2.py in {X_PATH}: {e}")
                    continue
                
                # Execute the second script with X_PATH as argument
                try:
                    subprocess.run([
                        "python3",
                        "/home/admin_flw/TUD_Thesis/RVT_FLW_Dataset_Github-main/FLW_Dataset/update/create_labelsv5.py",
                        X_PATH
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error executing create_labelsv5.py in {X_PATH}: {e}")
                    continue
            else:
                # If any required subfolder is missing, skip this directory
                print(f"Skipping this folder (missing subfolders) -> {X_PATH}")
        else:
            # If it is not a directory, skip it as well
            print(f"Skipping this item (not a directory) -> {X_PATH}")
            
        # Move "segmentation" and "output_masks" to "utils"
        if not os.path.exists(utils_dir):
            print(f"The destination folder '{utils_dir}' does not exist. Creating it...")
            os.makedirs(utils_dir)
        
        try:
            shutil.move(images_dir, utils_dir)
            print(f"Moved '{images_dir}' to '{utils_dir}'")
        except Exception as e:
            print(f"Error moving '{images_dir}': {e}")
            
        try:
            shutil.move(output_masks_dir, utils_dir)
            print(f"Moved '{output_masks_dir}' to '{utils_dir}'")
        except Exception as e:
            print(f"Error moving '{output_masks_dir}': {e}")


if __name__ == "__main__":
    main()
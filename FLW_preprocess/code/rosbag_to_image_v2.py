import os
import subprocess
from pathlib import Path

# Define paths
rosbags_dir = "/root/bags/"  # Replace with the folder containing .bag files
output_root_dir = "/root/bags/out/"  # Replace with the root output directory

# Function to execute a shell command
def execute_command(command, cwd=None, use_bash=False):
    print(f"Executing: {command}")
    if use_bash:
        command = f"bash -c \"{command}\""
    result = subprocess.run(command, shell=True, cwd=cwd, executable='/bin/bash')
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {command}")

# Expand user directory (~) in paths
def expand_path(path):
    return os.path.expanduser(path)

# Step 1: Build and configure ROS environment
catkin_ws_path = expand_path("~/catkin_ws")
execute_command("catkin_make", cwd=catkin_ws_path)
execute_command("source ~/catkin_ws/devel/setup.bash", cwd=catkin_ws_path, use_bash=True)
execute_command("export PYTHONPATH=$PYTHONPATH:~/catkin_ws/devel/lib/python3.x/site-packages", use_bash=True)


# Iterate over all .bag files in the directory
for carpeta in os.listdir(rosbags_dir):
    for bag_file in Path(rosbags_dir, carpeta).glob("*.bag"):
        print("Processing complete.")
        rosbag_path = str(bag_file)
        output_dir = str(bag_file.with_suffix(""))  # Remove the .bag extension to form the output directory

        # Check if the output directory already exists
        if Path(output_dir).exists():
            print(f"Skipping {rosbag_path} as {output_dir} already exists.")
            continue

        print(f"Processing rosbag file: {rosbag_path}")
        print(f"Output directory: {output_dir}")

        # Step 2: Extract left camera events
        if bag_file.stem.startswith("left"):
            execute_command("cd /root/TUD_Thesis/RVT_FLW_Dataset_Github-main/FLW_Dataset/update", use_bash=True)
            ros_events_script = "./rosbag_extraction.py"
            execute_command(f"python3.8 {ros_events_script} {rosbag_path} {output_dir}")
        
        # Step 3: Extract rgb images
        if not bag_file.stem.startswith("left") and not bag_file.stem.startswith("right"):
            execute_command("cd /root/TUD_Thesis/RVT_FLW_Dataset_Github-main/FLW_Dataset/update", use_bash=True)
            ros_events_script = "./rosbag_rgb_extraction.py"
            execute_command(f"python3.8 {ros_events_script} {rosbag_path} {output_dir}")

print("Processing complete.")


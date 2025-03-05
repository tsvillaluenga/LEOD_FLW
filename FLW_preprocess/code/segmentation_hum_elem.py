import subprocess
import os

# Path to the .txt file
input_file = "/home/admin_flw/TUD_Thesis/flw_dataset/coor_labels.txt"

# Paths to the scripts to be executed
script_segment_anything = "/home/admin_flw/TUD_Thesis/RVT_FLW_Dataset_Github-main/FLW_Dataset/update/segment_anything_flw.py"
script_seg_to_json = "/home/admin_flw/TUD_Thesis/RVT_FLW_Dataset_Github-main/FLW_Dataset/update/seg_to_json.py"

def rename_files_in_directory(directory_path):
    """
    Changes the extension of files from .png to .jpg in a given directory.
    """
    if not os.path.exists(directory_path):
        print(f"Warning: The folder '{directory_path}' does not exist. Skipping file renaming.")
        return

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.png'):
            old_file_path = os.path.join(directory_path, filename)
            new_file_name = filename[:-4] + '.jpg'
            new_file_path = os.path.join(directory_path, new_file_name)
            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_file_name}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

# Read the file line by line
with open(input_file, "r") as file:
    lines = file.readlines()

# Process each line of the file
for line in lines:
    parts = line.strip().split()
    if len(parts) < 5:
        print(f"Invalid line: {line.strip()}")
        continue

    root_dir = parts[0]
    video_dir = os.path.join(root_dir, "rgb")
    
    if not os.path.exists(video_dir) or not os.listdir(video_dir):
        print(f"Error: No images found in {video_dir}. Skipping this directory.")
        continue
    
    xmin2, ymin2, xmax2, ymax2 = parts[1:5]
    rename_files_in_directory(os.path.join(root_dir, "images"))
    rename_files_in_directory(os.path.join(root_dir, "rgb"))
    
    out_dir_element = f"{root_dir}/output_masks/element/"
    os.makedirs(out_dir_element, exist_ok=True)
    out_dir_human = None
    
    if len(parts) >= 9:
        xmin1, ymin1, xmax1, ymax1 = parts[1:5]
        out_dir_human = f"{root_dir}/output_masks/human/"
        os.makedirs(out_dir_human, exist_ok=True)
        
        command_human = [
            "python3", script_segment_anything,
            "--root_dir", root_dir,
            "--x_min", xmin1,
            "--y_min", ymin1,
            "--x_max", xmax1,
            "--y_max", ymax1,
            "--out_dir", out_dir_human
        ]
        print(f"Executing: {' '.join(command_human)}")
        subprocess.run(command_human)
        xmin2, ymin2, xmax2, ymax2 = parts[5:9]

    command_element = [
        "python3", script_segment_anything,
        "--root_dir", root_dir,
        "--x_min", xmin2,
        "--y_min", ymin2,
        "--x_max", xmax2,
        "--y_max", ymax2,
        "--out_dir", out_dir_element
    ]
    print(f"Executing: {' '.join(command_element)}")
    subprocess.run(command_element)

    last_dir_name = os.path.basename(os.path.normpath(root_dir))
    base_name = last_dir_name.split("_")[0]

    json_output_element = f"{root_dir}/segmentation/rgb_bounding_box_labels_{base_name}.json"
    os.makedirs(os.path.dirname(json_output_element), exist_ok=True)
    if not os.path.exists(json_output_element):
        with open(json_output_element, "w") as f:
            f.write("{}")
    
    if out_dir_human and os.listdir(out_dir_human):
        json_output_human = f"{root_dir}/segmentation/rgb_bounding_box_labels_human.json"
        if not os.path.exists(json_output_human):
            with open(json_output_human, "w") as f:
                f.write("{}")
        command_seg_to_json_human = [
            "python3", script_seg_to_json,
            out_dir_human,
            json_output_human
        ]
        print(f"Executing: {' '.join(command_seg_to_json_human)}")
        subprocess.run(command_seg_to_json_human)

    if os.listdir(out_dir_element):
        command_seg_to_json_element = [
            "python3", script_seg_to_json,
            out_dir_element,
            json_output_element
        ]
        print(f"Executing: {' '.join(command_seg_to_json_element)}")
        subprocess.run(command_seg_to_json_element)
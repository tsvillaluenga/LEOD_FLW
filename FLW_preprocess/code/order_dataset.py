import os
import shutil

# Example paths (adjust according to your scenario)
MAIN_DIR = "/home/admin_flw/TUD_Thesis/flw_dataset/zivid_SSOD"      # Original directory
OUT_PATH = "/media/admin_flw/Volume/LEOD_FLW/datasets/flw_dataset"  # Destination directory

# List of the 2 subfolders (Y_DIRS) that you want to copy in each XX_PATH folder
subfolders_to_copy = ["images", "event_representations_v2"]

def replicate_structure(main_dir, out_dir, subfolders):
    """
    Creates in out_dir the same folders (XXO_PATH) that exist in main_dir (XX_PATH),
    and copies only the subfolders listed in 'subfolders' into each one.
    Then, moves the subsubfolder 'images/visualized_images/labels_v2' to the root of xxo_path.
    """
    # 1. Create the base output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 2. Traverse the XX_PATH folders in MAIN_DIR
    for xx_folder in os.listdir(main_dir):
        xx_path = os.path.join(main_dir, xx_folder)
        
        # Check that it is a directory (skip files)
        if os.path.isdir(xx_path):
            # Create the corresponding folder in OUT_PATH with the same name
            xxo_path = os.path.join(out_dir, xx_folder)
            os.makedirs(xxo_path, exist_ok=True)

            # 3. Copy only the subfolders specified in 'subfolders_to_copy'
            for subf in subfolders:
                source_subfolder = os.path.join(xx_path, subf)
                target_subfolder = os.path.join(xxo_path, subf)
                
                # If the subfolder exists in the original folder, copy it
                if os.path.isdir(source_subfolder):
                    # Remove target if it already exists, to avoid copytree errors
                    if os.path.exists(target_subfolder):
                        shutil.rmtree(target_subfolder)
                    
                    shutil.copytree(source_subfolder, target_subfolder)
                    print(f"Copied: {source_subfolder} -> {target_subfolder}")
                else:
                    print(f"The subfolder '{subf}' does not exist in '{xx_path}'. Skipped.")

            # 4. Move the subfolder 'images/visualized_images/labels_v2' to xxo_path
            labels_v2_source = os.path.join(xxo_path, "images", "visualized_images", "labels_v2")
            labels_v2_target = os.path.join(xxo_path, "labels_v2")

            if os.path.exists(labels_v2_source):
                # Move the folder
                shutil.move(labels_v2_source, labels_v2_target)
                print(f"Moved: {labels_v2_source} -> {labels_v2_target}")

                # Optional: clean up the 'images/visualized_images' path if it is empty
                visualized_images_dir = os.path.join(xxo_path, "images", "visualized_images")
                try:
                    # os.removedirs deletes the specified folder and, recursively, its empty parents
                    os.removedirs(visualized_images_dir)
                except OSError:
                    # Occurs if the folder is not empty or does not exist
                    pass

if __name__ == "__main__":
    replicate_structure(MAIN_DIR, OUT_PATH, subfolders_to_copy)
import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

def transform_points(H, points):
    """
    Transforms a list of points (x, y) with a 3x3 transformation matrix.
    Assumes homogeneous coordinates [x, y, 1].
    """
    transformed_points = []
    for x, y in points:
        point_hom = np.array([x, y, 1.0], dtype=np.float64)
        result = H @ point_hom  # Matrix multiplication
        # Normalize by the third component
        if abs(result[2]) > 1e-12:
            x_t = result[0] / result[2]
            y_t = result[1] / result[2]
        else:
            # Avoid division by zero in case of anomalous values
            x_t = result[0]
            y_t = result[1]
        transformed_points.append((x_t, y_t))
    return transformed_points

def update_timestamps_json(json_labels_data, txt_file_path):
    """
    Matching RGB timestamps with Events timestamps of the labels
    """

    # 1. Build the dictionary by reading the TXT file.
    timestamp_map = {}
    with open(txt_file_path, 'r', encoding='utf-8') as f_txt:
        for line in f_txt:
            line = line.strip()
            if not line:
                continue
            # Split by ' -- '
            parts = line.split('--')
            if len(parts) == 2:
                ts1, ts2 = parts
                ts1 = ts1.strip()
                ts2 = ts2.strip()
                timestamp_map[ts1] = ts2

    # 2. For each JSON file in json_labels_data
    for json_file in json_labels_data:
        # Temporarily store modified lines
        modified_lines = []

        # Open the JSON and read it line by line
        with open(json_file, 'r', encoding='utf-8') as f_json:
            for json_line in f_json:
                json_line = json_line.strip()
                if not json_line:
                    continue
                # Convert each line to dict
                try:
                    data = json.loads(json_line)
                except json.JSONDecodeError:
                    # If a line is not valid JSON, ignore it or add it as is
                    modified_lines.append(json_line)
                    continue

                # Check if the timestamp is in our dictionary
                current_ts = data.get("timestamp", "")
                if current_ts in timestamp_map:
                    # Replace with the second timestamp
                    data["timestamp"] = timestamp_map[current_ts]

                # Convert back to JSON to store in modified_lines
                modified_lines.append(json.dumps(data))

        # Overwrite the same JSON file with the updated lines
        with open(json_file, 'w', encoding='utf-8') as f_json_out:
            for mod_line in modified_lines:
                f_json_out.write(mod_line + "\n")

def order_json_by_timestamp(json_labels_data):
    """
    Orders each file contained in json_labels_data (list of paths) 
    by the 'timestamp' value. Overwrites the original file with 
    the sorted lines and returns the same list of paths 
    (in case you want to read the content again).
    """
    for file_path in json_labels_data:
        # Read all lines (each line is a JSON object)
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        
        # Parse each line as a JSON object
        data_list = [json.loads(line) for line in lines if line.strip()]

        # Sort the list according to 'timestamp' (converted to integer for correct order)
        data_list.sort(key=lambda x: int(x["timestamp"]))

        # Write all lines back into the file in order
        with open(file_path, 'w') as f:
            for item in data_list:
                f.write(json.dumps(item) + '\n')
    
    return json_labels_data  # Returns the sorted files

def main(MAIN_PATH):
    # Directories based on MAIN_PATH
    image_dir = Path(MAIN_PATH) / 'images'
    visualization_dir = image_dir / 'visualized_images'
    visualization_dir.mkdir(exist_ok=True)
    labels_dir = visualization_dir / 'labels_v2'
    labels_dir.mkdir(exist_ok=True)
    txt_file_path = Path(MAIN_PATH) / 'utils/event_images/mapping.txt'

    # Dictionaries and lists to store results
    grouped_labels = {}
    labels = []
    image_timestamps = []
    objframe_idx_2_label_idx = []
    height, width = 240, 304

    # Labeling files (we search in the "segmentation" folder inside MAIN_PATH) and sort the data
    json_labels_data = list((Path(MAIN_PATH) / 'segmentation').glob("rgb_bounding_box_labels_*.json"))
    json_labels_data = order_json_by_timestamp(json_labels_data)
    
    # Transformation matrices (3x3) for each case (NOT inverted).
    # Rectangle 1 (Human)
    H_human = np.array([
        [0.1591,        -4.51e-15,     -11.4773],
        [1.47e-14,       0.1757,       -12.3579],
        [1.17e-16,      -7.02e-17,       1.0   ]
    ])

    # Rectangle 2 (Other element)
    H_other = np.array([
        [0.1647,        -1.23e-13,     -11.7647],
        [1.64e-13,       0.1619,        -3.0211],
        [1.81e-15,      -1.69e-15,       1.0   ]
    ])

    # Matching RGB timestamps with Events
    update_timestamps_json(json_labels_data, txt_file_path)
    
    print("Processing JSON files...")
    for json_file in json_labels_data:
        class_id = json_file.stem.split("_")[-1]  # Extracts "human" or other label from the file name
        print(f"Processing JSON file: {json_file}")
        with open(json_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                timestamp = int(data["timestamp"])
                bbox = [
                    int(data["xmin"]),
                    int(data["ymin"]),
                    int(data["xmax"]),
                    int(data["ymax"]),
                    class_id
                ]
                if timestamp not in grouped_labels:
                    grouped_labels[timestamp] = []
                grouped_labels[timestamp].append(bbox)

    print(f"Grouped labels by timestamp: {list(grouped_labels.keys())[:10]}...")

    # Process each timestamp and image
    print("Processing images...")
    for timestamp, bboxes in grouped_labels.items():
        # Search for image file with compatible extensions
        image_path = None
        for ext in ['.png', '.jpg']:
            temp_path = image_dir / f"{timestamp}{ext}"
            if temp_path.is_file():
                image_path = temp_path
                break
        
        if image_path is None:
            print(f"Image for timestamp {timestamp} not found in formats .png or .jpg")
            continue

        cv_image = cv2.imread(str(image_path))
        frame_labels = []

        print(f"Drawing bounding boxes for timestamp: {timestamp}")
        for bbox in bboxes:
            x1, y1, x2, y2, class_id = bbox
            points = [(x1, y1), (x2, y2)]
            
            # Choose the transformation matrix
            if class_id == 'human':
                H = H_human
                color = (0, 255, 0)
            else:
                H = H_other
                color = (255, 0, 0)
            
            # Transform points
            transformed_points = transform_points(H, points)
            print(f"Transformed points for '{class_id}': {transformed_points}")
            
            # Assign the new coordinates for the bounding box
            x1_t = int(round(transformed_points[0][0])) - 2
            y1_t = int(round(transformed_points[0][1])) - 2
            x2_t = int(round(transformed_points[1][0])) + 2
            y2_t = int(round(transformed_points[1][1])) + 2
            
            # Avoid exceptional cases
            if x1_t < 0: x1_t = 0
            elif x1_t > width: x1_t = width
            
            if x2_t < 0: x2_t = 0
            elif x2_t > width: x2_t = width
            
            if y1_t < 0: y1_t = 0
            elif y1_t > height: y1_t = height
            
            if y2_t < 0: y2_t = 0
            elif y2_t > height: y2_t = height

            if (x2_t - x1_t) <= 0 or (y2_t - y1_t) <= 0:
                continue
            
            # Draw the rectangle and the label on the image
            cv2.rectangle(cv_image, (x1_t, y1_t), (x2_t, y2_t), color, 1)
            label = f"{class_id}"
            cv2.putText(cv_image, label, (x1_t, y1_t - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            # Save the label data (width and height derived from x2_t - x1_t, etc.)
            frame_labels.append((
                timestamp, 
                x1_t, 
                y1_t, 
                x2_t - x1_t,  # width
                y2_t - y1_t,  # height
                1 if class_id == 'human' else 0,  # Numeric class ID
                1.0,  # confidence (dummy)
                1 if class_id == 'human' else 0     # track_id (dummy)
            ))

        if frame_labels:
            objframe_idx_2_label_idx.append(len(labels))  # Index of the first label of this frame
            labels.extend(frame_labels)
            image_timestamps.append(timestamp)

        # Save visualized image
        output_path = visualization_dir / f"{timestamp}.png"
        cv2.imwrite(str(output_path), cv_image)
        print(f"Saved visualized image: {output_path}")

    # Convert labels to structured array
    labels_dtype = [
        ('t', '<u8'), 
        ('x', '<f4'), 
        ('y', '<f4'), 
        ('w', '<f4'), 
        ('h', '<f4'), 
        ('class_id', 'u1'), 
        ('class_confidence', '<f4'), 
        ('track_id', '<u4')
    ]
    labels_array = np.array(labels, dtype=labels_dtype)

    # Convert objframe_idx_2_label_idx to numpy array and save it
    objframe_idx_2_label_idx = np.array(objframe_idx_2_label_idx, dtype=np.int64)
    np.savez(labels_dir / 'labels.npz',
             labels=labels_array,
             objframe_idx_2_label_idx=objframe_idx_2_label_idx)

    # Save timestamps
    image_timestamps = np.array(image_timestamps, dtype=np.int64)
    np.save(labels_dir / 'timestamps_us.npy', image_timestamps)

    print("Object detection and annotation complete. Visualized images saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to process and annotate bounding boxes on images."
    )
    parser.add_argument(
        "MAIN_PATH",
        type=str,
        help="Main path to the dataset (e.g., /root/TUD_Thesis/flw_dataset/)"
    )
    args = parser.parse_args()
    main(args.MAIN_PATH)
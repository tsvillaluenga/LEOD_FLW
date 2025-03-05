import os
import json
import cv2
from pathlib import Path
import numpy as np

# Paths
image_dir = Path('/home/admin_flw/TUD_Thesis/flw_dataset/zivid/zivid_2/rgb')  # RGB images
visualization_dir = Path('/home/admin_flw/TUD_Thesis/flw_dataset/zivid/zivid_2/prueba_labels_new')  # Labeled images
visualization_dir.mkdir(exist_ok=True)
labels_dir = visualization_dir / 'labels_v2'
labels_dir.mkdir(exist_ok=True)
segmentation_dir = Path('/home/admin_flw/TUD_Thesis/flw_dataset/zivid/zivid_2/segmentation')  # Segmentation JSONs
mapping_file = Path('/home/admin_flw/TUD_Thesis/flw_dataset/zivid/zivid_2/utils/event_images/mapping.txt')

# Read mapping file to create a dictionary
rgb_to_segmentation = {}
with open(mapping_file, 'r') as f:
    for line in f:
        rgb_ts, seg_ts = line.strip().split(" -- ")
        rgb_to_segmentation[int(rgb_ts)] = int(seg_ts)

# Dictionary to store bounding boxes
grouped_labels = {}
labels = []
image_timestamps = []
objframe_idx_2_label_idx = []

# Process JSON files
json_files = list(segmentation_dir.glob("rgb_bounding_box_labels_*.json"))
print("Processing JSON files...")

for json_file in json_files:
    class_id = json_file.stem.split("_")[-1]  # Extract class_id
    print(f"Processing JSON file: {json_file}")
    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            seg_timestamp = int(data["timestamp"])
            if seg_timestamp not in rgb_to_segmentation.values():
                continue  # Ignore timestamps not in the mapping

            # Find the corresponding RGB timestamp
            rgb_timestamp = next((k for k, v in rgb_to_segmentation.items() if v == seg_timestamp), None)
            if rgb_timestamp is None:
                continue

            bbox = [
                int(data["xmin"]),
                int(data["ymin"]),
                int(data["xmax"]),
                int(data["ymax"]),
                class_id
            ]
            if rgb_timestamp not in grouped_labels:
                grouped_labels[rgb_timestamp] = []
            grouped_labels[rgb_timestamp].append(bbox)

print(f"Grouped labels by timestamp: {list(grouped_labels.keys())[:10]}...")

# Process images
print("Processing images...")
for rgb_timestamp, bboxes in grouped_labels.items():
    image_path_png = image_dir / f"{rgb_timestamp}.png"
    image_path_jpg = image_dir / f"{rgb_timestamp}.jpg"

    if image_path_png.is_file():
        image_path = image_path_png
    elif image_path_jpg.is_file():
        image_path = image_path_jpg
    else:
        print(f"Image for timestamp {rgb_timestamp} not found: {image_path_png} or {image_path_jpg}")
        continue

    cv_image = cv2.imread(str(image_path))
    frame_labels = []

    print(f"Drawing bounding boxes for timestamp: {rgb_timestamp}")
    for bbox in bboxes:
        x1, y1, x2, y2, class_id = bbox
        x1, x2 = max(0, x1), min(cv_image.shape[1], x2)
        y1, y2 = max(0, y1), min(cv_image.shape[0], y2)

        color = (0, 255, 0) if class_id == 'human' else (255, 0, 0)
        cv_image = cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 3)
        label = f"{class_id}"
        cv2.putText(cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

        frame_labels.append((rgb_timestamp, x1, y1, x2 - x1, y2 - y1, 1 if class_id == 'human' else 0, 1.0, 0))

    if frame_labels:
        objframe_idx_2_label_idx.append(len(labels))
        labels.extend(frame_labels)
        image_timestamps.append(rgb_timestamp)

    output_path = visualization_dir / f"{rgb_timestamp}.png"
    cv2.imwrite(str(output_path), cv_image)
    print(f"Saved visualized image: {output_path}")

# Save labels
labels_dtype = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', '<i4'), ('class_confidence', '<f4'), ('track_id', '<i4')]
labels_array = np.array(labels, dtype=labels_dtype)
objframe_idx_2_label_idx = np.array(objframe_idx_2_label_idx, dtype=np.int64)
image_timestamps = np.array(image_timestamps, dtype=np.int64)

np.savez(labels_dir / 'labels.npz', labels=labels_array, objframe_idx_2_label_idx=objframe_idx_2_label_idx)
np.save(labels_dir / 'timestamps_us.npy', image_timestamps)

print("Object detection and annotation complete. Visualized images saved.")


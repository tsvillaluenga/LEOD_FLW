import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import argparse

def find_closest_timestamp(target_timestamp, timestamps):
    """Finds the closest timestamp in a list to a target timestamp."""
    return min(timestamps, key=lambda x: abs(x - target_timestamp))

def pad_timestamp(filename):
    """Ensures that a filename has a 19-digit timestamp, padding with zeros at the end if necessary."""
    stem = filename.stem
    if len(stem) < 19:
        padded_stem = stem.ljust(19, '0')
        return filename.with_name(f"{padded_stem}{filename.suffix}")
    return filename

def find_existing_file(timestamp, base_path):
    """Searches for a file with a given timestamp, removing trailing zeros until it is found."""
    while timestamp.endswith("0"):
        file_png = base_path / f"{timestamp}.png"
        file_jpg = base_path / f"{timestamp}.jpg"
        if file_png.exists():
            return file_png
        if file_jpg.exists():
            return file_jpg
        timestamp = timestamp[:-1]  # Remove a zero from the end

    # Final search without trailing zeros
    file_png = base_path / f"{timestamp}.png"
    file_jpg = base_path / f"{timestamp}.jpg"
    if file_png.exists():
        return file_png
    if file_jpg.exists():
        return file_jpg
    return None

def create_grayscale_image(image_path, output_path):
    """Converts an image to grayscale and saves it."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_path), grayscale_image)

def main(MAIN_PATH):
    # Paths within the dataset
    RGB_PATH = os.path.join(MAIN_PATH, "rgb")
    XX_PATH = os.path.join(MAIN_PATH, "rosbag")
    EVENT_IMAGES_PATH = os.path.join(MAIN_PATH, "utils", "event_images")
    OUTPUT_IMAGES_PATH = os.path.join(MAIN_PATH, "images")

    # Create output folders if they do not exist
    os.makedirs(EVENT_IMAGES_PATH, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)

    # Get the filenames in both folders
    rgb_images = sorted(Path(RGB_PATH).glob("*.png")) + sorted(Path(RGB_PATH).glob("*.jpg"))
    xx_images = sorted(Path(XX_PATH).glob("*.png")) + sorted(Path(XX_PATH).glob("*.jpg"))

    # Ensure that the filenames have 19 digits
    rgb_images = [pad_timestamp(img) for img in rgb_images]
    xx_images = [pad_timestamp(img) for img in xx_images]

    rgb_images = sorted(rgb_images, key=lambda p: int(p.stem))
    xx_images = sorted(xx_images, key=lambda p: int(p.stem))

    print(f"Number of images in /rgb: {len(rgb_images)}")
    print(f"Number of images in XX_PATH: {len(xx_images)}")

    if not rgb_images:
        print("No images found in the /rgb folder.")
    if not xx_images:
        print("No images found in the XX_PATH folder.")

    if rgb_images and xx_images:
        print(f"Found {len(rgb_images)} images in /rgb and {len(xx_images)} in XX_PATH.")

    # Convert filenames to timestamps
    rgb_timestamps = [int(img.stem) for img in rgb_images]
    xx_timestamps = [int(img.stem) for img in xx_images]

    mapping_file_path = os.path.join(EVENT_IMAGES_PATH, "mapping.txt")

    with open(mapping_file_path, "w") as mapping_file:
        for rgb_img, rgb_timestamp in zip(rgb_images, rgb_timestamps):
            # 1) Evaluate the current timestamp from /rgb
            print(f"Evaluating timestamp from /rgb: {rgb_timestamp}")

            # 2) Find the closest timestamp in XX_PATH
            closest_timestamp = find_closest_timestamp(rgb_timestamp, xx_timestamps)
            print(f"Closest timestamp found: {closest_timestamp}")

            # Search for the corresponding image in XX_PATH, adjusting trailing zeros if necessary
            closest_image = find_existing_file(str(closest_timestamp), Path(XX_PATH))

            if closest_image is None:
                print(f"Image not found for timestamp: {closest_timestamp}")
                continue

            # Copy the image to the event_images folder with the name of the event closest to the corresponding RGB
            original_stem = closest_image.stem  # Get the filename without extension
            new_image_name = f"{original_stem}.png"
            new_image_path = Path(EVENT_IMAGES_PATH) / new_image_name

            print(f"Copying image {closest_image} to {new_image_path}")
            shutil.copy(closest_image, new_image_path)

            # 3) Add information to the .txt file
            mapping_file.write(f"{rgb_timestamp} -- {closest_image.stem}\n")
            print(f"Added to mapping.txt file: {rgb_timestamp} -- {closest_image.stem}")

            # 4) Process the image to grayscale
            output_image_path = Path(OUTPUT_IMAGES_PATH) / new_image_name
            print(f"Converting image to grayscale: {new_image_path} -> {output_image_path}")
            create_grayscale_image(new_image_path, output_image_path)

    print("Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process dataset images.")
    parser.add_argument(
        "MAIN_PATH",
        type=str,
        help="Main path to the dataset (e.g., /root/TUD_Thesis/flw_dataset/element_X)"
    )

    args = parser.parse_args()
    main(args.MAIN_PATH)
import os
import numpy as np
import json
import argparse

def process_npy_files(directory, output_file):
    with open(output_file, "w") as json_file:
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                file_path = os.path.join(directory, filename)

                # Load the .npy file
                matrix = np.load(file_path)

                # Ensure the matrix is binary
                if not np.array_equal(matrix, matrix.astype(bool)):
                    print(f"Warning: {filename} contains non-binary values and will be skipped.")
                    continue

                # Reduce dimensions to keep only the two largest ones
                matrix_shape = sorted(enumerate(matrix.shape), key=lambda x: x[1], reverse=True)
                largest_dims = [matrix_shape[0][0], matrix_shape[1][0]]
                matrix = np.sum(matrix, axis=tuple(i for i in range(matrix.ndim) if i not in largest_dims))

                # Find the x and y bounds
                rows, cols = np.where(matrix == 1)
                if rows.size == 0 or cols.size == 0:
                    print(f"Warning: {filename} contains no 1s and will be skipped.")
                    continue

                xmin, xmax = cols.min(), cols.max()
                ymin, ymax = rows.min(), rows.max()

                # Extract the timestamp from the filename (without .npy)
                timestamp = os.path.splitext(filename)[0]

                # Write the data to the JSON file
                json_data = {
                    "timestamp": timestamp,
                    "xmin": int(xmin),
                    "xmax": int(xmax),
                    "ymin": int(ymin),
                    "ymax": int(ymax),
                }
                json_file.write(json.dumps(json_data) + "\n")

    print(f"Bounding box data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .npy files and save bounding box data to a JSON file.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing .npy files.")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file.")

    args = parser.parse_args()

    process_npy_files(args.directory_path, args.output_file)

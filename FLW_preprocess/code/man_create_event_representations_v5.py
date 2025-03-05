import os
import re
import rosbag
import numpy as np
import h5py

def loadTimestampsSetFromNpz(npz_path):
    """
    Checks if timestamp_ns exists in the first column of npz_path.
    Returns True if it exists, False otherwise.
    """
    # Load the data (use allow_pickle=True if your .npz requires it)
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract the "labels" matrix (adjust the name according to your .npz)
    labels = data['labels']
    
    # labels.shape is expected to be (N,). Each element of 'labels'
    # should be a sequence with 8 elements,
    # where the [0] is the large timestamp.
    first_col = [row[0] for row in labels]
    
    # Convert to int64, in case they come as float
    first_col = np.array(first_col, dtype=np.int64)
    
    # Return a set for average O(1) lookups
    return set(first_col)


def processBagDirectory(output_base_dir):
    """
    This function contains the logic to process a folder (output_base_dir) that
    contains exactly one .bag file. It performs the same operations as your
    original script.
    """

    # =========================================================================
    # 1. LOCATE THE .BAG FILE (it is assumed that there is only one in the folder)
    # =========================================================================
    bag_files = [f for f in os.listdir(output_base_dir) if f.endswith('.bag')]
    if len(bag_files) == 0:
        print(f"[WARNING] No '.bag' file found in {output_base_dir}. Skipping this folder.")
        return
    elif len(bag_files) > 1:
        print(f"[WARNING] More than one '.bag' file found in {output_base_dir}. The first one will be taken and the others will be ignored.")
    
    rosbag_path = os.path.join(output_base_dir, bag_files[0])
    print(f"\nProcessing folder: {output_base_dir}")
    print(f".bag file found: {rosbag_path}")

    # =========================================================================
    # 2. DEFINE OUTPUT PATHS
    # =========================================================================
    # Repeat the original logic to define your output paths
    output_dir = os.path.join(output_base_dir, "event_representations_v2/stacked_histogram_dt=50_nbins=10")
    hdf5_file_path = os.path.join(output_dir, 'event_representations.h5')
    timestamps_file_path = os.path.join(output_dir, 'timestamps_us.npy')
    objframe_idx_file_path = os.path.join(output_dir, 'objframe_idx_2_repr_idx.npy')
    external_timestamps_file_path = os.path.join(output_base_dir, "images/visualized_images/labels_v2/timestamps_us.npy")
    visualized_img_path = os.path.join(output_base_dir, "images/visualized_images")
    labels_path = os.path.join(output_base_dir, "images/visualized_images/labels_v2/labels.npz")
    
    # =========================================================================
    # 3. CREATE DIRECTORIES
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at: {output_dir}")

    # =========================================================================
    # 4. INITIALIZE VARIABLES
    # =========================================================================
    histogram_bins = 20      # Number of bins in each histogram
    time_interval = 50000000    # Interval of 50 us (in microseconds)
    height, width = 240, 304 # TARGET DIMENSIONS
    
    # ORIGINAL DIMENSIONS
    orig_height, orig_width = 480, 640  # Original event image size

    # Scaling factors
    scale_x = width / orig_width
    scale_y = height / orig_height

    stacked_histograms = []
    timestamps = []
    current_histogram = np.zeros((histogram_bins, height, width), dtype=np.uint8)
    start_time = None
    relative_start_time = 0
    current_bin_start_time = 0

    print("Histogram variables initialized.")

    # =========================================================================
    # 5. LOAD EXTERNAL TIMESTAMPS
    # =========================================================================
    try:
        external_timestamps = np.load(external_timestamps_file_path)
    except FileNotFoundError:
        print(f"[ERROR] External timestamps file not found: {external_timestamps_file_path}")
        print("Skipping this folder...\n")
        return

    print(f"External timestamps loaded from: {external_timestamps_file_path}")

    # Adjustment: if external timestamps are in ns, convert them to µs
    external_timestamps = external_timestamps ########// 1000

    # =========================================================================
    # 6. FUNCTION TO FIND MINIMUM AND MAXIMUM TIMES FROM PNG IMAGES
    # =========================================================================
    def low_high_num_finder(folder_path):
        # Verify that the folder exists
        if not os.path.isdir(folder_path):
            raise ValueError(f"The folder '{folder_path}' does not exist.")
        
        files_png = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        if not files_png:
            raise ValueError("No .png files found in the folder.")
        
        # Extract numbers from file names
        numbers = []
        for archivo in files_png:
            match = re.search(r'\d+', archivo)  # Search for a number in the name
            if match:
                numbers.append(int(match.group()))
        
        if not numbers:
            raise ValueError("No numbers found in the .png file names.")
        
        # Find the minimum and maximum values
        lower = min(numbers)
        higher = max(numbers)
        return lower, higher

    # =========================================================================
    # 7. DEFINE START AND END TIMES
    # =========================================================================
    try:
        specified_start_time, specified_end_time = low_high_num_finder(visualized_img_path)
    except ValueError as e:
        print(f"[ERROR]: {str(e)}")
        print("Skipping this folder...\n")
        return

    # timestamps in ns
    specified_start_time = specified_start_time 
    specified_end_time = specified_end_time 
    print(f"Specified_start_time: {specified_start_time}")
    print(f"Specified_end_time: {specified_end_time}")

    # Adjust the start time to match the interval
    start_time_adjustment = specified_start_time - (specified_start_time % time_interval)
    print(f"Start time adjusted to {start_time_adjustment} to include specified timestamps")

    # =========================================================================
    # 8. PROCESS EVENTS FROM THE ROSBAG
    # =========================================================================
    
    # Check labels and events timestamps
    timestamps_set = loadTimestampsSetFromNpz(labels_path)
    print(f"Loaded {len(timestamps_set)} timestamps from {labels_path}.")
    
    try:
        with rosbag.Bag(rosbag_path, 'r') as bag:
            print(f"Opened ROS bag file: {rosbag_path}")
            try:
                for topic, msg, t in bag.read_messages(topics=['/dvxplorer_left/events']):
                    for event in msg.events:
                        # Convert the event's timestamp to microseconds
                        timestamp_ns = t.to_nsec() 
                    ### üDEBUG] ### print(timestamp_ns)
                        
                        # Check if it exists in the first column of your .npz file
                        if timestamp_ns not in timestamps_set:
                            continue
                        else:
                            # # Initialize start_time with the first event
                            # if start_time is None:
                            #     start_time = timestamp_ns
                            #     relative_start_time = start_time_adjustment
                            #     current_bin_start_time = start_time_adjustment
                            #     print(f"Start time set to {start_time} microseconds")
                    ### üDEBUG] ### print(f"{timestamp_ns}: IN")
                            if start_time is None:
                                start_time = timestamp_ns
                                start_time_adjustment = start_time - (start_time % time_interval)  # Adjusted to the first valid event
                                relative_start_time = start_time_adjustment
                                current_bin_start_time = start_time_adjustment
                                print(f"Start time adjusted to {start_time_adjustment} to include specified timestamps")

                            
                            # Calculate the relative time from the adjusted start time
                            relative_time = timestamp_ns - start_time + start_time_adjustment
                            if relative_time < start_time_adjustment:
                    ### üDEBUG] ### print(f"{timestamp_ns}: relative_time < start_time_adjustment")
                                continue

                            # Determine the corresponding bin index for the event
                            bin_idx = (relative_time - current_bin_start_time) // time_interval
                            #print(f"{timestamp_ns}: ¿¿{bin_idx} >= {histogram_bins}??")
                            #print(f"{timestamp_ns}: ------------ ¿¿¿{relative_time} - ({current_bin_start_time}, {current_bin_start_time+time_interval}) = {relative_time - current_bin_start_time}???")
                            
                            
                            while relative_time >= (current_bin_start_time + time_interval): ##bin_idx >= histogram_bins:
                                stacked_histograms.append(current_histogram)
                                timestamps.append([current_bin_start_time, current_bin_start_time + time_interval])
                                print(f"Histogram and timestamps added. Total histograms: {len(stacked_histograms)}")
                                
                                # Reset the histogram for the next bin
                                current_histogram = np.zeros((histogram_bins, height, width), dtype=np.uint8)
                                current_bin_start_time += time_interval #(time_interval*histogram_bins)
                                relative_start_time = current_bin_start_time
                    ### üDEBUG] ### print(f"RESET HIST: ------------ current_bin_start_time: {current_bin_start_time} .. = {relative_time - current_bin_start_time}")
                                bin_idx = (relative_time - current_bin_start_time) // time_interval
                                print(f"New histogram bin initialized. Relative time updated to {relative_start_time} microseconds")

                            if current_bin_start_time > specified_end_time:
                                print(f"Reached the end time: {specified_end_time}. Finishing processing.")
                                raise StopIteration
                            
                            # MAP EVENTS TO NEW RESOLUTION
                            new_x = int(event.x * scale_x)
                            new_y = int(event.y * scale_y)
                            
                            # Fill the histogram with the event
                            if 0 <= new_x < width and 0 <= new_y < height:
                    ### üDEBUG] ### print("Saving event in histogram")
                                current_histogram[int(bin_idx), new_y, new_x] += 1

            except (KeyboardInterrupt, StopIteration):
                print("Stopping data processing...")

    except Exception as e:
        print(f"[ERROR] An issue occurred while reading {rosbag_path}: {str(e)}")
        print("Skipping this folder...\n")
        return

    # Ensure coverage until the specified time is reached
    while current_bin_start_time <= specified_end_time:
        stacked_histograms.append(current_histogram)
        timestamps.append([current_bin_start_time, current_bin_start_time + time_interval])
        print(f"Histogram and timestamps added. Total histograms: {len(stacked_histograms)}")
        
        current_histogram = np.zeros((histogram_bins, height, width), dtype=np.uint8)
        current_bin_start_time += time_interval

    # =========================================================================
    # 9. SAVE TIMESTAMPS IN NPY
    # =========================================================================
    print(f"Saving timestamps to: {timestamps_file_path}")
    np.save(timestamps_file_path, np.array(timestamps))
    print("Timestamps saved.")

    # =========================================================================
    # 10. SAVE HISTOGRAMS IN HDF5 FILE
    # =========================================================================
    print(f"Saving histograms to: {hdf5_file_path}")
    with h5py.File(hdf5_file_path, 'w') as f:
        blosc_filter = 32001  # Blosc filter ID in HDF5
        chunk_shape = (1, histogram_bins, height, width)  # Define chunk shape
        f.create_dataset(
            'data', 
            data=np.array(stacked_histograms),
            compression=blosc_filter,  # Use BLOSC
            chunks=chunk_shape
        )
    print("Histograms saved in HDF5 file (BLOSC compression).")
    # # with h5py.File(hdf5_file_path, 'w') as f:
    # #     f.create_dataset('data', data=np.array(stacked_histograms))
    # # print("Histograms saved in HDF5 file.")

    # =========================================================================
    # 11. GENERATE AND SAVE OBJFRAME_IDX_2_REPR_IDX BASED ON EXTERNAL TIMESTAMPS
    # =========================================================================
    print(f"Generating objframe_idx_2_repr_idx based on external timestamps...")

    objframe_idx_2_repr_idx = []
    unmatched_timestamps = []
    num_histograms = len(timestamps)

    # for ts in external_timestamps:
    #     matched = False
    #     for idx, (ts_start, ts_end) in enumerate(timestamps):
    #         if ts_start <= ts <= ts_end:
    #             objframe_idx_2_repr_idx.append(idx)
    #             matched = True
    #             break
    #     if not matched:
    #         unmatched_timestamps.append(ts)
    #         objframe_idx_2_repr_idx.append(-1)  # Add -1 if no bin is found
    #         print(f"[WARNING] External timestamp {ts} did not match any histogram bin.")
    
    for ts in external_timestamps:
        matched = False
        for idx, (ts_start, ts_end) in enumerate(timestamps):
            if ts_start <= ts <= ts_end:
                objframe_idx_2_repr_idx.append(idx)
                matched = True
                break
        if not matched:
            unmatched_timestamps.append(ts)
            objframe_idx_2_repr_idx.append(len(timestamps) - 1)  # Instead of -1, use the last valid index


    # Ensure that no index is greater than the number of histograms
    objframe_idx_2_repr_idx = np.array(objframe_idx_2_repr_idx)
    objframe_idx_2_repr_idx[objframe_idx_2_repr_idx >= num_histograms] = num_histograms - 1  # Correct out-of-range values
    objframe_idx_2_repr_idx[objframe_idx_2_repr_idx < 0] = 0  # Minimum limit to avoid errors

    # Sort the array in ascending order
    objframe_idx_2_repr_idx = sorted(objframe_idx_2_repr_idx)
    print(f"objframe_idx_2_repr_idx sorted: {objframe_idx_2_repr_idx}")
    
    ##################################################################
    # Ensure that objframe_idx_2_repr_idx is a numpy array
    objframe_idx_2_repr_idx = np.array(objframe_idx_2_repr_idx, dtype=int)

    # Verify indices before saving
    print(f"Total number of histograms generated: {len(timestamps)}")
    print(f"Total number of objframe_idx_2_repr_idx generated: {len(objframe_idx_2_repr_idx)}")
    # Count how many indices are out of the valid range
    out_of_range_count = sum(idx >= len(timestamps) for idx in objframe_idx_2_repr_idx)
    print(f"Indices out of range: {out_of_range_count}")

    # Show the first 10 out-of-range indices
    out_of_range_examples = [idx for idx in objframe_idx_2_repr_idx if idx >= len(timestamps)]
    print(f"Example of out-of-range indices: {out_of_range_examples[:10]}")

    # Force values within the valid range before saving
    objframe_idx_2_repr_idx = np.clip(objframe_idx_2_repr_idx, 0, len(timestamps) - 1)

    ####################################################################

    # Save the corrected file
    np.save(objframe_idx_file_path, objframe_idx_2_repr_idx)
    print(f"Frame indices saved based on external timestamps (total: {len(objframe_idx_2_repr_idx)}).")


    # Show warnings if there were external timestamps without an assigned bin
    if unmatched_timestamps:
        print(f"Warning: {len(unmatched_timestamps)} external timestamps did not match any histogram bin.")
    else:
        print("All external timestamps matched histogram bins.")

    print("Dataset creation completed.")
    print("------------------------------------------------------------\n")


if __name__ == "__main__":
    # =========================================================================
    # Root directory containing multiple subfolders, each with a .bag file
    # =========================================================================
    main_directory = "/root/TUD_Thesis/flw_dataset/zivid"  # <--- ADJUST THIS PATH AS NEEDED

    # Iterate over each item in the main directory
    for folder_name in os.listdir(main_directory):
        # Build the full path of the subfolder
        subdir_path = os.path.join(main_directory, folder_name)

        # We are only interested if it is a directory
        if os.path.isdir(subdir_path):
            # Call the function to process that particular folder
            processBagDirectory(subdir_path)

    print("Processing completed for all subfolders.")
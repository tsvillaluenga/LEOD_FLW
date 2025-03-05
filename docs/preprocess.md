# Preprocess Flow

The preprocessing of the data starts from the raw data obtained from the RGB camera and data camera managed through ROS Noetic. This data will be found in a single ROSBag with at least the topics `/dvxplorer_left/events` and `/rgb/image_raw`.

**IMPORTANT NOTE**: ALL THE ROSBAGS THAT YOU WANT TO BE PROCESSED MUST BE ENTERED IN A NEW SUBFOLDER (_“preprocess_dir”_). All subfolders in this main folder will be preprocessed.


## Scripts
All this scripts you can find them in ([folder FLW_preprocess/code](../FLW_preprocess/code)).


- **rosbag_to_image_v2.py**: This extracts the event information and creates a grayscale reconstruction of the events in the `/images` directory and the RGB images in the `/rosbag` directory.

***MODIFICATIONS:***
rosbags_dir(_“preprocess_dir”_) and output_root_dir (the folder where I want all the data from each rosbag in _“preprocess_dir”_ to be saved) must be modified. The name of the folder with the subfolders will be “output_root_dir/ROSBAG_NAME”. It must be run in Docker with ROS Noetic installed (in the case of Ubuntu 22.04 ROS Noetic can only be run in Docker).


```
List dockers:                          sudo docker ps -a
Init docker:                           sudo docker run -it docker_name
Reinit paused docker:                  sudo docker start -i docker_name
Create new docker sharing documents:   sudo docker run -it --name docker_name -v /path/main/directory/to/share:/root/name/of/directory/inside/docker ros:noetic
```




- **segmentation_hum_elem.py**: This is responsible for creating the masks of each element to be segmented through SAM2 from the RGB images and generating a _.json_ with the coordinates of each bounding box generated.

For this it is necessary to have SAM2 installed ([INSTALLATION VIDEO](https://www.youtube.com/watch?v=zr4yyu6a2UI ))


***MODIFICATIONS:***
You must adjust `input_file` (where the file with the segmentation coordinates for SAM2 is located). If is not created, create it. This must be in the following format:

```
For human mask + element mask:
”
PATH_ROSBAG_TO_MASK1 X_MIN_HUM Y_MIN_HUM X_MAX_HUM Y_MAX_HUM X_MIN_ELEM Y_MIN_ELEM X_MAX_ELEM Y_MAX_ELEM
PATH_ROSBAG_TO_MASK2 X_MIN_HUM Y_MIN_HUM X_MAX_HUM Y_MAX_HUM X_MIN_ELEM Y_MIN_ELEM X_MAX_ELEM Y_MAX_ELEM
PATH_ROSBAG_TO_MASK3 X_MIN_HUM Y_MIN_HUM X_MAX_HUM Y_MAX_HUM X_MIN_ELEM Y_MIN_ELEM X_MAX_ELEM Y_MAX_ELEM
”

For only element mask:
”
PATH_ROSBAG_TO_MASK1 X_MIN_ELEM Y_MIN_ELEM X_MAX_ELEM Y_MAX_ELEM
PATH_ROSBAG_TO_MASK2 X_MIN_ELEM Y_MIN_ELEM X_MAX_ELEM Y_MAX_ELEM
PATH_ROSBAG_TO_MASK3 X_MIN_ELEM Y_MIN_ELEM X_MAX_ELEM Y_MAX_ELEM
”
```




- **labeling_process.py**: This is the file that is responsible for matching the closest timestamps of the event representation with the timestamps of the RGB images. Once the matching is done, it is in charge of obtaining the configurations of the bounding boxes generated for the RGB image, transforms them to adjust them to the representation of events and finally paints them in `/visualized_images`.

***MODIFICATIONS:***
You only have to adjust `parent_path` as _“preprocess_dir”_.

Note that there is a file `utils/event_images/mapping.txt` where the names of the original timestamp files for each event reconstruction image are written, followed by the new timestamp name corresponding to the RGB image (`rgb` folder).




- **man_create_event_representations_v5.py**: This file is used to create the event representations, their linking with the labels and their compression. From here you can adjust the configuration of the histograms to be created.

***MODIFICATIONS:*** Only adjust `main_directory` to _“preprocess_dir”_ inside the main of the file.

- **order_dataset.py**: This file is used to send all the scenes with the relevant data (representation of events and labels only) to train a model to the location where the finished dataset is saved. Basically it copies each scene with its representation of events and labels and pastes them into the indicated destination folder.

***MODIFICATIONS:***
In this one, the MAIN_DIR (“preprocess_dir”) and OUT_DATA (directory where the dataset will be saved) are adjusted.





#### OTHERS:
- **filter_rosbag_all.py**: Allows you to eliminate unwanted topics from a rosbag to save disk space.
- **prueba_labels.py**: generates a new folder called `prueba_labels` where you can see the bounding boxes in the RGB images. It can be useful to know that the bboxes are being generated correctly.



## FINAL DATASET STRUCTURE:

- **event_representations_v2**: representations of the events.
- **images**: images of the reconstruction of the events at certain timestamps
- **images/visualized_images**: images of the reconstruction of the events at certain timestamps with bboxes graphed.
- **images/visualized_images/labels_v2**: labels.
- **rgb**: raw RGB images.
- **rosbag**: first images of the reconstruction of the events.
- **utils**: files used by the scripts to generate pre-processing



### STRUCTURE NEEDED

- scene_XX
  - event_representations_v2
    - stacked_histogram_dt=50_nbins=10
      - event_representations.h5
      - objframe_idx_2_repr_idx.npy
      - timestamps_us.npy
  - labels_v2
    - labels.npz
    - timestamps_us.npy




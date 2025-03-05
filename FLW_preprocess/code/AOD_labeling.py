import requests
import cv2
import numpy as np
from PIL import Image

# 1. Send the request to the API
url = "https://api.landing.ai/v1/tools/agentic-object-detection"
image_path = "/home/admin_flw/TUD_Thesis/flw_dataset/zivid/zivid_4/images/1729174733119152828.png"

# Use 'with' to properly handle file opening and closing
with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    data = {"prompts": ["human"], "model": "agentic"}
    headers = {
        "Authorization": "Basic dDd3YjZmN2s2ZDZvZWQ4Ynk4aWRvOlhsSmEyRXpISEJMUUVscEE1aHZjaWpKTGZ6RXdJOHNR"
    }

    response = requests.post(url, files=files, data=data, headers=headers)

# Check if the response is valid
if response.status_code != 200:
    print(f"Request error: {response.status_code} - {response.text}")
    exit()

# Extract data from the response
response_data = response.json()
print(response_data)

# Check if there are detections
if not response_data.get("data") or not response_data["data"][0]:
    print("No objects detected in the image.")
    exit()

# Extract the bounding box
bounding_box = response_data["data"][0][0]["bounding_box"]  # [x1, y1, x2, y2]
x1, y1, x2, y2 = map(int, bounding_box)  # Convert to integers

# Use 'with' to properly handle file opening and closing
with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    data = {"prompts": ["square"], "model": "agentic"}
    headers = {
        "Authorization": "Basic dDd3YjZmN2s2ZDZvZWQ4Ynk4aWRvOlhsSmEyRXpISEJMUUVscEE1aHZjaWpKTGZ6RXdJOHNR"
    }

    response2 = requests.post(url, files=files, data=data, headers=headers)

# Check if the response is valid
if response2.status_code != 200:
    print(f"Request error: {response2.status_code} - {response2.text}")
    exit()

# Extract data from the response
response2_data = response2.json()
print(response2_data)

# Check if there are detections
if not response2_data.get("data") or not response2_data["data"][0]:
    print("No objects detected in the image.")
    exit()

# Extract the bounding box
bounding_box2 = response2_data["data"][0][0]["bounding_box"]  # [x1, y1, x2, y2]
x1_2, y1_2, x2_2, y2_2 = map(int, bounding_box2)  # Convert to integers

# 3. Load the image with OpenCV
cv_image = cv2.imread(image_path)
if cv_image is None:
    print("Error: Could not load the image.")
    exit()

# 4. Draw the bounding box on the image
class_id = 'human'
color = (0, 255, 0)  # Green for humans
cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 1)
label = f"{class_id}"
cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

class_id = 'zivid'
color = (255, 0, 0)
cv2.rectangle(cv_image, (x1_2, y1_2), (x2_2, y2_2), color, 1)
label = f"{class_id}"
cv2.putText(cv_image, label, (x1_2, y1_2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# 5. Save the image with the bounding box
output_path = "/home/admin_flw/bbox_output.jpg"
cv2.imwrite(output_path, cv_image)

print(f"Image with bounding box saved at: {output_path}")

from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Load YOLO model
model = YOLO('detect_yolov8/last.pt')

# Define the directory containing the test images
image_directory = 'testdetect'

# Create a directory to save cropped images if it doesn't exist
base_output_dir = "save"
os.makedirs(base_output_dir, exist_ok=True)

# Define the subdirectories
output_dirs = ["shoes", "shirt", "pants", "hat", "glasses", "other"]

# Ensure each target subdirectory exists
for dir_name in output_dirs:
    os.makedirs(os.path.join(base_output_dir, dir_name), exist_ok=True)

# List to keep track of previous boxes
previous_boxes = []

for image_name in os.listdir(image_directory):
    # Construct the full path to the image
    image_path = os.path.join(image_directory, image_name)
    # Check if the path is a file (not a directory)
    if os.path.isfile(image_path):
        # Iterate over frames/results
        for frame_results in model.predict(source=image_path, stream=True):
            # Get current boxes and class IDs
            current_boxes = frame_results.boxes.xyxy.cpu().numpy()
            class_ids = frame_results.boxes.cls.cpu().numpy()  # Assuming 'cls' contains class IDs
            
            for box, class_id in zip(current_boxes, class_ids):
                if not any(np.allclose(box, prev_box, atol=10) for prev_box in previous_boxes):
                    # This is a new detection
                    im = Image.fromarray(frame_results.orig_img)
                    # Crop the image using the box coordinates
                    imcrop = im.crop(box[:4])
                    # Save the cropped image with the original name

                    class_labels = model.names[int(class_id)]
                    if class_labels == "shoes":
                        imcrop.save(os.path.join(base_output_dir, "shoes", image_name), "JPEG")
                    elif class_labels == "shirt":
                        imcrop.save(os.path.join(base_output_dir, "shirt", image_name), "JPEG")
                    elif class_labels == "pants":
                        imcrop.save(os.path.join(base_output_dir, "pants", image_name), "JPEG")
                    elif class_labels == "hat":
                        imcrop.save(os.path.join(base_output_dir, "hat", image_name), "JPEG")
                    elif class_labels == "glasses":
                        imcrop.save(os.path.join(base_output_dir, "glasses", image_name), "JPEG")
                    else:  # Default to "other" for any unspecified class labels
                        imcrop.save(os.path.join(base_output_dir, "other", image_name), "JPEG") 
                    # Append the current box to the list of previous boxes
                    previous_boxes.append(box)
                    # Get the class label using the class ID

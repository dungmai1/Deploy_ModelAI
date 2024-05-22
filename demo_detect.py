import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

# model = YOLO("detect_yolov8/last.pt")

# # Read the image
# image = cv2.imread("static/1dec401f-upload_9051bf22e39947af82339be61b7f1477_master.jpg")

# # Perform detection
# results = model.predict(image)

# boxes = results[0].boxes.xyxy.tolist()

# for i, box in enumerate(boxes):
#     x1, y1, x2, y2 = box
#     # Crop the object using the bounding box coordinates
#     ultralytics_crop_object = image[int(y1):int(y2), int(x1):int(x2)]
#     # Save the cropped object as an image
#     cv2.imwrite(f'cropped_object_{i}.jpg', ultralytics_crop_object)





# Load the YOLO model
model = YOLO("detect_yolov8/last.pt")  # load your trained model

# Read the image
image = cv2.imread("test/testdetect/hat.jpg")

# Predict with the model
results = model.predict(image)

# Process the results
for r in results:
    
    annotator = Annotator(image)
    
    boxes = r.boxes
    for box in boxes:
        
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
        
    img = annotator.result()  
    scale_percent = 50  # percent of original size, adjust as needed
    width = int(img.shape[1] * scale_percent / 200)
    height = int(img.shape[0] * scale_percent / 200)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    cv2.imshow('YOLO V8 Detection', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
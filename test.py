from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import numpy as np
from PIL import Image
from detect_yolov8 import image_retrieval
import pickle
import math

app = Flask(__name__)

upload_folder = os.path.join('static')
app.config['UPLOAD_FOLDER'] = upload_folder

def detect_yolo(image_path, model_path='detect_yolov8/last.pt', scale_percent=50):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Perform detection
    results = model.predict(image)
    
    for r in results:
        # Create an annotator
        annotator = Annotator(image)
        
        # Annotate each detected box
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
        
        # Get the resulting annotated image
        annotated_image = annotator.result()
        
        # Resize the image
        width = int(annotated_image.shape[1] * scale_percent / 100)
        height = int(annotated_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(annotated_image, dim, interpolation=cv2.INTER_AREA)

        previous_boxes = []
        current_boxes = r.boxes.xyxy.cpu().numpy()
        for box in current_boxes:
            if not any(np.allclose(box, prev_box) for prev_box in previous_boxes):
                # This is a new detection
                im = Image.fromarray(r.orig_img)
                imcrop = im.crop(box[:4])
                imcrop.save(f"search/crop_{len(previous_boxes)}.jpg", "JPEG")
                previous_boxes.append(box)

        # Save the resized annotated image
        cv2.imwrite(image_path, resized_image)

def testdetect(image_path):
    model = YOLO("detect_yolov8/last.pt")
    
    # Read the image
    image = cv2.imread(image_path)
    # Perform detection
    results = model.predict(image)
    
    boxes = results[0].boxes.xyxy.tolist()

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cropt_image = image[int(y1):int(y2), int(x1):int(x2)]
        # Save the cropped object as an image
        path = f'static\cropped_object_{index}.jpg'
        cv2.imwrite(path, cropt_image)  # Save each cropped object with a unique filename
    
    return path

def draw_boudingbox(image_path):

    model = YOLO("detect_yolov8/last.pt")

    image = cv2.imread(image_path)

    results = model.predict(image)
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
        cv2.imwrite(image_path, resized)

def testimage_retrieval(image_cropt):
    model = image_retrieval.get_extract_model()
    search_vector = image_retrieval.extract_vector(model, image_cropt)
    vectors = pickle.load(open("detect_yolov8/vectors.pkl","rb"))
    paths = pickle.load(open("detect_yolov8/paths.pkl","rb"))

    # Tinh khoang cach tu search_vector den tat ca cac vector
    distance = np.linalg.norm(vectors - search_vector, axis=1)

    # Sap xep va lay ra K vector co khoang cach ngan nhat
    K = 16
    ids = np.argsort(distance)[:K]

    # Tao oputput
    nearest_image = [(paths[id], distance[id]) for id in ids]

    # Ve len man hinh cac anh gan nhat do
    import matplotlib.pyplot as plt

    axes = []
    grid_size = int(math.sqrt(K))
    fig = plt.figure(figsize=(10,5))


    for id in range(K):
        draw_image = nearest_image[id]
        axes.append(fig.add_subplot(grid_size, grid_size, id+1))

        axes[-1].set_title(draw_image[1])
        plt.imshow(Image.open(draw_image[0]))

    fig.tight_layout()
    plt.show()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files['img']
        if image:
            # Secure the filename
            filename = secure_filename(image.filename)
            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("Save =", path_to_save)
            
            # Ensure the upload folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            # Save the file
            image.save(path_to_save)
            draw_boudingbox(path_to_save)
            path = testdetect(path_to_save)
            testimage_retrieval(path)
            return render_template("index.html", image_cropt = path,image_default = path_to_save)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

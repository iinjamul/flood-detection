import warnings
warnings.filterwarnings('ignore')

import cv2
import matplotlib.pyplot as plt

from temp import *
from temp import load_class_names
from mod import Darknet
from flask import Flask
import os
import shutil

app = Flask(__name__, template_folder='template')



def predict(filepath):
    # Set the location and name of the cfg file
    cfg_file = r'C:\Users\rupes\yolov3.cfg'

    # Set the location and name of the pre-trained weights file
    weight_file = r'C:\Users\rupes\yolov3.weights'

    # Set the location and name of the COCO object classes file
    namesfile = r'C:\Users\rupes\coco.names'

    # Load the network architecture
    m = Darknet(cfg_file)

    # Load the pre-trained weights
    m.load_weights(weight_file)

    # Load the COCO object classes
    class_names = load_class_names(namesfile)


    # Set the default figure size
    plt.rcParams['figure.figsize'] = [24.0, 14.0]

    # Load the image
    img = cv2.imread(filepath)

    # Convert the image to RGB
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # We resize the image to the input width and height of the first layer of the network.    
    resized_image = cv2.resize(original_image, (m.width, m.height))


    # Set the NMS threshold
    nms_thresh = 0.6  
    # Set the IOU threshold
    iou_thresh = 0.4

    # Set the default figure size
    plt.rcParams['figure.figsize'] = [24.0, 14.0]

    # Load the image
    img = cv2.imread(filepath)

    # Convert the image to RGB
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # We resize the image to the input width and height of the first layer of the network.    
    resized_image = cv2.resize(original_image, (m.width, m.height))

    # Set the IOU threshold. Default value is 0.4
    iou_thresh = 0.4

    # Set the NMS threshold. Default value is 0.6
    nms_thresh = 0.6

    # Detect objects in the image
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)

    # Print the objects found and the confidence level
    res = print_objects(boxes, class_names)
    
    return res



def get_filenames(root_dir):
    filenames = []
    for filename in os.listdir(root_dir):
        if filename.endswith(".jpg"):
            filenames.append(filename)
    return filenames




@app.route("/")
def home():
    
    root_dir = "Unprocessed"
    processed_dir = "Processed"
    

    for filename in get_filenames(root_dir):
        filepath = root_dir+"/"+filename
        # predict
        
        data = predict(filepath) # Pass filename here
        # Save to Database
        
        shutil.move(root_dir+"/"+filename, processed_dir+"/"+filename)
        
        with open("Result/"+filename+".txt", "w") as f:
            f.write(str(data))
        

    return "All files processed"
    



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)
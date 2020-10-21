import os
import numpy as np
import pandas as pd
from PIL import Image

annotations_path = "C:/dev/ML_Project/data/archive/annotations/Annotation/"
image_path = "C:/dev/ML_Project/data/archive/images/Images/"
data_path = "C:/dev/ML_Project/data/"

def get_paths():
    paths = []
    dirs = os.listdir(annotations_path)
    for dir in dirs:
        files = os.listdir(annotations_path + dir)
        for file in files:
            paths.append(dir + "/" + file)
    return paths

def save_array(data_array):
    np.save(data_path + 'data.csv', data_array, allow_pickle=True, fix_imports=True)

# Convert images from jpg to csv file
def convert_images():
    paths = get_paths()
    images = []
    for path in paths:
        image = Image.open(image_path + path + ".jpg")
        data = np.asanyarray(image)
        images.append(data)
    save_array(np.asanyarray(images))

convert_images()

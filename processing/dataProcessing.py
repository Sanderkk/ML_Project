import os
import numpy as np
import pandas as pd
from PIL import Image

data_path = "C:/dev/ML_Project/data/"
annotations_path = data_path + "archive/annotations/Annotation/"
image_path = data_path + "archive/images/Images/"

def get_paths():
    paths = []
    dirs = os.listdir(annotations_path)
    for dir in dirs:
        files = os.listdir(annotations_path + dir)
        for file in files:
            paths.append(dir + "/" + file)
    return paths

def save_array(data_array):
    np.savetxt(data_path + 'data.csv', np.asanyarray(data_array), delimiter=",")

# Convert images from jpg to csv file
def convert_images():
    paths = get_paths()
    images = []
    max_length = 0
    for path in paths:
        image = Image.open(image_path + path + ".jpg")
        data = np.asanyarray(image).flatten()
        max_length = len(data) if max_length < len(data) else max_length
        images.append(data)

    padded_images = np.zeros([len(images), max_length])
    for i, j in enumerate(images):
        padded_images[i][0:len(j)] = j

    lengthOfImage = 0
    for image in images:
        if (len(image) > lengthOfImage):
            lengthOfImage = len(image)
            print(lengthOfImage)
    save_array(np.asanyarray(images))


convert_images()

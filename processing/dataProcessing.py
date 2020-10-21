import os
import numpy as np
import pandas as pd
from PIL import Image
import progressbar

data_path = "/home/sander/dev/ML_Project/data/"
annotations_path = data_path + "archive/annotations/Annotation/"
image_path = data_path + "archive/images/Images/"


def get_paths(trainingData, trainingDataCount):
    paths = []
    dirs = os.listdir(annotations_path)
    # Tak only part of the data as training or test data
    if (trainingData):
        dirs = dirs[:trainingDataCount]
    else:
        dirs = dirs[trainingDataCount:]
    for dir in dirs:
        files = os.listdir(annotations_path + dir)
        for file in files:
            paths.append(dir + "/" + file)
    return paths


# This is slow as fuck. Uses a lot of ram
def pad_array(data, max_length):
    print("Pad arrays")
    updated_data = []
    for i in progressbar.progressbar(range(len(data))):
        updated_data.append(
            np.pad(
                data,
                (0, max_length - len(data)),
                "constant",
                constant_values=(0,0)
                )
        )
    return np.asanyarray(updated_data)


def read_images(paths):
    print("Read images")
    images = []
    max_length = 0
    for path_i in progressbar.progressbar(range(len(paths))):
        image = Image.open(image_path + paths[path_i] + ".jpg")
        data = np.asanyarray(image).flatten()
        max_length = len(data) if max_length < len(data) else max_length
        images.append(data)
    return np.asanyarray(images), max_length
    

# Convert images from jpg to csv file
def convert_images(trainingData=True, trainingDataCount=10):
    paths = get_paths(trainingData, trainingDataCount)
    # Read images from dish
    images, max_length = read_images(paths)
    # Pad image arrays
    # images = pad_array(images, max_length)
    return images


convert_images(True, 5)
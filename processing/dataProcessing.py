import os
import numpy as np
import pandas as pd
from PIL import Image
import progressbar
import xml.etree.ElementTree as ET

data_path = "/home/sander/dev/ML_Project/data/"
annotations_path = data_path + "archive/annotations/Annotation/"
image_path = data_path + "archive/images/Images/"


def get_paths(trainingData, trainingDataCount):
    paths = []
    dirs = os.listdir(annotations_path)
    # Tak only part of the data as training or test data
    for dir in dirs:
        files = os.listdir(annotations_path + dir)
        if (trainingData):
            files = files[:trainingDataCount]
        else:
            files = files[trainingDataCount:]
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


def read_images(paths, image_size):
    print("Read images")
    images = []
    labels = []
    for path_i in progressbar.progressbar(range(len(paths))):
        # Data
        image = Image.open(image_path + paths[path_i] + ".jpg")
        image.resize(image_size)
        image_data = np.asanyarray(image)
        images.append(image_data)
        # Label
        doc = doc1 = ET.parse(annotations_path + paths[path_i])
        root = doc.getroot()
        for element in root.findall("object"):
            label = element.find("name").text
            labels.append(label)
            continue

    return np.asanyarray(images), np.asanyarray(labels)
    

# Convert images from jpg to csv file
def load_data(training_data=True, data_count=0.7, image_size=(32,32)):
    paths = get_paths(training_data, data_count)
    # Read images from dish
    data, labels = read_images(paths, image_size)
    # Pad image arrays
    # images = pad_array(images, max_length)
    print(labels)
    return data, labels


load_data(True, 2)
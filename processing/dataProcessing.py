import os
import numpy as np
import pandas as pd
from PIL import Image
import progressbar
import xml.etree.ElementTree as ET
import math

data_path = "/home/sander/dev/ML_Project/data/"
annotations_path = data_path + "archive/annotations/Annotation/"
image_path = data_path + "archive/images/Images/"


def get_paths(trainingData, trainingDataCount):
    paths = []
    dirs = os.listdir(annotations_path)
    # Tak only part of the data as training or test data
    for dir in dirs:
        files = os.listdir(annotations_path + dir)
        files_count = len(files)
        selected_files = math.floor( files_count * trainingDataCount )
        if (trainingData):
            files = files[:selected_files]
        else:
            files = files[selected_files:]
        for file in files:
            paths.append(dir + "/" + file)
    return paths


# This is slow as fuck. Uses a lot of ram
def pad_array(data, max_length):
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
    images = []
    labels = []
    # Image "n02105855-Shetland_sheepdog/n02105855_2933" is fucked! RGBA
    for path_i in progressbar.progressbar(range(len(paths))):
        # Data
        image = Image.open(image_path + paths[path_i] + ".jpg")
        image = image.resize(image_size)
        image_data = np.asanyarray(image)
        if image_data.shape != (32,32,3):
            print(image_data.shape, paths[path_i])
        images.append(image_data)
        # Label
        doc = doc1 = ET.parse(annotations_path + paths[path_i])
        root = doc.getroot()
        for element in root.findall("object"):
            label = element.find("name").text
            labels.append(label)
            break
    return np.asanyarray(images), np.asanyarray(labels)
    

# TODO: Solve this
def convert_images_rgb():
    image_pahts = ["n02105855-Shetland_sheepdog/n02105855_2933"]
    for path in image_pahts:
        image = Image.open(image_path + path + ".jpg")
        image.load() # required for png.split()
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
        background.save(image_path + path + ".jpg", 'jpg', quality=80)


# Convert images from jpg to csv file
def load_data(training_data=True, data_count=0.8, image_size=(32,32)):
    paths = get_paths(training_data, data_count)
    # Read images and labels from disk
    data, labels = read_images(paths, image_size)
    return data, labels

if __name__ == "__main__":
    training_data, training_labels = load_data(True, 0.8, (32,32))
    test_data, test_labels = load_data(False, 0.8, (32,32))
    #convert_images_rgb()
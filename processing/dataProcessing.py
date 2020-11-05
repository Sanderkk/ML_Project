import os
import numpy as np
import pandas as pd
from PIL import Image
import progressbar
import xml.etree.ElementTree as ET
import math
from CONFIG import *

# example of saving an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

ANNOTATIONS_PATH = DATA_PATH + "archive/annotations/Annotation/"
IMAGE_DATA_PATH = DATA_PATH + "archive/images/Images/"
PROCESSED_ANNOTATIONS_PATH = DATA_PATH + "processed/annotations/"
PROCESSED_IMAGE_PATH = DATA_PATH + "processed/images/"

def get_file_paths(training_data=True, validation=False, validation_split=0.2, training_set_size=TRAINING_SET_SIZE, data_path=ANNOTATIONS_PATH):
    paths = {}
    dirs = os.listdir(data_path)
    # Tak only part of the data as training or test data
    for dir in dirs[:CLASS_COUNT]:
        paths[dir] = []
        files = os.listdir(data_path + dir)
        files_count = len(files)
        selected_files = math.floor( files_count * training_set_size )
        if (training_data):
            validation_marker = math.floor( selected_files * validation_split )
            if (validation):
                files = files[validation_marker:selected_files]
            else:
                files = files[:(selected_files - validation_marker)]
        else:
            files = files[selected_files:]
        for file in files:
            paths[dir].append(file)
    return paths


def read_label_contents_image_box(label_file):
    root = label_file.getroot()
    element = root.find("object")
    bndbox = element.find("bndbox")
    x_min = int(bndbox.find("xmin").text)
    x_max = int(bndbox.find("xmax").text)
    y_min = int(bndbox.find("ymin").text)
    y_max = int(bndbox.find("ymax").text)
    return x_min, x_max, y_min, y_max

def get_annotation_label(label_file):
    root = label_file.getroot()
    element = root.find("object")
    label = element.find("name").text
    return label

# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
def image_generation(label_doc, image, dir_path, path):
    img_array = img_to_array(image)

    # expand dimension to one sample
    samples = expand_dims(img_array, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=45,
        shear_range=0.2,
        horizontal_flip=True
    )
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # generate batch of images
        batch = it.next()
        image_data = batch[0].astype('uint8')
        save_img(PROCESSED_IMAGE_PATH + dir_path + "/" + path + str(i) + ".jpg", image_data)
        # label_doc.write(PROCESSED_ANNOTATIONS_PATH + dir_path + "/" + path + str(i))


def resize_images(paths, data_type="training"):
    for data_dir_path, image_path in paths.items():
        dir_path = data_type + "/" + data_dir_path
        os.mkdir(PROCESSED_ANNOTATIONS_PATH + dir_path)
        os.mkdir(PROCESSED_IMAGE_PATH + dir_path)
        for i in progressbar.progressbar(range(len(image_path))):
            path = image_path[i]
            # Get label
            doc = ET.parse(ANNOTATIONS_PATH + data_dir_path + "/" + path)
            x_min, x_max, y_min, y_max = read_label_contents_image_box(doc)
            # Image
            image = Image.open(IMAGE_DATA_PATH + data_dir_path + "/" + path + ".jpg")
            image = image.crop((x_min, y_min, x_max, y_max))
            image = image.convert('RGB')  # The one RGBA image
            image = image.resize(IMAGE_SIZE)
            # Generate more data with image augmentation
            image_generation(doc, image, dir_path, path)
            image.save(PROCESSED_IMAGE_PATH + dir_path + "/" + path + ".jpg")
            # doc.write(PROCESSED_ANNOTATIONS_PATH + dir_path + "/" + path)

def read_data(paths, annotations_path, image_file_path, preprocessed=True):
    images = []
    labels = []
    for dir_path, image_path in paths.items():
        for i in progressbar.progressbar(range(len(image_path))):
            path = image_path[i]
            # Get label
            doc = ET.parse(annotations_path + dir_path + "/" + path)
            labels.append(get_annotation_label(doc))
            image = Image.open(image_file_path + dir_path + "/" + path + ".jpg")
            if not preprocessed:
                x_min, x_max, y_min, y_max = read_label_contents_image_box(doc)
                image = image.crop((x_min, y_min, x_max, y_max))
                image = image.convert('RGB')  # The one RGBA image
                image = image.resize(IMAGE_SIZE)
            image_data = np.asanyarray(image)
            images.append(image_data)
            #print(dir_path+"/"+path, get_annotation_label(doc))
    return np.asanyarray(images), np.asanyarray(labels)


"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth

https://www.kaggle.com/hengzheng/dog-breeds-classifier
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
https://medium.com/@sourav_srv_bhattacharyya/image-augmentation-to-build-a-powerful-image-classification-model-3303e40af7b0
"""

def generate_data():
    paths = get_file_paths(training_set_size=TRAINING_SET_SIZE, validation_split=0.2)
    resize_images(paths, data_type="training")
    paths = get_file_paths(training_set_size=TRAINING_SET_SIZE, validation_split=0.2, validation=True)
    resize_images(paths, data_type="validation")

def read_training_set():
    paths = get_file_paths(training_set_size=TRAINING_SET_SIZE, validation_split=VALIDATION_SET_SIZE, data_path=ANNOTATIONS_PATH)
    data, targets = read_data(paths, ANNOTATIONS_PATH, IMAGE_DATA_PATH, preprocessed=False)
    training_set = tf.keras.preprocessing.timeseries_dataset_from_array(
        data, targets, len(data), sequence_stride=1, sampling_rate=1,
        batch_size=BATCH_SIZE, shuffle=False, seed=None, start_index=None, end_index=None
    )
    return training_set

def read_validation_set():
    paths = get_file_paths(training_set_size=TRAINING_SET_SIZE, validation=True, validation_split=VALIDATION_SET_SIZE, data_path=ANNOTATIONS_PATH)
    data, targets = read_data(paths, ANNOTATIONS_PATH, IMAGE_DATA_PATH, preprocessed=False)
    training_set = tf.keras.preprocessing.timeseries_dataset_from_array(
        data, targets, len(data), sequence_stride=1, sampling_rate=1,
        batch_size=BATCH_SIZE, shuffle=False, seed=None, start_index=None, end_index=None
    )
    return training_set

def read_test_set():
    paths = get_file_paths(training_data=False, training_set_size=TRAINING_SET_SIZE, data_path=ANNOTATIONS_PATH)
    data, targets = read_data(paths, ANNOTATIONS_PATH, IMAGE_DATA_PATH, preprocessed=False)
    training_set = tf.keras.preprocessing.timeseries_dataset_from_array(
        data, targets, len(data), sequence_stride=1, sampling_rate=1,
        batch_size=BATCH_SIZE, shuffle=False, seed=None, start_index=None, end_index=None
    )
    return training_set


if __name__ == "__main__":
    generate_data()
import os
import numpy as np
import pandas as pd
from PIL import Image
import progressbar
import xml.etree.ElementTree as ET
import math
from CONFIG import *

from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def get_file_paths(data_path=ANNOTATIONS_PATH, data_type="training"):
    paths = {}
    dirs = os.listdir(data_path)
    # Tak only part of the data as training or test data
    for dir in dirs[:CLASS_COUNT]:
        paths[dir] = []
        files = os.listdir(data_path + dir)

        training_set_splitter = math.floor(len(files) * TRAINING_SET_SIZE)
        validation_set_splitter = math.floor(training_set_splitter * VALIDATION_SET_SIZE)
        file_paths = []
        if data_type=="training":
            file_paths = files[:training_set_splitter-validation_set_splitter]
        elif data_type=="validation":
            file_paths = files[training_set_splitter-validation_set_splitter:training_set_splitter]
        else:
            file_paths = files[training_set_splitter:]

        for file_path in file_paths:
            paths[dir].append(file_path)
    return paths

def get_file_split_paths():
    training_set_paths = get_file_paths(data_type="training")
    validation_set_paths = get_file_paths(data_type="validation")
    test_set_paths = get_file_paths(data_type="testing")
    return training_set_paths, validation_set_paths, test_set_paths

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
def image_generation(image):
    img_array = img_to_array(image)

    # expand dimension to one sample
    samples = expand_dims(img_array, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=90,
        shear_range=0.2,
        horizontal_flip=True
    )
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    return it

def crop_image(image, label_doc):
    x_min, x_max, y_min, y_max = read_label_contents_image_box(label_doc)
    image = image.crop((x_min, y_min, x_max, y_max))
    image = image.convert('RGB')  # The one RGBA image
    return image

def save_image(image_iterator, path):
    for i in range(AUGMENT_DATA_NUMBER):
        # generate batch of images
        batch = image_iterator.next()
        image_data = batch[0].astype('uint8')
        save_img(path + "_" + str(i) + ".jpg", image_data)

def split_images(paths, data_type="training", crop=IMAGE_CROP, augment=False):
    os.mkdir(DATA_PATH + "processed/" + data_type)
    save_data_path = DATA_PATH + "processed/" + data_type + "/"

    for dir_path, image_path in paths.items():
        os.mkdir(save_data_path + dir_path)
        for i in progressbar.progressbar(range(len(image_path))):
            path = image_path[i]
            # Get label
            doc = ET.parse(ANNOTATIONS_PATH + dir_path + "/" + path)
            # Image
            image = Image.open(IMAGE_DATA_PATH + dir_path + "/" + path + ".jpg")
            if crop:
                image = crop_image(image, doc)

            image = image.resize(IMAGE_SIZE)
            # Generate more data with image augmentation

            if augment:
                image_generator_it = image_generation(image)
                save_image(image_generator_it, save_data_path + "/" + dir_path + "/" + path)
            else:
                image.save(save_data_path + "/" + dir_path + "/" + path + ".jpg")


def process_data():
    os.mkdir(DATA_PATH + "processed/")
    # Get paths
    training_set_paths, validation_set_paths, test_set_paths = get_file_split_paths()
    # Split data
    split_images(training_set_paths, data_type="training", augment=True)
    split_images(validation_set_paths, data_type="validation", augment=True)
    split_images(test_set_paths, data_type="testing")

"""
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

"""

if __name__ == "__main__":
    process_data()
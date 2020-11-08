import os
import math
from PIL import Image
from CONFIG import *
from processing.dataProcessing import get_file_paths, image_generation
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array

def min_max_data():
    images = data_info()
    return min(images), max(images)

def data_info():
    image_count = []
    dirs = os.listdir(IMAGE_DATA_PATH)
    for dir in dirs:
        data_count = len(os.listdir(IMAGE_DATA_PATH + dir))
        image_count.append(data_count)
    return image_count

def average_data_count():
    images = data_info()
    return sum(images) / len(images)

def data_count():
    return sum(data_info())

def show_data_examples(count=9):
    paths = get_file_paths()
    images = []
    labels = []
    keys = list(paths.keys())
    for key in keys[:count]:
        image = Image.open(IMAGE_DATA_PATH + key + "/" + paths[key][0] + ".jpg")
        image.resize((32,32))
        images.append(np.asarray(image))
        labels.append(key)

    plt.figure(figsize=(10, 10))
    for i in range(count):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(str(labels[i]))
        plt.axis("off")
    plt.show()

def show_augmented_images(count=9):
    paths = get_file_paths()
    images = []
    labels = []
    keys = list(paths.keys())
    for key in keys[:1]:
        image = Image.open(IMAGE_DATA_PATH + key + "/" + paths[key][0] + ".jpg")
        image.resize((32, 32))
        it = image_generation(image)
        for i in range(count):
            batch = it.next()
            augmented_image = batch[0].astype('uint8')
            images.append(np.asarray(augmented_image))
            labels.append(key)

    plt.figure(figsize=(10, 10))
    for i in range(count):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(str(labels[i]))
        plt.axis("off")
    plt.show()



if __name__ == "__main__":
    # Get min max image data count
    min_data, max_data = min_max_data()
    print("Min: ", str(min_data))
    print("Max: ", str(max_data))
    average = average_data_count()
    print("Average image count: ", average)
    count = data_count()
    print("The amount of data is: ", count)

    # Show 9 of the classes
    show_data_examples(count=9)

    # Show 9 augmented images
    show_augmented_images(count=9)
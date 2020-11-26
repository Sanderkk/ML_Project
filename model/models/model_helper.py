import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models, applications, optimizers
from tensorflow import keras
import matplotlib.pyplot as plt

from CONFIG import *
from data_processing.dataProcessing import *
from keras.models import load_model
from data_processing.dataProcessing import *
import numpy as np


def show_model_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()



# Use the fitted model to predict a class of a image
def load_trained_model(image, model_type="cnn"):
    if model_type == "cnn":
        model = load_model('../trained_models/CNN_Model')
    else:
        model = load_model('../trained_models/MLP_Model')
    return model.predict(image)

def get_prediction_from_result(predict_result=[]):
    return predict_result.index(max(predict_result))

def load_image(path):
    image = Image.open(path)
    image = np.asarray(image)
    input_arr = np.array([image])
    return image

def model_predict(model_type="cnn", base_path=IMAGE_PROCESS_PATH_TRAINING):
    paths = get_file_paths(data_path=base_path, data_type="training")
    class_list = paths.keys()
    actual_class = 0
    for key, path_list in paths.items():
        path = base_path + key + "/" + path_list[0]
        break
    image = load_image(path)
    predict_result = load_trained_model(image, "cnn")
    predicted_class = get_prediction_from_result(predict_result)
    return predict_result, class_list, predicted_class, actual_class

# model_predict()


"""
counter = 0
for dir_path, image_path in paths.items():
    for file_name in image_path:
        doc = ET.parse(ANNOTATIONS_PATH + dir_path + "/" + file_name)
        # Image
        image = Image.open(IMAGE_DATA_PATH + dir_path + "/" + file_name + ".jpg")
        image = crop_image(image, doc)
        image = image.resize((128,128))
        image = np.asarray(image)
        image = image * 1./255

        input_arr = np.array([image])
        classes = model.predict(input_arr)
        print(classes)
        max_value = max(classes[0])
        index = np.where(classes[0] == max_value)
        print(index)
        if (index == counter):
            print("For class: ", dir_path)
            print(classes)
            break
    counter += 1
"""
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


# Show the history of the model using matplotlib, drawing accuracy of training and validation dataset
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

# Get the predicted class from a prediction
def get_prediction_from_result(predict_result=[]):
    return predict_result.index(max(predict_result))

# Load image from path
def load_image(path):
    image = Image.open(path)
    image = np.asarray(image)
    input_arr = np.array([image])
    return input_arr

# Print info of the size of images of the original dataset
def image_size_info(augment=False, base_path=ANNOTATIONS_PATH):
    paths = get_file_paths(data_path=base_path, data_type=None)
    min_size_x = math.inf
    max_size_x = -math.inf
    min_size_y = math.inf
    max_size_y = -math.inf
    for key, image_list in paths.items():
        for i in range(len(image_list)):
            image_path = IMAGE_DATA_PATH + key + "/" + image_list[i] + ".jpg"
            image = Image.open(image_path)
            if (augment):
                doc = ET.parse(ANNOTATIONS_PATH + key + "/" + image_list[i])
                image = crop_image(image, doc)
            (x_size, y_size) = image.size
            min_size_x = x_size if x_size < min_size_x else min_size_x
            max_size_x = x_size if x_size > max_size_x else max_size_x
            min_size_y = y_size if y_size < min_size_y else min_size_y
            max_size_y = y_size if y_size > max_size_y else max_size_y
    print("Size of x coordinates:")
    print("Min:", str(min_size_x))
    print("Max:", str(max_size_x))
    print("Size of y coordinates:")
    print("Min:", str(min_size_y))
    print("Max:", str(max_size_y))

# Predict the class of a class from the test dataset
# Input model_type of value cnn to test the cnn model, and mlp to test the mlp model
def model_predict(model_type="cnn", actual_class_number=0, number_of_predictions=1, base_path=IMAGE_PROCESS_PATH_TRAINING):
    predictions = []
    paths = get_file_paths(data_path=base_path, data_type="testing")
    class_list = list(paths.keys())
    answer = (actual_class_number, class_list[actual_class_number])
    class_dir = class_list[actual_class_number]
    for file_name in paths[class_dir][:number_of_predictions]:
        path = base_path + class_dir + "/" + file_name
        image = load_image(path)
        predict_result = load_trained_model(image, model_type)
        predicted_class = get_prediction_from_result(predict_result.tolist()[0])
        predictions.append((predicted_class, class_list[predicted_class]))
    return predictions, answer

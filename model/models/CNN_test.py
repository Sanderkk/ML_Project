from keras.models import load_model
from data_processing.dataProcessing import *
import numpy as np
from tensorflow.keras import datasets, layers, models, applications, optimizers

model = load_model('../trained_models/CNN_Model')

# Image
image = None
paths = get_file_paths(data_type="training")
image_data_path = DATA_PATH + "/processed_images/training/"

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
paths = [
    "C:/dev/ML_Project/model/data/processed_images/training/Afghan_hound/n02088094_115_5.jpg",
    "C:/dev/ML_Project/model/data/processed_images/training/Blenheim_spaniel/n02086646_32_1.jpg",
    "C:/dev/ML_Project/model/data/processed_images/training/Chihuahua/n02085620_199_0.jpg",
    "C:/dev/ML_Project/model/data/processed_images/training/Japanese_spaniel/n02085782_2_1.jpg"
]
for path in paths:
    print("Image test")
    image = Image.open(path)
    image = np.asarray(image)
    input_arr = np.array([image])
    classes = model.predict(input_arr)
    a = classes[0].tolist()

    print(a)
    print(a.index(max(a)))
    print(sum(a))
    print(max(a))
    print()
    print()
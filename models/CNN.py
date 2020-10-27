import tensorflow as tf

from tensorflow.keras import datasets, layers, models, applications
import matplotlib.pyplot as plt
import numpy as np

from processing.dataProcessing import read_training_data, read_test_data

def normalize_data(data):
    return data/255

def create_model(classes_count, image_size_size):
    model = models.Sequential()
    # model.add(layers.BatchNormalization(input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(image_size_size[0], (3, 3), activation='relu',
                            input_shape=(image_size_size[0], image_size_size[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(254, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes_count, activation='softmax'))
    model.summary()
    return model
    """
    model.add(layers.Conv2D(image_size_size[0], (3, 3), activation='relu', input_shape=(image_size_size[0], image_size_size[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes_count, activation="softmax"))
    model.summary()
    return model
    """
    # For this model, the linear activation performes better than the softmax activation function.
    """
    # https://www.kaggle.com/criscastromaya/cnn-for-image-classification-in-coil-100-dataset
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_size_size[0], image_size_size[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(classes_count))
    model.summary()
    return model
    """

def compile_and_fit(model, training_data, training_labels, test_data, test_labels):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(training_data, training_labels, epochs=100, batch_size=128,
                validation_data=(test_data, test_labels))
    return history

def show_model_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

# Example with generating data on the fly
# https://towardsdatascience.com/a-simple-cnn-multi-image-classifier-31c463324fa
def CNN(image_size, training_set_size):
    # Load training and test data
    training_data, training_labels = read_training_data()
    test_data, test_labels = read_test_data()
    # Labels
    label_names = list(set(training_labels.copy()))
    training_labels = np.asanyarray([label_names.index(x) for x in training_labels])
    test_labels = np.asanyarray([label_names.index(x) for x in test_labels])
    # Normalize data
    training_data = normalize_data(training_data)
    test_data = normalize_data(test_data)
    # Create model
    model = create_model(len(training_labels), image_size)
    # Fit model
    # Test model
    history = compile_and_fit(model, training_data, training_labels, test_data, test_labels)
    # Show model accuracy
    show_model_history(history)

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    print(test_acc)
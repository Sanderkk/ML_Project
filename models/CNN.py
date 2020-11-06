import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models, applications, optimizers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from CONFIG import *
from processing.dataProcessing import read_training_set, read_validation_set, read_test_set


"""
def create_model(image_size_size, activation='softmax'):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding="same",
                            input_shape=(image_size_size[0], image_size_size[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    for size in [128, 256, 512, 728]:
        model.add(layers.Conv2D(size, (3, 3), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(size, (3, 3), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(1024, (3, 3), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(254, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(396, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(CLASS_COUNT, activation=activation))
    model.summary()
    return model
"""
def create_model(input_shape, activation='softmax', data_augmentation=None):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    # x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(96, 11, strides=4, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=5, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(384, kernel_size=3, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(384, kernel_size=3, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(384, kernel_size=3, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.MaxPooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(CLASS_COUNT, activation=activation)(x)
    return keras.Model(inputs, outputs)



def compile_and_fit(model, training_set, validation_set):
    callbacks = [
        keras.callbacks.ModelCheckpoint("callbacks/save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit_generator(
        training_set,
        epochs=CNN_EPOCHS,
        callbacks=callbacks,
        validation_data=validation_set
        )
    return history

def show_model_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

def convert_labels(training_labels, test_labels):
    # Labels
    label_names = list(set(training_labels.copy()))
    training_labels = np.asanyarray([label_names.index(x) for x in training_labels])
    test_labels = np.asanyarray([label_names.index(x) for x in test_labels])

# Example with generating data on the fly
# https://towardsdatascience.com/a-simple-cnn-multi-image-classifier-31c463324fa
def CNN(image_size):
    training_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed/training",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    validation_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed/validation",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed/testing",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    training_set = training_set.prefetch(buffer_size=32)
    validation_set = validation_set.prefetch(buffer_size=32)

    model = create_model(image_size + (3,), data_augmentation=None)
    history = compile_and_fit(model, training_set, validation_set)
    show_model_history(history)

    test_loss, test_acc = model.evaluate(test_set, verbose=2)
    print(test_acc)

    # Save model
    model.save('CNN_Model')
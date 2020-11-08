import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models, applications, optimizers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from CONFIG import *
from processing.dataProcessing import *
from models.model_helper import *


"""
def create_model(input_shape, activation='softmax'):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding="same",
                            input_shape=input_shape))
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
def create_model(input_shape, activation='softmax'):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    # x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(96, 11, strides=4, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=5, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(384, kernel_size=3, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(384, kernel_size=3, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="valid")(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

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
"""

def create_model(input_shape, activation='softmax'):
    model = models.Sequential()
    #models.add(layers.LayerNormalization())
    # model.add(layers.BatchNormalization(input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(96, (3, 3), activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(254, activation='relu'))
    model.add(layers.Dense(CLASS_COUNT))
    model.summary()
    return model
"""


def compile_and_fit(model, training_set, validation_set):
    callbacks = [
        #keras.callbacks.ModelCheckpoint("callbacks/save_at_{epoch}.h5"),
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

def CNN():
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

    model = create_model(IMAGE_SIZE + (3,))
    history = compile_and_fit(model, training_set, validation_set)
    show_model_history(history)

    test_loss, test_acc = model.evaluate(test_set, verbose=2)
    print(test_acc)

    # Save model
    model.save('CNN_Model')
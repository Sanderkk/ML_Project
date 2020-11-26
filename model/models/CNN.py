import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models, applications, optimizers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from CONFIG import *
from data_processing.dataProcessing import *
from models.model_helper import *

# Notes
def create_model(input_shape, activation='softmax'):
    inputs = keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(64, 5, strides=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(CLASS_COUNT, activation=activation)(x)
    return keras.Model(inputs, outputs)


def compile_and_fit(model, training_set, validation_set):
    callbacks = [
        #keras.callbacks.ModelCheckpoint("callbacks/save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=optimizers.Adam(0.0012158),
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
        DATA_PATH + "processed_images/training",
        seed=1436,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        label_mode='int'
    )
    validation_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed_images/validation",
        seed=1436,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        label_mode='int'
    )

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed_images/testing",
        seed=1436,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        label_mode='int'
    )

    training_set = training_set.prefetch(buffer_size=64)
    validation_set = validation_set.prefetch(buffer_size=64)

    model = create_model(IMAGE_SIZE + (3,))
    print(model.summary())
    history = compile_and_fit(model, training_set, validation_set)
    show_model_history(history)

    test_loss, test_acc = model.evaluate(test_set, verbose=2)
    print(test_acc)

    # Save model
    model.save('trained_models/CNN_Model')
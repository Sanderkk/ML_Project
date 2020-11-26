import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models, applications, optimizers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from CONFIG import *
from data_processing.dataProcessing import *
from models.model_helper import *

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.2)
    ]
)

def create_model(input_shape, activation='softmax'):
    inputs = keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    x = layers.Flatten()(x)

    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(CLASS_COUNT, activation=activation)(x)
    return keras.Model(inputs, outputs)

def compile_and_fit(model, training_set, validation_set):
    callbacks = [
        # keras.callbacks.ModelCheckpoint("callbacks/save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        training_set,
        epochs=CNN_EPOCHS,
        callbacks=callbacks,
        validation_data=validation_set
        )
    return history

def MLP():
    training_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed_images/training",
        seed=1436,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    validation_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed_images/validation",
        seed=1436,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed_images/testing",
        seed=1436,
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
    model.save('trained_models/MLP_Model')
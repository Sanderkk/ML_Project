from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models, applications, optimizers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from CONFIG import *
from data_processing.dataProcessing import *
from models.model_helper import *

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes, activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation

    def build(self, hp):
        inputs = keras.Input(shape=self.input_shape)
        # Image augmentation block
        # x = data_augmentation(inputs)

        # Entry block
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

        # Convolutional layer 1
        x = layers.Conv2D(
            filters=hp.Choice(
                'conv_filters_1',
                values=[64, 72, 96, 112],
                default=96,
            ),
            kernel_size=hp.Choice(
                'num_filters',
                values=[8,9,11,12],
                default=11,
            ),
            strides=4,
            padding="valid"
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_1',
                min_value=0.2,
                max_value=0.5,
                default=0.3,
                step=0.05
            )
        )(x)
        x = layers.BatchNormalization()(x)

        # Convolutional layer 2
        x = layers.Conv2D(
            filters=hp.Choice(
                'conv_filters_2',
                values=[256, 384, 512],
                default=384,
            ),
            kernel_size=5,
            strides=1,
            padding="valid"
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_2',
                min_value=0.2,
                max_value=0.5,
                default=0.3,
                step=0.05
            )
        )(x)
        x = layers.BatchNormalization()(x)

        # Convolutional layer 3
        x = layers.Conv2D(
            filters=hp.Choice(
                'conv_filters_3',
                values=[256, 384, 512],
                default=384,
            ),
            kernel_size=3,
            strides=1,
            padding="valid"
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_3',
                min_value=0.2,
                max_value=0.5,
                default=0.3,
                step=0.05
            )
        )(x)
        x = layers.BatchNormalization()(x)

        # Convolutional layer 4
        x = layers.Conv2D(
            filters=hp.Choice(
                'conv_filters_4',
                values=[256, 384, 512],
                default=384,
            ),
            kernel_size=3,
            strides=1,
            padding="valid"
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_4',
                min_value=0.2,
                max_value=0.5,
                default=0.3,
                step=0.05
            )
        )(x)
        x = layers.BatchNormalization()(x)

        # Convolutional layer 5
        x = layers.Conv2D(256, kernel_size=3, strides=1, padding="valid")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Flatten()(x)

        # Dense layer 1
        x = layers.Dense(
            units=hp.Int(
                'dense_units_1',
                min_value=2048,
                max_value=4096,
                step=512,
                default=4096
            ),
            activation='relu'
        )(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_5',
                min_value=0.2,
                max_value=0.5,
                default=0.3,
                step=0.05
            )
        )(x)
        x = layers.BatchNormalization()(x)

        # Dense layer 2
        x = layers.Dense(
            units=hp.Int(
                'dense_units_2',
                min_value=2048,
                max_value=4096,
                step=512,
                default=4096
            ),
            activation='relu'
        )(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_6',
                min_value=0.2,
                max_value=0.5,
                default=0.3,
                step=0.05
            )
        )(x)
        x = layers.BatchNormalization()(x)

        # Dense layer 3
        x = layers.Dense(
            units=hp.Int(
                'dense_units_3',
                min_value=1024,
                max_value=2048,
                step=512,
                default=1024
            ),
            activation='relu'
        )(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_7',
                min_value=0.2,
                max_value=0.5,
                default=0.3,
                step=0.05
            )
        )(x)
        x = layers.BatchNormalization()(x)

        # Dense layer 4
        outputs = layers.Dense(self.num_classes, activation=self.activation)(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

def CNN_Hyper():
    training_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed/training",
        seed=957,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    validation_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed/validation",
        seed=957,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH + "processed/testing",
        seed=957,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    training_set = training_set.prefetch(buffer_size=32)
    validation_set = validation_set.prefetch(buffer_size=32)

    hyperModel = CNNHyperModel(IMAGE_SIZE + (3,), CLASS_COUNT, "softmax")

    MAX_TRIALS = 20
    EXECUTION_PER_TRIAL = 1
    N_EPOCH_SEARCH = 25

    tuner = RandomSearch(
        hyperModel,
        objective='val_accuracy',
        seed=957,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='random_search',
        project_name='Stanford-Dogs-40_1'
    )

    tuner.search_space_summary()

    tuner.search(training_set,epochs=N_EPOCH_SEARCH, validation_data=validation_set)

    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(test_set)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    best_model.summary()
    # Save model
    best_model.save('CNN_Tuned_Best_Model')

# https://www.sicara.ai/blog/hyperparameter-tuning-keras-tuner
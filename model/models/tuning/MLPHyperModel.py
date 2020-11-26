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

class MLPHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes, activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation

    def build(self, hp):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

        x = layers.Flatten()(x)

        # Dense layer 1
        x = layers.Dense(
            units=hp.Int(
                'dense_units_1',
                min_value=512,
                max_value=2048,
                step=128,
                default=1024
            ),
            activation='relu'
        )(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_1',
                min_value=0.1,
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
                min_value=256,
                max_value=2048,
                step=128,
                default=512
            ),
            activation='relu'
        )(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_2',
                min_value=0.1,
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
                min_value=256,
                max_value=2048,
                step=128,
                default=512
            ),
            activation='relu'
        )(x)
        x = layers.Dropout(
            hp.Float(
                'dropout_3',
                min_value=0.1,
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

def MLP_Hyper():
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

    hyperModel = MLPHyperModel(IMAGE_SIZE + (3,), CLASS_COUNT, "softmax")

    MAX_TRIALS = 20
    EXECUTION_PER_TRIAL = 2
    N_EPOCH_SEARCH = 40

    tuner = RandomSearch(
        hyperModel,
        objective='val_accuracy',
        seed=1436,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='random_search',
        project_name='Stanford-Dogs-MLP_40_1'
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
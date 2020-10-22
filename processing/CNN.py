import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

from dataProcessing import load_data

def normalize_data(data):
    return data/255

def create_model(classes_count):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(classes_count))
    model.summary()
    return model

def compile_and_fit(model, training_data, training_labels, test_data, test_labels):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(training_data, training_labels, epochs=10, 
                validation_data=(test_data, test_labels))
    return history

def show_model_history(history):
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


# Main function
if __name__ == "__main__":
    # Load training data and normalize
    training_data, training_labels = load_data(True, 0.8, (32,32))
    label_names = list(set(training_labels.copy()))
    training_labels = [ label_names.index(x) for x in training_labels]
    training_data = normalize_data(training_data)
    test_data, test_labels = load_data(False, 0.8, (32,32))
    test_labels = [ label_names.index(x) for x in test_labels]
    # Create model
    model = create_model(len(training_labels))
    # Fit model
    # Test model
    history = compile_and_fit(model, training_data, training_labels, test_data, test_labels)
    # Show model accuracy
    show_model_history(history)

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    print(test_acc)
    

# Might need to add extra data

"""
###
Acc converging at 18% (64,64)
Overfitting quite a lot
Add aditional dropout?
###
model = models.Sequential()
#model.add(layers.BatchNormalization(input_shape=(64, 64, 3)))
model.add(layers.Conv2D(image_size_size[0], (3, 3), activation='relu', input_shape=(image_size_size[0], image_size_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(255, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(255, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(254, activation='relu'))
model.add(layers.Dense(classes_count))
model.summary()
return model
"""

"""
Acc converging at 10% (64,64)

model.add(layers.Conv2D(image_size_size[0], (3, 3), activation='relu',
                        input_shape=(image_size_size[0], image_size_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(255, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(255, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classes_count, activation="softmax"))
model.summary()
return model
"""

"""
Untested

model = models.Sequential()
#model.add(layers.BatchNormalization(input_shape=(64, 64, 3)))
model.add(layers.Conv2D(image_size_size[0], (3, 3), activation='relu',
                        input_shape=(image_size_size[0], image_size_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(255, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classes_count, activation="softmax"))
model.summary()
return model
"""

"""
# 1 %. Just trash

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



"""
    model = models.Sequential()

    # model.add(layers.BatchNormalization(input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(image_size_size[0], (3, 3), activation='relu',
                            input_shape=(image_size_size[0], image_size_size[1], 3)))
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(384, (3, 3), activation='relu'))

    model.add(layers.Conv2D(384, (3, 3), activation='relu'))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(layers.BatchNormalization())

    #model.add(layers.Dense(image_size_size[0], activation='relu', input_shape=(image_size_size[0], image_size_size[1], 3)))
    model.add(layers.Flatten())

    ### TEST
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes_count, activation='softmax'))
    """
    """
    model = models.Sequential()
    # model.add(layers.BatchNormalization(input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(image_size_size[0], (3, 3), activation='relu',
                            input_shape=(image_size_size[0], image_size_size[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(255, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(254, activation='relu'))
    model.add(layers.Dense(classes_count))
    model.summary()
    return model
    """
    """
    # 2048
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(classes_count, activation='softmax'))
    model.summary()
    return model
    """
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
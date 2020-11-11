# Data config
CLASS_COUNT = 40
IMAGE_SIZE = (224, 224)
TRAINING_SET_SIZE = 0.9
VALIDATION_SET_SIZE = 0.2
AUGMENT_DATA_NUMBER = 10
IMAGE_CROP=False

DATA_PATH = "C:/dev/ML_Project/data/"
ANNOTATIONS_PATH = DATA_PATH + "archive/annotations/Annotation/"
IMAGE_DATA_PATH = DATA_PATH + "archive/images/Images/"

# CNN config
BATCH_SIZE=32
CNN_EPOCHS=50

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth

https://www.kaggle.com/hengzheng/dog-breeds-classifier
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
https://medium.com/@sourav_srv_bhattacharyya/image-augmentation-to-build-a-powerful-image-classification-model-3303e40af7b0
"""
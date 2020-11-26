# Data config
GIVEN_CLASSES = []
CLASS_COUNT = 40
IMAGE_SIZE = (128, 128)
TRAINING_SET_SIZE = 0.9
VALIDATION_SET_SIZE = 0.2
AUGMENT_DATA_NUMBER = 9
IMAGE_CROP=True

# Network config
BATCH_SIZE=64
NETWORK_EPOCHS=50

# Data location config
DATA_PATH = "C:/dev/ML_Project/model/data/"
ANNOTATIONS_PATH = DATA_PATH + "StanfordDogs/archive/annotations/Annotation/"
IMAGE_DATA_PATH = DATA_PATH + "StanfordDogs/archive/images/Images/"

# Processed data path - No need to configure (should be left as is)
IMAGE_PROCESS_PATH_TRAINING = DATA_PATH + "processed_images/training/"
IMAGE_PROCESS_PATH_VALIDATION = DATA_PATH + "processed_images/validation/"
IMAGE_PROCESS_PATH_TESTING = DATA_PATH + "processed_images/testing/"
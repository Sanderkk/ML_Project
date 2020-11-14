# Data config
GIVEN_CLASSES = ['Chihuahua', 'Beagle', 'American Staffordshire Terrier', 'Labrador Retriever', 'Collie', 'Border Collie', 'Rottweiler', 'Miniature Pinscher', 'Boxer', 'Great Dane', 'Siberian Husky', 'Pug', 'Great Pyrenees', 'Pomeranian']
CLASS_COUNT = 14
IMAGE_SIZE = (224, 224)
TRAINING_SET_SIZE = 0.9
VALIDATION_SET_SIZE = 0.2
AUGMENT_DATA_NUMBER = 10
IMAGE_CROP=False

DATA_PATH = "C:/dev/ML_Project/model/data/"
ANNOTATIONS_PATH = DATA_PATH + "StanfordDogs/archive/annotations/Annotation/"
IMAGE_DATA_PATH = DATA_PATH + "StanfordDogs/archive/images/Images/"

DOG_BREED_DATA = DATA_PATH + "DogBreeds/archive/Dog_Breed_Recognition_Competition_Datasets"
DOG_BREED_ID_MAPPING = DOG_BREED_DATA + "/Dog_Breed_id_mapping.csv"
DOG_BREED_TRAINING_DATA_INFO = DOG_BREED_DATA + "/Dog_Breed_trainingdata.csv"
DOG_BREED_TEST_DATA_INFO = DOG_BREED_DATA + "/Dog_Breed_testdata_sorted.csv"
DOG_BREED_TRAINING_DATA = DOG_BREED_DATA + "/Dog_Breed_Training_Images/"
DOG_BREED_TEST_DATA = DOG_BREED_DATA + "/Dog_Breed_Test_Images/"

# CNN config
BATCH_SIZE=32
CNN_EPOCHS=50

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth

https://www.kaggle.com/hengzheng/dog-breeds-classifier
https://www.kaggle.com/malhotra1432/dog-breed-prediction-competition?fbclid=IwAR3c5F2od6oHAqirSu5t-IX76KCZLF7K28J8rTp4BXDdn16lBhwyDvXO3fw


https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
https://medium.com/@sourav_srv_bhattacharyya/image-augmentation-to-build-a-powerful-image-classification-model-3303e40af7b0
"""
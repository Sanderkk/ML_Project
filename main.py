from models.CNN import *
from processing.dataProcessing import *
from CONFIG import *

if __name__ == "__main__":
    # CNN
    CNN(IMAGE_SIZE, TRAINING_SET_SIZE)
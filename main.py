from models.CNN import *
from processing.dataProcessing import *
from CONFIG import *
import os

if __name__ == "__main__":
    # CNN

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    CNN(IMAGE_SIZE, TRAINING_SET_SIZE)
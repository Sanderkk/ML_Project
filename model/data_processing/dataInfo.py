from data_processing.dataProcessing import *
import matplotlib.pyplot as plt
import numpy as np

# Return a list of number of images in each class. The number of classes used are deffined as class count
# data_type null infers not filtering the data paths as training, validation and test-data paths
def data_info(paths):
    image_counts = [ len(x) for y, x in paths.items() ]
    return image_counts

# Get the minimum and maximum amount of images in the classes
def min_max_data(paths):
    images = data_info(paths)
    return min(images), max(images)

# Get the average amount of data in the classes in the dataset
def average_data_count(paths):
    images = data_info(paths)
    return sum(images) / len(images)

# Count the complete amount of data in the dataset
def data_count(paths):
    return sum(data_info(paths))

# Show an amount, given by "count", of data from the dataset.
def show_data_examples(count=9):
    name_dir_map = get_name_dir_mapping()
    paths = get_file_paths(data_type=None, class_count=9)
    images = []
    labels = []
    keys = list(paths.keys())
    for key in keys:
        image = Image.open(IMAGE_DATA_PATH + key + "/" + paths[key][0] + ".jpg")
        image.resize((32,32))
        images.append(np.asarray(image))
        labels.append(name_dir_map[key])

    plt.figure(figsize=(10, 10))
    for i in range(count):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(labels[i])
        plt.axis("off")
    plt.show()

# Show a number of augmented data. The number of augmentation of the data is given form the "count" argument
def show_augmented_images(count=9):
    name_dir_map = get_name_dir_mapping()
    paths = get_file_paths(data_type=None, class_count=1)
    images = []
    labels = []
    keys = list(paths.keys())
    for key in keys:
        image = Image.open(IMAGE_DATA_PATH + key + "/" + paths[key][0] + ".jpg")
        doc = ET.parse(ANNOTATIONS_PATH + key + "/" + paths[key][0])
        if IMAGE_CROP:
            image = crop_image(image, doc)
        image = image.resize(IMAGE_SIZE)
        image.resize((32, 32))
        it = image_generation(image)
        for i in range(count):
            batch = it.next()
            augmented_image = batch[0].astype('uint8')
            images.append(np.asarray(augmented_image))
            labels.append(name_dir_map[key])

    plt.figure(figsize=(10, 10))
    for i in range(count):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(str(labels[i]))
        plt.axis("off")
    plt.show()


# Print statistics derived from the dataset
def get_data_statistic(class_count=0, data_type=None):
    paths = get_file_paths(data_type=data_type, class_count=class_count)
    min_data, max_data = min_max_data(paths)
    print("The minimum amount of data in the dataset is:")
    print(str(min_data))
    print("The minimum amount of data in the dataset is:")
    print(str(max_data))
    print()

    average_count = average_data_count(paths)
    print("The average amount of data from the given dataset is:")
    print(average_count)
    print() # Newline

    count = data_count(paths)
    print("The total amount of data is:")
    print(count)

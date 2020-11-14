import csv
from CONFIG import *
from data_processing.dataProcessing import *
import numpy as np

def read_dog_races(file_path=DOG_BREED_ID_MAPPING):
    breeds = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                breed = [row[0], row[1]]
                breeds.append(breed)
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
    return breeds

def read_dog_breed_id(type="training"):
    image_names = {}
    with open(DOG_BREED_TRAINING_DATA_INFO if type == "training" else DOG_BREED_TEST_DATA_INFO) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                if not row[1] in image_names:
                    image_names[row[1]] = []
                image_names[row[1]].append(row[0])
                line_count += 1
    return image_names

def read_dog_breed_image(data_type="training", augment=True):
    image_names = read_dog_breed_id(data_type)
    image_classes = image_names.keys()
    image_classes = [ x for x in image_classes if x in GIVEN_CLASSES[:CLASS_COUNT]]
    image_class_dir_map = get_image_class_dir_map()
    save_data_path = DATA_PATH + "/processed_images/" + data_type + "/"
    for i in progressbar.progressbar(range(len(image_classes))):
        breed_name = image_classes[i]
        print(image_class_dir_map[breed_name], len(image_names[breed_name]))
        for image_name in image_names[breed_name][:300]:
            try:
                image = Image.open((DOG_BREED_TRAINING_DATA if data_type=="training" else DOG_BREED_TEST_DATA) + image_name + ".jpg")
                image = image.convert('RGB')
                image = image.resize(IMAGE_SIZE)
                if augment:
                    image_generator_it = image_generation(image)
                    save_image(image_generator_it, save_data_path + image_class_dir_map[breed_name] + "/" + image_name)
                else:
                    image.save(save_data_path + image_class_dir_map[breed_name] + "/" + image_name + ".jpg")
            except:
                pass

def get_image_class_dir_map():
    class_dir_map = {}
    breeds = read_dog_races()
    stanford = get_name_dir_mapping()

    extra_races = [x[0] for x in breeds]
    stanford_races = [x for y, x in stanford.items()]

    for race in stanford_races:
        race_name = " ".join([stanfor_breed.capitalize() for stanfor_breed in race.split("_")])
        if race_name in extra_races:
            class_dir_map[race_name] = race
    return class_dir_map


def get_similar_data(min_lim=300):
    image_names = read_dog_breed_id()
    breeds = read_dog_races()
    stanford = get_name_dir_mapping()

    extra_races = [x[0] for x in breeds]
    stanford_races = [x for y, x in stanford.items()]
    print(len(extra_races), extra_races)
    print(len(stanford_races), stanford_races)

    same_breed = []
    for race in stanford_races:
        race_name = " ".join([stanfor_breed.capitalize() for stanfor_breed in race.split("_")])
        if race_name in extra_races and race_name in image_names and len(image_names[race_name]) > min_lim:
            same_breed.append(race_name)
    print(same_breed[:CLASS_COUNT])
    print(len(same_breed))

if __name__ == "__main__":
    # process_data()
    read_dog_breed_image(augment=True)
    # get_similar_data()
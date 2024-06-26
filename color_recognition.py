# Based on the work of:
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

import csv
import math
import operator
import numpy as np
import os
import cv2

def color_histogram_of_test_image(test_src_image) -> str:
    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    red, green, blue = '', '', ''
    for (chan, _) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    return feature_data


def color_histogram_of_training_image(img_name: str, data_source: str) -> str:
    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    red, green, blue = '', '', ''
    for (chan, _) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    return f'{feature_data},{data_source}\n'


def training(training_dataset_dir: str, training_file_name: str):
    lines = []
    with os.scandir(training_dataset_dir) as color_dirs:
        for color_dirs_entry in color_dirs:
            if not color_dirs_entry.is_dir():
                continue

            with os.scandir(f'{training_dataset_dir}/{color_dirs_entry.name}') as color_dir:
                for color_dir_entry in color_dir:
                    if not color_dir_entry.is_file():
                        continue
                
                    lines.append(color_histogram_of_training_image(f'{training_dataset_dir}/{color_dirs_entry.name}/{color_dir_entry.name}', color_dirs_entry.name))

    with open(training_file_name, 'w') as training_file_file:
        training_file_file.write(''.join(lines))

# calculation of euclidead distance
def calculateEuclideanDistance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)

class KnnClassifier:
    def __init__(self, training_data: str) -> None:
        self.training_data = training_data
        self.training_feature_vector = []
        self.test_feature_vector = []
        self.load_training_dataset()

    def load_training_dataset(self) -> None:
        with open(self.training_data) as csvfile:
            lines = csv.reader(csvfile)
            dataset: list[list[str | float]] = list(lines) # pyright: ignore
            for x in range(len(dataset)):
                for y in range(3):
                    dataset[x][y] = float(dataset[x][y])
                self.training_feature_vector.append(dataset[x])

    def load_test_dataset(self, training_data: str) -> None:
        self.test_feature_vector = []
        lines = csv.reader([training_data])
        dataset = list(lines)
        f_dataset = np.empty((len(dataset), 3), dtype=float)
        for x in range(len(dataset)):
            for y in range(3):
                f_dataset[x][y] = float(dataset[x][y])
            self.test_feature_vector.append(f_dataset[x].tolist())

    def predict(self, training_data: str) -> str:
        classifier_prediction = []
        k = 3
        self.load_test_dataset(training_data)
        for x in range(len(self.test_feature_vector)):
            neighbors = self.kNearestNeighbors(self.test_feature_vector[x], k)
            result = self.responseOfNeighbors(neighbors)
            classifier_prediction.append(result)
        return classifier_prediction[0]

    # get k nearest neigbors
    def kNearestNeighbors(self, testInstance, k) -> list[float]:
        distances = []
        length = len(testInstance)
        for x in range(len(self.training_feature_vector)):
            dist = calculateEuclideanDistance(testInstance,
                    self.training_feature_vector[x], length)
            distances.append((self.training_feature_vector[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    # votes of neighbors
    def responseOfNeighbors(self, neighbors) -> str:
        all_possible_neighbors = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in all_possible_neighbors:
                all_possible_neighbors[response] += 1
            else:
                all_possible_neighbors[response] = 1
        sortedVotes = sorted(all_possible_neighbors.items(),
                             key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

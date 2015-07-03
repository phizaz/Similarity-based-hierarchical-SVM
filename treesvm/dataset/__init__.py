# dataset
import numpy as np
import csv

class Dataset:

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    # load dataset from file
    # the user can provide some adapter
    # adapter: input: a row , output: feature, label of that row
    @staticmethod
    def load(file, adapter=None):
        features = []
        labels = []

        with open(file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if adapter != None:
                    feature, label = adapter(row)
                else:
                    feature, label = row[:-1], row[-1]

                features.append(feature)
                labels.append(label)

        data = Dataset(np.array(features).astype(float), np.array(labels))
        return data

    # split a dataset into groups according to its class
    @staticmethod
    def split(dataset):
        classes = {}

        for idx, row in enumerate(dataset.features):
            label = dataset.labels[idx]
            if not label in classes:
                classes[label] = []

            classes[label].append(row)

        # convert normal list into numpy array
        for key, val in classes.items():
            classes[key] = np.array(val)

        return classes

# dataset
import numpy as np
import math
import csv

class Dataset:
    features = None
    labels = None
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

# load dataset from file
def Load(file):
    features = []
    labels = []

    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            features.append( row[:-1] )
            labels.append( row[-1] )

    data = Dataset( np.array(features).astype(float), np.array(labels) )
    return data


# split a dataset into groups according to its class
def Split(dataset):
    classes = {}

    for idx, row in enumerate(dataset.features):
        label = dataset.labels[ idx ]
        if not label in classes:
            classes[label] = []

        classes[label].append( row )

    # convert normal list into numpy array
    for key, val in classes.items():
        classes[key] = np.array(val)

    return classes

# find the squared radius of a given class
def SquaredRadius(points, kernel):
    def InSqrt(point):
        # first term of the equation goes here
        a = kernel(point, point)

        # second term of the equation
        def MiddleTerm(point):
            summation = 0
            for p in points:
                summation += kernel(point, p)
            return -2. / points.size * summation

        b = MiddleTerm(point)

        # last term of the equation
        # this function doesn't accept any parameters
        def LastTerm():
            # summation = 0
            # for xx in points:
            #     for yy in points:
            #         summation += kernel(xx, yy)
            # this is the optimized version of the code above
            summation = 0
            for i, xx in enumerate(points):
                summation += kernel(xx, xx)
                for yy in points[i + 1:]:
                    summation += 2 * kernel(xx, yy)
            return 1. / points.size ** 2 * summation

        c = LastTerm()

        # put it all together
        return a + b + c

    # calculate the radius (max of the sqrt thing)
    squaredRadius = [ InSqrt(point) for point in points ]
    # return the maximum
    return max( squaredRadius )


# squared distance
def SquaredDistance(classA, classB, kernel):
    # square distance calculations (without sqrt)
    # first term
    def FirstTerm():
        summation = 0
        for i, xx in enumerate(classA):
            summation += kernel(xx, xx,)
            for yy in classA[i + 1:]:
                summation += 2 * kernel(xx, yy)

        return 1. / classA.size ** 2 * summation

    # second term
    def SecondTerm():
        summation = 0
        # this loop cannot be optmized by the factor of two
        for xx in classA:
            for yy in classB:
                summation += kernel(xx, yy)
        return -2. / ( classA.size * classB.size ) * summation

    # third term
    def ThirdTerm():
        summation = 0
        for i, xx in enumerate(classB):
            summation += kernel(xx, xx)
            for yy in classB[i + 1:]:
                summation += 2 * kernel(xx, yy)
        return 1. / classB.size ** 2 * summation

    return FirstTerm() + SecondTerm() + ThirdTerm()

# dataset
import numpy as np
import csv


class Dataset:
    features = None
    labels = None

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

    # find the squared radius of a given class
    @staticmethod
    def squared_radius(all_points, kernel):
        # last term of the equation
        # this function doesn't accept any parameters
        def last_term():
            # summation = 0
            # for xx in points:
            #     for yy in points:
            #         summation += kernel(xx, yy)
            # this is the optimized version of the code above
            summation = 0
            for i, xx in enumerate(all_points):
                summation += kernel(xx, xx)
                for yy in all_points[i + 1:]:
                    summation += 2 * kernel(xx, yy)
            return 1. / all_points.shape[0] ** 2 * summation

        # we cache the result for future uses
        c = last_term()

        def in_sqrt(a_point):
            # first term of the equation goes here
            a = kernel(a_point, a_point)

            # second term of the equation
            def middle_term(point):
                summation = 0
                for p in all_points:
                    summation += kernel(point, p)
                return -2. / all_points.shape[0] * summation

            b = middle_term(a_point)
            # put it all together
            # c from above
            return a + b + c

        # calculate the radius (max of the sqrt thing)
        squared_radius = [in_sqrt(point) for point in all_points]
        # for each in squared_radius:
        #     assert each >= 0
        # return the maximum
        return max(squared_radius)

    # squared distance maker
    @staticmethod
    def squared_distance_maker():
        # caching techniques
        cache_first_term = {}
        cache_third_term = {}

        def squared_distance(label_a, class_a, label_b, class_b, kernel):
            # square distance calculations (without sqrt)
            # first term
            def first_term():
                if label_a in cache_first_term:
                    return cache_first_term[label_a]

                summation = 0
                # normally is the faster form of full nested for loops
                for i, xx in enumerate(class_a):
                    summation += kernel(xx, xx)
                    for yy in class_a[i + 1:]:
                        summation += 2 * kernel(xx, yy)

                result = 1. / class_a.shape[0] ** 2 * summation
                cache_first_term[label_a] = result
                return result

            # second term
            def second_term():
                summation = 0
                # this loop cannot be optimized by the factor of two
                for xx in class_a:
                    for yy in class_b:
                        summation += kernel(xx, yy)
                return -2. / (class_a.shape[0] * class_b.shape[0]) * summation

            # third term
            def third_term():
                if label_b in cache_third_term:
                    return cache_third_term[label_b]

                summation = 0
                for i, xx in enumerate(class_b):
                    summation += kernel(xx, xx)
                    for yy in class_b[i + 1:]:
                        summation += 2 * kernel(xx, yy)
                result = 1. / class_b.shape[0] ** 2 * summation
                cache_third_term[label_b] = result
                return result

            distance = first_term() + second_term() + third_term()
            # assert distance > 0
            return distance

        return squared_distance

    def squared_distance(class_a, class_b, kernel):
        # square distance calculations (without sqrt)
        # first term
        def first_term():
            summation = 0
            # normally is the faster form of full nested for loops
            for i, xx in enumerate(class_a):
                summation += kernel(xx, xx)
                for yy in class_a[i + 1:]:
                    summation += 2 * kernel(xx, yy)

            result = 1. / class_a.shape[0] ** 2 * summation
            return result

        # second term
        def second_term():
            summation = 0
            # this loop cannot be optimized by the factor of two
            for xx in class_a:
                for yy in class_b:
                    summation += kernel(xx, yy)
            return -2. / (class_a.shape[0] * class_b.shape[0]) * summation

        # third term
        def third_term():
            summation = 0
            for i, xx in enumerate(class_b):
                summation += kernel(xx, xx)
                for yy in class_b[i + 1:]:
                    summation += 2 * kernel(xx, yy)
            result = 1. / class_b.shape[0] ** 2 * summation
            return result

        distance = first_term() + second_term() + third_term()
        # assert distance > 0
        return distance

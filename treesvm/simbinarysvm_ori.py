import numpy
from . import graph
from . import multitree
import sklearn.svm
import time
from treesvm.dataset.tools import Tools
from treesvm.group.group import Group
from treesvm.group.groupmanager import GroupManager


class SimBinarySVMORI:

    def __init__(self, gamma=0.1, C=1.0, verbose=False):
        self.int_to_label = None
        self.label_to_int = None
        self.gamma = gamma
        self.C = C
        self.verbose = verbose
        self.group_mgr = None

    def make_gram_matrix(self, vectors, gamma):
        if self.verbose:
            start_time = time.process_time()
        matrix = sklearn.metrics.pairwise.rbf_kernel(vectors, gamma=gamma)
        if self.verbose:
            print('gram matrix: %.4f' % (time.process_time() - start_time))

        def rbf(a, b):
            # vector a, b need to have index at the first element
            return matrix[int(a[0])][int(b[0])]
        return rbf

    def _create_mapping(self, training_classes):
        # create mapping function from labels to integers
        class_cnt = len(training_classes.keys())
        label_to_int = {}
        int_to_label = [None for i in range(class_cnt)]
        for i, label in enumerate(training_classes.keys()):
            label_to_int[label] = i
            int_to_label[i] = label
        return label_to_int, int_to_label

    def _create_tree(self, training_classes, label_to_int):
        # create a matrix list and give them indexes
        vectors = []
        training_classes_with_idx = {}
        idx = 0
        for name, points in training_classes.items():
            this_class = training_classes_with_idx[name] = []
            for point in points:
                # give it an index
                vector = point.tolist()
                vector_with_idx = [idx] + vector
                idx += 1
                vectors.append(vector)
                this_class.append(vector_with_idx)
            training_classes_with_idx[name] = numpy.array(this_class)

        vectors = numpy.array(vectors)
        kernel = self.make_gram_matrix(vectors, self.gamma)

        # group manager
        group_mgr = GroupManager(kernel)

        # add all classes to the group manager
        # use the int-based group name
        for name, point in training_classes_with_idx.items():
            universe = {
                label_to_int[name]: point
            }
            group = group_mgr.create_group(universe)
            # add this group to the manager
            group_mgr.add(group)

        # starting to merge them one by one
        while len(group_mgr.groups.keys()) > 1:
            most_similar = group_mgr.most_similar()
            # merge them
            group_mgr.merge(most_similar[0], most_similar[1])

        return group_mgr

    def train(self, training_classes):
        label_to_int, int_to_label = self._create_mapping(training_classes)
        self.label_to_int = label_to_int
        self.int_to_label = int_to_label
        group_mgr = self._create_tree(training_classes, label_to_int)

        def runner(current):
            if current.children == None:
                return

            # create train samples
            samples = None
            labels = []
            # this is int-based name
            for name in current.universe:
                points = training_classes[int_to_label[name]]
                if samples == None:
                    samples = points
                else:
                    samples = numpy.append(samples, points, axis=0)

                for idx, child in enumerate(current.children):
                    if name in child.universe.keys():
                        labels += [idx for i in points]

            current.svm = sklearn.svm.SVC(kernel='rbf', gamma=self.gamma, C=self.C).fit(samples, labels)
            runner(current.children[0])
            runner(current.children[1])

        # train according to the tree structure in group manager
        root_group = next(iter(group_mgr.groups.values()))
        runner(root_group)
        self.group_mgr = group_mgr
        return group_mgr

    def predict(self, sample):
        # current is the only member in group manager
        current = next(iter(self.group_mgr.groups.values()))
        iterations = 0
        while current.children != None:
            prediction = current.svm.predict(sample)
            iterations += 1
            current = current.children[prediction]

        one_left = list(current.universe.keys())[0]
        return self.int_to_label[one_left], iterations

    def test(self, testing_classes):
        total = 0
        errors = 0
        total_itr = 0

        for name, points in testing_classes.items():
            for test in points:
                total += 1
                prediction, iterations = self.predict(test)
                total_itr += iterations
                if prediction != name:
                    errors += 1

        return total, errors, total_itr

    def cross_validate(self, folds, training_classes):
        total = 0
        for key, val in training_classes.items():
            total += val.size

        random_list = [i % folds for i in range(total)]
        # should  we shuffle it ?
        # random.shuffle( ramdom_list )
        acc_total = 0
        acc_errors = 0
        for i in range(folds):
            training = {}
            testing = {}
            # select a portion to be left
            no = 0
            training_cnt = 0
            testing_cnt = 0
            for class_name, class_samples in training_classes.items():
                training[class_name] = []
                testing[class_name] = []
                for sample in class_samples:
                    if no % folds != i:
                        # this is in
                        training[class_name].append(sample)
                        training_cnt += 1
                    else:
                        # keep this for testing
                        testing[class_name].append(sample)
                        testing_cnt += 1
                    no += 1
                training[class_name] = numpy.array(training[class_name])
                testing[class_name] = numpy.array(testing[class_name])

                # print('training: ', 'name: ', class_name, 'size: ', training[class_name].size)
                # print('testing: ', 'name: ', class_name, 'size: ', testing[class_name].size)

            # train the rest
            self.train(training)

            # test with the leftover
            test_result = self.test(testing)
            acc_total += test_result[0]
            acc_errors += test_result[1]

        # average the error
        cross_validation_error = acc_errors / acc_total
        return cross_validation_error, acc_total, acc_errors
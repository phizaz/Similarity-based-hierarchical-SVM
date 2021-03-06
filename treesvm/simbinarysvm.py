import sys
import numpy
import time
from .dataset import Dataset
from . import graph
from . import binarytree
import sklearn.svm
import time
from treesvm.dataset.tools import Tools


class SimBinarySVM:
    def __init__(self, gamma=0.1, C=1.0, verbose=False):
        self.gamma = gamma
        self.C = C
        self.verbose = verbose
        self.tools = None

    # this use precomputed kernel matrix
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

    def _find_separability(self, training_classes):
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
        self.tools = Tools(kernel)

        # calculate all the sqRadiuses
        if self.verbose:
            start_time = time.process_time()
        sq_radiuses = {}
        for name, points in training_classes_with_idx.items():
            sq_radiuses[name] = self.tools.squared_radius(name, points)
        if self.verbose:
            print('sq_radiuses: %.4f' % (time.process_time() - start_time))

        # separability section
        # use the precalculated squared radiuses from above
        def find_separability(name_a, name_b):
            sq_ra = sq_radiuses[name_a]
            sq_rb = sq_radiuses[name_b]
            sq_dist = self.tools.squared_distance(
                name_a,
                training_classes_with_idx[name_a],
                name_b,
                training_classes_with_idx[name_b],
            )

            return sq_dist / (sq_ra + sq_rb)

        # create mapping function from labels to integers and vice ver
        class_cnt = len(training_classes.keys())
        label_to_int = {}
        int_to_label = [None for i in range(class_cnt)]
        for i, label in enumerate(training_classes.keys()):
            label_to_int[label] = i
            int_to_label[i] = label

        # 2d matrix showing separability of each
        if self.verbose:
            start_time = time.process_time()
        separability = numpy.empty((class_cnt, class_cnt))
        separability.fill(float('inf'))
        for i, a in enumerate(training_classes.keys()):
            int_a = label_to_int[a]
            # should be no separability with itself
            separability[int_a][int_a] = 0
            for b in list(training_classes.keys())[i + 1:]:
                int_b = label_to_int[b]
                separability[int_a][int_b] = separability[int_b][int_a] = find_separability(a, b)
        if self.verbose:
            print('separability: %.4f' % (time.process_time() - start_time))
            
        return separability, label_to_int, int_to_label

    def _construct_mst_graph(self, training_classes, separability):
        # construct a graph, and find its MST
        class_cnt = len(training_classes.keys())
        mesh = graph.Graph(class_cnt)
        for i, row in enumerate(separability):
            for j, sep in enumerate(row):
                mesh.link(i, j, sep)
        # find its MST
        mst_list = mesh.mst()
        mst_list.sort(key=lambda x: -x[2])
        # print(mst_list)

        # creat a graph of MST()
        mst_graph = graph.Graph(class_cnt)
        for link in mst_list:
            mst_graph.double_link(link[0], link[1], link[2])
        return mst_graph, mst_list

    def _construct_tree(self, mst_graph, mst_list):
        tree = binarytree.BinaryTree()
        # the root of the tree is a list of every node
        tree.add_root(binarytree.BinaryTreeNode(mst_graph.connected_with(0)))

        for link in mst_list:
            # remove this link
            mst_graph.double_unlink(link[0], link[1])
            parent = None
            # find where the link in the binary tree
            parent = tree.find(link[0])
            # explode this binarytree node into two
            left = binarytree.BinaryTreeNode(mst_graph.connected_with(link[0]))
            right = binarytree.BinaryTreeNode(mst_graph.connected_with(link[1]))
            tree.add_left(parent, left)
            tree.add_right(parent, right)
        return tree

    def train(self, training_classes):
        (self.separability, self.label_to_int, self.int_to_label) = \
            (separability, label_to_int, int_to_label) = self._find_separability(training_classes)

        self.class_cnt = class_cnt = len(training_classes.keys())

        (self.mst_graph, self.mst_list) = (mst_graph, mst_list) = self._construct_mst_graph(training_classes,
                                                                                            separability)

        # recursively disconnect the largest distance link of the MST
        self.tree = tree = self._construct_tree(mst_graph, mst_list)

        # create SVMs according to this tree
        # train svm ..
        svm_cnt = 0
        def train(training_classes):
            # svm must be recursively trained
            def runner(current, universe):
                # if the current has no children, cannot separate anymore
                if current.left == None and current.right == None:
                    return

                # details of training is here
                left_class = {}
                right_class = {}
                for class_name, class_samples in universe.items():
                    # decide if this label is left hand side or right ?
                    if class_name in current.left.val:
                        # it belongs to the left group
                        left_class[class_name] = class_samples
                    else:
                        # add the class into the dropbox
                        right_class[class_name] = class_samples

                # the label of the left side is '0'
                # the lable of the right side is '1'
                training = []
                label = []

                for class_name, class_samples in left_class.items():
                    samples = class_samples.tolist()
                    training += samples
                    label += [0 for i in samples]

                for class_name, class_samples in right_class.items():
                    samples = class_samples.tolist()
                    training += samples
                    label += [1 for i in samples]

                training = numpy.array(training)
                label = numpy.array(label)

                svm = sklearn.svm.SVC(kernel='rbf', gamma=self.gamma, C=self.C).fit(training, label)
                nonlocal svm_cnt
                svm_cnt += 1
                # we will use the 'svm' attribute of each node (arbitrarily added)
                current.svm = svm

                runner(current.left, left_class)
                runner(current.right, right_class)

            # start training from the tree's root
            universe = {}
            for key, val in training_classes.items():
                universe[self.label_to_int[key]] = val
            runner(tree.root, universe)

        # the result is stored in the tree , self.tree
        if self.verbose:
            start_time = time.process_time()
        train(training_classes)
        if self.verbose:
            print('train: %.4f' % (time.process_time() - start_time))
        return svm_cnt

    def predict(self, sample):
        iterations = 0
        def runner(current):
            # if it is the leaf of the tree, return its value
            if current.left == None and current.right == None:
                return current.val[0]

            prediction = current.svm.predict(sample)
            nonlocal iterations
            iterations += 1

            if prediction[0] == 0:
                # goes left
                return runner(current.left)
            else:
                # goes right
                return runner(current.right)

        return self.int_to_label[runner(self.tree.root)], iterations

    def test(self, testing_classes):
        total = 0
        errors = 0
        total_itr = 0

        for class_name, tests in testing_classes.items():
            for test in tests:
                total += 1
                prediction, iterations = self.predict(test)
                total_itr += iterations
                if prediction != class_name:
                    errors += 1

        return total, errors, total_itr

    def cross_validate(self, folds, training_classes):
        total = 0
        for key, val in training_classes.items():
            total += val.size

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

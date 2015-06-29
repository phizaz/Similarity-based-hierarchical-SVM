import numpy
import time
from sklearn import svm as sklearnSVM
from .dataset import Dataset
from . import graph
from . import binarytree

class SimBinarySVM:

    def __init__(self, gamma=0.1, C=1.0):
        # kernel should have its own gamma
        # but, since we're using the traditional scikit 'rbf' version as well, we need another gamma for it as well
        self.kernel = self.make_rbf_kernel(gamma)
        self.gamma = gamma
        self.C = C

    def make_rbf_kernel(self, gamma):
        cache = {}

        def rbf(a, b):
            label_a = hash(tuple(a))
            label_b = hash(tuple(b))
            if label_a in cache:
                cached_label_a = cache[label_a]
                if label_b in cached_label_a:
                    return cached_label_a[label_b]
            else:
                cached_label_a = cache[label_a] = {}
            cached_label_a[label_b] = res = numpy.exp(-gamma * numpy.linalg.norm(a - b) ** 2)
            return res

        return rbf

    def _find_similarity(self, training_classes):
        find_squared_distance = Dataset.squared_distance_maker()
        # calculate all the sqRadiuses
        sq_radiuses = {}
        for name, points in training_classes.items():
            startTime = time.time()
            sq_radiuses[name] = Dataset.squared_radius(points, self.kernel)
            elapsedTime = time.time() - startTime

        # similarity section
        # use the precalculated squared radiuses from above
        def pair_similarity(nameA, nameB):
            sqRA = sq_radiuses[nameA]
            sqRB = sq_radiuses[nameB]
            sqDist = find_squared_distance(
                nameA,
                training_classes[nameA],
                nameB,
                training_classes[nameB],
                self.kernel, )

            return (sqRA + sqRB) / sqDist

        # create mapping function from labels to integers and vice versa
        classCnt = len( training_classes.keys() )
        labelToInt = {}
        intToLabel = [ None for i in range(classCnt) ]
        for i, label in enumerate(training_classes.keys()):
            labelToInt[label] = i
            intToLabel[i] = label

        # 2d matrix showing similarity of each
        similarity = numpy.zeros(( classCnt, classCnt ))
        for i, classA in enumerate(training_classes.keys()):
            # convert to int
            a = labelToInt[classA]
            for classB in list( training_classes.keys() )[i+1:]:
                # convert to int
                b = labelToInt[classB]

                similarity[a][b] = similarity[b][a] = pair_similarity(classA, classB)
                # print( 'similarity of %s and %s : %f' % ( classA, classB, similarity[a][b] ) )
        return similarity, labelToInt, intToLabel

    def _construct_mst_graph(self, training_classes, similarity):
        # construct a graph, and find its MST
        classCnt = len( training_classes.keys() )
        mesh = graph.Graph(classCnt)
        for i, row in enumerate(similarity):
            for j, col in enumerate(row):
                mesh.link(i, j, col)
        # find its MST
        mst_list = mesh.mst()
        mst_list.sort(key=lambda x: -x[2])
        # print(mst_list)

        # creat a graph of MST
        mst_graph = graph.Graph(classCnt)
        for link in mst_list:
            mst_graph.double_link(link[0], link[1], link[2])
        return (mst_graph, mst_list)

    def _construct_tree(self, mst_graph, mst_list):
        tree = binarytree.BinaryTree()
        # the root of the tree is a list of every node
        tree.add_root( binarytree.BinaryTreeNode( mst_graph.connected_with(0) ) )
        left = tree.root
        right = None
        for link in mst_list:
            # remove this link
            mst_graph.unlink(link[0], link[1])
            mst_graph.unlink(link[1], link[0])
            parent = None
            # find where the link in the binary tree
            if link[0] in left.val:
                parent = left
            else:
                parent = right
            # explode this binarytree node into two
            left = binarytree.BinaryTreeNode( mst_graph.connected_with(link[0]) )
            right = binarytree.BinaryTreeNode( mst_graph.connected_with(link[1]) )
            tree.add_left(parent, left)
            tree.add_right(parent, right)
        return tree

    def train(self, training_classes):

        (self.similarity, self.labelToInt, self.intToLabel) = \
        (similarity, labelToInt, intToLabel) = self._find_similarity(training_classes)

        self.classCnt = classCnt = len( training_classes.keys() )

        (self.mst_graph, self.mst_list) = (mst_graph, mst_list) = self._construct_mst_graph(training_classes, similarity)

        # recursively disconnect the largest distance link of the MST
        self.tree = tree = self._construct_tree(mst_graph, mst_list)

        # create SVMs according to this tree
        # train svm ..
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
                    # just put in this var that's all
                    dropbox = None
                    # decide if this label is left hand side or right ?
                    if class_name in current.left.val:
                        # it belongs to the left group
                        dropbox = left_class
                    else:
                        dropbox = right_class
                    # add the class into the dropbox
                    dropbox[class_name] = class_samples

                # the label of the left side is '0'
                # the lable of the right side is '1'
                training = []
                label = []

                for class_name, class_samples in left_class.items():
                    samples = class_samples.tolist()
                    training += samples
                    label += [ 0 for i in range( len(samples) ) ]

                for class_name, class_samples in right_class.items():
                    samples = class_samples.tolist()
                    training += samples
                    label += [ 1 for i in range( len(samples) ) ]

                training = numpy.array(training)
                label = numpy.array(label)

                svm = sklearnSVM.SVC(kernel='rbf', gamma=self.gamma, C=self.C).fit(training, label)
                # we will use the 'svm' attribute of each node (arbitrarily added)
                current.svm = svm

                runner(current.left, left_class)
                runner(current.right, right_class)

            # start training from the tree's root
            universe = {}
            for key, val in training_classes.items():
                universe[ self.labelToInt[key] ] = val
            runner( tree.root, universe )

        # the result is stored in the tree , self.tree
        train( training_classes )
        return self.tree

    def predict(self, sample):
        def runner(current):
            # if it is the leaf of the tree, return its value
            if current.left == None and current.right == None:
                return current.val[0]

            prediction = current.svm.predict(sample)

            if prediction == 0:
                # goes left
                return runner(current.left)
            else:
                # goes right
                return runner(current.right)

        return self.intToLabel[ runner(self.tree.root) ]

    def test(self, testing_classes):
        total = 0
        errors = 0

        for class_name, tests in testing_classes.items():
            for test in tests:
                total += 1
                prediction = self.predict(test)
                if prediction != class_name:
                    errors += 1

        return total, errors

    def cross_validate(self, folds, training_classes):
        total = 0
        for key, val in training_classes.items():
            total += val.size

        random_list = [ i % folds for i in range(total) ]
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

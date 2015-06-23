import numpy as np
import random
import simbinarysvm.dataset as dataset
import simbinarysvm.graph as graph
import simbinarysvm.binarytree as binarytree
import math
import time
from sklearn import svm as sklearnSVM

class SimBinarySVM:

    def __init__(self, gamma=0.1, C=1.0):
        # kernel should have its own gamma
        # but, since we're using the traditional scikit 'rbf' version as well, we need another gamma for it as well
        self.kernel = self.MakeRBFKernel(gamma)
        self.gamma = gamma
        self.C = C

    def MakeRBFKernel(self, gamma):
        def rbf(a, b):
            return np.exp( gamma * np.linalg.norm(a - b) ** 2)
        return rbf

    def _FindSimilarity(self, trainingClasses):
        # calculate all the sqRadiuses
        sqRadiuses = {}
        for name, points in trainingClasses.items():
            startTime = time.time()
            sqRadiuses[name] = dataset.SquaredRadius(points, self.kernel)
            elapsedTime = time.time() - startTime

        # similarity section
        # use the precalculated squared radiuses from above
        def Similarity(nameA, nameB):
            sqRA = sqRadiuses[nameA]
            sqRB = sqRadiuses[nameB]
            sqDist = dataset.SquaredDistance(
                trainingClasses[nameA],
                trainingClasses[nameB],
                self.kernel, )

            return (sqRA + sqRB) / sqDist

        # create mapping function from labels to integers and vice versa
        classCnt = len( trainingClasses.keys() )
        labelToInt = {}
        intToLabel = [ None for i in range(classCnt) ]
        for i, label in enumerate(trainingClasses.keys()):
            labelToInt[label] = i
            intToLabel[i] = label

        # 2d matrix showing similarity of each
        similarity = np.zeros(( classCnt, classCnt ))
        for i, classA in enumerate(trainingClasses.keys()):
            # convert to int
            a = labelToInt[classA]
            for classB in list( trainingClasses.keys() )[i+1:]:
                # convert to int
                b = labelToInt[classB]

                similarity[a][b] = similarity[b][a] = Similarity(classA, classB)
                # print( 'similarity of %s and %s : %f' % ( classA, classB, similarity[a][b] ) )
        return (similarity, labelToInt, intToLabel)

    def _ConstructMSTGraph(self, trainingClasses, similarity):
        # construct a graph, and find its MST
        classCnt = len( trainingClasses.keys() )
        mesh = graph.Graph(classCnt)
        for i, row in enumerate(similarity):
            for j, col in enumerate(row):
                mesh.Link(i, j, col)
        # find its MST
        mst_list = mesh.MST()
        mst_list.sort(key=lambda x: -x[2])
        # print(mst_list)

        # creat a graph of MST
        mst_graph = graph.Graph(classCnt)
        for link in mst_list:
            mst_graph.DoubleLink(link[0], link[1], link[2])
        return (mst_graph, mst_list)

    def _ConstructTree(self, mst_graph, mst_list):
        tree = binarytree.BinaryTree()
        # the root of the tree is a list of every node
        tree.AddRoot( binarytree.Node( mst_graph.ConnectedWith(0) ) )
        left = tree.root
        right = None
        for link in mst_list:
            # remove this link
            mst_graph.Unlink(link[0], link[1])
            mst_graph.Unlink(link[1], link[0])
            parent = None
            # find where the link in the binary tree
            if link[0] in left.value:
                parent = left
            else:
                parent = right
            # explode this binarytree node into two
            left = binarytree.Node( mst_graph.ConnectedWith(link[0]) )
            right = binarytree.Node( mst_graph.ConnectedWith(link[1]) )
            tree.AddLeft(parent, left)
            tree.AddRight(parent, right)
        return tree

    def Train(self, trainingClasses):

        (self.similarity, self.labelToInt, self.intToLabel) = \
        (similarity, labelToInt, intToLabel) = self._FindSimilarity(trainingClasses)

        self.classCnt = classCnt = len( trainingClasses.keys() )

        (self.mst_graph, self.mst_list) = (mst_graph, mst_list) = self._ConstructMSTGraph(trainingClasses, similarity)

        # recursively disconnect the largest distance link of the MST
        self.tree = tree = self._ConstructTree(mst_graph, mst_list)

        # create SVMs according to this tree
        # train svm ..
        def train(trainingClasses):
            # svm must be recursively trained
            def runner(current, trainingUniverse):
                # if the current has no children, cannot separate anymore
                if current.left == None and current.right == None:
                    return

                # details of training is here

                left_class = {}
                right_class = {}
                for class_name, class_samples in trainingUniverse.items():
                    # just put in this var that's all
                    dropbox = None
                    # decide if this label is left hand side or right ?
                    if class_name in current.left.value:
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

                training = np.array(training)
                label = np.array(label)

                svm = sklearnSVM.SVC(kernel='rbf', gamma=self.gamma, C=self.C).fit(training, label)
                # we will use the 'svm' attribute of each node (arbitrarily added)
                current.svm = svm

                runner(current.left, left_class)
                runner(current.right, right_class)

            # start training from the tree's root
            runner( tree.root, trainingClasses )

        # the result is stored in the tree , self.tree
        universe = {}
        for key, val in trainingClasses.items():
            universe[ self.labelToInt[key] ] = val

        train( universe )
        return self.tree

    def Predict(self, sample):
        def runner(current):
            # if it is the leaf of the tree, return its value
            if current.left == None and current.right == None:
                return current.value[0]

            prediction = current.svm.predict(sample)

            if prediction == 0:
                # goes left
                return runner(current.left)
            else:
                # goes right
                return runner(current.right)

        return self.intToLabel[ runner(self.tree.root) ]

    def CrossValidate(self, folds, trainingClasses):
        total = 0
        for key, val in trainingClasses.items():
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
            for class_name, class_samples in trainingClasses.items():
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
                training[class_name] = np.array(training[class_name])
                testing[class_name] = np.array(testing[class_name])

                # print('training: ', 'name: ', class_name, 'size: ', training[class_name].size)
                # print('testing: ', 'name: ', class_name, 'size: ', testing[class_name].size)

            # train the rest
            self.Train(training)

            # test with the leftover
            for class_name, class_tests in testing.items():
                for test in class_tests:
                    prediction = self.Predict(test)
                    acc_total += 1
                    if prediction != class_name:
                        acc_errors += 1

        # average the error
        cverror = acc_errors / acc_total
        return (cverror, acc_total, acc_errors)

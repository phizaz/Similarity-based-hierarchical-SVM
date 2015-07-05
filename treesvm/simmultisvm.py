import time
import numpy
import sklearn
from treesvm.binarytree import BinaryTree, BinaryTreeNode

from treesvm.dataset import Dataset
from treesvm.dataset.tools import Tools
from treesvm.graph import Graph
from treesvm.multitree import MultiTree, MultiTreeNode


class SimMultiSVM:
    def __init__(self, gamma=0.1, C=1, verbose=False):
        self.label_to_int = None
        self.int_to_label = None
        self.tree = None
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

    # this uses normal caching techinque
    # def make_rbf_kernel(self, gamma):
    #     cache = {}
    #
    #     def rbf(a, b):
    #         # this one is very fast!!
    #         stra = hash(a.tostring())
    #         strb = hash(b.tostring())
    #         if stra < strb:
    #             label_a = stra
    #             label_b = strb
    #         else:
    #             label_a = strb
    #             label_b = stra
    #         if label_a in cache:
    #             cached_label_a = cache[label_a]
    #             if label_b in cached_label_a:
    #                 return cached_label_a[label_b]
    #         else:
    #             cached_label_a = cache[label_a] = {}
    #         cached_label_a[label_b] = res = numpy.exp(-gamma * numpy.linalg.norm(a - b) ** 2)
    #         return res
    #
    #     return rbf

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

        # find radius of each class
        if self.verbose:
            start_time = time.process_time()
        sq_radiuses = {}
        for name, points in training_classes_with_idx.items():
            sq_radiuses[name] = self.tools.squared_radius(name, points)
        if self.verbose:
            print('train: %.4f' % (time.process_time() - start_time))

        def find_separability(a, b):
            sq_ra = sq_radiuses[a]
            sq_rb = sq_radiuses[b]
            sq_dist = self.tools.squared_distance(
                a,
                training_classes_with_idx[a],
                b,
                training_classes_with_idx[b],
            )
            return sq_dist / (sq_ra + sq_rb)

        # relabelling
        class_cnt = len(training_classes.keys())
        label_to_int = {}
        int_to_label = [None for i in range(class_cnt)]
        for i, label in enumerate(training_classes.keys()):
            label_to_int[label] = i
            int_to_label[i] = label

        # find separability of each pair
        # default value is very high separability
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
            print('train: %.4f' % (time.process_time() - start_time))

        return separability, label_to_int, int_to_label

    def train(self, training_classes):
        separability, label_to_int, int_to_label = self._find_separability(training_classes)
        # create a mesh
        class_cnt = len(training_classes.keys())
        mesh = Graph(class_cnt)
        for i, row in enumerate(separability):
            for j, sep in enumerate(row):
                mesh.link(i, j, sep)

        # create the mst of this mesh
        mst_list = mesh.mst()
        mst_graph = Graph(class_cnt)
        for link in mst_list:
            mst_graph.double_link(link[0], link[1], link[2])

        # recursively remove links (that are greater than the average of the mst)
        # at the same time create the binary tree
        tree = MultiTree()
        # the root node is the node of all members, assume that all connected with 0
        all_classes = mst_graph.connected_with(0)
        tree.add_root(MultiTreeNode(all_classes))

        def runner(current):
            # loop to remove every link that larger than average
            sum_weight, edges = mst_graph.sum_weight(current.val[0])
            # terminate when there is not edge to go
            if len(edges) is 0:
                return
            # sort weight desc
            edges.sort(key=lambda x: -x[2])
            avg_weight = sum_weight / len(edges)

            btree = BinaryTree()
            btree.add_root(BinaryTreeNode(current.val))
            for edge in edges:
                # remove this link
                if edge[2] >= avg_weight:
                    # remove the link
                    mst_graph.double_unlink(edge[0], edge[1])
                    # add this link to the binary tree
                    parent = btree.find(edge[0])
                    # if edge[0] in left.val:
                    #     parent = left
                    # else:
                    #     parent = right
                    left = btree.add_left(parent, BinaryTreeNode(mst_graph.connected_with(edge[0])))
                    right = btree.add_right(parent, BinaryTreeNode(mst_graph.connected_with(edge[1])))
                # else or the last one
                if edge[2] < avg_weight or edge == edges[-1]:
                    # groups are the display output of the btree
                    groups = btree.leaves()
                    for group in groups:
                        new_node = MultiTreeNode(group)
                        # add new group to the leat of the multi tree
                        tree.add_child(current, new_node)
                        # recursively run it
                        runner(new_node)
                    # if the link's weight has become smaller than avg_weight,
                    # there is no need to keep going on
                    break

        if self.verbose:
            start_time = time.process_time()
        runner(tree.root)
        if self.verbose:
            print('train: %.4f' % (time.process_time() - start_time))


        # now got the tree
        # train svm according to the mulitree
        def train(training_classes):
            def runner(current, universe):
                if current.children == None:
                    return

                child_universes = [{} for each in current.children]
                for class_name, samples in universe.items():
                    for i, child in enumerate(current.children):
                        # the class belongs to this child
                        if class_name in child.val:
                            child_universes[i][class_name] = samples

                current.svms = [None for child in current.children]
                # one against the rest method
                for i, child in enumerate(current.children):
                    training = []
                    labels = []

                    for class_int, samples in universe.items():
                        # class in this child is marked as 0
                        if class_int in child.val:
                            training += samples.tolist()
                            labels += [0 for each in samples]
                        else:
                            # put to one labeled box
                            training += samples.tolist()
                            labels += [1 for each in samples]

                    training = numpy.array(training)
                    labels = numpy.array(labels)

                    # train the svms
                    # using one against the rest method
                    current.svms[i] = sklearn.svm.SVC(kernel='rbf', gamma=self.gamma, C=self.C) \
                        .fit(training, labels)
                    # the recursive part
                    runner(child, child_universes[i])
            # relabel all the classes to int based
            universe = {}
            for key, val in training_classes.items():
                universe[label_to_int[key]] = val
            runner(tree.root, universe)

        train(training_classes)
        # make these vars visible to the outsiders
        self.tree = tree
        self.int_to_label = int_to_label
        self.label_to_int = label_to_int
        return tree

    def predict(self, sample):
        iterations = 0
        def runner(current):
            if current.children is None:
                return current.val[0]
            # use confidence score
            confidence = [svm.decision_function(sample) for svm in current.svms]
            nonlocal iterations
            iterations += len(current.svms)
            # since the more the confidence is the more likely its gonna be '1' class
            # so we find the minimum to find the most likely to be '0' class
            min_pos, min_val = min(enumerate(confidence), key=lambda x: x[1])
            # recursively call down the tree
            return runner(current.children[min_pos])

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
        acc_total = 0
        acc_errors = 0
        for fold in range(folds):
            training = {}
            testing = {}
            # select a portion to be left
            i = 0
            training_cnt = 0
            testing_cnt = 0
            for name, samples in training_classes.items():
                training[name] = []
                testing[name] = []
                for sample in samples:
                    if i % folds != fold:
                        # this is in
                        training[name].append(sample)
                        training_cnt += 1
                    else:
                        # keep this for testing
                        testing[name].append(sample)
                        testing_cnt += 1
                    i += 1
                training[name] = numpy.array(training[name])
                testing[name] = numpy.array(testing[name])

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

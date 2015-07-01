from unittest import TestCase
import numpy
import random
from treesvm import SimBinarySVM
from treesvm.dataset import Dataset
import pytest

class TestSimBinarySVM(TestCase):
    training_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/satimage/sat-train-s.csv'
    training_set = Dataset.load(training_file)
    training_classes = Dataset.split(training_set)
    class_cnt = len(training_classes.keys())
    gamma = 1e-6
    C = 0.01
    svm = SimBinarySVM(gamma=gamma, C=C)

    # def test_MakeRBFKernel(self):
    #     self.fail()

    def test__FindSeparability(self):
        # svm = SimBinarySVM(Kernel)
        (self.svm.separability, self.svm.label_to_int, self.svm.int_to_label) = self.svm._find_separability(self.training_classes)
        # print('similarity', similarity)
        assert self.svm.separability.size == self.class_cnt * self.class_cnt
        assert self.svm.separability[0].size == self.class_cnt

        # print('labelToINt:', labelToInt)
        assert len(self.svm.label_to_int.keys()) == 6

        # print('intToLabel', intToLabel)
        for idx, val in enumerate(self.svm.int_to_label):
            assert self.svm.label_to_int[val] == idx

    @pytest.mark.run(after='test__FindSimilarity')
    def test__ConstructMSTGraph(self):
        (self.svm.mst_graph, self.svm.mst_list) = self.svm._construct_mst_graph(self.training_classes, self.svm.separability)
        assert len(self.svm.mst_list) == self.class_cnt - 1
        assert len(self.svm.mst_graph.connected_with(0)) == self.class_cnt

        cnt = 0
        for i, row in enumerate(self.svm.mst_graph.connection):
            for j, dist in enumerate(row):
                if dist != float('inf'):
                    cnt += 1

        # the graph bidirectional
        assert cnt == (self.class_cnt - 1) * 2

    @pytest.mark.run(after='test__ConstructMSTGraph')
    def test__ConstructTree(self):
        self.svm.tree = self.svm._construct_tree(self.svm.mst_graph, self.svm.mst_list)

        def runner(current):
            if current.left is None and current.right is None:
                return

            assert len(current.val) == len(current.left.val) + len(current.right.val)

            assert set(current.val) == set(current.left.val + current.right.val)

            runner(current.left)
            runner(current.right)

        runner(self.svm.tree.root)

    @pytest.mark.run(after='test_ConstructTree')
    def test_Train(self):
        self.svm.train(self.training_classes)

        def runner(current):
            if current.left is None and current.right is None:
                return

            assert current.svm
            runner(current.left)
            runner(current.right)

        runner(self.svm.tree.root)

    @pytest.mark.run(after='test_Train')
    def test_Predict(self):
        errors = 0
        total = 0
        for class_name, class_samples in self.training_classes.items():
            for sample in class_samples:
                total += 1
                if self.svm.predict(sample) != class_name:
                    # wrong prediction
                    errors += 1
        # just to see the idea
        print('errors:', errors, ' total:', total)
        assert errors == 0

    @pytest.mark.run(after='test_Predict')
    def test_CrossValidate(self):
        # 10 folds validation
        res = self.svm.cross_validate(10, self.training_classes)
        # this just to get the idea
        assert res == 0

    def test_make_rbf_kernel(self):
        gamma = 0.1
        rbf_kernel = self.svm.make_rbf_kernel(gamma=gamma)

        def original_kernel(a, b):
            import numpy

            return numpy.exp(-gamma * numpy.linalg.norm(a - b) ** 2)

        cnt = 0
        for class_name, samples in self.training_classes.items():
            a = samples
            b = a[:].tolist()
            random.shuffle(b)
            b = numpy.array(b)

            for i in range(a.shape[0]):
                cnt += 1
                assert rbf_kernel(a[i], b[i]) == original_kernel(a[i], b[i])

        assert cnt == 0

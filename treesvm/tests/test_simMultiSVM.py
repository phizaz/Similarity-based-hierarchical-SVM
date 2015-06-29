from unittest import TestCase
import numpy
import random
from treesvm.simmultisvm import SimMultiSVM
from treesvm.dataset import Dataset
import pytest

__author__ = 'phizaz'


class TestSimMultiSVM(TestCase):
    training_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/satimage/sat-train-s.csv'
    training_set = Dataset.load(training_file)
    training_classes = Dataset.split(training_set)
    class_cnt = len(training_classes.keys())
    gamma = 0.1
    svm = SimMultiSVM(gamma=gamma)

    def test__find_similarity(self):
        # svm = SimBinarySVM(Kernel)
        (self.svm.similarity, self.svm.label_to_int, self.svm.int_to_label) = self.svm._find_similarity(
            self.training_classes)
        # print('similarity', similarity)
        assert self.svm.similarity.size == self.class_cnt * self.class_cnt
        assert self.svm.similarity[0].size == self.class_cnt

        # print('labelToINt:', labelToInt)
        assert len(self.svm.label_to_int.keys()) == 6

        # print('int_to_label', int_to_label)
        for idx, val in enumerate(self.svm.int_to_label):
            assert self.svm.label_to_int[val] == idx

    @pytest.mark.run(after='test__find_similarity')
    def test_Train(self):
        self.svm.train(self.training_classes)

        def runner(current):
            if current.children == None:
                return

            assert len(current.svms) == len(current.children)
            for child in current.children:
                runner(child)

        runner(self.svm.tree.root)

    @pytest.mark.run(after='test_train')
    def test_predict(self):
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

    @pytest.mark.run(after='test_predict')
    def test_cross_validate(self):
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


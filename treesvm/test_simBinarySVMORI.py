from unittest import TestCase
import numpy
import pytest
import random
from treesvm.dataset import Dataset
from treesvm.simbinarysvm_ori import SimBinarySVMORI

__author__ = 'phizaz'


class TestSimBinarySVMORI(TestCase):
    training_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/satimage/sat-train-s.csv'
    training_set = Dataset.load(training_file)
    training_classes = Dataset.split(training_set)
    class_cnt = len(training_classes.keys())
    gamma = 1e-6
    C = 0.01
    svm = SimBinarySVMORI(gamma=gamma, C=C)

    def test_create_mapping(self):
        self.label_to_int, self.int_to_label = self.svm._create_mapping(self.training_classes)

    @pytest.mark.run(after='test_create_mapping')
    def test_create_tree(self):
        self.label_to_int, self.int_to_label = self.svm._create_mapping(self.training_classes)
        self.group_mgr = self.svm._create_tree(self.training_classes, self.label_to_int)

        def runner(current):
            if current.children == None:
                return

            child_universe = []
            for child in current.children:
                child_universe += list(child.universe.keys())
            assert set(current.universe.keys()) == set(child_universe)

            for child in current.children:
                runner(child)

        runner(next(iter(self.group_mgr.groups.values())))

    @pytest.mark.run(after='test_construct_tree')
    def test_train(self):
        group_mgr = self.svm.train(self.training_classes)

        def runner(current):
            if current.children == None:
                return

            assert current.svm
            for child in current.children:
                runner(child)

        runner(next(iter(group_mgr.groups.values())))

    @pytest.mark.run(after='test_train')
    def test_predict(self):
        group_mgr = self.svm.train(self.training_classes)
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
        group_mgr = self.svm.train(self.training_classes)
        # 10 folds validation
        res = self.svm.cross_validate(10, self.training_classes)
        # this just to get the idea
        assert res == 0

    def test_make_gram_matrix(self):
        gamma = 0.1
        vectors = []
        training_classes_with_idx = {}
        idx = 0
        for name, points in self.training_classes.items():
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
        kernel = self.svm.make_gram_matrix(vectors, gamma)

        def original_kernel(a, b):
            import numpy

            return numpy.exp(-gamma * numpy.linalg.norm(a - b) ** 2)

        for class_name, samples in training_classes_with_idx.items():
            a = samples
            b = a[:].tolist()
            random.shuffle(b)
            b = numpy.array(b)

            for i in range(a.shape[0]):
                assert abs(kernel(a[i], b[i]) - original_kernel(a[i][1:], b[i][1:])) < 1e-5

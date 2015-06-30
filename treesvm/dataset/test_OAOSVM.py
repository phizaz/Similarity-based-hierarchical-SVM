from unittest import TestCase

from treesvm.OAOsvm import OAOSVM
from treesvm.dataset import Dataset

__author__ = 'phizaz'


class TestOAOSVM(TestCase):
    training_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/satimage/sat-train-s.csv'
    training_set = Dataset.load(training_file)
    training_classes = Dataset.split(training_set)
    class_cnt = len(training_classes.keys())
    gamma = 0.1
    svm = OAOSVM(gamma=gamma)

    def test_train(self):
        self.svm.train(self.training_classes)

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

    def test_cross_validate(self):
        # 10 folds validation
        res = self.svm.cross_validate(10, self.training_classes)
        # this just to get the idea
        assert res == 0

from unittest import TestCase
from treesvm.dataset import Dataset
from treesvm.group.group import Group

__author__ = 'phizaz'


class TestGroup(TestCase):
    training_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/satimage/sat-train-s.csv'
    training_set = Dataset.load(training_file)
    training_classes = Dataset.split(training_set)
    class_cnt = len(training_classes.keys())

    def test___init__(self):
        pass

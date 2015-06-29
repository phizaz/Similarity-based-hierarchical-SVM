from unittest import TestCase
from treesvm.dataset import Dataset

__author__ = 'phizaz'


class TestDataset(TestCase):
    file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/iris.csv'
    dataset = Dataset.load(file)
    splitted = Dataset.split(dataset)

    def test_Load(self):
        assert len(self.dataset.features[0]) == 4

    def test_Split(self):
        assert len(self.splitted.keys()) == 3

        sum_splitted = 0
        for name, members in self.splitted.items():
            sum_splitted += len(members)
            for each in members:
                assert len(each) == 4
        assert sum_splitted == len(self.dataset.features)

    # def test_SquaredRadius(self):
    #     self.fail()
    #
    # def test_SquaredDistance(self):
    #     self.fail()

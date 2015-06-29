from unittest import TestCase
from treesvm import SimBinarySVM
from treesvm.dataset import Dataset
import pytest

# this might be wrong, because of the very high cross-validation error
# def rbf(a, b):
#     # it turned out to be that a, b are equal !
#     gamma = 0.1
#     # if input is single vector, expand it to 2 dimensions
#     if len(a.shape) == 1:
#         a = np.array([a])
#     if len(b.shape) == 1:
#         b = np.array([b])
#
#     print('a:', a.shape)
#     print('b:', b.shape)
#
#
#     # assert np.array_equal(a, b)
#
#     b = np.repeat(np.expand_dims(b, axis=1), a.shape[0], axis=1)
#     diff = a - b
#     c = np.exp( gamma * np.linalg.norm(diff, axis=2) ).T
#
#     print('c:', c.shape)
#     return c

class TestSimBinarySVM(TestCase):
    trainingFile = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/iris.csv'
    trainingSet = Dataset.load(trainingFile)
    trainingClasses = Dataset.split(trainingSet)
    classCnt = len(trainingClasses.keys())
    gamma = 0.1
    svm = SimBinarySVM(gamma=gamma)

    # def test_MakeRBFKernel(self):
    #     self.fail()

    def test__FindSimilarity(self):
        # svm = SimBinarySVM(Kernel)
        (self.svm.similarity, self.svm.labelToInt, self.svm.intToLabel) = self.svm._find_similarity(self.trainingClasses)
        # print('similarity', similarity)
        assert self.svm.similarity.size == self.classCnt * self.classCnt
        assert self.svm.similarity[0].size == self.classCnt

        # print('labelToINt:', labelToInt)
        assert len(self.svm.labelToInt.keys()) == 3
        assert 'Iris-setosa' in self.svm.labelToInt.keys()
        assert 'Iris-virginica' in self.svm.labelToInt.keys()
        assert 'Iris-versicolor' in self.svm.labelToInt.keys()

        # print('intToLabel', intToLabel)
        for idx, val in enumerate(self.svm.intToLabel):
            assert self.svm.labelToInt[val] == idx

    @pytest.mark.run(after='test__FindSimilarity')
    def test__ConstructMSTGraph(self):
        (self.svm.mst_graph, self.svm.mst_list) = self.svm._construct_mst_graph(self.trainingClasses, self.svm.similarity)
        assert len(self.svm.mst_list) == self.classCnt - 1
        assert len(self.svm.mst_graph.connected_with(0)) == self.classCnt

        cnt = 0
        for i, row in enumerate(self.svm.mst_graph.connection):
            for j, dist in enumerate(row):
                if dist != float('inf'):
                    cnt += 1

        # the graph bidirectional
        assert cnt == (self.classCnt - 1) * 2

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
        self.svm.train(self.trainingClasses)

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
        for class_name, class_samples in self.trainingClasses.items():
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
        res = self.svm.cross_validate(10, self.trainingClasses)
        # this just to get the idea
        assert res == 0

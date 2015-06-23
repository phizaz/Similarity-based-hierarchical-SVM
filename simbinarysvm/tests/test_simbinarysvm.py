from simbinarysvm import SimBinarySVM
import simbinarysvm.dataset as dataset
import math
import numpy as np
from sklearn import svm as sklearnSVM
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

trainingFile = 'iris.csv'
trainingSet = dataset.Load(trainingFile)
trainingClasses = dataset.Split( trainingSet )
classCnt = len( trainingClasses.keys() )
svm = None
gamma = 0.01

@pytest.mark.run('first')
def test_init_simbinarysvm():
    global svm
    svm = SimBinarySVM(gamma=gamma)

@pytest.mark.run(after='test_init_simbinarysvm')
def test_finding_similarity():
    # svm = SimBinarySVM(Kernel)
    (svm.similarity, svm.labelToInt, svm.intToLabel) = svm._FindSimilarity(trainingClasses)
    # print('similarity', similarity)
    assert svm.similarity.size == classCnt * classCnt
    assert svm.similarity[0].size == classCnt

    # print('labelToINt:', labelToInt)
    assert len( svm.labelToInt.keys() ) == 3
    assert 'Iris-setosa' in svm.labelToInt.keys()
    assert 'Iris-virginica' in svm.labelToInt.keys()
    assert 'Iris-versicolor' in svm.labelToInt.keys()

    # print('intToLabel', intToLabel)
    for idx, val in enumerate(svm.intToLabel):
        assert svm.labelToInt[ val ] == idx

@pytest.mark.run(after='test_finding_similarity')
def test_constructing_mst_graph():
    (svm.mst_graph, svm.mst_list) = svm._ConstructMSTGraph(trainingClasses, svm.similarity)
    assert len( svm.mst_list ) == classCnt - 1
    assert len( svm.mst_graph.ConnectedWith(0) ) == classCnt

    cnt = 0
    for i, row in enumerate(svm.mst_graph.connection):
        for j, dist in enumerate(row):
            if dist != float('inf'):
                cnt += 1

    #  the graph bidirectional
    assert cnt == (classCnt - 1) * 2

@pytest.mark.run(after='test_constructing_mst_graph')
def test_constructing_tree():
    svm.tree = svm._ConstructTree(svm.mst_graph, svm.mst_list)

    def runner(current):
        if current.left == None and current.right == None:
            return

        assert len(current.value) == len(current.left.value) + len(current.right.value)

        assert set(current.value) == set(current.left.value + current.right.value)

        runner(current.left)
        runner(current.right)

    runner(svm.tree.root)
@pytest.mark.run(after = 'test_constructing_tree')
def test_train_simbinarysvm():
    svm.Train(trainingClasses)
    def runner(current):
        if current.left == None and current.right == None:
            return

        assert current.svm
        runner(current.left)
        runner(current.right)

    runner(svm.tree.root)

@pytest.mark.run(after = 'test_train_simbinarysvm')
def test_predict_simbinarysvm():
    errors = 0
    total = 0
    for class_name, class_samples in trainingClasses.items():
        for sample in class_samples:
            total += 1
            if svm.Predict(sample) != class_name:
                # wrong prediction
                errors += 1
    # just to see the idea
    print('errors:', errors, ' total:', total)
    assert errors == 0
#
@pytest.mark.run(after = 'test_predict_simbinarysvm')
def test_cross_validation_simbinarysvm():
    # 10 folds validation
    res = svm.CrossValidate(10, trainingClasses)
    # this just to get the idea
    assert res == 0

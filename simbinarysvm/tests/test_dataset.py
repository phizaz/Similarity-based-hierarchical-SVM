from simbinarysvm.dataset import *

# this is the sharde code section
file = 'iris.csv'
dataset = Load(file)
splitted = Split(dataset)

# default RBF kernel
def KernelRBF(a, b, gamma):
    prefix = - 1. / 2 * gamma ** 2
    diff = a - b
    return math.exp( prefix * np.linalg.norm( diff ) ** 2 )

# using kernel function
# use this one instead of the above
def Kernel(a, b):
    gamma = 0.5
    return KernelRBF(a, b, gamma)
# end of kernel declarations

def test_datatest_init():
    features = [(1,2,3), (2,3,4)]
    labels = [0,1]
    dataset = Dataset(features, labels)
    assert dataset.features == features
    assert dataset.labels == labels

def test_load_features_count():
    assert len( dataset.features[0] ) == 4

def test_split_keys_count():
    assert len( splitted.keys() ) == 3

def test_split_members_count():
    sumSplitted = 0
    for name, members in splitted.items():
        sumSplitted += len(members)
        for each in members:
            assert len(each) == 4
    assert sumSplitted == len(dataset.features)

def test_squared_radius():
    # this is extremely hard to test
    assert 0

def test_squared_distance():
    assert 0

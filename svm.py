import numpy as np
import math
import sys
from simbinarysvm import SimBinarySVM, dataset


# load dataset
trainingFile = 'iris.csv'
trainingSet = dataset.Load(trainingFile)
trainingClasses = dataset.Split( trainingSet )

def MakeRBFKernel(gamma):
    def rbf(a, b):
        return np.exp( gamma * np.linalg.norm(a - b) ** 2)
    return rbf

gamma = 0.001
C = 1.0
if len(sys.argv) >= 2:
    gamma = float(sys.argv[1])
if len(sys.argv) >= 3:
    C = float(sys.argv[2])

print('gamma : ', gamma)

svm = SimBinarySVM(gamma=gamma, C=C)
cross_validation = svm.CrossValidate(10, trainingClasses)
print(cross_validation)

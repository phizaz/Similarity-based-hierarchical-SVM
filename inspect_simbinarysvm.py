import time
from treesvm.dataset import Dataset
from treesvm import SimBinarySVM

__author__ = 'phizaz'

def timer(func):
    start_time = time.process_time()
    func()
    return time.process_time() - start_time


training_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/satimage/sat-train.csv'
training_set = Dataset.load(training_file)
training_classes = Dataset.split(training_set)

testing_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/satimage/sat-test.csv'
testing_set = Dataset.load(testing_file)
testing_classes = Dataset.split(testing_set)

svm = SimBinarySVM(gamma=0.0001, C=10, verbose=True)
def train():
    svm.train(training_classes)
print('training: %.4f' % (timer(train)))

result = None
def test():
    global result
    result = svm.test(testing_classes)
print('testing: %.4f' % (timer(test)))
print(result)

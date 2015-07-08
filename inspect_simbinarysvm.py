import time
from treesvm.dataset import Dataset
from treesvm import SimBinarySVM

__author__ = 'phizaz'

def timer(func):
    start_time = time.process_time()
    func()
    return time.process_time() - start_time

# ('letter', 'datasets/letter/letter-train.txt', 'datasets/letter/letter-test.txt', lambda row: (row[1:], row[0]))
training_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/treesvm/datasets/letter/letter-train.txt'
# training_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/simbinarysvm/generated/generated.csv'
training_set = Dataset.load(training_file, adapter=lambda row: (row[1:], row[0]))
training_classes = Dataset.split(training_set)

testing_file = '/Users/phizaz/Dropbox/waseda-internship/svm-implementations/treesvm/datasets/letter/letter-test.txt'
testing_set = Dataset.load(testing_file, adapter=lambda row: (row[1:], row[0]))
testing_classes = Dataset.split(testing_set)

svm = SimBinarySVM(gamma=0.001, C=10, verbose=True)
def train():
    svm.train(training_classes)
print('training: %.4f' % (timer(train)))

result = None
def test():
    global result
    result = svm.test(testing_classes)
print('testing: %.4f' % (timer(test)))
print(result)

import sys
import multiprocessing
import concurrent.futures
import time
import numpy
from treesvm import SimBinarySVM
from treesvm.dataset import Dataset

start_time = time.time()
num_workers = multiprocessing.cpu_count()
print('workers: ', num_workers)

# load dataset
training_file = 'iris.csv'
training_set = Dataset.load(training_file)
training_classes = Dataset.split(training_set)

gammas = numpy.logspace(-9, 3, 13)
Cs = numpy.logspace(-2, 10, 13)


def instance(SVM, gamma, C):
    svm = SVM(gamma=gamma, C=C)
    cross_validation, total, errors = svm.cross_validate(10, training_classes)
    return cross_validation, total, errors


results = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    jobs = {}
    for gamma in gammas:
        for C in Cs:
            jobs[executor.submit(instance, SimBinarySVM, gamma, C)] = gamma, C
    # gamma, C
    for job in concurrent.futures.as_completed(jobs):
        gamma, C = jobs[job]
        cross_validation, total, errors = job.result()
        print('finished! gamma: ', gamma, ' C:', C, ' cross_validation: ', cross_validation)
        # store the result
        results[gamma] = {}
        results[gamma][C] = cross_validation, total, errors

print('time elapsed: ', time.time() - start_time)
print('results:')
print(results)
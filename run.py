import multiprocessing
import concurrent.futures
import time
import numpy
import json
from treesvm import SimBinarySVM
from treesvm.dataset import Dataset
from treesvm.simmultisvm import SimMultiSVM

print('creating svm and testing with cross_validation')

num_workers = multiprocessing.cpu_count()
print('workers: ', num_workers)

training_files = [
    ('iris', 'iris.csv'),
]

for training in training_files:
    project_name = training[0]
    print('working on project: ', project_name)

    # load dataset
    training_file = training[1]
    training_set = Dataset.load(training_file)
    training_classes = Dataset.split(training_set)

    for i, SVM in enumerate((SimBinarySVM, SimMultiSVM)):
        if i == 0:
            svm_type = 'SimBinarySVM'
            print('with: ', svm_type)
        else:
            svm_type = 'SimMultiSVM'
            print('with: ', svm_type)

        start_time = time.time()

        # normally it's 13 steps each
        gammas = numpy.logspace(-9, 3, 2)
        Cs = numpy.logspace(-2, 10, 2)

        def instance(SVM, gamma, C):
            svm = SVM(gamma=gamma, C=C)
            cross_validation_error, total, errors = svm.cross_validate(10, training_classes)
            accuracy = 1 - cross_validation_error
            return accuracy, total, errors

        results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            jobs = {}
            for gamma in gammas:
                for C in Cs:
                    jobs[executor.submit(instance, SVM, gamma, C)] = gamma, C
            # gamma, C
            for job in concurrent.futures.as_completed(jobs):
                gamma, C = jobs[job]
                cross_validation, total, errors = job.result()
                print('finished! gamma: ', gamma, ' C:', C, ' accuracy (cross_validated): ', cross_validation)
                # store the result
                results[gamma] = {}
                results[gamma][C] = cross_validation, total, errors

        print('time elapsed: ', time.time() - start_time)
        print('results:')
        print(results)

        # save results into a file
        json.dump(results, open(project_name + '-' + svm_type + '.txt', 'w'))
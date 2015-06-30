import multiprocessing
import concurrent.futures
import time
import numpy
import json
import random
from treesvm import SimBinarySVM
from treesvm.oaosvm import OAOSVM
from treesvm.dataset import Dataset
from treesvm.simmultisvm import SimMultiSVM

print('creating svm and testing with supplied test data')

num_workers = multiprocessing.cpu_count()
print('workers: ', num_workers)

training_files = [
    ('satimage', 'satimage/sat-train-s.csv', 'satimage/sat-test.csv'),
]

for training in training_files:
    project_name = training[0]
    print('working on project: ', project_name)

    # load dataset
    training_file = training[1]
    training_set = Dataset.load(training_file)
    training_classes = Dataset.split(training_set)

    testing_file = training[2]
    testing_set = Dataset.load(testing_file)
    testing_classes = Dataset.split(testing_set)

    best = {}

    for each in (
            ('OAO', OAOSVM),
            ('SimBinarySVM', SimBinarySVM),
            ('SimMultiSVM', SimMultiSVM),
    ):

        svm_type = each[0]
        SVM = each[1]
        print('with: ', svm_type)

        best[svm_type] = {
            'gamma': None,
            'C': None,
            'accuracy': 0
        }

        start_time = time.time()

        # normally it's 9 steps each
        gammas = numpy.logspace(-6, 2, 9)
        print('gammas: ', gammas)
        # it's 9 steps
        Cs = numpy.logspace(-2, 6, 9)
        print('Cs: ', Cs)

        def instance(SVM, gamma, C):
            svm = SVM(gamma=gamma, C=C)
            svm.train(training_classes)
            result = svm.test(testing_classes)
            total = result[0]
            errors = result[1]
            accuracy = (total - errors) / total
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
                accuracy, total, errors = job.result()
                print('finished! gamma: ', gamma, ' C:', C, ' accuracy: ', accuracy)
                # store the result
                results[gamma] = {}
                results[gamma][C] = accuracy, total, errors

                if accuracy > best[svm_type]['accuracy']:
                    tmp = best[svm_type]
                    tmp['accuracy'] = accuracy
                    tmp['C'] = C
                    tmp['gamma'] = gamma

        print('time elapsed: ', time.time() - start_time)
        print('results:')

        # save results into a file
        json.dump(results, open(project_name + '-' + svm_type + '.txt', 'w'))

    for svm_type, each in best.items():
        print('best of ', svm_type, ' with ', project_name)
        print('accuracy: ', each['accuracy'])
        print('C": ', each['C'])
        print('gamma: ', each['gamma'])

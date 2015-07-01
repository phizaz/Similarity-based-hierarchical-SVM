import multiprocessing
import concurrent.futures
import time
import gc
import numpy
import json
import random
from treesvm import SimBinarySVM
from treesvm.oaosvm import OAOSVM
from treesvm.dataset import Dataset
from treesvm.simmultisvm import SimMultiSVM

print('creating svm and testing with supplied test data')

num_workers = multiprocessing.cpu_count()
num_workers = 2
print('workers: ', num_workers)

training_files = [
    ('satimage', 'satimage/sat-train.csv', 'satimage/sat-test.csv'),
]

for training in training_files:
    project_name = training[0]
    print('working on project: ', project_name)

    # load dataset
    training_file = training[1]
    print('train: ', training_file)
    training_set = Dataset.load(training_file)
    training_classes = Dataset.split(training_set)

    testing_file = training[2]
    print('test:  ', testing_file)
    testing_set = Dataset.load(testing_file)
    testing_classes = Dataset.split(testing_set)

    best = {}
    time_used = {}

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

        time_used[svm_type] = 0

        start_time = time.process_time()

        # normally it's 9 steps each
        gammas = numpy.logspace(-6, 2, 9)
        print('gammas: ', gammas)
        # it's 9 steps
        Cs = numpy.logspace(-2, 6, 9)
        print('Cs: ', Cs)

        instance_cnt = gammas.size * Cs.size

        def instance(SVM, gamma, C):
            gc.collect()
            start_time = time.process_time()
            print('started gamma: ', gamma, ' C: ', C)
            svm = SVM(gamma=gamma, C=C)

            # start_time = time.process_time()
            svm.train(training_classes)
            # print('gamma: ', gamma, ' C: ', C, ' training time: %f' % (time.process_time() - start_time))

            # start_time = time.process_time()
            result = svm.test(testing_classes)
            # print('gamma: ', gamma, ' C: ', C, ' testing time: %f' % (time.process_time() - start_time))

            total = result[0]
            errors = result[1]
            accuracy = (total - errors) / total
            time_elapsed = time.process_time() - start_time
            return accuracy, total, errors, time_elapsed

        results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            jobs = {}
            for gamma in gammas:
                for C in Cs:
                    jobs[executor.submit(instance, SVM, gamma, C)] = gamma, C
            # gamma, C
            for job in concurrent.futures.as_completed(jobs):
                gamma, C = jobs[job]
                accuracy, total, errors, time_elapsed = job.result()
                print('finished! gamma: ', gamma, ' C:', C, ' accuracy: ', accuracy)
                # store the result
                results[gamma] = {}
                results[gamma][C] = accuracy, total, errors
                time_used[svm_type] += time_elapsed / instance_cnt

                if accuracy > best[svm_type]['accuracy']:
                    tmp = best[svm_type]
                    tmp['accuracy'] = accuracy
                    tmp['C'] = C
                    tmp['gamma'] = gamma

        print('time elapsed: ', time.process_time() - start_time)
        print('results:')

        # save results into a file
        json.dump(results, open(project_name + '-' + svm_type + '.txt', 'w'))

    for svm_type, each in best.items():
        print('best of ', svm_type, ' with ', project_name)
        print('accuracy: ', each['accuracy'])
        print('C": ', each['C'])
        print('gamma: ', each['gamma'])
        print('time avg: ', time_used[svm_type])

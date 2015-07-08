import multiprocessing
import concurrent.futures
import time
import gc
import numpy
import json
import random
from treesvm import SimBinarySVM
from treesvm.oaasvm import OAASVM
from treesvm.oaosvm import OAOSVM
from treesvm.dataset import Dataset
from treesvm.simbinarysvm_ori import SimBinarySVMORI
from treesvm.simmultisvm import SimMultiSVM

print('creating svm and testing with supplied test data')

num_workers = multiprocessing.cpu_count()
num_workers = 2
print('workers: ', num_workers)

training_files = [
    # ('satimage', 'datasets/satimage/sat-train.csv', 'datasets/satimage/sat-test.csv', lambda row: (row[:-1], row[-1])),
    # ('pendigits', 'datasets/pendigits/pendigits.tra', 'datasets/pendigits/pendigits.tes', lambda row: (row[:-1], row[-1])),
    ('letter', 'datasets/letter/letter-train.txt', 'datasets/letter/letter-test.txt', lambda row: (row[1:], row[0])),
]

for training in training_files:
    project_name = training[0]
    print('working on project: ', project_name)

    # load dataset
    given_adapter = None
    if len(training) > 3:
        given_adapter = training[3]

    training_file = training[1]
    print('train: ', training_file)
    training_set = Dataset.load(training_file, adapter=given_adapter)
    training_classes = Dataset.split(training_set)

    testing_file = training[2]
    print('test:  ', testing_file)
    testing_set = Dataset.load(testing_file, adapter=given_adapter)
    testing_classes = Dataset.split(testing_set)

    best = {}
    avg = {}

    for each in (
            ('OAO', OAOSVM),
            ('OAA', OAASVM),
            ('SimMultiSVM', SimMultiSVM),
            # ('SimBinarySVM_ORI', SimBinarySVMORI),
            ('SimBinarySVM', SimBinarySVM),
    ):

        svm_type = each[0]
        SVM = each[1]
        print('with: ', svm_type)

        best[svm_type] = {
            'gamma': None,
            'C': None,
            'accuracy': 0
        }

        avg[svm_type] = {
            'training_time': 0,
            'testing_time': 0,
            'svm_cnt': 0,
        }

        start_time = time.process_time()

        # normally it's 9 steps each
        # gammas = numpy.logspace(-6, 2, 9)
        gammas = numpy.logspace(-6, 0, 7)
        # gammas = numpy.array([0.1])
        print('gammas: ', gammas)
        # it's 9 steps
        # Cs = numpy.logspace(-2, 6, 9)
        Cs = numpy.logspace(-2, 4, 7)
        # Cs = numpy.array([10.0])
        print('Cs: ', Cs)

        instance_cnt = gammas.size * Cs.size

        def instance(SVM, gamma, C):
            # force calling garbage collection (solves memory leaks)
            gc.collect()
            print('started gamma: ', gamma, ' C: ', C)

            start_time = time.process_time()
            svm = SVM(gamma=gamma, C=C)
            svm_cnt = svm.train(training_classes)
            training_time = time.process_time() - start_time

            # start_time = time.process_time()
            start_time = time.process_time()
            result = svm.test(testing_classes)
            testing_time = time.process_time() - start_time

            testing_cnt = 0
            for name, points in testing_classes.items():
                testing_cnt += points.shape[0]

            total = result[0]
            errors = result[1]
            total_itr = result[2]
            avg_itr = total_itr / testing_cnt
            accuracy = (total - errors) / total

            print('finished! gamma: ', gamma, ' C:', C, ' accuracy: ', accuracy, ' avg_itr: ', avg_itr,
                  ' svm_cnt: ', svm_cnt,
                  ' training_time: %.4f' % (training_time), ' testing_time: %.4f' % (testing_time))
            return accuracy, total, errors, (training_time, testing_time), avg_itr, svm_cnt

        results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            jobs = {}
            for gamma in gammas:
                for C in Cs:
                    jobs[executor.submit(instance, SVM, gamma, C)] = gamma, C
            # gamma, C
            for job in concurrent.futures.as_completed(jobs):
                gamma, C = jobs[job]
                accuracy, total, errors, time_elapsed, avg_itr, svm_cnt \
                    = job.result()
                # store the result
                if gamma not in results:
                    results[gamma] = {}
                results[gamma][C] = accuracy, total, errors, time_elapsed, avg_itr, svm_cnt
                avg[svm_type]['training_time'] += time_elapsed[0] / instance_cnt
                avg[svm_type]['testing_time'] += time_elapsed[1] / instance_cnt
                avg[svm_type]['svm_cnt'] += svm_cnt / instance_cnt

                if accuracy > best[svm_type]['accuracy']:
                    tmp = best[svm_type]
                    tmp['accuracy'] = accuracy
                    tmp['C'] = C
                    tmp['gamma'] = gamma
                    tmp['training_time'] = time_elapsed[0]
                    tmp['testing_time'] = time_elapsed[1]
                    tmp['avg_itr'] = avg_itr
                    tmp['svm_cnt'] = svm_cnt

        # show report after each svm type
        print('time elapsed: ', time.process_time() - start_time)
        print('results:')
        print('best of ', svm_type, ' with ', project_name)
        print('accuracy: ', best[svm_type]['accuracy'])
        print('best C: ', best[svm_type]['C'])
        print('best gamma: ', best[svm_type]['gamma'])
        print('best training_time: ', best[svm_type]['training_time'])
        print('best testing_time: ', best[svm_type]['testing_time'])
        print('best avg_itr: ', best[svm_type]['avg_itr'])
        print('best svm_cnt: ', best[svm_type]['svm_cnt'])

        # save results into a file
        json.dump(results, open('results/' + project_name + '-' + svm_type + '.txt', 'w'))

    # sum up all the reports again
    for svm_type, each in best.items():
        print('best of ', svm_type, ' with ', project_name)
        print('accuracy: ', each['accuracy'])
        print('C": ', each['C'])
        print('gamma: ', each['gamma'])
        print('best training_time: ', best[svm_type]['training_time'])
        print('best testing_time: ', best[svm_type]['testing_time'])
        print('avg_itr: ', each['avg_itr'])
        print('svm_cnt: ', each['svm_cnt'])
        print('training_time avg: ', avg[svm_type]['training_time'])
        print('testing_time avg: ', avg[svm_type]['testing_time'])
        print('svm_cnt avg: ', avg[svm_type]['svm_cnt'])
    # save all the reports back to a file
    json.dump(best, open('results/' + project_name + '-best.txt', 'w'))
    json.dump(avg, open('results/' + project_name + '-avg.txt', 'w'))

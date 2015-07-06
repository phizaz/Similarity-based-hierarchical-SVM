import numpy
import sklearn


class OAASVM:
    def __init__(self, gamma=0.1, C=1.0, verbose=False):
        self.int_to_label = None
        self.label_to_int = None
        self.svms = None
        self.gamma = gamma
        self.C = C
        self.verbose = verbose

    def train(self, training_classes):
        self.svms = {}
        svm_cnt = 0
        for focus_name in training_classes:
            samples = None
            labels = None
            for name, points in training_classes.items():
                # just add all the points to this list
                if samples == None:
                    samples = points
                else:
                    samples = numpy.append(samples, points, axis=0)

                if focus_name is name:
                    # mark it as 0
                    if labels == None:
                        labels = numpy.array([0 for i in points])
                    else:
                        labels = numpy.append(labels, [0 for i in points])
                else:
                    # mark it as 1
                    if labels == None:
                        labels = numpy.array([1 for i in points])
                    else:
                        labels = numpy.append(labels, [1 for i in points])
            self.svms[focus_name] = sklearn.svm.SVC(kernel='rbf', gamma=self.gamma, C=self.C).fit(samples, labels)
            svm_cnt += 1
        return svm_cnt

    def predict(self, sample):
        iterations = len(self.svms.keys())
        confidence = {name: svm.decision_function(sample) for name, svm in self.svms.items()}
        min_name = min(confidence, key=confidence.get)
        return min_name, iterations

    def test(self, testing_classes):
        total = 0
        errors = 0
        total_itr = 0

        for class_name, tests in testing_classes.items():
            for test in tests:
                total += 1
                prediction, iterations = self.predict(test)
                total_itr += iterations
                if prediction != class_name:
                    errors += 1

        return total, errors, total_itr

    def cross_validate(self, folds, training_classes):
        total = 0
        for key, val in training_classes.items():
            total += val.size

        random_list = [ i % folds for i in range(total) ]
        # should  we shuffle it ?
        # random.shuffle( ramdom_list )
        acc_total = 0
        acc_errors = 0
        for i in range(folds):
            training = {}
            testing = {}
            # select a portion to be left
            no = 0
            training_cnt = 0
            testing_cnt = 0
            for class_name, class_samples in training_classes.items():
                training[class_name] = []
                testing[class_name] = []
                for sample in class_samples:
                    if no % folds != i:
                        # this is in
                        training[class_name].append(sample)
                        training_cnt += 1
                    else:
                        # keep this for testing
                        testing[class_name].append(sample)
                        testing_cnt += 1
                    no += 1
                training[class_name] = numpy.array(training[class_name])
                testing[class_name] = numpy.array(testing[class_name])

                # print('training: ', 'name: ', class_name, 'size: ', training[class_name].size)
                # print('testing: ', 'name: ', class_name, 'size: ', testing[class_name].size)

            # train the rest
            self.train(training)

            # test with the leftover
            test_result = self.test(testing)
            acc_total += test_result[0]
            acc_errors += test_result[1]

        # average the error
        cross_validation_error = acc_errors / acc_total
        return cross_validation_error, acc_total, acc_errors
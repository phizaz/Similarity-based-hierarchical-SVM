import numpy
import sklearn


class OAOSVM:
    def __init__(self, gamma=0.1, C=1.0):
        self.int_to_label = None
        self.label_to_int = None
        self.svm = None
        self.gamma = gamma
        self.C = C

    def train(self, training_classes):
        training = []
        label = []

        self.label_to_int = {}
        self.int_to_label = {}
        i = 0
        for class_name, samples in training_classes.items():
            training += samples.tolist()
            self.label_to_int[class_name] = i
            self.int_to_label[i] = class_name
            label += [i for j in range(samples.shape[0])]
            i += 1
        training = numpy.array(training)
        label = numpy.array(label)
        self.svm = sklearn.svm.SVC(kernel='rbf', gamma=self.gamma, C=self.C).fit(training, label)

    def predict(self, sample):
        prediction = self.svm.predict(sample)
        return self.int_to_label[prediction[0]]

    def test(self, testing_classes):
        total = 0
        errors = 0

        for class_name, tests in testing_classes.items():
            for test in tests:
                total += 1
                prediction = self.predict(test)
                if prediction != class_name:
                    errors += 1

        return total, errors

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
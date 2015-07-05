__author__ = 'phizaz'


class Tools:

    def __init__(self, kernel):
        # full combination cache
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.kernel = kernel

    def full_combination(self, label, points):
        if label in self.cache:
            # self.hits += 1
            # print('hits: ', self.hits)
            return self.cache[label]

        # self.misses += 1
        # print('misses: ', self.misses)
        sum = 0
        for i, xx in enumerate(points):
            sum += self.kernel(xx, xx)
            for j, yy in enumerate(points[i+1:]):
                sum += 2 * self.kernel(xx, yy)
        self.cache[label] = sum
        return sum

    # find the squared radius of a given class
    def squared_radius(self, label, points):
        # last term of the equation
        # this function doesn't accept any parameters
        def last_term():
            return 1. / points.shape[0] ** 2 * self.full_combination(label, points)

        # we cache the result for future uses
        c = last_term()

        def in_sqrt(a_point):
            # first term of the equation goes here
            a = self.kernel(a_point, a_point)

            # second term of the equation
            def middle_term(point):
                summation = 0
                for p in points:
                    summation += self.kernel(point, p)
                return -2. / points.shape[0] * summation

            b = middle_term(a_point)
            # put it all together
            # c from above
            return a + b + c

        # calculate the radius (max of the sqrt thing)
        squared_radius = [in_sqrt(point) for point in points]
        # for each in squared_radius:
        #     assert each >= 0
        # return the maximum
        return max(squared_radius)

    # squared distance maker
    def squared_distance(self, label_a, points_a, label_b, points_b):
        # square distance calculations (without sqrt)
        # first term
        def first_term():
            return 1. / points_a.shape[0] ** 2 * self.full_combination(label_a, points_a)

        # second term
        def second_term():
            summation = 0
            # this loop cannot be optimized by the factor of two
            for xx in points_a:
                for yy in points_b:
                    summation += self.kernel(xx, yy)
            return -2. / (points_a.shape[0] * points_b.shape[0]) * summation

        # third term
        def third_term():
            return 1. / points_b.shape[0] ** 2 * self.full_combination(label_b, points_b)

        distance = first_term() + second_term() + third_term()
        # assert distance > 0
        return distance
# graph's node
# deprecated
class Node:

    def __init__(self, name, dist):
        self.name = name
        self.dist = dist

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, val):
        self._dist = val

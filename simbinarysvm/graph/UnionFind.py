class UnionFind:
    _parents = None

    def __init__(self, size):
        self._parents = [ i for i in range(0, size) ]

    def Find(self, member):
        # it is the root
        if self._parents[member] == member:
            return member
        else:
            self._parents[member] = self.Find(self._parents[member])
            return self._parents[member]

    def Union(self, a, b):
        parentA = self.Find(a)
        parentB = self.Find(b)
        if parentA == parentB:
            return False

        self._parents[parentA] = self._parents[parentB]
        return True

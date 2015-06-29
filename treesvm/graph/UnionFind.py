class UnionFind:
    _parents = None

    def __init__(self, size):
        self._parents = [ i for i in range(0, size) ]

    def find(self, member):
        # it is the root
        if self._parents[member] == member:
            return member
        else:
            self._parents[member] = self.find(self._parents[member])
            return self._parents[member]

    def union(self, a, b):
        parent_a = self.find(a)
        parent_b = self.find(b)
        if parent_a == parent_b:
            return False

        self._parents[parent_a] = self._parents[parent_b]
        return True

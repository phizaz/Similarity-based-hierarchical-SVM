# tree's node
class Node:
    def __init__(
        self,
        value = None,
        parent = None,
        left = None,
        right = None):
        self.value = value
        self.parent = parent
        self.left = left
        self.right = right

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val):
        self._parent = val

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, val):
        self._left = val

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, val):
        self._right = val

    # @property
    # def children(self):
    #     return (self._left, self._right)
    #
    # @children.setter
    # def children(self, children):
    #     if children[0] != 0:
    #         self._left = children[0]
    #     if children[1] != 0:
    #         self._right = children[1]

from .Node import Node

class BinaryTree:

    def __init__(self):
        self.root = None

    def AddRoot(self, node):
        if self.root != None:
            return

        self.root = node
        node.parent = None
        node.children = (None, None)
        return node

    # adding should work correctly only when the newnode has no children
    def AddLeft(self, parent, newnode):
        old = parent.left
        # add link to newnode
        # 0 is don't care
        parent.left = newnode
        newnode.parent = parent

        # short circuit, because the parent has no left child
        if old == None:
            return newnode

        newnode.left = old
        old.parent = newnode

        return newnode

    def AddRight(self, parent, newnode):
        old = parent.right
        # add link to newnode
        # 0 is don't care
        parent.right = newnode
        newnode.parent = parent

        # short circuit, because the parent has no left child
        if old == None:
            return newnode

        newnode.right = old
        old.parent = newnode

        return newnode

    def Left(self, node):
        return node.left

    def Right(self, node):
        return node.right

    def First(self):
        current = self.root
        while current.left != None:
            current = current.left
        return current

    def Display(self):
        def travel(current):
            if current == None:
                return
            left = current.left
            right = current.right
            travel(left)
            print(current.value)
            travel(right)
        travel(self.root)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

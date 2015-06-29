from .binarytree_node import BinaryTreeNode


class BinaryTree:
    def __init__(self):
        self.root = None

    def add_root(self, node):
        if self.root != None:
            return

        self.root = node
        node.parent = None
        node.children = (None, None)
        return node

    # adding should work correctly only when the newnode has no children
    def add_left(self, parent, newnode):
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

    def add_right(self, parent, newnode):
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

    def left(self, node):
        return node.left

    def right(self, node):
        return node.right

    def first(self):
        current = self.root
        while current.left != None:
            current = current.left
        return current

    def inorder(self):
        order = []

        def travel(current):
            if current == None:
                return
            travel(current.left)
            order.append(current.val)
            travel(current.right)

        travel(self.root)
        return order

    def leaves(self):
        leaves = []

        def travel(current):
            if current.left == None and current.right == None:
                leaves.append(current.val)
                return

            travel(current.left)
            travel(current.right)

        travel(self.root)
        return leaves

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

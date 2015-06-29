from treesvm.multitree.multitree_node import MultiTreeNode

__author__ = 'phizaz'

class MultiTree:

    def __init__(self):
        self.root = None

    def add_root(self, node):
        assert isinstance(node, MultiTreeNode)
        self.root = node
        return node

    def add_child(self, parent, node):
        assert isinstance(parent, MultiTreeNode)
        assert isinstance(node, MultiTreeNode)

        if parent.children is None:
            parent.children = []

        parent.children.append(node)
        node.parent = parent
        return node

    def preorder(self):
        output_list = []
        def runner(current):
            output_list.append(current.val)
            if current.children == None:
                return
            for child in current.children:
                runner(child)

        runner(self.root)
        return output_list
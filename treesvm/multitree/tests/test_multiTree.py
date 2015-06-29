import pytest
from unittest import TestCase
from treesvm.multitree import MultiTree, MultiTreeNode

__author__ = 'phizaz'

class TestMultiTree(TestCase):
    tree = MultiTree()
    i = 0

    @pytest.mark.first
    def test_add_root(self):
        self.tree.add_root(MultiTreeNode(val=10))
        assert self.tree.root.val == 10

    def test_add_child(self):
        self.tree.add_root(MultiTreeNode(val=10))
        root = self.tree.root
        adding_node = self.tree.add_child(root, MultiTreeNode(val=1))
        assert root.children[0].val == 1
        assert adding_node.parent.val == 10

    def test_preorder(self):
        self.tree.add_root(MultiTreeNode(val=10))
        first_node = self.tree.add_child(self.tree.root, MultiTreeNode(val=2))
        second_node = self.tree.add_child(first_node, MultiTreeNode(val=3))
        third_node = self.tree.add_child(self.tree.root, MultiTreeNode(val=4))
        fourt_node = self.tree.add_child(first_node, MultiTreeNode(val=5))

        result = self.tree.preorder()
        correct = [10, 2, 3, 5, 4]
        assert correct == result

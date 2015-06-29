from unittest import TestCase
from treesvm.binarytree.binarytree_node import BinaryTreeNode

__author__ = 'phizaz'


class TestNode(TestCase):
    def test_binary_node(self):
        node = BinaryTreeNode(1)
        assert node._right == node.right == None
        assert node._left == node.left == None
        node.left = 1
        assert node.left == 1
        node.right = 2
        assert node.right == 2

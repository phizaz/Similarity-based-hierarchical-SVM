from unittest import TestCase
from treesvm.binarytree import BinaryTree
from treesvm.binarytree.binarytree_node import BinaryTreeNode

__author__ = 'phizaz'


class TestBinaryTree(TestCase):
    def test_AddRoot(self):
        tree = BinaryTree()
        root = tree.add_root(BinaryTreeNode(1))
        assert root == tree.root
        assert root.parent == None
        assert root.children == (None, None)

    def test_AddLeft(self):
        tree = BinaryTree()
        root = tree.add_root( BinaryTreeNode(1) )

        # complex adding case, because normal ones are covered with test_add()
        left = tree.add_left(root, BinaryTreeNode(2))
        assert root.right == None
        assert root.left.val == 2
        right = tree.add_right(root, BinaryTreeNode(3))

        tmp = tree.add_left(root, BinaryTreeNode(4))
        assert tmp.parent == root
        assert root.left == tmp
        assert tmp.left == left
        assert tmp.right == None
        assert root.right == right

    def test_AddRight(self):
        tree = BinaryTree()
        root = tree.add_root( BinaryTreeNode(1) )

        #complex adding case, becasue nromal ones are covered with test_add()
        leftNode = BinaryTreeNode(2)
        rightNode = BinaryTreeNode(3)
        right = tree.add_right(root, rightNode)
        assert root.right == rightNode
        assert root.left == None
        left = tree.add_left(root, leftNode)

        tmp = tree.add_right(root, BinaryTreeNode(4))
        assert tmp.parent == root
        assert root.right == tmp
        assert tmp.left == None
        assert tmp.right == right
        assert root.left == left

    def test_Left(self):
        tree = BinaryTree()
        root = tree.add_root(BinaryTreeNode(5))
        self.create_binary_search_tree(tree, [3, 1, 2, 7, 6])
        assert tree.left(root) != None
        assert tree.left(root).val == 3

    def test_Right(self):
        tree = BinaryTree()
        root = tree.add_root(BinaryTreeNode(5))
        self.create_binary_search_tree(tree, [3, 1, 2, 7, 6])
        # tree.Display()
        assert tree.right(root) != None
        assert tree.right(root).val == 7

    def test_First(self):
        tree = BinaryTree()
        root = tree.add_root(BinaryTreeNode(1))
        for i in range(2,10):
            node = BinaryTreeNode(i)
            current = root
            while current.left != None:
                current = current.left
            current = tree.add_left(current, node)

        assert tree.first().val == 9

    def create_binary_search_tree(self, tree, insertion_list):
        current = root = tree.root
        for each in insertion_list:
            num = each
            current = root
            # adding the num to the tree
            while True:
                left = current.left
                right = current.right
                if num < current.val:
                    # goes left
                    if left == None:
                        new = tree.add_left(current, BinaryTreeNode(num) )
                        # check normal addintg case
                        assert new == current.left
                        assert new.parent == current
                        break
                    else:
                        current = left

                elif num >= current.val:
                    # goes right
                    if right == None:
                        new = tree.add_right(current, BinaryTreeNode(num))
                        # check normal adding case
                        assert new == current.right
                        assert new.parent == current
                        break
                    else:
                        current = right

        # test binary search tree itself
        def travel(current):
            left = current.left
            right = current.right
            if left != None:
                assert left.val < current.val
                travel(left)
            if right != None:
                assert current.val <= right.val
                travel(right)
        travel(root)

    def test_inorder(self):
        tree = BinaryTree()
        root = tree.add_root(BinaryTreeNode(5))
        self.create_binary_search_tree(tree, [3, 1, 2, 7, 6])
        assert tree.inorder() == [1, 2, 3, 5, 6, 7]

    def test_leaves(self):
        tree = BinaryTree()
        root = tree.add_root(BinaryTreeNode(5))
        self.create_binary_search_tree(tree, [3,1,4,7,6,8])
        assert tree.leaves() == [1, 4, 6, 8]


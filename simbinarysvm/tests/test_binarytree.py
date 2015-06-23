import random

from simbinarysvm.binarytree import BinaryTree
from simbinarysvm.binarytree.Node import Node

def test_binarytree_node():
    node = Node(1)
    assert node._right == node.right == None
    assert node._left == node.left == None
    node.left = 1
    assert node.left == 1
    node.right = 2
    assert node.right == 2

def test_first():
    tree = BinaryTree()
    root = tree.AddRoot(Node(1))
    for i in range(2,10):
        node = Node(i)
        current = root
        while current.left != None:
            current = current.left
        current = tree.AddLeft(current, node)

    assert tree.First().value == 9

def test_add_left():
    tree = BinaryTree()
    root = tree.AddRoot( Node(1) )

    # complex adding case, because normal ones are covered with test_add()
    left = tree.AddLeft(root, Node(2))
    assert root.right == None
    assert root.left.value == 2
    right = tree.AddRight(root, Node(3))

    tmp = tree.AddLeft(root, Node(4))
    assert tmp.parent == root
    assert root.left == tmp
    assert tmp.left == left
    assert tmp.right == None
    assert root.right == right

def test_add_right():
    tree = BinaryTree()
    root = tree.AddRoot( Node(1) )

    #complex adding case, becasue nromal ones are covered with test_add()
    leftNode = Node(2)
    rightNode = Node(3)
    right = tree.AddRight(root, rightNode)
    assert root.right == rightNode
    assert root.left == None
    left = tree.AddLeft(root, leftNode)

    tmp = tree.AddRight(root, Node(4))
    assert tmp.parent == root
    assert root.right == tmp
    assert tmp.left == None
    assert tmp.right == right
    assert root.left == left

def test_root():
    tree = BinaryTree()
    root = tree.AddRoot(Node(1))
    assert root == tree.root
    assert root.parent == None
    assert root.children == (None, None)

# create a binary search tree with a given list
def random_binary_search_tree(tree, insertion_list):
    current = root = tree.root
    for each in insertion_list:
        num = each
        current = root
        # adding the num to the tree
        while True:
            left = current.left
            right = current.right
            if num < current.value:
                # goes left
                if left == None:
                    new = tree.AddLeft(current, Node(num) )
                    # check normal addintg case
                    assert new == current.left
                    assert new.parent == current
                    break
                else:
                    current = left

            elif num >= current.value:
                # goes right
                if right == None:
                    new = tree.AddRight(current, Node(num))
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
            assert left.value < current.value
            travel(left)
        if right != None:
            assert current.value <= right.value
            travel(right)
    travel(root)

def test_left():
    tree = BinaryTree()
    root = tree.AddRoot(Node(5))
    random_binary_search_tree(tree, [3, 1, 2, 7, 6])
    assert tree.Left(root) != None
    assert tree.Left(root).value == 3

def test_right():
    tree = BinaryTree()
    root = tree.AddRoot(Node(5))
    random_binary_search_tree(tree, [3, 1, 2, 7, 6])
    # tree.Display()
    assert tree.Right(root) != None
    assert tree.Right(root).value == 7

# test normal adding both left and right
def test_add():
    tree = BinaryTree()
    root = tree.AddRoot( Node( random.randint(0,100) ) )
    # tests are in the following line`
    random_binary_search_tree(tree, [5, 3, 2, 7, 6, 8, 1])

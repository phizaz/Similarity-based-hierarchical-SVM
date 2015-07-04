import numpy
from treesvm.binarytree import BinaryTree, BinaryTreeNode

__author__ = 'phizaz'

class Group():

    def __init__(self, universe, tools):
        assert isinstance(universe, dict)

        self.tools = tools
        # calling name of this group, may be used as a label
        self.name = tuple(universe.keys())
        # caveat: universe is just a link (not actual value)
        self.universe = universe
        self.children = None
        self.parent = None
        self.svm = None
        # gather the list of points in this scope
        self.points = None
        for name, points in self.universe.items():
            if self.points == None:
                self.points = points
            else:
                self.points = numpy.append(self.points, points, axis=0)
        # calculate its radius
        self.sq_radius = self.tools.squared_radius(self.name, self.points)

    def add_child(self, group):
        assert isinstance(group, Group)

        group.parent = self
        if self.children == None:
            self.children = []
        self.children.append(group)


from treesvm.dataset.tools import Tools
from treesvm.group.group import Group

__author__ = 'phizaz'


class GroupManager:
    def __init__(self, kernel):
        self.kernel = kernel
        self.groups = {}
        self.similarity = {}
        self.tools = Tools(kernel)

    def create_group(self, universe):
        return Group(universe, self.tools)

    def _similarity(self, group_a, group_b):
        sq_ra = group_a.sq_radius
        sq_rb = group_b.sq_radius
        sq_dist = self.tools.squared_distance(
            group_a.name, group_a.points,
            group_b.name, group_b.points,
        )
        return (sq_ra + sq_rb) / sq_dist

    def add(self, group):
        assert isinstance(group, Group)

        # update similarity
        self.similarity[group.name] = {}
        # self similarity is inf
        self.similarity[group.name][group.name] = float('inf')
        for name, g in self.groups.items():
            self.similarity[name][group.name] = \
                self.similarity[group.name][name] = \
                self._similarity(group, g)
        # add to groups
        self.groups[group.name] = group
        return group

    def delete(self, group):
        # delete from groups
        del self.groups[group.name]
        # delete from similarity
        del self.similarity[group.name]
        for name, to_class in self.similarity.items():
            del to_class[group.name]

    def merge(self, group_a, group_b):
        for each in (group_a, group_b):
            assert isinstance(each, Group)

        # calculate universe
        universe = {}
        for group in (group_a, group_b):
            for name, points in group.universe.items():
                universe[name] = points
        # create new node
        merged = self.create_group(universe)
        merged.add_child(group_a)
        merged.add_child(group_b)
        self.delete(group_a)
        self.delete(group_b)
        # add merged group into the manager
        self.add(merged)
        return merged

    def most_similar(self):
        most = 0
        result = None
        for a, to_groups in self.similarity.items():
            for b, similarity in to_groups.items():
                if a != b and similarity > most:
                    most = similarity
                    result = (self.groups[a], self.groups[b])
        return result
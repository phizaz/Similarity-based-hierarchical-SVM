from unittest import TestCase
from treesvm.graph.Graph import Graph

__author__ = 'phizaz'


class TestGraph(TestCase):
    def test_Link(self):
        for implement in ('matrix', 'hash'):
            g = Graph(10, implement=implement)

            assert g.link(0, 1, 2)[0][1] == 2

            assert g.link(0, 1, 10)[0][1] == 10

            assert g.link(1, 2, 20)[1][2] == 20

            connection = g.link(2, 1, 10)
            assert connection[2][1] == 10
            assert connection[1][2] == 20

            assert g.link(0, 1, 1)[0][1] == 1


    def test_DoubleLink(self):
        for implement in ('matrix', 'hash'):
            g = Graph(10, implement=implement)

            connection = g.double_link(0, 1, 2)
            assert connection[0][1] == 2
            assert connection[1][0] == 2


    def test_MST(self):
        for implement in ('matrix', 'hash'):
            g = Graph(10, implement=implement)

            def Link(link):
                g.link(link[0], link[1], link[2])
                g.link(link[1], link[0], link[2])

            links = [
                (0, 1, 2),
                (1, 2, 3),
                (0, 2, 1),
                (3, 4, 2),
                (4, 1, 1),
            ]

            for i, link in enumerate(links, 1):
                Link(link)

            MST = g.mst()

            def Assert(link):
                assert (link in MST) or ((link[1], link[0], link[2]) in MST)

            assert len(MST) == 4
            # note that the MST might have many results, and those should not be clarified as wrong ones
            Assert(links[0])
            Assert(links[2])
            Assert(links[4])
            Assert(links[3])

    def test_Unlink(self):
        for implement in ('matrix', 'hash'):
            g = Graph(10, implement=implement)

            g.link(0, 1, 10)
            g.link(1, 0, 1)

            tmp = g.unlink(0, 1)
            if implement == 'matrix':
                assert tmp[0][1] == float('inf')
            else:
                assert not 1 in tmp[0]

            assert tmp[1][0] == 1

    def test_ConnectedWith(self):
        for implement in ('matrix', 'hash'):
            g = Graph(10, implement=implement)

            def Link(link):
                g.link(link[1], link[0], link[2])
                g.link(link[0], link[1], link[2])

            links = [
                (0, 1, 2),
                (1, 2, 3),
                (0, 2, 1),
                (3, 4, 2),
                (4, 1, 1),
            ]

            for i, link in enumerate(links, 1):
                Link(link)

            assert set(g.connected_with(0)) == set([0, 1, 2, 3, 4])

            g.unlink(4,1)
            g.unlink(1,4)
            assert set(g.connected_with(4)) == set([3,4])
            assert set(g.connected_with(1)) == set([0,1,2])


    def test_sum_weight(self):
        for implement in ('matrix', 'hash'):
            g = Graph(10, implement=implement)


            def Link(link):
                g.link(link[1], link[0], link[2])
                g.link(link[0], link[1], link[2])

            links = [
                (0, 1, 2),
                (1, 2, 3),
                (0, 2, 1),
                (3, 4, 2),
                (4, 1, 1),
            ]

            for i, link in enumerate(links, 1):
                Link(link)

            res = g.sum_weight(0)
            assert res[0] == 9 and len(res[1]) == 5

            g.unlink(1,4)
            g.unlink(4,1)
            res = g.sum_weight(0)
            assert  res[0] == 6 and len(res[1]) == 3

            res = g.sum_weight(4)
            assert res[0] == 2 and len(res[1]) == 1

    def test_double_unlink(self):
        for implement in ('matrix', 'hash'):
            g = Graph(10, implement=implement)


            def Link(link):
                g.link(link[1], link[0], link[2])
                g.link(link[0], link[1], link[2])

            links = [
                (0, 1, 2),
                (1, 2, 3),
                (0, 2, 1),
                (3, 4, 2),
                (4, 1, 1),
            ]

            for i, link in enumerate(links, 1):
                Link(link)

            g.double_unlink(1,4)
            res = g.sum_weight(0)
            assert res[0] == 6 and len(res[1]) == 3

            res = g.sum_weight(4)
            assert res[0] == 2 and len(res[1]) == 1


from .UnionFind import UnionFind


class Graph:
    def __init__(self, node_cnt, implement='matrix'):
        self.connection = None
        self.is_matrix = True
        self.node_cnt = node_cnt

        if implement == 'matrix':
            # construct a 2d nodeCnt * nodeCnt dimensional matrix
            # and set all of its members to be inf
            self.connection = [[float('inf') for i in range(0, node_cnt)] for i in range(0, node_cnt)]
        elif implement == 'hash':
            self.is_matrix = False
            # construct using python's dictionary
            self.connection = [{} for i in range(0, node_cnt)]
        else:
            print('graph type is not compatible')

    # it's a unidirectional link
    def link(self, a, b, dist):
        self.connection[a][b] = dist
        return self.connection

    def double_link(self, a, b, dist):
        self.link(a, b, dist)
        self.link(b, a, dist)
        return self.connection

    # return the links of subgraph that represents the MST of the main graph
    def mst(self):
        # sort the link according it its distance
        links = []
        if self.is_matrix:
            #  create a list of links represented by connection
            for i in range(self.node_cnt):
                for j in range(self.node_cnt):
                    val = self.connection[i][j]
                    if val != float('inf'):
                        links.append((i, j, val))
        else:
            # this is for things implemented in array of hash tables
            for i in range(self.node_cnt):
                for j, val in self.connection[i].items():
                    links.append((i, j, val))

        sorted_links = sorted(links, key=lambda x: x[2])

        # put in the union find datastruct
        mst_links = []
        uf = UnionFind(len(self.connection))
        for link in sorted_links:
            if uf.union(link[0], link[1]):
                # union success
                # they don't share the same root
                mst_links.append(link)

        return mst_links

    def unlink(self, a, b):
        if self.is_matrix:
            self.connection[a][b] = float('inf')
        else:
            del self.connection[a][b]

        return self.connection

    def double_unlink(self, a, b):
        self.unlink(a, b)
        self.unlink(b, a)

    def connected_with(self, node):
        node_list = []
        visited = [False for i in range(self.node_cnt)]

        def runner(current):
            node_list.append(current)
            nexts = self.connection[current]
            iterable = enumerate(nexts) if self.is_matrix else nexts.items()
            for next, dist in iterable:
                if dist != float('inf') and not visited[next]:
                    visited[next] = True
                    runner(next)

        # set the node to be visited
        visited[node] = True
        runner(node)

        return node_list

    # return the sum of the edges that connected with a given node
    def sum_weight(self, node):
        visited = [{} for i in range(self.node_cnt)]
        edges = []

        def runner(current):
            nexts = self.connection[current]
            iterable = enumerate(nexts) if self.is_matrix else nexts.items()
            sum_weight = 0
            for next, dist in iterable:
                (a, b) = (min(next, current), max(next, current))
                if not b in visited[a] and dist != float('inf'):
                    visited[a][b] = True
                    edges.append((a, b, dist))
                    sum_weight += runner(next) + dist
            return sum_weight

        return runner(node), edges

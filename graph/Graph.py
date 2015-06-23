from .Node import Node
from .UnionFind import UnionFind

class Graph:

    def __init__(self, nodeCnt, implement='matrix'):
        self.connection = None
        self.isMatrix = True
        self.nodeCnt = nodeCnt

        if implement == 'matrix':
            # consturct a 2d nodeCnt * nodeCnt dimensional matrix
            # and set all of its members to be inf
            self.connection = [ [ float('inf') for i in range(0, nodeCnt) ] for i in range(0, nodeCnt) ]
        elif implement == 'hash':
            self.isMatrix = False
            # construct using python's dictionary
            self.connection = [ {} for i in range(0, nodeCnt) ]
        else:
            print('graph type is not compatible')

    # it's a unidirectional link
    def Link(self, a, b, dist):
        self.connection[a][b] = dist
        return self.connection

    def DoubleLink(self, a, b, dist):
        self.Link(a, b, dist)
        self.Link(b, a, dist)
        return self.connection

    # return the links of subgraph that represents the MST of the main graph
    def MST(self):
        # sort the link according it its distance
        links = []
        if self.isMatrix:
            #  create a list of links represented by connection
            for i in range(self.nodeCnt):
                for j in range(self.nodeCnt):
                    val = self.connection[i][j]
                    if val != float('inf'):
                        links.append((i, j, val))
        else:
            # this is for things implemented in array of hash tables
            for i in range(self.nodeCnt):
                for j, val in self.connection[i].items():
                    links.append((i,j, val))

        sortedLinks = sorted(links, key=lambda x: x[2])

        # put in the union find datastruct
        mstLinks = []
        uf = UnionFind( len(self.connection) )
        for link in sortedLinks:
            if uf.Union( link[0], link[1] ):
                # union success
                # they don't share the same root
                mstLinks.append(link)

        return mstLinks

    def Unlink(self, a, b):
        if self.isMatrix:
            self.connection[a][b] = float('inf')
        else:
            del self.connection[a][b]

        return self.connection

    def ConnectedWith(self, node):
        nodeList = []
        visited = [ False for i in range(self.nodeCnt) ]

        def runner(current):
            nodeList.append(current)
            nexts = self.connection[current]
            iterable = enumerate(nexts) if self.isMatrix else nexts.items()
            for next, dist in iterable:
                if dist != float('inf') and not visited[next]:
                    visited[next] = True
                    runner(next)

        # set the node to be visited
        visited[node] = True
        runner(node)

        return nodeList

    # getters and setters
    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, connection):
        self._connection = connection

    @property
    def isMatrix(self):
        return self._isMatrix

    @isMatrix.setter
    def isMatrix(self, isMatrix):
        self._isMatrix = isMatrix

    @property
    def nodeCnt(self):
        return self._nodeCnt

    @nodeCnt.setter
    def nodeCnt(self, val):
        self._nodeCnt = val

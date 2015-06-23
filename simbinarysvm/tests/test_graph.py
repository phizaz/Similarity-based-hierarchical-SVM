from simbinarysvm.graph import Graph

def test_init():
    for implement in ('matrix', 'hash'):
        g = Graph(10, implement=implement)
        assert len(g._connection) == 10
        assert g._isMatrix == (implement == 'matrix')

def test_link():

    for implement in ('matrix', 'hash'):
        g = Graph(10, implement=implement)

        assert g.Link(0,1,2)[0][1] == 2

        assert g.Link(0,1,10)[0][1] == 10

        assert g.Link(1,2,20)[1][2] == 20

        connection =  g.Link(2,1,10)
        assert connection[2][1] == 10
        assert connection[1][2] == 20

        assert g.Link(0,1,1)[0][1] == 1

def test_MST():
    for implement in ('matrix', 'hash'):
        g = Graph(10, implement=implement)
        def Link(link):
            g.Link(link[0], link[1], link[2])
            g.Link(link[1], link[0], link[2])

        links = [
            (0,1,2),
            (1,2,3),
            (0,2,1),
            (3,4,2),
            (4,1,1),
        ]

        for i, link in enumerate(links, 1):
            Link(link)

        MST = g.MST()
        def Assert(link):
            assert (link in MST) or ( (link[1], link[0], link[2]) in MST)
        assert len(MST) == 4
        # note that the MST might have many results, and those should not be clarified as wrong ones
        Assert(links[0])
        Assert(links[2])
        Assert(links[4])
        Assert(links[3])

def test_unlink():
    for implement in ('matrix', 'hash'):
        g = Graph(10, implement=implement)

        g.Link(0,1,10)
        g.Link(1,0,1)

        tmp = g.Unlink(0,1)
        if implement == 'matrix':
            assert tmp[0][1] == float('inf')
        else:
            assert not 1 in tmp[0]

        assert tmp[1][0] == 1

def test_connected_with():
    for implement in ('matrix', 'hash'):
        g = Graph(10, implement=implement)
        def Link(link):
            g.Link(link[0], link[1], link[2])
            g.Link(link[1], link[0], link[2])

        links = [
            (0,1,2),
            (1,2,3),
            (0,2,1),
            (3,4,2),
            (4,1,1),
        ]

        for i, link in enumerate(links, 1):
            Link(link)

        assert set(g.ConnectedWith(4)) == set([0,1,2,3,4])
        assert set(g.ConnectedWith(0)) == set([0,1,2,3,4])

        g.Unlink(1,4)
        g.Unlink(4,1)

        assert set(g.ConnectedWith(4)) == set([3,4])
        assert set(g.ConnectedWith(1)) == set([0,1,2])

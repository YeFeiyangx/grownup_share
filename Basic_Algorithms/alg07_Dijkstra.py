# %%
"""Bellman-Ford算法"""
"获得与起始点S，所有点都为最小的其它点的路径和值"

inf = float("inf")
def relax(W,u,v,D,P):
    """
    W是字典图
    u 起始点
    v 是终点
    D 是初始只含起始点权为0的，权值字典，起始点外都inf，最终结果是起点至其它所有点的最小距离
    P 是记录所有最短路径的节点 @@ 拿例子中的点7来说，距离0最近的点7的路径遇到的第一个最近的点是 5
    点5最近的点是0
    "松弛的意思就是尝试一下能不能改进，如果不能改进就不改"
    """
    # d是权值，最早如果未登记在D中，就是无穷大。如果起点有登记在D中，就是起点的权值+起终点的权值
    d = D.get(u,inf) + W[u][v]
    if d < D.get(v,inf):
        ## 当条件成立 则 D[v] 和 P[v]都分别更新为有限权值的d,和路径P[v] = u
        D[v], P[v] = d, u
        return True


def bellman_ford(G,s):
    D, P = {s:0}, {}
    for rnd in G:
        changed = False
        for u in G:
            for v in G[u]:
                if relax(G, u, v, D, P):
                    changed = True
        print("D:",D)
        print("P:",P)
        if not changed: break
    else:
        raise ValueError("negative cycle")

    return D,P

a,b,c,d,e,f,g,h = range(8)
# N是一张图，表示了a-h可以到达的位置,并设置路径权重
N = {
    a:{b:2,c:1,d:3,e:9,f:4},    # a
    b:{c:4,e:3},                # b
    c:{d:8},                    # c
    d:{e:7},                    # d
    e:{f:5},                    # e
    f:{c:2,g:2,h:2},            # f
    g:{f:1,h:6},                # g
    h:{f:9,g:8}                 # h
    }

bellman_ford(N,a)

# %%
from heapq import heappush, heappop

inf = float("inf")
def relax(W,u,v,D,P):
    """
    W是字典图
    u 起始点
    v 是终点
    D 是初始只含起始点权为0的，权值字典，起始点外都inf，最终结果是起点至其它所有点的最小距离
    P 是记录所有最短路径的节点 @@ 拿例子中的点7来说，距离0最近的点7的路径遇到的第一个最近的点是 5
    点5最近的点是0
    "松弛的意思就是尝试一下能不能改进，如果不能改进就不改"
    """
    # d是权值，最早如果未登记在D中，就是无穷大。如果起点有登记在D中，就是起点的权值+起终点的权值
    d = D.get(u,inf) + W[u][v]
    if d < D.get(v,inf): # 如果遇起点未更新过的，就更新起点的信息
        ## 当条件成立 则 D[v] 和 P[v]都分别更新为有限权值的d,和路径P[v] = u
        D[v], P[v] = d, u
        return True


def dijkstra(G, s):

    D, P, Q, S = {s:0}, {}, [(0,s)], set()      # Est., tree, queue, visited
    
    while Q:                                    # Still unprocessed nodes?
        N = [0]
        ## heappop 取点，都是取到起始点最小的那个
        _, u = heappop(Q)     # 点都访, Q取光    # Node with lowest estimate
        if u in S: continue   # 点都访, Q取光    # Already visited? Skip it
        S.add(u)                                # We've visited it now
        print("***---"*5)
        print(S)
        for v in G[u]:                          # Go through all its neighbors
            ## 根据已有的起点，去松弛终点，松弛后的结果会修改D,P
            relax(G, u, v, D, P)                # Relax the out-edge
            ## 堆的目的在于，每次取起始点，都是从最小权的点开始，去做终点的松弛，做过的点就不再做了
            heappush(Q, (D[v], v))              # Add to queue, w/est. as pri

            print("D:",D)
            print("P:",P)
            print("Q:",Q)
            print("--end--{}--".format(N[0])*5)
            N[0] += 1
    return D, P                                 # Final D and P returned

## 起始点s，以及起始点s的权为0 这就是 (0,s)的含义
#while Q
#0      _ = 0, u = 0, S"visited" = set()
#1      if u = 0 not in S"visited" pass
#2      S"visited" add u这个站点
#for 开始对G[u=0]的迭代 0:{1:10, 3:5}
#3      relax 已有的 连同界 0-1,0-3
#4      获得 [(5, 3), (10, 1)]

#0      _ = 5, u = 3, S"visited" = set(0,)
#1      if u = 3 not in S"visited" pass
#2      S"visited" add u这个站点
#for 开始对G[u=3]的迭代 3:{1:3, 2:9, 4:2}
#3      relax 已有的 连通界 u=3-v=1, 3-v=2,3-v=4



s, t, x, y, z, k, l = range(7)
W = {
    s: {t:10, y:5},
    t: {x:1, y:2},
    x: {l:4},
    y: {t:3, x:9, k:6},
    z: {x:6, s:7},
    k: {z:1},
    l: {z:20}
}

D, P = dijkstra(W, s)
print("D:",D)
print("P:",P)


#%%
from copy import deepcopy

def johnson(G):                                 # All pairs shortest paths
    G = deepcopy(G)                             # Don't want to break original
    s = object()                                # Guaranteed unique node
    G[s] = {v:0 for v in G}                     # Edges from s have zero wgt
    h, _ = bellman_ford(G, s)                   # h[v]: Shortest dist from s
    del G[s]                                    # No more need for s
    for u in G:                                 # The weight from u...
        for v in G[u]:                          # ... to v...
            G[u][v] += h[u] - h[v]              # ... is adjusted (nonneg.)
    D, P = {}, {}                               # D[u][v] and P[u][v]
    for u in G:                                 # From every u...
        D[u], P[u] = dijkstra(G, u)             # ... find the shortest paths
        for v in G:                             # For each destination...
            D[u][v] += h[v] - h[u]              # ... readjust the distance
    return D, P                                 # These are two-dimensional

a, b, c, d, e = range(5)
W = {
    a: {c:1, d:7},
    b: {a:4},
    c: {b:-5, e:2},
    d: {c:6},
    e: {a:3, b:8, d:-4}
}

D, P = johnson(W)
print("D:",D)
print("P:",P)


#%%
from copy import deepcopy
"""floyd算法"""

def test_floyd_warshall1():
    """
    >>> a, b, c, d, e = range(1,6) # One-based
    >>> W = {
    ...     a: {c:1, d:7},
    ...     b: {a:4},
    ...     c: {b:-5, e:2},
    ...     d: {c:6},
    ...     e: {a:3, b:8, d:-4}
    ... }
    >>> for u in W:
    ...     for v in W:
    ...         if u == v: W[u][v] = 0
    ...         if v not in W[u]: W[u][v] = inf
    >>> D = floyd_warshall1(W)
    >>> [D[a][v] for v in [a, b, c, d, e]]
    [0, -4, 1, -1, 3]
    >>> [D[b][v] for v in [a, b, c, d, e]]
    [4, 0, 5, 3, 7]
    >>> [D[c][v] for v in [a, b, c, d, e]]
    [-1, -5, 0, -2, 2]
    >>> [D[d][v] for v in [a, b, c, d, e]]
    [5, 1, 6, 0, 8]
    >>> [D[e][v] for v in [a, b, c, d, e]]
    [1, -3, 2, -4, 0]
    """

def floyd_warshall1(G):
    D = deepcopy(G)                             # No intermediates yet
    for k in G:                                 # Look for shortcuts with k
        for u in G:
            for v in G:
                D[u][v] = min(D[u][v], D[u][k] + D[k][v])
    return D

#%%
import time
import copy

start = time.perf_counter()
inf = float("inf")
a, b, c, d, e = range(1,6) # One-based
W0 = {
    a: {c:1, d:7},
    b: {a:4},
    c: {b:-5, e:2},
    d: {c:6},
    e: {a:3, b:8, d:-4}
}

for i in range(16000):
    W = copy.deepcopy(W0)
    for u in W:
        for v in W:
            if u == v: W[u][v] = 0
            if v not in W[u]: W[u][v] = inf
    D = floyd_warshall1(W)

end = time.perf_counter()   #结束计时
print('Running time: %f seconds'%(end-start))  #程序运行时间 2s

#%%

def test_floyd_warshall():
    """
    >>> a, b, c, d, e = range(5)
    >>> W = {
    ...     a: {c:1, d:7},
    ...     b: {a:4},
    ...     c: {b:-5, e:2},
    ...     d: {c:6},
    ...     e: {a:3, b:8, d:-4}
    ... }
    >>> for u in W:
    ...     for v in W:
    ...         if u == v: W[u][v] = 0
    ...         if v not in W[u]: W[u][v] = inf
    >>> D, P = floyd_warshall(W)
    >>> [D[a][v] for v in [a, b, c, d, e]]
    [0, -4, 1, -1, 3]
    >>> [D[b][v] for v in [a, b, c, d, e]]
    [4, 0, 5, 3, 7]
    >>> [D[c][v] for v in [a, b, c, d, e]]
    [-1, -5, 0, -2, 2]
    >>> [D[d][v] for v in [a, b, c, d, e]]
    [5, 1, 6, 0, 8]
    >>> [D[e][v] for v in [a, b, c, d, e]]
    [1, -3, 2, -4, 0]
    >>> [P[a,v] for v in [a, b, c, d, e]]
    [None, 2, 0, 4, 2]
    >>> [P[b,v] for v in [a, b, c, d, e]]
    [1, None, 0, 4, 2]
    >>> [P[c,v] for v in [a, b, c, d, e]]
    [1, 2, None, 4, 2]
    >>> [P[d,v] for v in [a, b, c, d, e]]
    [1, 2, 3, None, 2]
    >>> [P[e,v] for v in [a, b, c, d, e]]
    [1, 2, 3, 4, None]
    """

def floyd_warshall(G):
    D, P = deepcopy(G), {}
    for u in G:
        for v in G:
            if u == v or G[u][v] == inf:
                P[u,v] = None
            else:
                P[u,v] = u
    for k in G:
        for u in G:
            for v in G:
                shortcut = D[u][k] + D[k][v]
                if shortcut < D[u][v]:
                    D[u][v] = shortcut
                    P[u,v] = P[k,v]
    return D, P

#%%
import time
import copy

start = time.perf_counter()

a, b, c, d, e = range(5)
W0 = {
    a: {c:1, d:7},
    b: {a:4},
    c: {b:-5, e:2},
    d: {c:6},
    e: {a:3, b:8, d:-4}
}

W = copy.deepcopy(W0)
for u in W:
    for v in W:
        if u == v: W[u][v] = 0
        if v not in W[u]: W[u][v] = inf
D, P = floyd_warshall(W)

end = time.perf_counter()   #结束计时
print('Running time: %f seconds'%(end-start))  #程序运行时间 2s

#%%
a = 'OPPAK14，OPPAK05'
b = set(a)
b

#%%

def test_idijkstra():
    """
    >>> s, t, x, y, z = range(5)
    >>> W = {
    ...     s: {t:10, y:5},
    ...     t: {x:1, y:2},
    ...     x: {z:4},
    ...     y: {t:3, x:9, z:2},
    ...     z: {x:6, s:7}
    ... }
    >>> D = dict(idijkstra(W, s))
    >>> [D[v] for v in [s, t, x, y, z]]
    [0, 8, 9, 5, 7]
    """

def idijkstra(G, s):
    Q, S = [(0,s)], set()                       # Queue w/dists, visited
    while Q:                                    # Still unprocessed nodes?
        d, u = heappop(Q)                       # Node with lowest estimate
        if u in S: continue                     # Already visited? Skip it
        S.add(u)                                # We've visited it now
        yield u, d                              # Yield a subsolution/node
        for v in G[u]:                          # Go through all its neighbors
            heappush(Q, (d+G[u][v], v))         # Add to queue, w/est. as pri


#%%
def test_bidir_dijkstra_et_al():
    """
    >>> W = {
    ...     'hnl': {'lax':2555},
    ...     'lax': {'sfo':337, 'ord':1743, 'dfw': 1233},
    ...     'sfo': {'ord':1843},
    ...     'dfw': {'ord':802, 'lga':1387, 'mia':1120},
    ...     'ord': {'pvd':849},
    ...     'lga': {'pvd':142},
    ...     'mia': {'lga':1099, 'pvd':1205}
    ... }
    >>> nodes = list(W)
    >>> for u in nodes:
    ...     for v in W[u]:
    ...         if not v in W: W[v] = {}
    ...         W[v][u] = W[u][v]
    ...
    >>> for u in W:
    ...     W[u][u] = 0
    ...
    >>> for u in W:
    ...     for v in W[u]:
    ...         assert W[u][v] == W[v][u]
    ...
    >>> for u in W:
    ...     Dd, _ = dijkstra(W, u)
    ...     Db, _ = bellman_ford(W, u)
    ...     for v in W:
    ...         d = bidir_dijkstra(W, u, v)
    ...         assert d == Dd[v], (d, Dd[v])
    ...         assert d == Db[v], (d, Db[v])
    ...         a = a_star_wrap(W, u, v, lambda v: 0)
    ...         assert a == d
    ...
    >>> G = {0:{0:0}, 1:{1:0}}
    >>> bidir_dijkstra(G, 0, 1)
    inf
    >>> bidir_dijkstra(G, 0, 0)
    0
    >>> G = {0:{1:7}, 1:{0:7}}
    >>> bidir_dijkstra(G, 0, 1)
    7
    >>> bidir_dijkstra(G, 0, 1)
    7
    >>> D, P = dijkstra(W, 'hnl')
    >>> P['pvd'], P['ord'], P['lax']
    ('ord', 'lax', 'hnl')
    >>> D['pvd'] == W['hnl']['lax'] + W['lax']['ord'] + W['ord']['pvd']
    True
    >>> D['pvd']
    5147
    >>> bidir_dijkstra(W, 'hnl', 'pvd')
    5147
    >>> bidir_dijkstra(W, 'pvd', 'sfo')
    2692
    """

from itertools import cycle

def bidir_dijkstra(G, s, t):
    Ds, Dt = {}, {}                             # D from s and t, respectively
    forw, back = idijkstra(G,s), idijkstra(G,t) # The "two Dijkstras"
    dirs = (Ds, Dt, forw), (Dt, Ds, back)       # Alternating situations
    try:                                        # Until one of forw/back ends
        for D, other, step in cycle(dirs):      # Switch between the two
            v, d = next(step)                   # Next node/distance for one
            D[v] = d                            # Demember the distance
            if v in other: break                # Also visite by the other?
    except StopIteration: return inf            # One ran out before they met
    m = inf                                     # They met; now find the path
    for u in Ds:                                # For every visited forw-node
        for v in G[u]:                          # ... go through its neighbors
            if not v in Dt: continue            # Is it also back-visited?
            m = min(m, Ds[u] + G[u][v] + Dt[v]) # Is this path better?
    return m                                    # Return the best path

def a_star(G, s, t, h):
    P, Q = {}, [(h(s), None, s)]                # Pred and queue w/heuristic
    while Q:                                    # Still unprocessed nodes?
        d, p, u = heappop(Q)                    # Node with lowest heuristic
        if u in P: continue                     # Already visited? Skip it
        P[u] = p                                # Set path predecessor
        if u == t: return d - h(t), P           # Arrived! Ret. dist and preds
        for v in G[u]:                          # Go through all neighbors
            w = G[u][v] - h(u) + h(v)           # Modify weight wrt heuristic
            heappush(Q, (d + w, u, v))          # Add to queue, w/heur as pri
    return inf, None                            # Didn't get to t

def a_star_wrap(G, s, t, h):
    return a_star(G, s, t, h)[0]

#%%
from string import ascii_lowercase as chars

class WordSpace:                                # An implicit graph w/utils

    def __init__(self, words):                  # Create graph over the words
        self.words = words
        self.M = M = dict()                     # Reachable words

    def variants(self, wd, words):              # Yield all word variants
        wasl = list(wd)                         # The word as a list
        for i, c in enumerate(wasl):            # Each position and character
            for oc in chars:                    # Every possible character
                if c == oc: continue            # Don't replace with the same
                wasl[i] = oc                    # Replace the character
                ow = ''.join(wasl)              # Make a string of the word
                if ow in words:                 # Is it a valid word?
                    yield ow                    # Then we yield it
            wasl[i] = c                         # Reset the character

    def __getitem__(self, wd):                  # The adjacency map interface
        if wd not in self.M:                    # Cache the neighbors
            self.M[wd] = dict.fromkeys(self.variants(wd, self.words), 1)
        return self.M[wd]

    def heuristic(self, u, v):                  # The default heuristic
        return sum(a!=b for a, b in zip(u, v))  # How many characters differ?

    def ladder(self, s, t, h=None):             # Utility wrapper for a_star
        if h is None:                           # Allows other heuristics
            def h(v):
                return self.heuristic(v, t)
        _, P = a_star(self, s, t, h)            # Get the predecessor map
        if P is None:
            return [s, None, t]                 # When no path exists
        u, p = t, []
        while u is not None:                    # Walk backward from t
            p.append(u)                         # Append every predecessor
            u = P[u]                            # Take another step
        p.reverse()                             # The path is backward
        return p




# %%

# %%
"""
遍历一个表示为邻接集的图结构的连通分量
"""
# S 代表一个禁区
# 字典P 代表已经访问过的前驱节点
# 找到单个连通分量
# G 是邻接图，s 是访问起始节点，S禁止访问点的集合




def walk(G, s, S=set()):
    P, Q = dict(), set()                    # 前置任务，将做队列
    P[s] = None                             # s为起点，前置任务s起点为None
    Q.add(s)                                # Q是将做的遍历的队列
    while Q:
        u = Q.pop()                         # 暴力遍历，取出一个就是
        for v in G[u].difference(P, S):      # 艾豆，这用法，仙人粑粑，把已经搞完的，和不允许搞的进行遍历
            Q.add(v)                        # Q 增加一个被端掉的点为v
            P[v] = u                        # 把被端掉的连通点位的信息记录下来
    return P


# %%
# @@difference 小方法
a = set([1, 2, 3])
b = set([8, 9])
c = set(range(10))
print(c.difference(a, b))
print(c)
d = {5: "s", 6: "e"}
print(c.difference(b, d))  # {0, 1, 2, 3, 4, 7} 可以可以这个仙人粑粑


# %%
a, b, c, d, e, f, g, h = range(8)

N = [
    {b, c, d, e, f},     # a
    {c, e},           # b
    {d},             # c
    {e},             # d
    {f},             # e
    {c, g, h},         # f
    {f, h},           # g
    {f, g}            # h
]

walk(N, s=1, S={2, 3})


# %%
"""
找出图的所有连通分量
"""


def components(G):
    comp = []
    seen = set()
    for u in G:
        if u in seen:
            continue
        C = walk(G, u)
        seen.update(C)
        comp.append(C)
    return comp

# %%
# a,b,c,d,e,f,g,h = range(8)


N = {
    "a": set("bcdef"),
    "b": set("ce"),
    "c": set("d"),
    "d": set("e"),
    "e": set("f"),
    "f": set("cgh"),
    "g": set("fh"),
    "h": set("fg")
}

components(N)


# %%
"""
递归树的遍历方法
"""


def tree_walk(T, r):
    # 遍历每一个子节点
    for u in T[r]:
        # 在递归中继续遍历每一个子节点
        tree_walk(T, u)


"""递归版的深度优先搜索"""


def rec_dfs(G, s, S=None):
    if S is None:
        S = set()
    S.add(s)                    # 将搜索图起点加入至待搜索集合
    for u in G[s]:              # 遍历对应节点下方子点
        if u in S:
            continue     # 如果子点已添加至S路径中，就跳过这个节点
        rec_dfs(G, u, S)          # 如果未添加，就再运行递归，把起始点，和路径记录状态进入下一次迭代


# %%
"""迭代版深度优先搜索"""


def iter_dfs(G, s):
    S, Q = set(), []            # S是站点集合，Q是待迭代的集合
    Q.append(s)                 # Q加入需要开始的点位置
    while Q:
        u = Q.pop()             # u是Q待迭代集合中的站点, @@用pop起到了深度优先的特点
        print("---"*10)
        print(Q)
        print(G[u])
        print(u)
        if u in S:
            continue      # 如果u在S站点中，那就别看了，进入下一个
        S.add(u)                # 如果u不在S站点中，那么S站点添加u
        Q.extend(G[u])          # 待迭代的集合把 G[u] 都放进去
        yield u                 # 把u yield出来


# %%
a = {1, 2, 3, 4}
a.pop()


# %%
a, b, c, d, e, f, g, h = range(8)

N = [
    {b, c, d, e, f},     # a
    {c, e},           # b
    {d},             # c
    {e},             # d
    {f},             # e
    {c, g, h},         # f
    {f, h},           # g
    {f, g}            # h
]


list(iter_dfs(N, 0))

# %%
for i in iter_dfs(N, 0):
    print(i)

# %%
"""
通用性的图遍历函数
"""
# 没太大明白为什么通用了，把上例的list，提出来作为一个方法qtype，就通用了？


def traverse(G, s, qtype=set):
    S, Q = set(), qtype()
    Q.add(s)
    while Q:
        u = Q.pop()
        if u in S:
            continue
        S.add(u)
        for v in G[u]:
            Q.add(v)
        yield u


# %%
a, b, c, d, e, f, g, h = range(8)

N = [
    {b, c, d, e, f},     # a
    {c, e},           # b
    {d},             # c
    {e},             # d
    {f},             # e
    {c, g, h},         # f
    {f, h},           # g
    {f, g}            # h
]


class stack(list):
    add = list.append


list(traverse(N, 0, stack))


# %%
"""带时间戳的深度优先搜索"""


def dfs(G, s, d=dict(), f=dict(), S=None, t=0):
    if S is None:
        S = set()
    d[s] = t
    t += 1
    S.add(s)
    for u in G[s]:
        if u in S:
            continue
        t, d, f = dfs(G, u, d, f, S, t)
    f[s] = t
    t += 1
    return t, d, f


# %%
a, b, c, d, e, f, g, h = range(8)

N = [
    {b, c, d, e, f},     # a
    {c, e},              # b
    {d},                 # c
    {e},                 # d
    {f},                 # e
    {c, g, h},           # f
    {f, h},              # g
    {f, g}               # h
]


a, b, c = dfs(N, 0)
print(a)
print(b)
print(c)


# %%
"""基于深度优先搜索的拓扑排序"""


def dfs_topsort(G):
    S, res = set(), []

    def recurse(u):     # 递归
        if u in S:      # 如果点已经在S中，就直接退出
            return
        S.add(u)        # 如果不在S中，加入S，
        for v in G[u]:  # 遍历G图中可到达的位置
            recurse(v)  # 每个v又要recurse，深度优先到底
        res.append(u)   # res从最深的开始append点位u，append到最外层
        # 如果S中有就不要再append了

    for u in G:         # 把所有点丢尽recurse玩一遍，recurse里面也不断递归
        recurse(u)

    res.reverse()       # 从最深开始的列表，一个逆序就好
    return res


# %%
# a -> b -> c -> d -> e -> f -> c(reject) -> g -> h 完毕
# @@ 到处运用动态数据类型的特性
a, b, c, d, e, f, g, h = range(8)

N = {
    a: {b, c, d, e, f},     # a
    b: {c, e},              # b
    c: {d},                 # c
    d: {e},                 # d
    e: {f},                 # e
    f: {c, g, h},           # f
    g: {f, h},              # g
    h: {f, g}               # h
}

dfs_topsort(N)

# %%
"""
迭代深度的深度优先搜索
"""


def iddfs(G, s):
    yielded = set()

    def recurse(G, s, d, S=None):
        if s not in yielded:
            yield s
            yielded.add(s)
        if d == 0:
            return
        if S is None:
            S = set()
        S.add(s)
        for u in G[s]:
            if u in S:
                continue
            for v in recurse(G, u, d-1, S):
                yield v
    n = len(G)
    for d in range(n):
        if len(yielded) == n:
            break
        for u in recurse(G, s, d):
            yield u


# %%


from collections import deque

# 广度优先搜索
# @@ deque 用双向链表从左侧取就好了，稳稳稳。。。。尴尬，好气啊。。。
def bfs(G, s):
    P, Q = {s: None}, deque([s])
    while Q:
        u = Q.popleft()
        for v in G[u]:
            if v in P:
                continue
            P[v] = u
            Q.append(v)
    return P


# %%

a, b, c, d, e, f, g, h = range(8)

N = {
    a: {b, c, d, e, f},     # a
    b: {c, e},              # b
    c: {d},                 # c
    d: {e},                 # d
    e: {f},                 # e
    f: {c, g, h},           # f
    g: {f, h},              # g
    h: {f, g}               # h
}

bfs(N, 0)

# %%
## Kosaraju 的查找强连同分量算法
def tr(G):
    GT = {}
    for u in G:
        GT[u] = set()
    for u in G:
        for v in G[u]:
            GT[v].add(u)

    return GT

def scc(G):
    GT = tr(G)
    sccs, seen = [], set()
    for u in dfs_topsort(G):
        if u in seen: continue
        C = walk(GT,u,seen)
        seen.update(C)
        sccs.append(C)
    return sccs
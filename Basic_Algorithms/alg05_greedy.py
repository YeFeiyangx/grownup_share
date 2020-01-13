#%%
"""
哈夫曼算法
"""
#节点类
class Node(object):
    def __init__(self,name=None,value=None):
        self._name=name
        self._value=value
        self._left=None
        self._right=None

#哈夫曼树类
class HuffmanTree(object):

    #根据Huffman树的思想：以叶子节点为基础，反向建立Huffman树
    def __init__(self,char_weights):
        self.a=[Node(part[0],part[1]) for part in char_weights]  #根据输入的字符及其频数生成叶子节点
        print("看这里，查这里，理这里:",self.a)
        self.a.sort(key=lambda node: node._value, reverse=True)
        print("看这里，查这里，理这里:",self.a)
        while len(self.a)!=1:
            ## @@非常棒的用法， 排序，根据指定对象的属性排序
            ## 还有这种方式 edges = sorted(edges, key=lambda element: element[2])
            self.a.sort(key=lambda node: node._value, reverse=True)
            c = Node(value=(self.a[-1]._value + self.a[-2]._value))
            c._left = self.a.pop(-1)
            c._right = self.a.pop(-1)
            self.a.append(c)
        self.root = self.a[0]
        self.b = list(range(10))  #self.b用于保存每个叶子节点的Haffuman编码,range的值只需要不小于树的深度就行

    #用递归的思想生成编码
    def pre(self,tree,length):
        node=tree
        if (not node):
            return
        elif node._name:
            print (node._name + '的编码为:',
            [self.b[i] for i in range(length)], '\n')
            return
        print("lengthlength:",length)
        self.b[length]=0
        self.pre(node._left,length+1)
        self.b[length]=1
        self.pre(node._right,length+1)
     #生成哈夫曼编码   
    def get_code(self):
        self.pre(self.root,0)

if __name__=='__main__':
    #输入的是字符及其频数
    char_weights=[('a',5),('b',4),('c',10),('d',8),('f',15),('g',2)]
    tree=HuffmanTree(char_weights)
    tree.get_code()

#%%
import time
start = time.perf_counter()
"""Kruskal算法"""
## https://blog.csdn.net/weixin_44193909/article/details/88774567
## 找到无向图的最小生成树
## 1. 因为所有边是以最小权排序的，所以都是从小到大搜索
## 2. 所有记录的边，都以边的右节点记录下一节点，以字典的格式做记录
## 3. 当新增边进入最小生成树的时候，如果按照所有边的节点传递成环，那么就不记录该边
N = 0
class DisjointSet(dict):
    '''不相交集，传入dict方法后，定义属性就可以用self[key] = value'''
    # @@init可以用pass跳过
    def __init__(self, dict):
        pass
    # 把添加对象中key的方法赋予add
    def add(self, item):
        self[item] = item

    def find(self, item):
        global N
        temp_n = N
        N += 1
        print("--{}--".format(temp_n)*10)
        print("self[item]",self[item])
        print("item",item)
        if self[item] != item:
            self[item] = self.find(self[item])
        print("__{}__".format(temp_n)*10)
        print("self[item]",self[item])
        print("item",item)
        return self[item]
    ## 这个方法就是用右节点不断记录位置传递
    def unionset(self, item1, item2):
        print("&&&***&&&***"*5)
        print("key-item1:",item1,"value-self[item1]",self[item1])
        print("key-item2:",item2,"value-self[item2]",self[item2])
        self[item2] = self[item1]
        print("key-item1:",item1,"value-self[item1]",self[item1])
        print("key-item2:",item2,"value-self[item2]",self[item2])

def Kruskal(nodes, edges):
    '''基于不相交集实现Kruskal算法'''
    ## 只是先实例化而已，至于为啥要nodes，似乎没毛用
    forest = DisjointSet(nodes)
    ## 建立最小生成树的边列表
    MST = []
    # 将所有单点都加入
    for item in nodes:
        forest.add(item)
    # 把边按照其权进行排序
    edges = sorted(edges, key=lambda element: element[2])
    num_sides = len(nodes)-1  # 最小生成树的边数等于顶点数减一
    ## 遍历每一个边，由小到大
    _num = 0
    
    for e in edges:
        print("$${}@@".format(_num)*10)
        _num += 1
        node1, node2, _ = e
        parent1 = forest.find(node1)
        parent2 = forest.find(node2)
        print("start:",MST)
        if parent1 != parent2: # 判断是否成环，如果成环，则该点跳过
            MST.append(e)
            num_sides -= 1
            
            if num_sides == 0:
                return MST
            else:
                forest.unionset(parent1, parent2)
        print("end:",MST)
    pass

def main():
    nodes = set(list('ABCDEFG'))
    edges = [("A", "B", 7), ("A", "B", 5),
             ("B", "C", 8), ("B", "D", 9), ("B", "E", 7), 
             ("C", "E", 5), ("D", "E", 15), ("D", "F", 6),
             ("E", "F", 8), ("E", "G", 9), ("F", "G", 11)]
    print("\n\nThe undirected graph is :", edges)
    print("\n\nThe minimum spanning tree by Kruskal is : ")
    print(Kruskal(nodes, edges))

if __name__ == '__main__':
    main()
end = time.perf_counter()   #结束计时
print('Running time: %f seconds'%(end-start))  #程序运行时间

# %%
"""Kruskal算法实现的朴素版"""
def naive_find(C,u):
    while C[u] != u:
        u = C[u]
    return u

def naive_union(C,u,v):
    u = naive_find(C,u)
    v = naive_find(C,v)
    C[u] = v

def naive_kruskal(G):
    E = [(G[u][v], u, v) for u in G for v in G[u]]
    T = set()
    C = {u:u for u in G}
    for _, u, v in sorted(E):
        if naive_find(C,u) != naive_find(C, v):
            T.add((u,v))
            naive_union(C, u, v)
    return T

# %%
"""Kruskal算法"""
def find(C,u):
    if C[u] != u:
        C[u] = find(C,C[u])
    return C[u]

def union(C, R, u,v):
    u,v = find(C,u),find(C,v)
    if R[u] > R[v]:
        C[v] = u
    else:
        C[u] = v
    if R[u] == R[v]:
        R[v] += 1

def kruskal(G):
    E = [(G[u][v], u, v) for u in G for v in G[u]]
    T = set()
    C,R = {u:u for u in G},{u:0 for u in G}
    for _, u, v in sorted(E):
        if find(C,u) != find(C, v):
            T.add((u,v))
            union(C, R, u, v)
    return T

# %%
from collections import defaultdict
"""用迭代解决最优搜索树问题"""
def opt_tree(p):
    n = len(p)
    s, e = defaultdict(int), defaultdict(int)
    for k in range(1,n+1):
        for i in range(n-k+1):
            j = i + k
            s[i,j] = s[i,j-1] + p[j-1]
            e[i,j] = min(e[i,r] + e[r+1,j] for r in range(i,j))
            e[i,j] += s[i,j]
    return e[0,n]







# %%

#%%
"""
BFS;
DFS;
BFS_Shortest 最短路径无权;
BFS_Dijkstra 最短路径带权;
DP -> floyd_warshall 最短路径带权;
广度优先加回溯解决迷宫问题;
递归数独;

"""


"邻接集的字典表示法,有向图"

graph = {
    "a":set("bc"),
    "b":set("acd"),
    "c":set("abde"),
    "d":set("bcef"),
    "e":set("cd"),
    "f":set("d"),
    # "g":set("fh"),
    # "h":set("fg")
    }
#%%
"图无向化处理"
for i,j in graph.items():
    for k in j:
        graph[k].add(i)

#%%
"""
BFS，广度优先
"""
def BFS(graph, s):
    collection_result = []
    queue = []
    queue.append(s)
    seen = set()
    seen.add(s)
    while (len(queue)>0):
        vertex = queue.pop(0)
        collection_result.append(vertex)
        nodes = graph[vertex]
        for w in nodes:
            if w not in seen:
                queue.append(w)
                seen.add(w)
    print(collection_result)
        
BFS(graph, "e")
# ['e', 'd', 'c', 'f', 'b', 'a']
#%%
"""
DFS,深度优先
"""
def DFS(graph, s):
    collection_result = []
    stack = []
    stack.append(s)
    seen = set()
    seen.add(s)
    while (len(stack)>0):
        vertex = stack.pop()
        collection_result.append(vertex)
        nodes = graph[vertex]
        for w in nodes:
            if w not in seen:
                stack.append(w)
                seen.add(w)
        # print(vertex)
    print(collection_result)
DFS(graph, "e")
# ['e', 'c', 'b', 'a', 'd', 'f']
#%%
"""
BFS，广度优先,拓展最短路径
"""
def BFS_Shortest(graph, s):
    collection_result = []
    queue = []
    queue.append(s)
    seen = set()
    seen.add(s)
    parent = {s:None}
    while (len(queue)>0):
        vertex = queue.pop(0)
        collection_result.append(vertex)
        nodes = graph[vertex]
        for w in nodes:
            if w not in seen:
                queue.append(w)
                seen.add(w)
                parent[w] = vertex
    print(collection_result)
    return parent
parent = BFS_Shortest(graph, "e")
print(parent)
# ['e', 'd', 'c', 'f', 'b', 'a']
# f的前一个点是d，d的前一个点是e.
# {'e': None, 'd': 'e', 'c': 'e', 'f': 'd', 'b': 'd', 'a': 'c'}

# %%
"""
BFS，广度优先,拓展最短路径,拓展带权
"""
import heapq
import math

# heapq.heappush(obj,input())
# heapq.heappop(obj)
graph = {
    "a":{"b":5,"c":1},
    "b":{"a":5,"c":2,"d":1},
    "c":{"a":1,"b":2,"d":4,"e":8},
    "d":{"b":1,"c":2,"e":4,"f":8},
    "e":{"c":8,"d":3},
    "f":{"d":6},
    }

def init_distance(graph,s):
    distance = {s:0}
    for vertex in graph:
        if vertex != s:
            distance[vertex] = math.inf
    return distance


def BFS_Dijkstra(graph, s):
    pqueue = []
    heapq.heappush(pqueue,(0,s))
    seen = set()
    parent = {s:None}
    distance = init_distance(graph,s)
    
    while (len(pqueue)>0):
        pair = heapq.heappop(pqueue)
        dist = pair[0]
        vertex = pair[1]
        seen.add(vertex)
        nodes = graph[vertex].keys()
        for w in nodes:
            if w not in seen:
                if dist + graph[vertex][w] < distance[w]:
                    heapq.heappush(pqueue,(dist + graph[vertex][w], w))
                    parent[w] = vertex
                    distance[w] = dist + graph[vertex][w]
                
    return parent, distance

parent, distance = BFS_Dijkstra(graph, "a")
print(parent)
print(distance)

# %%
"""
DP -> floyd_warshall
"""
import math


class Graph:
    def __init__(self, N=0):  # a graph with Node 0,1,...,N-1
        self.N = N
        self.W = [
            [math.inf for j in range(0, N)] for i in range(0, N)
        ]  # adjacency matrix for weight
        self.dp = [
            [math.inf for j in range(0, N)] for i in range(0, N)
        ]  # dp[i][j] stores minimum distance from i to j

    def addEdge(self, u, v, w):
        self.dp[u][v] = w

    def floyd_warshall(self):
        for k in range(0, self.N):
            for i in range(0, self.N):
                for j in range(0, self.N):
                    self.dp[i][j] = min(self.dp[i][j], self.dp[i][k] + self.dp[k][j])

    def showMin(self, u, v):
        return self.dp[u][v]

graph = Graph(5)
graph.addEdge(0, 2, 9)
graph.addEdge(0, 4, 10)
graph.addEdge(1, 3, 5)
graph.addEdge(2, 3, 7)
graph.addEdge(3, 0, 10)
graph.addEdge(3, 1, 2)
graph.addEdge(3, 2, 1)
graph.addEdge(3, 4, 6)
graph.addEdge(4, 1, 3)
graph.addEdge(4, 2, 4)
graph.addEdge(4, 3, 9)
graph.floyd_warshall()
print(graph.showMin(1, 4))
print(graph.showMin(0, 3))

#%%
"""
广度优先加回溯解决迷宫问题
"""
try:
    while True:
        row,col = map(int,input().split())
        maze = []
        for i in range(row):
            maze.append(list(map(lambda x:-x,map(int,input().split()))))
        queue = [[0,0]]
        maze[0][0] = 1
        while queue:
            x,y = queue.pop(0)
            if x == row-1 and y == col-1:
                break
            if x+1 < row and maze[x+1][y] == 0:
                maze[x+1][y] = maze[x][y]+1
                queue.append([x+1,y])
            if y+1 < col and maze[x][y+1] == 0:
                maze[x][y+1] = maze[x][y]+1
                queue.append([x,y+1])
            if x-1 >= 0 and maze[x-1][y] == 0:
                maze[x-1][y] = maze[x][y]+1
                queue.append([x-1,y])
            if y-1 >= 0 and maze[x][y-1] == 0:
                maze[x][y-1] = maze[x][y]+1
                queue.append([x,y-1])
        result = [[row-1,col-1]]
        for i in range(maze[-1][-1]-1,0,-1):
            tempRow = result[0][0]
            tempCol = result[0][1]
            if tempRow-1>=0 and maze[tempRow-1][tempCol] == i:
                result.insert(0,[tempRow-1,tempCol])
            elif tempCol-1>=0 and maze[tempRow][tempCol-1] == i:
                result.insert(0,[tempRow,tempCol-1])
            elif tempRow+1<row and maze[tempRow+1][tempCol] == i:
                result.insert(0,[tempRow+1,tempCol])
            elif tempCol+1<col and maze[tempRow][tempCol+1] == i:
                result.insert(0,[tempRow,tempCol+1])
        for i in result:
            print('(%d,%d)'%(i[0],i[1]))
except Exception:
    pass

#%%
"""
递归数独
"""
#-*- coding: utf8 -*-
 
def check(matrix,row,col,value):
    """
    检测在(row,col)放value是否合适
    1.每行含1-9,不含重复值value
    2.每列含1-9,不含重复值value
    3.3*3区块含1-9,不含重复值value
    """
    #检测每行
    for j in range(9):
        if matrix[row][j]==value:
            return False
    #检测每列
    for i in range(9):
        if matrix[i][col]==value:
            return False
    #检测元素所在3*3区域
    area_row=(row/3)*3
    area_col=(col/3)*3
    for i in range(area_row,area_row+3):
        for j in range(area_col,area_col+3):
            if matrix[i][j]==value:
                return False
    return True
 
def solveSudoku(matrix,count=0):
    """
    遍历每一个未填元素，遍历1-9替换为合适的数字
    """
    if (count==81):#递归出口
        return True
    row=count/9#行标
    col=count%9#列标
    if (matrix[row][col]!=0):#已填充
        return solveSudoku(matrix,count=count+1)
    else:#未填充
        for i in range(1,10):
            if check(matrix,row,col,i):#找到可能的填充数
                matrix[row][col]=i
                if solveSudoku(matrix,count=count+1):#是否可完成
                    return True#可完成
                #不可完成
                matrix[row][col]=0#回溯
        return False#不可完成
 
matrix=[]
for i in range(9):
    matrix.append(map(int,raw_input().split(' ')))
solveSudoku(matrix)
for i in range(9):
    print ' '.join(map(str,matrix[i]))
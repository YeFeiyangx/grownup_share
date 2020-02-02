#%%
"邻接集表示法"
## 在无序的情况下，集合的形式，优于列表的形式
a,b,c,d,e,f,g,h = range(8)
# N是一张图，表示了a-h可以到达的位置
N = [
    {b,c,d,e,f},     # a
    {c,e},           # b
    {d},             # c
    {e},             # d
    {f},             # e
    {c,g,h},         # f
    {f,h},           # g
    {f,g}            # h
    ]

print("查找a点所能到达的位置:",N[a])

#%%
"加权邻接字典"

a,b,c,d,e,f,g,h = range(8)
# N是一张图，表示了a-h可以到达的位置,并设置路径权重
N = [
    {b:2,c:1,d:3,e:9,f:4},    # a
    {c:4,e:3},                # b
    {d:8},                    # c
    {e:7},                    # d
    {f:5},                    # e
    {c:2,g:2,h:2},            # f
    {f:1,h:6},                # g
    {f:9,g:8}                 # h
    ]

"加权邻接字典2"

a,b,c,d,e,f,g,h = range(8)
# N是一张图，表示了a-h可以到达的位置,并设置路径权重
N = {
    "a":{"b":5,"c":1},
    "b":{"a":5,"c":2,"d":1},
    "c":{"a":1,"b":2,"d":4,"e":8},
    "d":{"b":1,"c":2,"e":4,"f":8},
    "e":{"c":8,"d":3},
    "f":{"d":6},
    }
#%%
"邻接集的字典表示法"

N = {
    "a":set("bcdef"),
    "b":set("ce"),
    "c":set("d"),
    "d":set("e"),
    "e":set("f"),
    "f":set("cgh"),
    "g":set("fh"),
    "h":set("fg")
    }

#%%
"邻接矩阵的列表嵌套表示法"
a,b,c,d,e,f,g,h = range(8)
#     a b c d e f g h
N = [[0,1,1,1,1,1,0,0], #a
     [0,0,1,0,1,0,0,0], #b
     [0,0,0,1,0,0,0,0], #c
     [0,0,0,0,1,0,0,0], #d
     [0,0,0,0,0,1,0,0], #e
     [0,0,1,0,0,0,1,1], #f
     [0,0,0,0,0,1,0,1], #g
     [0,0,0,0,0,1,1,0]] #h

#%%
"加权，不存在的边赋予无限大的权值"

a,b,c,d,e,f,g,h = range(8)
inf = float("inf")
#       a   b   c   d   e   f   g   h
W = [[  0,  2,  1,  3,  9,  4,inf,inf], #a
     [inf,  0,  4,inf,  3,inf,inf,inf], #b
     [inf,inf,  0,  8,inf,inf,inf,inf], #c
     [inf,inf,inf,  0,  7,inf,inf,inf], #d
     [inf,inf,inf,inf,  0,  5,inf,inf], #e
     [inf,inf,  2,inf,inf,  0,  2,  2], #f
     [inf,inf,inf,inf,inf,  1,  0,  6], #g
     [inf,inf,inf,inf,inf,  9,  8,  0]] #h

#%%
import math
matrix_map = [[-math.inf for j in range(col+1)] for i in range(row+1)]
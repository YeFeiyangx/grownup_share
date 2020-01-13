# %%
import random
# 找出数组中最接近，但是不相等的两个数字
random.seed = 6
seq = [random.randrange(10**10) for i in range(100)]
# @@ 注意dd用法中的（“inf")
dd = float("inf")
for x in seq:
    for y in seq:
        if x == y:
            continue
        d = abs(x-y)
        if d < dd:
            xx, yy, dd = x, y, d

print(xx, yy)

# %%
# 排序是个 nlogn的操作，所以先排序，再求解
seq.sort()
dd = float("inf")
for i in range(len(seq)-1):
    x, y = seq[i], seq[i+1]
    if x == y:
        continue
    d = abs(x-y)
    if d < dd:
        xx, yy, dd = x, y, d

print(xx, yy)

# %%
# 棋盘拼接问题
# 将一个偶数边长的棋盘，用L型方块填充，填充满即可


def cover(board, lab=1, top=0, left=0, side=None):
    # 递归中side 等于递归中 len(board)的长度
    if side is None:
        side = len(board)

    # side length of subboard
    # 将其一分为二
    s = side // 2

    # offsets for outer/inner squares of subboards
    # @@ tuple组
    # @@ 除去递归本身，精髓就是这个tuple对于问题的归化
    # 根据四个映射的corner，获得矩阵内相应小corner的位置
    offsets = (0, -1), (side-1, 0)
    print("--{}--".format(lab)*10)
    print("offsets,lab:", offsets, lab)
    print("top,left:", top, left)
    for dy_outer, dy_inner in offsets:
        for dx_outer, dx_inner in offsets:
            # If the outer corner is not set...
            if not board[top+dy_outer][left+dx_outer]:
                print("top,left,side,dy_outer,dy_inner,dx_outer,dx_inner")
                print(top, left, side, dy_outer, dy_inner, dx_outer, dx_inner)
                print(top+dy_outer, left+dx_outer)
                print(top+s+dy_inner, left+s+dx_inner)
                # ...label the inner corner
                board[top+s+dy_inner][left+s+dx_inner] = lab

    # next label:
    lab += 1
    if s > 1:
        for dy in [0, s]:
            for dx in [0, s]:
                # Recursive calls, if s is at least 2:
                lab = cover(board, lab, top+dy, left+dx, s)

    # Return the next available label
    return lab


# %%
board = [[0]*8 for i in range(8)]
board[7][7] = -1

for row in board:
    print((" %2i"*8) % tuple(row))

# %%
cover(board)

for row in board:
    print((" %2i"*8) % tuple(row))

# %%
offsets = ((0, -1), (7, 0))
for i, j in offsets:
    print(i, j)

# %%
if not 0:
    print("yes")

# %%
"""
充分运用动态数据类型的作用
"""
# 递归版插入排序
# 把最小组逐渐拓展开始排序


def ins_sort_rec(seq, i):
    if i == 0:
        return
    ins_sort_rec(seq, i-1)  # 从最前面两个开始排，逐渐拓展
    j = i
    while j > 0 and seq[j-1] > seq[j]:
        seq[j-1], seq[j] = seq[j], seq[j-1]
        j -= 1


# 插入排序
def sort_rec(seq, i):
    for i in range(1, len(seq)):
        j = i
        while j > 0 and seq[j-1] > seq[j]:
            seq[j-1], seq[j] = seq[j], seq[j-1]
            j -= 1


# 递归版选择排序
# 把最大的挑出来排
def sel_sort_rec(seq, i):
    if i == 0:
        return
    max_j = i
    for i in range(i):
        if seq[j] > seq[max_j]:
            max_j = j
    seq[i], seq[max_j] = seq[max_j], seq[i]
    sel_sort_rec(seq, i-1)


# 选择排序
def sel_sort(seq):
    # @ 逆序遍历的一种方式 range(5,0,-1) => [5,4,3,2,1]
    # @@ 用索引作为遍历的元素,巧用动态数据类型
    for i in range(len(seq)-1, 0, -1):
        max_j = i
        for j in range(i):
            if seq[j] > seq[max_j]:
                max_j = j
        seq[i], seq[max_j] = seq[max_j], seq[i]


# %%
testa = [3, 21, 4, 5, 2, 10, 6]
sel_sort(testa)
print(testa)

# %%
list(range(4, 0, -1))

# %%
# p84 电影院排序
## 朴素实现方法
## 无权的，所以1号工具人比较悲剧
def naive_max_perm(M, A=None):
    if A is None:
        A = set(range(len(M)))
    if len(A) == 1:
        return A
    ## 邪恶的注释菌，又来了，这题有点简单中透着奇思妙想
    #@@ 从工具人喜欢的座位中剔除没人喜欢的位置
    ##  没人喜欢的位置上的人是没机会换了，所以进而把那些人喜欢的位置也剔除就是了
    ##  剔除老多悲剧的人后，最后再求剩余人喜欢的位置，与剩余人的差集时，
    ##  就发现差集C为空，那就没什么号剔除了，返回最后剩余的集合A就好了
    B = set(M[i] for i in A)
    C = A - B
    print("C:",C)
    if C:
        print("--*--"*10,"begin")
        print(A)
        A.remove(C.pop())
        print(A)
        print("--*--"*10,"end")
        return naive_max_perm(M, A)
    return A

## 0号工具人，喜欢2号位；1号工具人，喜欢2号位；2号工具人，喜欢0号位；3号工具人，喜欢5号位。。。。
M = [2,2,0,5,3,5,7,4]

naive_max_perm(M)

# %%
A = {10,6}
print(A.pop())
print(A)

# %%
## 减低计算复杂度
def max_perm(M):
    n = len(M)
    A = set(range(n))
    # 建立一个计数器，映射关系下，记录每个位置被喜好的程度。
    ## 没被人喜好的，那么他就凉了咯
    count = [0]*n
    Q = [i for i in A if count[i] == 0]
    ## 没被人喜欢的位置，自然这个人喜好也就麽用了，没有价值
    ## 所以这个位置对应的喜好也要从计数器中删除
    ## 计数器删除到0，那么这个位置就要回到 嫌弃的列表Q
    ## Q对应的位置又要被删删删
    while Q:
        i = Q.pop()
        A.remove(i)
        j = M[i]
        count[j] -= 1
        if count[j] == 0:
            Q.append(j)
    return A


#%%
"""
题外进阶道具
from collections import defaultdict
"""
from collections import defaultdict
## @@defaultdict(list),当字典被赋值时，空值就是[]，无需手动输入
## 当为空值时，自然而然就无法append了
## 有重复的key就有问题咯
def counting_sort(A, key=lambda x:x):
    B,C = [],defaultdict(list)
    print("B,C",B,C)
    for x in A:
        C[key(x)].append(x)
    print("C",C)
    print()
    for k in range(min(C),max(C)+1):
        B.extend(C[k])
    return B

counting_sort([4,2,3,10,8,7,1,5])


# %%
# 朴素版的明星问题方案
# 找出人人都认识他，但是他却人人都不认识的明星
# u是明星
# 暴力破解，哼，嫌弃
def naive_celeb(G):
    n = len(G)
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if G[u][v]:
                break
            if not G[v][u]:
                break
        else:
            return u
    return None

#%%
## 补充小知识
## for 循环完整进行后才运行
for i in range(4):
    if i == 2:
        break
    print(i)
else:
    print("yes1")

for i in range(4):
    a = 1
        # break
else:
    print("yes2")

# %%
# 既然明星队人人都不认识，那么事情就容易了，只需要确凿某个有认识的人，也有不认识的人，那妥妥不是明星
# 同时，那个人不认识的人，也妥妥的不是明星
# 那么解决起来就容易了，因为进入平民集合的人就不用了再搜索了
def celeb(G):
    n = len(G)
    u,v = 0,1               # 先找两个人的索引
    for c in range(2,n+1):  # 检查剩余的宝宝们索引 介个。。。n+1是什么鬼哦
        if G[u][v]:         # 如果两个人认识，那么记录u为下一个人的索引
            u = c
        else:               # 如果两人不认识，那就检查u是否认识下一个人的索引，这需要把这个索引给V
            v = c
    if u == n:              # 如果最后u和最大人数N一样大
        c = v               # c就等于v,v有可能是那个被所有人认识的明星
    else:
        c = u               # 否则有可能C等于U，因为U似乎谁都不认识
    for v in range(n):      # 遍历n
        if c == v:          # 如果c和v相当，那么就是同一个人，就continue到下一个
            continue
        if G[c][v]:         # 但凡有c认识一个v那么久结束了，这个人不满足条件
            break
        if not G[v][c]:     # 但凡有一个人不认识C，那么就也结束了，这个人不满足全明星条件
            break
    else:
        return c            # 找到了明星，返回C
    return None             # 找不到任何人


#%%
from random import randrange

n = 100
# 构建平民和明星图
#@@ 这是什么鬼才方法？
G = [[randrange(2) for i in range(n)] for i in range(n)]
c = randrange(n)
print("看看是哪个小可爱成为了明星：",c)

for i in range(n):
    G[i][c] = True
    G[c][i] = False

print(naive_celeb(G))   # 平方级的暴力法
print(celeb(G))         # 指数级的集合法

#%%
# 拓扑排序问题
def naive_topsort(G, _n, S=None):
    if S is None:                   # 初始化，获得所有点
        S = set(G)              
    if len(S) == 1:                 # 剩下一个点了，就没啥好排的
        return list(S)
    print("--{}--".format(_n)*10)
    _n += 1
    print("S:",S)
    v = S.pop()                     # 取出S的第一个点
    print("vb:",v)
    seq = naive_topsort(G, _n, S)   # 把剩余的点和全集合送入递归函数中，获得新seq
    print("--seq--{}--:".format(_n), seq)
    print("ve:", v)
    ## @@ 这一块比较懵，但是应该是精髓
    min_i = 0                       # min_i 起始为0
    for i,u in enumerate(seq):      # 对seq取元素和索引
        print("i,u,G[u]:", i, u, G[u])
        if v in G[u]:               # 判断依赖关系，如存在依赖关系，min_i的索引要相应的发生变化
            min_i = i + 1           # 如果存在依赖关系，自然其索引应该后推一位
        print("min_i:",min_i)
    seq.insert(min_i, v)            # 通过变化的索引，进行列表的插入。
    return seq

#%%
# naive_topsort([1,5],[2,3,5],[3],[4,5],[5],[5]])
n = 0
M = {0:{1,5},1:{2,3,5},2:{3},3:{4,5},4:{5},5:{}}
naive_topsort(M,n,S = None)


# %%
#有向无环图的拓扑排序
#@@ 采用计数器的方式进行拓扑排序 
def topsort(G):
    ## 建立每一个点的计数器
    count = dict((u, 0) for u in G)
    for u in G:
        for v in G[u]:
            # 任何有被引用的计数器，其计数+1
            count[v] += 1
    # Q为未被引用的计数器的key，开始一定有一个未被引用的计数器，不然循环依赖，就没起点了
    Q = [u for u in G if count[u] == 0]
    S = []
    while Q:                    # 不断从Q中取数u，放入S，直到Q取空了
        u = Q.pop()             # 
        S.append(u)             # 
        for v in G[u]:          # 把数u指向的索引计数，都咔咔减光
            count[v] -= 1       #
            if count[v] == 0:   # 如果减到0了，那么又有新的成员可以进入 Q集合了
                Q.append(v)
    return S                    # 最后返回S集合

#%%



# %%

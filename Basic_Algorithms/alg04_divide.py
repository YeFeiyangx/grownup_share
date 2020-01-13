# %%
# ---------------------天际线问题-----------------------------
# 题设给定矩形
# @@ 使用堆的数据结构来完成该问题
import heapq
rectangles = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]


class Solution(object):
    def getSkyline(self, buildings):
        # 第一步 把所有的矩阵化成线条的形式，用左-右-负高度的形式表示，第二项表示右边界也就是“零开始的线条”，用set是为了防止两个矩形在同一点结束，用tuple是因为set的元素不能是list
        events = sorted([(L, -H, R) for L, R, H in buildings] +
                        list({(R, 0, None) for _, R, _ in buildings}))
        # res是最终结果，hp表示“现在需要考虑的线段们”
        res, hp = [[0, 0]], [(0, float("inf"))]

        # 对所有的线条进行遍历
        for x, negH, R in events:
            # 当一个线条的左边界大于hp中线条的右边界时，说明这个线条已经没有任何作用了，可以删去
            while x >= hp[0][1]:
                heapq.heappop(hp)
            # 如果这是一个非零线段，我们把它放入hp中
            if negH:
                heapq.heappush(hp, (negH, R))
            # 这里需要注意，heapq保证列表中的第一个元素一定是最小元素，取相反数也就是目前最高元素。
            # res[-1][1]是结果中最后一个元素的高度，hp[0][0]是目前最高那个元素的高度，如果不相等，就需要更新res。
            # 为什么我们只考虑dp中第一个元素？首先，这个元素在dp里面，所以它是有影响的，其次，它是最高的，当前我们只要最高的元素，至于比它低的元素，都被覆盖了
            if res[-1][1] + hp[0][0]:
                res.append([x, -hp[0][0]])

        return res[1:]


a = Solution()
a.getSkyline(rectangles)
# %%
"""二分法标准模式"""


def divide_andconquer(S, divide, combine):
    if len(S) == 1:
        return S
    L, R = divide(S)
    A = divide_andconquer(L, divide, combine)
    B = divide_andconquer(R, divide, combine)
    return combine(A, B)

# %%


class Node:
    lft = None
    rgt = None

    def __init__(self, key, val):
        self.key = key
        self.val = val


def insert(node, key, val):
    if node is None:
        return Node(key, val)
    if node.key == key:
        node.val = val
    elif key < node.key:
        node.lft = insert(node.lft, key, val)
    else:
        node.rgt = insert(node.rgt, key, val)
    return node


def search(node, key):
    if node is None:
        raise KeyError
    if node.key == key:
        return node.val
    elif key < node.key:
        return search(node.lft, key)
    else:
        return search(node.rgt, key)

# @@ __setitem__ class 中的魔法方法，可以直接通过实例化属性实现该方法的调用
# @@ __getitem__ class 中的魔法方法，可以直接print实现属性值的显示
# @@ __contain__ class 中的魔法方法，确认一个键值是否再在字典的key中


class Tree:
    root = None

    def __setitem__(self, key, val):
        self.root = insert(self.root, key, val)

    def __getitem__(self, key):
        return search(self.root, key)

    def __contain__(self, key):
        try:
            search(self.root, key)
        except KeyError:
            return False
        return True


# %%
def partition(seq):
    pi, seq = seq[0], seq[1:]
    lo = [x for x in seq if x <= pi]
    hi = [x for x in seq if x > pi]
    return lo, pi, hi

def select(seq, k):
    lo, pi, hi = partition(seq)
    m = len(lo)
    if m == k:
        return pi
    elif m < k:
        return select(hi, k-m-1)
    else:
        return select(lo, k)

#%%
a = [1,2,3,4,5,6,7]
select(a,3)

# %%
# 快速排序
def quicksort(seq):
    if len(seq) <= 1: return seq
    lo, pi, hi = partition(seq)
    return quicksort(lo) + [pi] + quicksort(hi)

# 归并排序
"""
和前文的归并方法如出一辙
"""
def mergesort(seq):
    mid = len(seq)//2
    lft, rgt = seq[:mid], seq[mid:]
    if len(lft) > 1: lft = mergesort(lft)
    if len(rgt) > 1: rgt = mergesort(rgt)
    res = []
    while lft and rgt:
        if lft[-1] >= rgt[-1]:
            res.qppend(lft.pop())
        else:
            res.qppend(rgt.pop())

    res.reverse()
    return (lft or rgt) + res

# %%
import matplotlib.pyplot as plt
import math
 
import sklearn.datasets as datasets
 
"""
使用Graham扫描法计算凸包
网上的代码好多运行效果并不好
算法参见《算法导论》第三版 第605页
"""
 
 
def get_bottom_point(points):
    """
    返回points中纵坐标最小的点的索引，如果有多个纵坐标最小的点则返回其中横坐标最小的那个
    :param points:
    :return:
    """
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i
    return min_index
 
 
def sort_polar_angle_cos(points, center_point):
    """
    按照与中心点的极角进行排序，使用的是余弦的方法
    :param points: 需要排序的点
    :param center_point: 中心点
    :return:
    """
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0]-center_point[0], point_[1]-center_point[1]]
        rank.append(i)
        norm_value = math.sqrt(point[0]*point[0] + point[1]*point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)
 
    for i in range(0, n-1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index-1] or (cos_value[index] == cos_value[index-1] and norm_list[index] > norm_list[index-1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index-1]
                rank[index] = rank[index-1]
                norm_list[index] = norm_list[index-1]
                cos_value[index-1] = temp
                rank[index-1] = temp_rank
                norm_list[index-1] = temp_norm
                index = index-1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])
 
    return sorted_points
 
 
def vector_angle(vector):
    """
    返回一个向量与向量 [1, 0]之间的夹角， 这个夹角是指从[1, 0]沿逆时针方向旋转多少度能到达这个向量
    :param vector:
    :return:
    """
    norm_ = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1])
    if norm_ == 0:
        return 0
 
    angle = math.acos(vector[0]/norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2*math.pi - angle
 
 
def coss_multi(v1, v2):
    """
    计算两个向量的叉乘
    :param v1:
    :param v2:
    :return:
    """
    return v1[0]*v2[1] - v1[1]*v2[0]
 
 
def graham_scan(points):
    # print("Graham扫描法计算凸包")
    bottom_index = get_bottom_point(points)
    bottom_point = points.pop(bottom_index)
    sorted_points = sort_polar_angle_cos(points, bottom_point)
 
    m = len(sorted_points)
    if m < 2:
        print("点的数量过少，无法构成凸包")
        return
 
    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])
 
    for i in range(2, m):
        length = len(stack)
        top = stack[length-1]
        next_top = stack[length-2]
        v1 = [sorted_points[i][0]-next_top[0], sorted_points[i][1]-next_top[1]]
        v2 = [top[0]-next_top[0], top[1]-next_top[1]]
 
        while coss_multi(v1, v2) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length-1]
            next_top = stack[length-2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]
 
        stack.append(sorted_points[i])
 
    return stack
 
 
def test1():
    points = [[1.1, 3.6],
                       [2.1, 5.4],
                       [2.5, 1.8],
                       [3.3, 3.98],
                       [4.8, 6.2],
                       [4.3, 4.1],
                       [4.2, 2.4],
                       [5.9, 3.5],
                       [6.2, 5.3],
                       [6.1, 2.56],
                       [7.4, 3.7],
                       [7.1, 4.3],
                       [7, 4.1]]
 
    for point in points:
        plt.scatter(point[0], point[1], marker='o', c='y')
 
    result = graham_scan(points)
 
    length = len(result)
    for i in range(0, length-1):
        plt.plot([result[i][0], result[i+1][0]], [result[i][1], result[i+1][1]], c='r')
    plt.plot([result[0][0], result[length-1][0]], [result[0][1], result[length-1][1]], c='r')
 
    plt.show()
 
 
def test2():
    """
    使用复杂一些的数据测试程序运行效果
    :return:
    """
    iris = datasets.load_iris()
    data = iris.data
    points_ = data[:, 0:2]
    points__ = points_[0:50, :]
    points = points__.tolist()
 
    temp_index = 0
    for point in points:
        plt.scatter(point[0], point[1], marker='o', c='y')
        index_str = str(temp_index)
        plt.annotate(index_str, (point[0], point[1]))
        temp_index = temp_index + 1
 
    result = graham_scan(points)
    print(result)
    length = len(result)
    for i in range(0, length-1):
        plt.plot([result[i][0], result[i+1][0]], [result[i][1], result[i+1][1]], c='r')
    plt.plot([result[0][0], result[length-1][0]], [result[0][1], result[length-1][1]], c='r')
 
    # for i in range(0, len(rank)):
 
    plt.show()
 
 
if __name__ == "__main__":
    test2()
    
# %%
"""
用AA树结构实现再平衡的二分搜索树
"""
class Node:
    lft = None
    rgt = None
    lvl = 1
    def __init__(self,key,val):
        self.key = key
        self.val = val

def skew(node):
    if None in [node, node.lft]:
        return node
    if node.lft.lvl != node.lvl:
        return node
    lft = node.lft
    node.lft = lft.rgt
    lft.rgt = node
    return lft

def split(node):
    if None in [node, node.rgt, node.rgt.rgt]:
        return node
    if node.rgt.rgt.lvl != node.lvl:
        return node

    rgt = node.rgt
    node.rgt = rgt.lft
    rgt.lft = node
    rgt.lvl += 1
    return rgt

def insert(node,key,val):
    if node is None:
        return Node(key,val)
    if node.key == key:
        node.val = val
    elif key < node.key:
        node.lft = insert(node.lft,key,val)
    else:
        node.rgt = insert(node.rgt,key,val)
    node = skew(node)
    node = split(node)
    return node

#%%
def sift_up(heap,startpos,pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos-1) >> 1 # 位操作符
        parent = heap[parentpos]
        if parent <= newitem:
            break
        heap[pos] = parent
        pos = parentpos
    heap[pos] = newitem
    
##
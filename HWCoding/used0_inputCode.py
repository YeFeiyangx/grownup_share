#%%
# goods_num 物品数，vlue_limit总限制
goods_num, vlue_limit = map(int, input().split())
goods_info = []

for i in range(goods_num):
    goods_info.append(list(map(int, input().split())))
    
# %%
while True:
    try:
        ## codehere
        _var = input().split()
        _var = map(int,input().split())
        
    except:
        break

# %%
"""
另一种while True
"""
try:
    while True:
        N = int(input())
        max_num = int((N+1)*N/2)
        num_list = list(range(1,max_num+1))
        output_list = [[] for i in range(N)]
        cal_num_end = 0
        for i in range(N):
            cal_num_start = cal_num_end
            cal_num_end += i+1
            for j,k in zip(num_list[cal_num_start:cal_num_end],list(range(i+1))[::-1]):
                output_list[k].append(str(j))
        for i in output_list:
            print(" ".join(i))
except:
    pass
        
    
    
#%%
import sys
# input == sys.stdin, input 可用input("explain")
goods_info = []
for line in sys.stdin:
    # code here
    goods_info.append(list(map(int, line.split())))
    pass


#%%
from collections import defaultdict,Counter

_dict = defaultdict(dict)
_strings = Counter("test_tte")
# _strings.most_common() -> [('t', 4), ('e', 2), ('s', 1), ('_', 1)]

#%%
from collections import deque

# init_list = deque()
# init_list = deque([1,2,3])

# init_list.pop() -> 3 & init_list: [1,2]
# init_list.popleft() -> 1 & init_list: [2,3]
# init_list.append(4) -> init_list: [1,2,3,4]
# init_list.appendleft(4) -> init_list: [4,1,2,3]
# init_list.rotate(): rotate(1) 1，2，3 -> 3,1,2; rotate(2) 1，2，3 -> 2, 3, 1。

# %%
from bisect import bisect,insort

# bisect==bisect_right -> 有序数组中找给定数值的index，在满足比较条件的值右侧
# bisect([1,2,2,2,3,4,7,11] ,2) -> index = 4

# insort==insort_right -> 有序数组中找给定数值的index，在满足比较条件的值右侧
# insort([1,2,2,2,3,4,7,11] ,2) -> [1,2,2,2,2,3,4,7,11]

# %%
from itertools import permutations,combinations,combinations_with_replacement
from itertools import accumulate
# combinations_with_replacement -> 可复选
# list(combinations_with_replacement([1,2],2))
# [(1, 1), (1, 2), (2, 2)]

# permutations -> 带顺序，不可复选
# list(permutations([1,2],2))
# [(1, 2), (2, 1)]

# combinations -> 不带顺序，不可复选
# list(combinations([1,2],2))
# [(1, 2)]

# accumulation -> 累加
# list(accumulate([1,2,3]))
# [1,3,6]

# %%

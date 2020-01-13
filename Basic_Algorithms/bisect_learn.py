from bisect import bisect
from bisect import bisect_left
from bisect import insort_left
a = [0,2,3,5,6,8,8,9]
print("bisect",bisect(a,5))            # 不插入，就给要插入的5的索引 
print("bisect_left:",bisect_left(a,5)) # 不插入，就给要插入的5的索引
print("insort_left:",insort_left(a,5)) # 插入，返回的是被插入的列表
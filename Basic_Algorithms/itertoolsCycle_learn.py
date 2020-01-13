# %%
from itertools import cycle

cs = cycle('ABC') # 注意字符串也是序列的一种
n = 0
for c in cs:
    n+=1
    print (c)
    if n == 10:
        break
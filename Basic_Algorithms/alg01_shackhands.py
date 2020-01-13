#%%
"""侏儒排序法"""

def gnomesort(seq):
    i = 0
    while i < len(seq):
        if i == 0 or seq[i-1] < seq[i]:
            i += 1
        else:
            ## 如果不满足顺序，则交换后回头再来一次while循环
            seq[i],seq[i-1] = seq[i-1],seq[i]
            i -= 1

"""归并排序法"""
def mergesort(seq,strt):
    global _n
    print("--{}--{}--{}--".format(_n,strt,seq)*3)
    _n += 1
    mid = len(seq)//2
    ## 先将列表一分为二
    lft,rgt = seq[:mid],seq[mid:]
    # 如果左侧列表大于1，再进入一分为二的循环
    if len(lft)>1: 
        lft = mergesort(lft,"lft")
        print("--mainlft--",lft)
    # 如果右侧列表大于1，再进入一分为二的循环
    if len(rgt)>1: 
        rgt=mergesort(rgt,"rgt")
        print("--mainrgt--",rgt)
    # @@ 列表可以直接丢在while语句中，如果为空，等同于False
    ## insert,消耗是比较大的
    res = []
    while lft and rgt:
        if lft[-1] >= rgt[-1]:
            res.insert(0,lft.pop())
        else:
            res.insert(0,rgt.pop())

    ## append消耗相对比较小，最后带个reverse
    # res = []
    # while lft and rgt:
    #     if lft[-1] >= rgt[-1]:
    #         res.append(lft.pop())
    #     else:
    #         res.append(rgt.pop())
    # res.reverse()

    print("observation1:",strt,(lft or rgt))
    print("observation2:",strt,res)
    return (lft or rgt) + res

#%%
_n = 0
mergesort([4,1,3,8,9,0,2],"main")

# %%
a = [1]
b = [1,2,3]
_n = 0
while a and _n < 5 and b:
    _n += 1
    a.pop()
    b.pop()
    print("yes")

# %%
a = []
b = [1,2,3,4]
c = [5,6,7]
(a or b) + c

# %%

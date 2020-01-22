# %%
#最长上升子序列
import bisect 
while True:
    try:
        a, b = int(input()), map(int, input().split())
        q = []
        for v in b:
            pos = bisect.bisect_left(q, v)
            if pos == len(q):
                q.append(v)
            else:
                q[pos] = v
        print(len(q),q)
 
    except:
        break

# %%
import bisect 
help(bisect.bisect_left)

# %%
q = []
q[0] = 1
q

# %%
v = 1
pos = bisect.bisect_right(q, v)
pos

# %%
q

# %%
n=7
while n<5:
    try:
        dsfhkjfadsh
    except:
        break
    n-=1
    print("loop")
print("end")

# %%
try:
    hsdhfh
except:
    print("No")

# %%
import itertools
M = input("输入元素集合:->[1 2 3 4...]")
N = input("输入组合的个数:->num")
M = list(map(int, M.split()))
N = int(N)
# M = M*N
# print(len(list(itertools.permutations(M,N))))
print(len(list(itertools.combinations_with_replacement(M,N))))
print(list(itertools.combinations_with_replacement(M,N)))

# %%

while True:
    try:
        a, res = int(input()), 0
        for i in range(0, a + 1):
            if str(i ** 2).endswith(str(i)):
                res += 1
        print(res)
    except:
        break
#%%

while True:
    try:
        a, res, isNum = input(), "", False
        for i in a:
 
            if i.isdigit():
                if not isNum:
                    res = res + "*" + i
                else:
                    res += i
                isNum = True
            else:
                if isNum:
                    res = res + "*" + i
                else:
                    res += i
                isNum = False
        if a[-1].isdigit():
            res+="*"
        print(res)
 
 
    except:
        break
    
#%%
from itertools import combinations

a = [1,2,3,4,5,6,7,8]

mean_value = sum(a)/2
collection = set()
for i in range(1, len(a)):
    for j in combinations(a, i):
        # print(j)
        if sum(j) == int(mean_value):
            collection.add(j)

Result = []
if len(collection) == 0:
    print("not found")
else:
    cal_list = Counter(a)
    for i in collection:
        cal_list_temp = Counter(i)
        if cal_list[5] == cal_list_temp[5]:
            print("found")
            Result.append([i,tuple(set(a)-set(i))])

if len(Result) == 0:
    print("not found")
else:
    print(Result)
    
#%%
a = "sd4545asdasf78werwqrsadfas45sad65a4sdff1316546zxc"
print(list(enumerate(a)))
# for i, v in enumerate(a):
    

# %%
a = "13"
a.isdigit()

# %%
dp = [[0 for i in range(4)] for j in range(5)]
dp

# %%
from itertools import combinations

a = [1,2,3,4,5,6,7,8]

mean_value = sum(a)/2
collection = set()
for i in range(1, len(a)):
    for j in combinations(a, i):
        # print(j)
        if sum(j) == int(mean_value):
            collection.add(j)

Result = []
if len(collection) == 0:
    print("not found")
else:
    cal_list = Counter(a)
    for i in collection:
        cal_list_temp = Counter(i)
        if cal_list[5] == cal_list_temp[5]:
            print("found")
            Result.append([i,tuple(set(a)-set(i))])

if len(Result) == 0:
    print("not found")
else:
    print(Result)

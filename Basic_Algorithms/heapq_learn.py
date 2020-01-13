# %%
import heapq
import random

# Top-K
mylist = list(random.sample(range(100), 10))
k = 3
largest = heapq.nlargest(k, mylist)
smallest = heapq.nsmallest(k, mylist)
print('original list is', mylist)
print('largest-'+str(k), '  is ', largest)
print('smallest-'+str(k), ' is ', smallest)

# heapify 堆化
print('original list is', mylist)
heapq.heapify(mylist)
print('heapify  list is', mylist)

# heappush 加数入堆 & heappop 取顶堆数出堆
heapq.heappush(mylist, 105)
print('pushed heap is', mylist)
heapq.heappop(mylist)
print('popped heap is', mylist)

# heappushpop & heapreplace
heapq.heappushpop(mylist, 130)    # heappush -> heappop 先加后取
print('heappushpop', mylist)
heapq.heapreplace(mylist, 2)      # heappop -> heappush 先取后加
print('heapreplace', mylist)

# %%
heapq.heapify()


#%%
mylist = [(0,0),(1,1),(2,2),(3,3)]
a,b = heapq.heappop(mylist)
print(mylist)
print(a)
print(b)

# %%
test_list = random.sample(range(100),10)
print(test_list)

# %%
random.shuffle(test_list)
Q = []
for i in test_list:
    heapq.heappush(Q,i)
    print(Q)

# %%

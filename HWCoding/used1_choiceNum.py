#%%
"""
dp_opt 选数字，不相邻的数字选最大;
max_subarray_sum 最大加和子串;
isSumSubset 寻找指定加和数值;
deal 正反序最大连续子串;
字符串排序 filter;
偶数分解最接近的素数和;
1~10 24点四则运算;
扑克牌四则运算;
矩阵乘法;
百钱买百鸡;
整数的连续奇数和;
求立方根 牛顿迭代法;
求最小公倍数 辗转相除法求最大公约数;
考察数据栈 火车进出;
自动售货机 系统工程;
"""





"""
选数字，不相邻的数字选最大
"""
arr = [1,2,4,1,7,8,3]

def rec_opt(arr,i):
    if i == 0:
        return arr[0]
    elif i==1:
        return max(arr[0],arr[1])
    else:
        A = rec_opt(arr,i-2) + arr[i]
        B = rec_opt(arr,i-1)
        return max(A,B)

print(rec_opt(arr,6))


def dp_opt(arr):
    # create [0,0,...0,0] 记录前N个数字可选过程中，每一位选择的时候可选的最大值
    opt = [0 for i in range(len(arr))]
    opt[0] = arr[0]
    opt[1] = max(arr[0],arr[1])
    for i in range(2,len(arr)):
        A = opt[i-2] + arr[i]
        B = opt[i-1]
        opt[i] = max(A,B)
    return opt[len(arr)-1]

print(dp_opt(arr))

# %%
"""
最大加和子串
"""

def max_subarray_sum(nums: list) -> int:
    """
    >>> max_subarray_sum([6, 9, -1, 3, -7, -5, 10])
    17
    """
    if not nums:
        return 0
    n = len(nums)
    s = [0] * n
    res, s, s_pre = nums[0], nums[0], nums[0]
    for i in range(1, n):
        
        print("start look->%s" % i)
        print(res, s, s_pre)
        s = max(nums[i], s_pre + nums[i])
        s_pre = s
        res = max(res, s)
    return res


if __name__ == "__main__":
    
    nums = [6 , 9, -100, 3, -7, 300, 10]
    print(max_subarray_sum(nums))



#%%
"""
isSumSubset 寻找指定加和数值
"""

def isSumSubset(arr, arrLen, requiredSum):
    """
    >>> isSumSubset([2, 4, 6, 8], 4, 5)
    False
    >>> isSumSubset([2, 4, 6, 8], 4, 14)
    True
    """
    # a subset value says 1 if that subset sum can be formed else 0
    # initially no subsets can be formed hence False/0
    subset = [[False for i in range(requiredSum + 1)] for i in range(arrLen + 1)]

    # for each arr value, a sum of zero(0) can be formed by not taking any element hence True/1
    for i in range(arrLen + 1):
        subset[i][0] = True

    # sum is not zero and set is empty then false
    for i in range(1, requiredSum + 1):
        subset[0][i] = False

    for i in range(1, arrLen + 1):
        for j in range(1, requiredSum + 1):
            if arr[i - 1] > j:
                subset[i][j] = subset[i - 1][j]
            if arr[i - 1] <= j:
                subset[i][j] = subset[i - 1][j] or subset[i - 1][j - arr[i - 1]]

    # uncomment to print the subset
    for i in range(arrLen+1):
        print(subset[i])
    print(subset[arrLen][requiredSum])

#%%
"""
deal 正反序最大连续子串

8
186 186 150 200 160 130 197 200

186 200 160 130 -> len = 4
"""

import bisect
def deal(l,res):
    #每次向b中加一个list中的元素
    b    = [9999]*len(l)
    b[0] = l[0]
    res  = res+[1]
    for i in range(1,len(l)):
        pos =bisect.bisect_left(b,l[i])
        res += [pos+1]
        b[pos]=l[i]
    return res
    
while True:
    try:
        n=int(input())
        s=list(map(int,input().split()))
        dp1=[]
        dp2=[]
        dp1 =deal(s,dp1)#正序遍历位置
        dp2=deal(s[::-1],dp2)[::-1]#逆序遍历位置
        a=max(dp1[i]+dp2[i]for i in range(n))#两次遍历的结果相加
        print(n-a+1)#a中的那个人多加了一次 故要+1
    except:
        break

# %%
"""
素数伴侣
"""

import math

def prime_judge(n):
    m=int(math.sqrt(n))
    if n%2==0:
        return False
    else:
        for i in range(m+1)[3::2]:
            if n%i==0:
                return False
    return True
def group_lst(lst):
    a=[]
    b=[]
    for i in lst:
        if int(i)%2==1:
            a.append(int(i))
        else:
            b.append(int(i))
    return (a,b)
def matrix_ab(a,b):
    matrix=[[0 for i in range(len(b))]for i in range(len(a))]
    for ii,i in enumerate(a):
        for ij,j in enumerate(b):
            if prime_judge(i+j)==True:
                matrix[ii][ij]=1
    return matrix
def find(x):
    for index,i in enumerate(b):
        if matrix[x][index]==1 and used[index]==0:
            used[index]=1
            if connect[index]==-1 or find(connect[index])!=0:
                connect[index]=x
                return 1
    return 0

while True:
    try:
        n=input()
        m=input().split()
        (a,b)=group_lst(m)
        matrix=matrix_ab(a,b)
        connect=[-1 for i in range(len(b))]
        count=0
        for i in range(len(a)):
            used=[0 for j in range(len(b))]
            if find(i):
                count+=1
        print(count)
    except:
        break

# %%
"""
整数与IP地址转化
10.0.3.193
"""
print(int("".join(map(lambda c:bin(c).replace("0b","").rjust(8,"0"),map(int,"10.0.3.193".split(".")))),2))

# %%
"""
偶数分解最接近的素数和 -> 判断是否是素数
"""
import math
 
 
def isPrime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
 
 
while True:
    try:
        num ,start= int(input()) // 2,1
        if num%2==1:
            start=0
        for i in range(start, num, 2):
            a, b = num + i, num - i
            if isPrime(a) and isPrime(b):
                print(b)
                print(a)
                break
 
    except:
        break
    
#%%
"""
递归 -> 1~10 24点四则运算
"""
# 题目的隐藏条件好像是，不考虑使用括号，数字位置可调
def helper(arr, item):
    if item < 1:
        return False
    if len(arr) == 1:
        return arr[0] == item
    for i in range(len(arr)):
        L = arr[:i] + arr[i+1:]
        v = arr[i]
        if helper(L, item-v) or helper(L, item+v) or helper(L, item*v) or helper(L, item/v):
            return True
    return False
while True:
    try:
        arr = list(map(int, input().split(' ')))
        if helper(arr, 24):
            print("true")
        else:
            print("false")
    except:
        break

#%%
"""
扑克牌四则运算
"""

import itertools
data=['K','A','2','4']
map2={'J':'11','Q':'12','K':'13','A':'1'}
new_data=[]
for d in data:
    if d in map2:
        new_data.append(map2[d])
    else:
        new_data.append(d)
 
map1={'0':'+','1':'-','2':'*','3':'/'}   
flag=0
for o in (''.join(x) for x in itertools.product(map(str,range(4)), repeat=3)):
    for i in itertools.permutations(range(4),4):
        temp1='(('+new_data[i[0]]+map1[o[0]]+new_data[i[1]]+')'+map1[o[1]]+new_data[i[2]]+')'+map1[o[2]]+new_data[i[3]]
        temp2=data[i[0]]+map1[o[0]]+data[i[1]]+map1[o[1]]+data[i[2]]+map1[o[2]]+data[i[3]]
        if ('joker' in temp1) or ('JOKER' in temp1):
            flag=1
            print('ERROR')
        elif eval(temp1)==24:
                print(temp2)
                flag=2
                '''
                break
    if flag!=0:
        break'''
if flag==0:
    print('NONE')
    
    
#%%
"""
矩阵乘法
"""
while True:
    try:
        row1=int(input())
        row2=int(input())
        col2=int(input())
        matrix1=[]
        matrix2=[]
        t2=[]
        def muti(s1,s2):
            add=0
            for j in range(len(s1)):
                add+=s1[j]*s2[j]
            return add
        for i in range(row1):
            line=input().split()
            line=[int(x) for x in line]
            matrix1.append(line)
        for i in range(row2):
            line=input().split()
            line=[int(x) for x in line]
            matrix2.append(line)
        for i in range(col2):
            raw=[]
            for j in range(row2):
                raw.append(matrix2[j][i])
            t2.append(raw)
   
        out=[]
        for i in range(row1):
            raw=[]
            for j in range(col2-1):
                print(muti(matrix1[i],t2[j]),end=' ')
            print(muti(matrix1[i],t2[-1]))
    except:
        break


#%%    
"""
百钱买百鸡
"""
while True:
    try:
        num = input()
        for i in range(0,21):
            for j in range(0,34):
                if 5*i + 3*j + (100-i-j)/3.0 == 100:
                    print i,j,100-i-j
    except:
        break
    
#%%
"""
整数的连续奇数和
"""
while True:
    try:
        n = int(raw_input())
        num = n ** 2 - n + 1
        string = str(num)
        for i in range(1, n):
            num = num + 2
            string = string + '+' + str(num)
        print string
    except:
        break
    
#%%
"""
求立方根 牛顿迭代法
"""

while True:
    
    try:
        num_raw = int(input())
        target_num = num_raw
        while target_num*target_num*target_num - num_raw > 0.0001:
            target_num -= (target_num*target_num*target_num-num_raw)/target_num/target_num/3
            
        print('%.1f' % target_num)
    except:
        break
    
import math
while True:
    try:
        num = float(input())
        num2 = math.pow(num, 1.0/3)
        print('%.1f' % num2)
    except:
        break
    
#%%
"""
求最小公倍数 辗转相除法求最大公约数
"""
def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:
        a, b = b, a % b
    return a
def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)
while True:
    try:
        a,b=map(int,input().split())
        print(lcm(a,b))
    except:
        break

import math
math.gcd(12,13) # -> 最大公约数1

# %%
"""
考察数据栈 火车进出
"""


def handle(pre_station, in_station, after_station):
    if not pre_station and not in_station:  # 没有待进站的，也没有待出站的车，一种情况产生了
        result.append(" ".join(after_station))
    else:
        if in_station:  # 出站作业，先检查站内是否有车
            after_station.append(in_station.pop())
            handle(pre_station,in_station,after_station)
            in_station.append(after_station.pop())
        if pre_station:  # 进站作业，先检查是否还有待进站车辆
            in_station.append(pre_station.pop(0))
            handle(pre_station,in_station,after_station)
            pre_station.insert(0,in_station.pop())
 
# count = int(raw_input())  # 火车数量，没有用到，但是是题目输入格式要求，故保留
# row_2 = raw_input()
result = []  # 记录最终数据
# pre_station = [x for x in row_2.split(" ")]  # 待进站的车辆
pre_station = ["1","2","3"]
in_station = []  # 待出站车辆
after_station = []  # 出站后的车辆
handle(pre_station, in_station, after_station)
result.sort() # 要字典序输出，排个序咯
for rs in result:
    print (rs)

#%%
"""
自动售货机 系统工程

"""
import sys
while True:
    try:
        priceGoods = {'A1':2, 'A2':3, 'A3':4, 'A4':5, 'A5':8, 'A6':6}
        priceMoney = [1 , 2 , 5 , 10]
        numGoods = {'A1':0, 'A2':0, 'A3':0, 'A4':0, 'A5':0, 'A6':0}
        numMoney = [0] * 4
         #1 2 5 10
        balance = 0
  
        def printMoney(line):
            print '1 yuan coin number=%s' % (line[0])
            print '2 yuan coin number=%s' % (line[1])
            print '5 yuan coin number=%s' % (line[2])
            print '10 yuan coin number=%s' % (line[3])
        def printGoods(priceGoods,numGoods,flag):# 0:sorted goods name;1:sorted num of goods
            if flag == 0:
                for i in range(6):
                    good = 'A'+str(i+1)
                    print good+' '+str(priceGoods[good])+' '+str(numGoods[good])
            if flag == 1:
                print numGoods
                numGoodsSorted = sorted(numGoods.items(),key = lambda a:a[1],reverse = True)
                for i in range(6):
                    print numGoodsSorted[i][0]+' '+str(priceGoods[numGoodsSorted[i][0]])+' '+str(numGoodsSorted[i][1])
  
        line = raw_input().split(';')[:-1]
        for i in line:
            func = i.split()
            if func[0] == 'r':
                func[1] = func[1].split('-')
                for i in range(6):
                    numGoods['A'+str(i+1)] += int(func[1][i])
                for i in range(4):
                    numMoney[i] += int(func[2].split('-')[i]) #1 2 5 10
                print 'S001:Initialization is successful'
  
            elif func[0] == 'p':
                if int(func[1]) not in priceMoney:
                    print 'E002:Denomination error'
                elif int(func[1]) in [5,10] and numMoney[0] + numMoney[1] * 2 < int(func[1]):
                    print 'E003:Change is not enough, pay fail'
                elif int(func[1]) == 10 and balance > 10:# only print when $10 input
                    print 'E004:Pay the balance is beyond the scope biggest'
                elif numGoods['A1'] == numGoods['A2'] == numGoods['A3'] == numGoods['A4'] == numGoods['A5'] == numGoods['A6'] == 0:
                    print 'E005:All the goods sold out'
                else:
                    numMoney[priceMoney.index(int(func[1]))] += 1
                    balance += int(func[1])
                    print 'S002:Pay success,balance=%d'%(balance)
  
            elif func[0] == 'b':
                if func[1] not in ['A1','A2','A3','A4','A5','A6']:
                    print 'E006:Goods does not exist'
                elif numGoods[func[1]] == 0:
                    print 'E007:The goods sold out'
                elif balance < priceGoods[func[1]]:
                    print 'E008:Lack of balance'
                else:
                    balance -= priceGoods[func[1]]
                    numGoods[func[1]] -= 1
                    print 'S003:Buy success,balance=%d'%(balance)
  
            elif func[0] == 'c':
                if balance == 0:
                    sys.stdout.write('E009:Work failure')#no line break
                else:
                    numCall = [0] * 4 #1 2 5 10
                    for i in range(-1,-5,-1):
                        numCall[i] = min(balance / priceMoney[i] , numMoney[i])
                        balance -= numCall[i] * priceMoney[i]
                        numMoney[i] -= numCall[i]
                    printMoney(numCall)
                    balance = 0
  
            elif func[0] == 'q':
                if func[1] == '0':
                    printGoods(priceGoods,numGoods,1)
                elif func[1] == '1':
                    printMoney(numMoney)
            else:
                sys.stdout.write('E010:Parameter error')#no line break
  
    except:
        break
    
# %%
"""
最长上升子序列

"""
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
        print(len(q))
    except:
        break
    
#%%
"""
计算字符串最后一个单词的长度，单词以空格隔开;
字符串中的count用法;
求一个数的质因数分解;
最大公约数转化为最小公倍数;
求立方根 损失函数; Xn+1 = Xn - f(Xn)/f(Xn)';
最长上升子序列;

"""


#%%
# 计算字符串最后一个单词的长度，单词以空格隔开。
input_str = input("请输入字符串,多组字符串，请用“；”隔开:  ")

temp_procession = input_str

value_list = temp_procession.split(";")
_n = len(value_list)
_caln = 0
print("共%s组字符串！" % _n)
result_list = []
while _caln < _n:
    temp_str = value_list[_caln]
    temp_str_list = temp_str.split(" ")

    print("第%s组字符串最后一个词语为：%s" % (_caln, temp_str_list[-1]))
    _caln += 1
    result_list.append(value_list[-1])
# %%
## 字符串中的count用法
a=input().lower()
b=input().lower()
print(a.count(b))

# %%
## 求一个数的质因数分解
a, res = int(input()), []
for i in range(2, a // 2 + 1):
    while a % i == 0:
        a = a / i
        res.append(i)
print(" ".join(map(str, res)) + " " if res else str(a) + " ")

# %%
# 最大公约数转化为最小公倍数 & 辗转相除法
list_num = input("请输入两个求公倍数的正整数，以逗号隔开:").split(",")
A,B = map(int, list_num)

def gcd(a, b):
    """最大公约数."""
    while b:
        a, b = b, a % b
    return a
def lcm(a, b):
    """最小公倍数."""
    return a * b // gcd(a, b)  #//表示整数除法
while True:
    try:
        a,b=map(int,input().split())
        print(lcm(a,b))
    except:
        break

#%%
# 求立方根 损失函数
# temp = input()
temp = 4
a = float(temp)
r = 0.001
b = a
while abs(b*b*b-a) > r:
    b = b - (b*b*b-a)/b/b/10
print(b)
    
# %%
# 最长上升子序列
# 不停改写列表长度
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

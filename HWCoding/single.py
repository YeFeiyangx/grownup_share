
# %%
def editDistance(str1, str2):
    len1, len2 = len(str1) + 1, len(str2) + 1
    dp = [[0 for i in range(len2)] for j in range(len1)]
    for i in range(len1):
        dp[i][0] = i
    for j in range(len2):
        dp[0][j] = j
    print(dp)
    for i in range(1, len1):
        for j in range(1, len2):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (str1[i - 1] != str2[j - 1]))
    return dp[-1][-1]
 
while True:
    try:
        print("1/" + str(editDistance(input(), input()) + 1))
    except:
        break
    
#%%
from collections import Counter

a = "dsgfuigasdiufguiasdfhjgdhfhsaj"

b = Counter(a)
list(sorted(b.values()))

# %%
for i in set(a):
    print(b[i],i)

# %%
dir(b)

# %%
a = "asdsa"
a.isalnum()

# %%

import math
math.sqrt(4)

# %%
from collections import defaultdict
rabit = dict()
rabit[0] = 0
month = 6
while month > 0:
    max_value = max(rabit.keys())
    temp_dict = dict()
    for i,j in rabit.items():
        if j >= 2:
            temp_dict[max_value+1] = 0
            max_value += 1
        rabit[i] += 1
    rabit.update(temp_dict)
    month -= 1
    print(rabit)
# print(rabit)

# %%n
n = 10
num = n
init_num = n
while n//3 != 0:
    num += n//3
    n = n//3 + n%3
if n == 2:
    num += 1
print(num-init_num)

# %%
from collections import Counter


while True:
    try:
        test_str_input = input("输入您需要测试的字符串")
        target_str_input = input("输入您需要检测的目标字符")
        test_str = Counter(test_str_input)
        if len(target_str_input)==1:
            print(test_str[target_str_input])
        else:
            print("请输入正确的目标字符")
    except:
        print("输入的字符有误")
        break

# %%
from collections import Counter

test_str_input = input("输入您需要测试的字符串")
target_str_input = input("输入您需要检测的目标字符")
test_str_input = test_str_input.lower()
target_str_input = target_str_input.lower()
test_str = Counter(test_str_input)
print(str(test_str[target_str_input]))


# %%
target_str_input

# %%
from collections import Counter

test_str_input = input("输入您需要测试的字符串")
target_str_input = input("输入您需要检测的目标字符")
test_str_input = test_str_input.lower()
target_str_input = target_str_input.lower()
num = test_str_input.count(target_str_input)
print(num)

# %%

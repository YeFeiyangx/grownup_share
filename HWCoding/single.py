
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
from collections import defaultdict
a = "123456"
a = a[::-1]
a[-3:]

# %%
import sys
from collections import defaultdict

record_dict = dict()
index = 0
for line in ["E:\V1R2\product\\fpgadrive.c   1325"]:
    temp_str_input = "E:\V1R2\product\\fpgadrive.c   1325"
    if temp_str_input not in record_dict:
        index += 1
        cal_num = 1
        road_record, line = temp_str_input.split()
        file_name_check = road_record.split("\\")[-1]
        file_name_path = road_record[:-len(file_name_check)]
        if len(file_name_check) > 16:
            file_name_check = file_name_check[-16:]
        file_name_path = file_name_path + file_name_check
        record_dict[temp_str_input] = [index,file_name_path," ",line," ",cal_num]
    
    else:
        record_dict[temp_str_input][-1] += 1
print_list = sorted(record_dict.values())
print(print_list)
for i in print_list:
    print("".join(list(map(str, i[1:]))))


# %%
import sys
from collections import defaultdict
while True:
    try:
        record_dict = dict()
        index = 0
        for line_temp in sys.stdin:
            temp_str_input, line = line_temp.split()
            if "\\" in temp_str_input:
                file_name_check = temp_str_input.split("\\")[-1]
            else:
                file_name_check = temp_str_input
            if len(file_name_check) > 16:
                file_name_check = file_name_check[-16:]     
            if line_temp not in record_dict:
                index += 1
                cal_num = 1
                record_dict[temp_str_input] = [index,file_name_check," ",line," ",cal_num]

#%%
record_dict = dict()
index = 0          
for line_temp in ["G:\\rsle\lsax\yalcxu\\vwhysms 637"]:
    temp_str_input, line = line_temp.split()
    if "\\" in temp_str_input:
        file_name_check = temp_str_input.split("\\")[-1]
    else:
        file_name_check = temp_str_input
    if len(file_name_check) > 16:
        file_name_check = file_name_check[-16:]     
    if line_temp not in record_dict:
        index += 1
        cal_num = 1
        record_dict[temp_str_input] = [index,file_name_check," ",line," ",cal_num]
    else:
        record_dict[temp_str_input][-1] += 1
print_list = sorted(record_dict.values())
for i in print_list[:8]:
    print("".join(list(map(str, i[1:]))))

# %%
import sys
from collections import defaultdict
f = open("D:\\HWCoding\\fortest.txt","r")   
lines = f.readlines()      #读取全部内容 ，并以列表方式返回  
 
record_dict = dict()
index = 10000
for line_temp in lines:
    temp_str_input, line = line_temp.split()
    if "\\" in temp_str_input:
        file_name_check = temp_str_input.split("\\")[-1]
    else:
        file_name_check = temp_str_input
    file_dict_key = file_name_check+line
    if len(file_name_check) > 16:
        file_name_check = file_name_check[-16:]     
    if file_dict_key not in record_dict:
        index -= 1
        cal_num = 1
        record_dict[file_dict_key] = [index,file_name_check," ",line," ",cal_num]
    else:
        record_dict[file_dict_key][-1] += 1
print_list = sorted(record_dict.values())[:8]

for i in print_list[::-1]:
    print("".join(list(map(str, i[1:]))))

# %%
from itertools import combinations
def detection_func(input_str):
    str_length = len(input_str)
    if str_length > 8:
        pass
    else:
        return "NG"
    if str_length == 3:
        pass
    else:
        copy_2_skip = []
        detective_rest_skip = []
        for i in range(str_length-1):
            copy_2_skip.append([i,i+1])
            temp_list = list(range(i))
            temp_list.extend(list(range(i+2, str_length)))
            detective_rest_skip.append(temp_list)
        for i,k in zip(copy_2_skip,detective_rest_skip):
            sub_signal = input_str[i[0]] + input_str[i[1]]
            for j in list(combinations(k,2)):
                if max(j) - min(j) != 1:
                    pass
                else:
                    sub_find = input_str[min(j)] + input_str[max(j)]
                    if sub_signal == sub_find:
                        print("sub_signal","sub_find")
                        return "NG"

    tag_list = set()

    while str_length>0:
        str_length -= 1
        tag = input_str[str_length]
        if tag.isdigit():
            tag_list.add("digit")
        elif tag.isalpha():
            temp_tag = tag
            tag = tag.upper()
            if temp_tag == tag:
                tag_list.add("upalpha")
            else:
                tag_list.add("loweralpha")
        else:
            tag_list.add("special")
    print(tag_list)
    if len(tag_list) >= 3:
        return "OK"
    else:
        return "NG"

    
#%%
print(detection_func("021Abc9Abc1"))


#%%
import sys
output_result = []
for line in sys.stdin:
    temp_value = detection_func(line)
    output_result.append(temp_value)
    
for i in output_result:
    print(i)
    
#%%
from itertools import combinations
def detection_func(input_str):
    input_str = input_str[:-1]
    str_length = len(input_str)
    if str_length > 8:
        pass
    else:
        return "NG"

    copy_2_skip = []
    detective_rest_skip = []
    for i in range(str_length-2):
        copy_2_skip.append([i,i+1,i+2])
        temp_list = list(range(i))
        if len(temp_list)<3:
            temp_list = []
        temp_list_rest = list(range(i+3, str_length))
        if len(temp_list_rest)<3:
            temp_list_rest = []
        temp_list.extend(temp_list_rest)
        detective_rest_skip.append(temp_list)
    for i,k in zip(copy_2_skip,detective_rest_skip):
        sub_signal = input_str[i[0]] + input_str[i[1]] + input_str[i[2]]
        for j in list(combinations(k,3)):
            if max(j) - min(j) != 2:
                pass
            else:
                sub_find = input_str[min(j)] + input_str[min(j)+1] + input_str[max(j)]
                if sub_signal == sub_find:
                    return "NG"
    tag_list = set()
    while str_length>0:
        str_length -= 1
        tag = input_str[str_length]
        if tag.isdigit():
            tag_list.add("digit")
        elif tag.isalpha():
            temp_tag = tag
            tag = tag.upper()
            if temp_tag == tag:
                tag_list.add("upalpha")
            else:
                tag_list.add("loweralpha")
        else:
            tag_list.add("special")

    if len(tag_list) >= 3:
        return "OK"
    else:
        return "NG"
#%%
import sys
f = open("D:\\HWCoding\\fotest2.txt","r")   
lines = f.readlines()      #读取全部内容 ，并以列表方式返回  

output_result = []
for line in lines:
    temp_value = detection_func(line)
    print(temp_value)
    output_result.append(temp_value)
    
# for i in output_result:
#     print(i)

# %%
len(lines[0])

# %%
lines[0][:10]

# %%
Result=[]
for line in sys.stdin:
    str_temp = ""
    for i in line:
        if i.isupper():
            str_temp += recognize_up_dict[i]
        elif i.islower():
            str_temp += recognize_low_dict[i]
        elif i.isdigit():
            str_temp += i
    Result.append(str_temp)
for i in Result:
    print(i)
    
#%%
from collections import Counter

# test_str = input()
test_str = "sjrnqlzzzz"
test_dict = Counter(test_str)

tag_num = min(sorted(test_dict.values()))
used_str = set()
for i,j in test_dict.items():
    if j == tag_num:
        pass
    else:
        used_str.add(i)
output_str = ""
for i in test_str:
    if i in used_str:
        output_str += i

print(output_str)
# %%
a = "3+2*{1+2*\[-4/(8-6)+7\]}"
eval(a)

# %%

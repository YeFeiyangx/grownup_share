#%%
"""
四舍五入
"""
## 要么用decimal,要么使用+0.0001方法处理精度问题
print(round(float(4.5001))) # -> 5
print(round(float(4.50))) # -> 4

# %%
"""
16进制BIT获得
"""

int("0x"+"a",16) # :-> a 转为16进制的BIT

# %%
"""
字符排序 直接就是按照ASCII
"""
while True:
    try:
        print("".join(sorted(input())))
    except:break

# %%
N = 5
max_num = int((N+1)*N/2)
num_list = list(range(1,max_num+1))
output_list = [[] for i in range(N)]
cal_num_end = 0
for i in range(N):
    cal_num_start = cal_num_end
    cal_num_end += i+1
    for j,k in zip(num_list[cal_num_start:cal_num_end],list(range(i+1))[::-1]):
        output_list[k].append(j)
for i in output_list:
    print(i)
    
#%%
#%%
import collections

#%%
# ------collections.namedtuple------
"""
无需创建类对象，可直接获得能使用.X的方法调用的元祖
详见 test_list[0].info_0
"""

test_task_type = collections.namedtuple('test_title',
                                                'info_0 info_1 info_2 info_3')

test_list = []


for i in range(10):

    test_list.append(test_task_type(info_0=1, info_1=2, info_2=3, info_3=4))                               

print("test_list:\n",test_list)

# %%
print(dir(test_list[0]))
test_list[0].info_0


# %%
"""
字典的键值添加
test_dict[i,j] = i
"""
test_dict = {}
for i,j in zip(range(4),range(4)):
    # @@ 不需要元祖括号，可直接形成元祖
    test_dict[i,j] = i

print(test_dict)

# %%
"""
字符对象的创建，也可以通过 %i 的方式来
"""
## %i 和 %d 表示一致

suffix = '_%i_%i' % (0, 1)

print(suffix)

# %%

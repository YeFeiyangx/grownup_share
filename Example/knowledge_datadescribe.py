# %%
"""
数据描述符
是用于类中，对数据描述类进行实例化，然后进行相关数据验证，再返回设定值
"""
class atest:                                        # 1
    def __init__(self,a,b):                         # 2 # 7
        self._value = None                              # 7+
        print(a,b)                                      # 7++

    def __get__(self,instance,owner):               # 3 # 10+
        """
        只有在调用属性的时候才会用到，即a.name
        owner -> 指向调用的原类方法 fatherclass
        """
        print("owner:",owner)                   
        print("self._value:",self._value)           # 10++
        return self._value                          # 10+++
    def __set__(self,instance,value):               # 4 # 9+
        print("instance:", instance)                # 9++
        self._value = value                         # 9+++

class fatherclass:                                  # 5
    name = atest(4,5)                               # 6
    # def __init__(self,c):
    #     print(c)

a = fatherclass()                                   # 8         

a.name = 4                                          # 9

print("a.name:",a.name)                             # 10 # 11
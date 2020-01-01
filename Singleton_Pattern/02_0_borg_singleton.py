__author__ = ['Chetan',"IvanYoung"]
#%%
"""
采用__init__方法
调用类（Borg）的属性方法，self.__dict__，这是个属于Borg的对象属性，而不是类的实例化属性。
所有调用Borg进行的实例化，都将继承Borg的属性；
同时，这属于动态数据类型，所有实例化的类共享并同步该属性，不区分创建时间早晚！

_shared_state = {}
obj.__dict__ = cls._shared_state

单例模式，主要是为了共享所有该类的属性数据。可以想象为全局变量
"""

class Borg:
    __shared_state = {"1":"2"}
    def __init__(self):
        # self.x = 1
        self.__dict__ = self.__shared_state
        pass

b = Borg()
b1 = Borg()
b.x = 4

print("Borg Object 'b': ", b)  ## b and b1 are distinct objects
print("Borg Object 'b1': ", b1)
print("Object State 'b':", b.__dict__) ## b and b1 share same state
print("Object State 'b1':", b1.__dict__)

# %%

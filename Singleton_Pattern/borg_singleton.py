__author__ = 'Chetan'
#%%
"""
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

__author__ = 'Chetan'
#%%
"""创建同一个内存空间的对象"""
class Singleton(object):
     
    def __new__(cls):
       if not hasattr(cls, 'instance'):
        #  cls.instance = super(Singleton, cls).__new__(cls)
         cls.instance = super().__new__(cls)
       return cls.instance

s = Singleton()
print("Object created", s)

s1 = Singleton()
print("Object created", s1)

#%%
"""对super拓展一下"""
class A:
  def add(self, x):
      y = x+1
      print(y)
class B(A):
  def add(self, x):
      super().add(x*x)
b = B()
b.add(2)  # 5 = 1 + 2*2

# %%
class A:
  def add(self, x):
      y = x+1
      print(y)
class B(A):
  def add(self, x):
      super().add(x)
b = B()
b.add(2)  # 3 = 1 + 2
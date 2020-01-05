__author__ = ['Chetan',"IvanYoung"]
#%%
"""
用VSCODE可以获得更好的调试体验
VSCODE-Python套装见#TODO
用__new__和if not hasattr(cls, 'instance')方法 \
创建同一个内存空间的不同名称的实例化对象
这个是对于单个类属性继承而言的，拓展性差，仅供理解
"""
class Singleton(object):
    ## __new__方法优先级大于__init__,创建一个类的类实例化对象，返回给Singleton
    def __new__(cls):
        ## cls -> class '__main__.Singleton'
        # print(cls)
        if not hasattr(cls, 'instance'):
            # cls.instance = super(Singleton, cls).__new__(cls) -> python 2.X +
            cls.instance = super().__new__(cls)     # python -> 3.X +
        # 造成实例化的对象 指向该类本身，同时具有该实例化对象属性具有instance属性，\
        # 也指向本身
        return cls.instance

s = Singleton()
print("Object created", s)
print(s.instance)
s1 = Singleton()
print("Object created", s1)

#%%
"""
对super拓展一下
先进行 x^2的计算，然后再向上搜索进入到父类继承属性 X(-> x^2) + 1
继承相当重要，面对对象四特性 封装，继承，多态，抽象（JAVA->接口）
"""
class A:
    def add(self, X):
        y = X+1
        print(y)
class B(A):
    def add(self, x):
        super().add(x*x)

b = B()
b.add(2)  # 5 = 2*2 + 1

class C(B):
    def add(self, x):
        xt = pow(x,3)
        super().add(xt)
b = C()
b.add(2)  # 65 = (2^3)^2 +1 -> 2^3 -> (2^3)^2 -> (2^3)^2 +1

# %%

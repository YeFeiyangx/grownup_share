__author__ = ['Chetan',"IvanYoung"]

#%%
"""
元类概念 元类的基类一定要用 type: cls,*args,**kwds;
*arg -> name[类目],base[基类],attrs[内属性] !


延申 __call__ 可以用来写 装饰器；
[闭包]<https://foofish.net/python-closure.html>；
"""
## 右侧调用顺序，为元编程特有
class MyInt(type):
    def __call__(cls, *args, **kwds):                           # 2
        print("***** Here's My int *****", args)                # 3
        print("Now do whatever you want with these objects...") # 4
        print(cls)
        return type.__call__(cls, *args, **kwds)                # 5 # 9

## metaclass 创建类实例
class int(metaclass=MyInt):
    def __init__(self, x, y):                                   # 6
        self.x = x                                              # 7
        self.y = y                                              # 8

## 元类编程中__call__方法直接付给类，而不是实例，所以运行类实例化的时候，__call__触发
i = int(4,5)                                                    # 1 # 10

#%%
"""
__call__方法案例
"""
class TestFun:
    def __init__(self,*args):
        n = 0
        for i in args:
            n += i
        self.n = n
        print("__init__",self.n)

    def __call__(self,*args):
        print("__call__:",args)
        # return "why here?"  ## 如果__init__中有func，且return该func,就能成为装饰器
        return self.__init__(*args)

a = TestFun(4)
a(4) # -> __init__方法无return（过程中会执行中的print），最后返回None
print(a(4)) # -> None

#%%
"""
属性描述符案例
[__get__,__set__]<https://blog.csdn.net/sjyttkl/article/details/80655421>
"""
class RevealAccess:
    def __init__(self,initval = None,name='var'):
        self.val = initval
        self.name = name

    def __get__(self, instance, owner):
        print("获取..",self.name)
        return self.val

    def __set__(self, instance, value):
        print("设置值：",self.name)
        self.val = value

class MyClass:
    x = RevealAccess(10,"var 'x'") # 开始的时候set self.name和默认value
    y = 5

    def __init__(self):
       self.x = 'self x'
        # pass
m = MyClass()   ## 实例化的时候，进入self.x触发 -> set方法 -> 默认self.val设置为str"self x"
print(m.x)      ## __get__方法获取
print("-----------------")
m1 = MyClass()
m1.x = 20       ## __set__方法设置
print(m1.x)     ## __get__方法获取


# %%
"""
装饰器
[装饰器]<https://foofish.net/decorator.html>
"""
class Counter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@Counter
def foo():
    pass

for i in range(10):
    foo()

print(foo.count)  # 10


# %%

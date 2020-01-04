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
a(4)

#%%
class MyInt(type):
    def __new__(cls, name,bases,attrs, **kwds):
        print("***** Here's My name *****", name)
        print("***** Here's My bases *****", bases)
        print("***** Here's My attrs *****", attrs)
        print("***** Here's My kwds *****", kwds)
        print("Now do whatever you want with these objects...")
        return type.__new__(cls, name, bases, attrs, **kwds)

def testfunc():
    pass

class testclass:
    def __init__(self, x, y):
        self._value = None
        self.col1 = x
        self.col2 = y
        print("====self.col1====:",self.col1)
        print("====self.col2====:",self.col2)

    def __get__(self,instance,owner):
        print("self._value",self._value)
        return self._value
    def __set__(self,instance,value):
        print("instance:", instance)
        self._value = value

## metaclass 创建类实例
class int(metaclass=MyInt):
    test = testfunc()
    test1 = testclass(x=1,y=2)
    def __init__(self, x, y,z=None):
        self.x = x
        self.y = y
        self.z = z
        print("self.__dict__:", self.__dict__)
        print("int.__dict__:", int.__dict__)
        print(help(int.__dict__["__init__"]))
        print(help(type(self)))
        print(int.__dict__["__dict__"])
        print(int.__dict__["__weakref__"])

i = int(4,5, z=6)
print(dir(i))
# ***** Here's My int ***** (4, 5)
# Now do whatever you want with these objects...


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

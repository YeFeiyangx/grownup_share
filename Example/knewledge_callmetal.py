# %%
"""
元类编程只能用 type作为父类
https://www.cnblogs.com/busui/p/7283137.html?utm_source=itdadao&utm_medium=referral
"""
class MyInt(type):
    
    def __call__(cls, *args, **kwds):
        print("call**call@@"*10)
        print("***** Here's My cls *****", cls)
        print("***** Here's My int *****", args)
        print("***** Here's My kwds *****", kwds)
        print("Now do whatever you want with these objects...")
        # return type.__call__(cls, *args, **kwds) ## i = int(4,5) -> object
        return 1000 ## i = int(4,5) -> int 跳出__init__

    def __new__(cls, name,bases,attrs, **kwds):
        a = 4
        print("new**new@@"*10)
        print("a:",a)
        print("***** Here's My cls *****", cls)
        attrs["a"] = a
        print("***** Here's My int *****", attrs)
        print("***** Here's My kwds *****", kwds)
        print("Now do whatever you want with these objects...")
        return super().__new__(cls, name,bases,attrs, **kwds)

## metaclass 创建类实例
class int(metaclass=MyInt):
    print("int**int@@"*10)
    # print(dir(int))
    def __init__(self, x, y, z = None):
        self.x = x
        self.y = y
        self.z = z
        print("self.__dict__:", self.__dict__)
        print("int.__dict__:", int.__dict__)
        print(help(int.__dict__["__init__"]))
        # print(help(type(self)))
        print(int.__dict__["__dict__"])
        print(int.__dict__["__weakref__"])

i = int(4,5)
print(i)
print(dir(i))


# %%

__author__ = 'Chetan'
#%%
"""
元类概念

"""
class MyInt(type):
    
    # def __call__(cls, *args, **kwds):
    #     print("***** Here's My int *****", args)
    #     print("Now do whatever you want with these objects...")
    #     return type.__call__(cls, *args, **kwds)
    def __new__(cls, name,bases,attrs, **kwds):
        print("***** Here's My name *****", name)
        print("***** Here's My bases *****", bases)
        print("***** Here's My attrs *****", attrs)
        print("***** Here's My kwds *****", kwds)
        print("Now do whatever you want with these objects...")
        return type.__new__(cls, name,bases,attrs, **kwds)

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

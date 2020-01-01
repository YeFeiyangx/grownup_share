__author__ = ['Chetan',"IvanYoung"]
#%%
"""
创建不同空间的内存对象，但是内部是同一内存地址的动态数据类型(dict)
_shared_state = {}
obj.__dict__ = cls._shared_state
单例模式，主要是为了共享所有该类的属性数据。可以想象未全局变量
"""

class Borg(object):
    _shared_state = {}
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_state
        return obj

b = Borg()
b1 = Borg()
b.x = 4

print("Borg Object 'b': ", b)  ## b and b1 are distinct objects
print("Borg Object 'b1': ", b1)
print("Object State 'b':", b.__dict__) ## b and b1 share same state
print("Object State 'b1':", b1.__dict__)

#%%
class Borg(object):
    _shared_state = {}
    # def __init__(self, a, b):
    #   print("__init__")
    #   print("a:",a)
    #   print("b:",b)
  
    def __new__(cls, a, b):
        cls.a = a
        cls.b = b
        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_state
        return obj
 
b = Borg(5,6)
b1 = Borg(7,8)
b.x = 4

print(b)
print(b.__dict__)
print(dir(b))


#%%
## 嘿嘿嘿顺路看下去就通透了
class Borg(int):
    _shared_state = {}
    def __init__(self, a, *args, **kwargs):
        for i in args:
            print(i)
    def __new__(cls, a, *args, **kwargs):
        obj = super().__new__(cls, a, *args, **kwargs)
        obj.__dict__ = cls._shared_state ## 起到共享的主要语句
        return obj
    def testSysStep(self, var_a, var_b):
        self.total = var_a + var_b
        print(self.total)

b = Borg(5)
b1 = Borg(6)
b.testSysStep(6,60)
b1.testSysStep(8,80)
b.x = 4
print("dir(b):",dir(b))
print("Borg Int 'b': ", b)  ## b and b1 are distinct objects
# Borg Int 'b':  5
print("Borg Int 'b1': ", b1)
# Borg Int 'b1':  6

print("Object State 'b':", b.__dict__) ## b and b1 share same state
# Object State 'b': {'x': 4}
print("Object State 'b1':", b1.__dict__)
# Object State 'b1': {'x': 4}

print("==*=="*10)
"""
object和int差在一些类方法的继承上
"""
class Borg(object):
    _shared_state = {}
    def __init__(self, a, *args, **kwargs):
        print("a:",a)
        self.a = a
        for i in args:
            print(i)
    def __new__(cls, a, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_state
        return obj

b = Borg(5)
b1 = Borg(6)
b.x = 4

print("Borg Object 'b': ", b)  ## b and b1 are distinct objects
# Borg Object 'b':  <__main__.Borg object at 0x000002317A152EF0>
print("Borg Object 'b1': ", b1)
# Borg Object 'b1':  <__main__.Borg object at 0x000002317B54DD68>

print("Object State 'b':", b.__dict__) ## b and b1 share same state
# Object State 'b': {'x': 4}
print("Object State 'b1':", b1.__dict__)
# Object State 'b1': {'x': 4}

#%%
"""
是他是她就是它，把介个玩意搞定啦，啦啦啦。
"""
class Borg(object):
    _shared_state = {}
    def __init__(self, *args, **kwargs):
        pass
    def __new__(cls, *args, **kwargs):

        for i in args:
            setattr(cls, str(i), args[i])

        for i,j in kwargs.items():
            setattr(cls, i, j)

        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_state
        return obj

b = Borg(a = 3)
b1 = Borg(c = 5)
b.x = 4

print(b)
print(b1)
print("dir(b):",dir(b),"\n", b.c)
print("dir(b1):",dir(b1),"\n", b.a)


# %%
class Borg(object):
    _shared_state = {}
    def __new__(cls, x):
        cls.a = x
        print(x)
        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_state
        return obj

b = Borg(5)
b1 = Borg(6)
b.x = 4

print("dir(b):",dir(b))
print("dir(b):",dir(b1))
print("Borg Object 'b': ", b)             ## 5
print("Borg Object 'b1': ", b1)           ## 6
print("Object State 'b':", b.__dict__)    ## Object State 'b': {'x': 4}
print("Object State 'b1':", b1.__dict__)  ## Object State 'b1': {'x': 4}
print("b.a",b.a)
print("b1.a",b1.a)

# %%

class CapStr(str):
    def __new__(cls,string):
        if isinstance(string, str):
            string2 = string.upper()
        else:
            string2 = "Not Str Type!"
        return super().__new__(cls,string2)
 
a = CapStr("I love China!")

print(a)  # I LOVE CHINA!

b =  CapStr(1)
print(b)  # Not Str Type!

print("===="*10)

class CapStr2(str):
    def __new__(cls,string):
        if isinstance(string, str):
            string2 = string.upper()
        else:
            string2 = "Not Str Type!"
        return super().__new__(cls,string2)
    def __init__(self, a):
        self.strings = a
        print(self.strings) # 1
c = CapStr2(1)
print(type(c),c)          # <class '__main__.CapStr2'> Not Str Type!
print(c.strings)          # 1


# %%
class TestPrivate:
    def __init__(self):
        self.__testattr = "nihao"

a = TestPrivate()
a._TestPrivate__testattr



# %%

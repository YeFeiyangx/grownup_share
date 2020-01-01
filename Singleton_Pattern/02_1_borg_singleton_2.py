__author__ = ['Chetan',"IvanYoung"]
#%%
"""
懒汉模式

类采用 __new__ 方法 input: -> any
ctrl+左键 点new看源码
但是，元类编程 type:-> *arg -> name[类目],base[基类],attrs[内属性] !

创建不同空间的内存对象，但是内部是同一内存地址的动态数据类型(dict)

先通过继承的方式，获取原类属性；
调用类的__dict__，共享类属性cls._shared_state；
返回继承的原类

P.S. 通过obj.__dict__ = cls._shared_state共享内存地址
即每个实例的Borg的__dict__是同一地址，但是实例却不是！

obj.__dict__ = cls._shared_state
单例模式的一种，主要是为了共享所有该类的属性数据。可以想象为全局变量
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
print(dir(Borg))
print(dir(b))
print("Borg Object 'b': ", b)  ## b and b1 are distinct objects
print("Borg Object 'b1': ", b1)
print("Object State 'b':", b.__dict__) ## b and b1 share same state
print("Object State 'b1':", b1.__dict__)
print("Object State id'b1':", id(b.__dict__)) ## b and b1 share same state
print("Object State id'b1':", id(b1.__dict__))

#%%
class Borg(object):
    _shared_state = {}
    def __new__(cls, a, b, *arg, **kwargs):         # def __new__(cls, a, b):
        obj = super().__new__(cls,*arg,**kwargs)    # obj = super().__new__(cls) 
        obj.__dict__ = cls._shared_state
        obj.__dict__["a"] = a
        obj.__dict__["b"] = b
        return obj

b = Borg(5,6)
b1 = Borg(7,8)
b.x = 4

print(b)
print(b.__dict__)
print(dir(b))


#%%
## 嘿嘿嘿顺路看下去就通透了
print("==="*5+"step1"+"==="*5)
class Borg(object):
    _shared_state = {}
    def __init__(self, *args, **kwargs):
        for i in args:
            print("__init__:",i)
    def __new__(cls, *args, **kwargs):
        print("cls:",cls)
        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_state ## 起到共享的主要语句
        for i in args:
            obj.__dict__[str(i)] = i     ## dir会读取__dict__,这东西里头不能有数值类型的数据哈！
        for i,j in kwargs:
            obj.__dict__[i] = j
        return obj
    def testSysStep(self, var_a, var_b):
        self.total = var_a + var_b
        print("testSysStep:",self.total)

b = Borg(5,7,8)
b1 = Borg(6)
b.testSysStep(6,60)
b1.testSysStep(8,80)
b.x = 4
"""
'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 
'numerator', 'real', 'testSysStep', 'to_bytes', 'total', 'x'
"""
print("Borg:",dir(Borg))
print("dir(b):", dir(b))

print("Borg Int 'b': ", b)  ## b and b1 are distinct objects
print("Borg Int 'b1': ", b1)

print("Object State 'b':", b.__dict__) ## b and b1 share same state
# {'5': 5, '7': 7, '8': 8, '6': 6, 'total': 88, 'x': 4}
print("Object State 'b1':", b1.__dict__)
# {'5': 5, '7': 7, '8': 8, '6': 6, 'total': 88, 'x': 4}

#%%
"""
观察__new__中的arg和kwargs

"""
print("==="*5+"step2"+"==="*5)
class Borg(object):
    _shared_state = {}
    # def __init__(self, a, *args, **kwargs):
    def __init__(self, a):
        print("a:",a)
        self.a = a
    def __new__(cls, a, *args, **kwargs):
        print("=====args:",args)
        print("=====kwargs:",kwargs)
        obj = super().__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_state
        obj.__dict__[str(a)] = a
        return obj

b = Borg(5)

#%%
"""
采用cls.var = value的方式 和 setattr(cls,"var",value的方式)  -> 进dir不进__dict__
与obj.__dict__[var] = value的区别                           -> 都进

"""
print("==="*5+"step3"+"==="*5)
class Borg(object):
    _shared_state = {}
    def __new__(cls, x):
        cls.a = x # "a"直接进入dir
                  # setattr(cls,"a",x) 这个要这样
                  # 不进入__dict__
        setattr(cls,"aa",x)
        print("test=====",dir(cls))
        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_state
        obj.__dict__["aaa"] = x ## 进入__dict__
        return obj

b = Borg(5)
b.x = 4

print("dir(b):",dir(b))
print(" b.__dict__:", b.__dict__)
print("Borg Object 'b': ", b)
print("b.a",b.a)


#%%
"""
以**kwargs巩固
"""
print("==="*5+"step4"+"==="*5)
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
b.x = 4

print(b)
print("dir(b):",dir(b),"\n", b.c)

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

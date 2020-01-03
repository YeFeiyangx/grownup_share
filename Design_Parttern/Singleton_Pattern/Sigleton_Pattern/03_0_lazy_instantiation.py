__author__ = ['Chetan',"IvanYoung"]

#%%
"""
采用类的静态方法 classmethod；
实例化的时候并未创建，调用静态方法时方能创建；
下次再实例化的时候，直接从if-else转出去。
"""

class Singleton:
    
    __instance = None
    
    def __init__(self):
        if not Singleton.__instance:
            print(" __init__ method called..")
        else:
            print("Instance already created:", self.getInstance()) ## 用于第二次实例化的时候显示地址
    
    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = Singleton()
        return cls.__instance

s = Singleton() ## class initialized, but object not created
print("Object created", Singleton.getInstance()) ## Gets created here
s1 = Singleton() ## instance already created
print(id(s.getInstance()))
print(id(s1.getInstance()))
#%%
A = 4

class Singleton:
    
    __instance = None
    
    def __init__(self):
        if not Singleton.__instance:
            print(" __init__ method called..")
        else:
            print("Instance already created:", self.getInstance())
    
    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = 4
        return cls.__instance
# Singleton.__instance 无法获得 
s = Singleton() ## class initialized, but object not created
print("Object created", Singleton.getInstance(),id(Singleton.getInstance())) ## Gets created here
s1 = Singleton() ## instance already created
print(id(s.getInstance()))
id(s1.getInstance())
# %%


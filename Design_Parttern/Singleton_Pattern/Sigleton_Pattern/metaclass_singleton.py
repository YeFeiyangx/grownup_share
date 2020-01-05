__author__ = 'Chetan'
#%%
"""
元编程中的懒汉模式
"""
class MetaSingleton(type):
    
    _instances = {}
    def __call__(cls, *args, **kwargs):
        print("cls:",cls)
        print("args:",args)
        print("kwargs:",kwargs)
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton,cls).__call__(*args, **kwargs)
            print("cls._instances:",cls._instances)
        return cls._instances[cls]

class Logger(metaclass=MetaSingleton):
    pass
class Pullclass(metaclass=MetaSingleton):
    pass
logger1 = Logger()
logger2 = Logger()
logger3 = Pullclass()
logger4 = Pullclass()
print(logger1, logger2) # -> 同一内存地址
print(logger3, logger4) # -> 同一内存地址，但与loger1不同

#%%
class MetaSingleton(type):
    
    _instances = {}
    def __call__(cls, *args, **kwargs):
        print("cls:",cls)
        print("args:",args)
        print("kwargs:",kwargs)
        if cls not in cls._instances and len(cls._instances)==0:
            cls._instances[cls] = super(MetaSingleton,cls).__call__(*args, **kwargs)
            print("cls._instances:",cls._instances)
        # else:
        #     cls,value = cls._instances.items
        cls,value = list(cls._instances.items())[0]
        print("else,cls:",cls)
        print("value:",value)

        return value

class Logger(metaclass=MetaSingleton):
    pass
class Pullclass(metaclass=MetaSingleton):
    pass
logger1 = Logger()
logger2 = Logger()
logger3 = Pullclass()
logger4 = Pullclass()
# 四个都为同一内存地址
print(logger1, logger2)
print(logger3, logger4)


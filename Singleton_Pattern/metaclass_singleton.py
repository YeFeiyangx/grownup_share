__author__ = 'Chetan'
#%%
class MetaSingleton(type):
    
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=MetaSingleton):
    pass
class Pullclass(metaclass=MetaSingleton):
    pass
logger1 = Logger("a","b")
logger2 = Logger()
logger3 = Pullclass()
logger4 = Pullclass()
print(logger1, logger2)
print(logger3, logger4)

# %%

__author__ = ['Chetan',"IvanYoung"]

#%%
import sqlite3 # anaconda 3.7python么有sqlite3

# [解决方案]<https://blog.csdn.net/frostime/article/details/86762858>

# [sqlite3略读]<https://blog.csdn.net/sinat_35886587/article/details/80561959>

class MetaSingleton(type):
    
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Database(metaclass=MetaSingleton):
    connection = None
    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect("db.sqlite3")
            self.cursorobj = self.connection.cursor()           # 建立游标，开启操作
        return self.cursorobj

db1 = Database().connect()
db2 = Database().connect()

## 链接同一个数据库
print ("Database Objects DB1", db1)
print ("Database Objects DB2", db2)




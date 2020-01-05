import numbers
"""
MyOrm
使用元类编程
数据描述符
继承
__new__
"""

class Field:            # 0：过；2：过；
    pass                # 1：过；

class IntField(Field):  # 3：过；7：过；
    """
    数据描述符
    """
    def __init__(self, db_column, min_value=None, max_value=None):  # 4：过；
        self._value = None                                          # 32：过;
        self.min_value = min_value                                  # 33：过;
        self.max_value = max_value                                  # 34：过;
        self.db_column = db_column                                  # 35：过;
        
        if min_value is not None:                                   # 36：过 (下述皆过至==*==);
            if not isinstance(min_value, numbers.Integral):
                raise ValueError("min_value must be int")
            elif min_value < 0:
                raise ValueError("min_value must be positive int")
        if max_value is not None:
            if not isinstance(max_value, numbers.Integral):
                raise ValueError("max_value must be int")
            elif max_value < 0:
                raise ValueError("max_value must be positive int")
        if max_value is not None and min_value is not None:
            if min_value > max_value:
                raise ValueError("min_value must be smaller than max_value") # 37：过 (上述皆过至==*==);

    def __get__(self, instance, owner):         # 5：过；
        return self._value

    def __set__(self, instance, value):         # 6：过；
                                                # 56：self.db_column:'age_col' \ self.max_value:100 \ self.min_value:1 \ 
                                                # self._value:None \ value:28 \ instance:<__main__.User object at 0x000001891ED62C50>

        if not isinstance(value, numbers.Integral):
            raise ValueError("int value need")
        if value < self.min_value or value > self.max_value:
            raise ValueError("value must between min_value and max_value")
        self._value = value                         # 57：self._value:28；


class CharField(Field):                             # 8：过；12：过；
    """
    数据描述符
    """
    def __init__(self, db_column, max_length=None): # 9：过；
        
        self._value = None                          # 26：db_column="name_col", max_length =10;
        
        self.db_column = db_column                  # 27：self.db_column="name_col"；
        if max_length is None:                      # 28：过；
            raise ValueError("you must spcify max_lenth for charfiled")
        self.max_length = max_length                # 29：self.max_length=10；

    def __get__(self, instance, owner):             # 10：过；
        return self._value

    def __set__(self, instance, value):             # 11：过；
         
        if not isinstance(value, str):              # 52：self.db_column:'name_col' \ self.max_length:10 \ self._value:None \
                                                    # value:'bobby' \ instance:<__main__.User object at 0x000001891ED62C50>
            raise ValueError("string value need")
        if len(value) > self.max_length:
            raise ValueError("value len excess len of max_length")  # 53：过；
        self._value = value                                         # 54：self._value:'bobby'；

class ModelMetaClass(type):                                         # 13：过；# 15：过；

    def __new__(cls, name, bases, attrs, **kwargs):   # 14：过；
                                                      # 20：name="BaseModel",；attrs:{'__classcell__': <cell at 0x000001E03...38: empty>, \
                                                      # '__init__': <function BaseModel....030CF2378>, '__module__': '__main__', \
                                                      # '__qualname__': 'BaseModel', 'save': <function BaseModel....030C9CAE8>}
                                                      # 20 -> 能有这么多内容，主要源于他是元类
        if name == "BaseModel": # 21：过；
                                # 38：name='User',{'Meta': <class '__main__.User.Meta'>, '__module__': '__main__', \
                                # '__qualname__': 'User', 'age': <__main__.IntField o...91ED9DC18>, 'name': <__main__.CharField ...91ED19C88>};
            
            return super().__new__(cls, name, bases, attrs, **kwargs)    # 22：过
        fields = {}
        
        for key, value in attrs.items():        # 39：{'Meta': <class '__main__.User.Meta'>, '__module__': '__main__', '__qualname__': 'User', \
                                                # 'age': <__main__.IntField o...91ED9DC18>, 'name': <__main__.CharField ...91ED19C88>}；
            if isinstance(value, Field):
                
                fields[key] = value             # 40：fields：{'age': <__main__.IntField o...91ED9DC18>, 'name': <__main__.CharField ...91ED19C88>}
        
        attrs_meta = attrs.get("Meta", None)    # 41：attrs_meta -> <class '__main__.User.Meta'>；
        _meta = {}
        db_table = name.lower()       # 42：db_table: 'User'->'user'；
        if attrs_meta is not None:
            
            table = getattr(attrs_meta, "db_table", None) # 43：table -> <class '__main__.User.Meta.db_table'> -> 'user_2'；
            if table is not None:
                db_table = table      # 44：db_table: table -> 'user_2'；
       
        _meta["db_table"] = db_table  # 45：_meta：{'db_table': 'user_2'}；
        attrs["_meta"] = _meta
        attrs["fields"] = fields
          
        del attrs["Meta"] # 46：attrs：{'__module__': '__main__', '__qualname__': 'User', \
                          # '_meta': {'db_table': 'user_2'}, 'age': <__main__.IntField o...91ED9DC18>, \
                          # 'fields': {'age': <__main__.IntField o...91ED9DC18>, 'name': <__main__.CharField ...91ED19C88>}, \
                          # 'name': <__main__.CharField ...91ED19C88>}；
        
        return super().__new__(cls, name, bases, attrs, **kwargs) # 47：name='User',bases=(<class '__main__.BaseModel'>,),attrs(46.)

#! ModelMetaClass 一定要在之前创建


class BaseModel(metaclass=ModelMetaClass):          # 16：过；# 19：过；# 23：过
    def __init__(self, *args, **kwargs):            # 17：过；
        for key, value in kwargs.items():           # 51：kwargs：{'age': 28, 'name': 'bobby'} \ args:()；
                                                    # 55：过；
            
            setattr(self, key, value) # 52：self.age: None \ self.name: 'bobby' \ self：<__main__.User object at 0x000001891ED62C50> \
                                      # self.fields:{'age': <__main__.IntField o...91ED9DC18>, 'name': <__main__.CharField ...91ED19C88>} \
                                      # self._meta：{'db_table': 'user_2'}

                                      # 58：self.age: 28 \ self.name: 'bobby' \ self：<__main__.User object at 0x000001891ED62C50> \ 
                                      # self.fields:{'age': <__main__.IntField o...91ED9DC18>, 'name': <__main__.CharField ...91ED19C88>} \ 
                                      # self._meta：{'db_table': 'user_2'}
        
        return super().__init__()     # 59：arg:() \ kwargs(51.) \ self(58) \ <class '__main__.BaseModel'>

    def save(self):                   # 18：过；60：(58.);
        fields = []
        values = []
        
        for key, value in self.fields.items():                  # 61：self.fields:{'age': <__main__.IntField o...91ED9DC18>, \
                                                                # 'name': <__main__.CharField ...91ED19C88>} -> 内存空间有之前的值；
            
            db_column = value.db_column                         # 62：key:'name',value:value.db_column:'name_col' \
                                                                # value.max_length:10 \ value._value:"bobby"；

            if db_column is None:                               # 67：key:'age',value:value.db_column:'age_col' \ value.min_value:1 \
                                                                # value.max_value:100 \ value._value:28；

                db_column = key.lower()                         # 63：db_column:'name_col'；68：db_column:'age_col'；
            
            fields.append(db_column)                            # 64：fields:['name_col']；69：fields:['name_col','age_col']；
            
            value = getattr(self, key)                          # 65：value -> self.name -> 'bobby'；70：value -> self.age -> 28；
            
            values.append(str(value))                           # 66：values:['bobby']；71：values:['bobby', '28']；

        sql = "insert {db_table}({fields})value({values})".format(db_table=self._meta["db_table"],
                                                                  fields=",".join(fields), values=",".join(values))
        pass

#! 类但凡是实例化之前，无论在哪里被实例化，一定要在提前创建


class User(BaseModel):                                                  # 24：过；48：过；

    
    name = CharField(db_column="name_col", max_length=10)               # 25：过；30：过；
    age = IntField(db_column="age_col", min_value=1, max_value=100)     # 31：过；

    class Meta:                                                         # 37+：过；# 37+++：过；
        db_table = "user_2"                                             # 37++：过；


if __name__ == "__main__":                  # 49：过；
    user = User(name="bobby", age=28)       # 50：过；
    user.save()                             # 50+：过；
    print(User.__mro__)

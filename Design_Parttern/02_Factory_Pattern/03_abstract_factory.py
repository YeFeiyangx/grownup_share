__author__ = 'Chetan'
from abc import ABCMeta, abstractmethod
"""
抽象工厂模式

[抽象类方法]<https://www.cnblogs.com/baxianhua/p/10876181.html>

含抽象类方法的不能实例化

1. 搞两种pizza大类，印度的和美国的,两大类下各有创建小类的方法 见4；
2. 两大类里分别有创建荤pizza和素pizza；

3. 建立荤素两种抽象pizza；
4. 为每一个pizza大类建立pizza小类，分别为一荤一素，共四类；
    [DeluxVeggiePizza,ChickenPizza]->Indian,
    [MexicanVegPizza,HamPizza]->USA；

5. 将三个基类进行组合 -> 可以搞以系列为级别的披萨。

抽象工厂 与 工厂方法 区别在于：
前者使用组合，创建相关产品的系列，一或多种创建方法；
后者使用继承，创建一个产品，仅一种创建方法。
"""



class PizzaFactory(metaclass=ABCMeta):
    """
    抽象工厂
    """
    @abstractmethod
    def createVegPizza(self):
        pass
    
    @abstractmethod
    def createNonVegPizza(self):
        pass

class IndianPizzaFactory(PizzaFactory):
    """
    实例工厂
    """
    def createVegPizza(self):
        return DeluxVeggiePizza()
    def createNonVegPizza(self):
        return ChickenPizza()

class USPizzaFactory(PizzaFactory):
    """
    实例工厂
    """
    def createVegPizza(self):
        return MexicanVegPizza()
    def createNonVegPizza(self):
        return HamPizza()

class VegPizza(metaclass=ABCMeta):
    """
    抽象产品
    """
    @abstractmethod
    def prepare(self, VegPizza):
        pass

class NonVegPizza(metaclass=ABCMeta):
    """
    抽象产品
    """
    @abstractmethod
    def serve(self, VegPizza):
        pass

class DeluxVeggiePizza(VegPizza):
    
    def prepare(self):
        print("Prepare ", type(self).__name__)

class ChickenPizza(NonVegPizza):
    
    def serve(self, VegPizza):
        print(type(self).__name__, " is served with Chicken on ", type(VegPizza).__name__)

class MexicanVegPizza(VegPizza):
    
    def prepare(self):
        print("Prepare ", type(self).__name__)

class HamPizza(NonVegPizza):
    
    def serve(self, VegPizza):
        print(type(self).__name__, " is served with Ham on ", type(VegPizza).__name__)

class PizzaStore:
    
    def __init__(self):
        pass
    
    def makePizzas(self):
        for factory in [IndianPizzaFactory(), USPizzaFactory()]:
            self.factory = factory
            self.NonVegPizza = self.factory.createNonVegPizza()
            self.VegPizza = self.factory.createVegPizza()
            self.VegPizza.prepare()
            self.NonVegPizza.serve(self.VegPizza)


pizza = PizzaStore()
pizza.makePizzas()
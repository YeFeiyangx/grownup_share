__author__ = ['Chetan',"IvanYoung"]

from abc import ABCMeta, abstractmethod
"""
工厂模式

[抽象类方法]<https://www.cnblogs.com/baxianhua/p/10876181.html>

含抽象类方法的不能实例化
"""
class Animal(metaclass = ABCMeta):
    """
    抽象产品
    """
    @abstractmethod
    def do_say(self):
        pass

## 所有子类中，必须再次实现父类的方法
class Dog(Animal):
    """
    实例产品
    """
    def do_say(self):
        print("Bhow Bhow!!")

class Cat(Animal):
    """
    实例产品
    """
    def do_say(self):
        print("Meow Meow!!")


## forest factory defined
class ForestFactory(object):
    """
    带有方法的工厂
    """
    def make_sound(self, object_type):
        return eval(object_type)().do_say()

## client code
if __name__ == '__main__':
    ff = ForestFactory()
    animal = input("Which animal should make_sound Dog or Cat?")  ## 客户端，选择输入Dog or Cat
    ff.make_sound(animal)

__author__ = ['Chetan',"IvanYoung"]

from abc import ABCMeta, abstractmethod

"""
模板模式 - 

"""

class AbstractClass(metaclass=ABCMeta):
    """
    抽象元类，强制要求子类重写抽象元类中的方法;
    子类强制重写操作方法1，操作方法2
    """
    def __init__(self):
        pass

    @abstractmethod
    def operation1(self):
        pass

    @abstractmethod
    def operation2(self):
        pass

    def template_method(self):
        print("Defining the Algorithm. Operation1 follows Operation2")
        self.operation2()
        self.operation1()


class ConcreteClass(AbstractClass):

    def operation1(self):
        print("My Concrete Operation1")

    def operation2(self):
        print("Operation 2 remains same")


class Client:
    def main(self):
        self.concreate = ConcreteClass()
        self.concreate.template_method()

client = Client()
client.main()
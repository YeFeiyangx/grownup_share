__author__ = ['Chetan',"IvanYoung"]

from abc import ABCMeta, abstractmethod

"""
模板方法模式
1. 使用基本操作定义算法的框架；
2. 重新定义子类的某些操作，而无需修改算法的结构；
3. 实现代码重用并避免重复工作；
4. 利用通用接口或实现。

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
        # 需要被强制重写的方法1
        pass

    @abstractmethod
    def operation2(self):
        # 需要被强制重写的方法2
        pass

    def template_method(self):
        # 方法执行集合
        print("Defining the Algorithm. Operation1 follows Operation2")
        self.operation2()
        self.operation1()


class ConcreteClass(AbstractClass):
    """
    编写实例方法
    """
    def operation1(self):
        print("My Concrete Operation1")

    def operation2(self):
        print("Operation 2 remains same")


class Client:
    """
    客户端，调用模板算法，使其实例化
    """
    def main(self):
        self.concreate = ConcreteClass()
        self.concreate.template_method()

client = Client()
client.main()
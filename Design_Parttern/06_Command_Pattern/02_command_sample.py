__author__ = ['Chetan',"IvanYoung"]

"""
命令模式 含 方法名称 拥有方法的对象 方法的参数值；


"""
from abc import ABCMeta, abstractmethod
class Command(metaclass=ABCMeta):
    """
    抽象元类，强制要求子类重写抽象元类中的方法
    """
    def __init__(self, recv):
        self.recv = recv
    
    def execute(self):
        pass

class ConcreteCommand(Command):
    """
    将 命令接收方 调入命令执行实体，
    将所需执行命令加入执行函数execute集合，按书写逻辑执行集合中的action
    """
    def __init__(self, recv):
        self.recv = recv
    
    def execute(self):
        self.recv.action()

class Receiver:
    """
    命令接收方，赋予命令接收方行为action
    """
    def action(self):
        print("Receiver Action")

class Invoker:
    """
    请求者集合
    把执行实体放入 请求者中，通过execute方法，调用 执行实体 的execute，
    通过 执行实体 的execute集合，调用Receiver的action。
    """
    def command(self, cmd):
        self.cmd = cmd
    
    def execute(self):
        self.cmd.execute()

if __name__ == '__main__':
    recv = Receiver()
    cmd = ConcreteCommand(recv)
    invoker = Invoker()
    invoker.command(cmd)
    invoker.execute()

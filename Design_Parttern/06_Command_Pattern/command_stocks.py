__author__ = ['Chetan',"IvanYoung"]

from abc import ABCMeta, abstractmethod


class StockTrade:
    """
    命令接收方（实际命令模块），赋予 命令接收方 执行方法
    """
    def buy(self):
        print("You will buy stocks")
    
    def sell(self):
        print("You will sell stocks")


class Order(metaclass=ABCMeta):
    """
    抽象元类，强制要求子类重写抽象元类中的方法
    """
    @abstractmethod
    def execute(self):
        pass

class BuyStockOrder(Order):
    """
    “买执行方法实体” 将 命令接收方（StockTrade） 加入买执行实体，调用命令接收方的“买”命令
    """
    def __init__(self, stock):
        self.stock = stock
    
    def execute(self):
        self.stock.buy()

class SellStockOrder(Order):
    """
    “卖执行方法实体” 将 命令接收方（StockTrade） 加入卖执行实体，调用命令接收方的“卖”命令
    """
    def __init__(self, stock):
        self.stock = stock
    
    def execute(self):
        self.stock.sell()

class Agent:
    """
    调用者
    将命令加入__orderQueue作记录，然后执行命令
    """
    def __init__(self):
        self.__orderQueue = []
    
    def placeOrder(self, order):
        self.__orderQueue.append(order)
        order.execute()


if __name__ == '__main__':
    #Client
    stock = StockTrade()
    buyStock = BuyStockOrder(stock)
    sellStock = SellStockOrder(stock)
    
    #Invoker
    agent = Agent()
    agent.placeOrder(buyStock)
    agent.placeOrder(sellStock)



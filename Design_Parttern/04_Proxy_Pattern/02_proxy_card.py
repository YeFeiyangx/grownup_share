__author__ = ['Chetan',"IvanYoung"]
#%%
from abc import ABCMeta, abstractmethod
"""
代理模式
为对象提供一个代理，从而实现对原始对象的访问控制
它可以用作一个层或接口，以支持分布式访问
它通过增加代理，保护真正的组件不受意外的影响
"""
#%%
class Payment(metaclass=ABCMeta):
    "元对象,赋予主题do_pay方法，强制重写do_pay"
    @abstractmethod
    def do_pay(self):
        pass

class Bank(Payment):
    """
    主题
    """
    def __init__(self):
        self.card = None
        self.account = None
    
    def __getAccount(self):
        """
        类的私有方法，保证该方法在类的内部被调用
        单一功能，获取账户
        """
        # Assume card number is account number
        self.account = self.card 
        return self.account
    
    def __hasFunds(self):
        """
        类的私有方法，保证该方法在类的内部被调用
        单一功能，验证是否有钱
        """
        print("Bank:: Checking if Account", self.__getAccount(), "has enough funds")
        # return False
        return True
    
    def setCard(self, card):
        self.card = card
    
    def do_pay(self):
        if self.__hasFunds():
            print("Bank:: Paying the merchant")
            return True
        else:
            print("Bank:: Sorry, not enough funds!")
            return False

class DebitCard(Payment):
    """
    代理
    """
    def __init__(self):
        self.bank = Bank()
    
    def do_pay(self):
        card = input("Proxy:: Punch in Card Number: ")
        self.bank.setCard(card)
        return self.bank.do_pay()

class You:
    """
    客户端
    """
    def __init__(self):
        print("You:: Lets buy the Denim shirt")
        self.debitCard = DebitCard()
        self.isPurchased = None
    
    def make_payment(self):
        self.isPurchased = self.debitCard.do_pay()
    
    def __del__(self):
        """
        所有程序运行完，回收；
        python自带垃圾回收机制；
        理论上运行完就结束了，这个功能看上去没啥用，但是最后可以确保结束前完成当中的内容；
        具体和jupyter对比运行即可
        
        """
        if self.isPurchased:
            print("You:: Wow! Denim shirt is Mine :-)")
        else:
            print("You:: I should earn more :(")

you = You()
you.make_payment()

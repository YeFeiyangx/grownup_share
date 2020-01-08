__author__ = ['Chetan',"IvanYoung"]

class NewsPublisher:
    """
    主题
    """
    def __init__(self):
        self.__subscribers = []
        self.__latestNews = None
    
    def attach(self, subscriber):
        """
        用于登记需要通知的媒体
        """
        self.__subscribers.append(subscriber)
    
    def detach(self):
        return self.__subscribers.pop()
    
    def subscribers(self):
        return [type(x).__name__ for x in self.__subscribers]
    
    def notifySubscribers(self):
        """
        让被登记的观察者更新新闻
        """
        for sub in self.__subscribers:
            sub.update()
    
    def addNews(self, news):
        self.__latestNews = news
    
    def getNews(self):
        return "Got News:", self.__latestNews


from abc import ABCMeta, abstractmethod

class Subscriber(metaclass=ABCMeta):
    """
    抽象元类，子类必须重写update方法
    """
    @abstractmethod
    def update(self):
        pass


class SMSSubscriber:
    
    def __init__(self, publisher): ## 互选，观察者也可以选其他主题
        """
        实例化时，就登记进入主题
        """
        self.publisher = publisher
        self.publisher.attach(self) ## 使观察者被登记入主题
    
    def update(self):
        """
        获得被登入的新闻
        """
        print(type(self).__name__, self.publisher.getNews())

class EmailSubscriber:
    
    def __init__(self, publisher):
        self.publisher = publisher
        self.publisher.attach(self)
    
    def update(self):
        print(type(self).__name__, self.publisher.getNews())

class AnyOtherSubscriber:
    
    def __init__(self, publisher):
        self.publisher = publisher
        self.publisher.attach(self)
    
    def update(self):
        print(type(self).__name__, self.publisher.getNews())


if __name__ == '__main__':
    news_publisher = NewsPublisher()
    
    for Subscribers in [SMSSubscriber, EmailSubscriber, AnyOtherSubscriber]:
        Subscribers(news_publisher)
        print("\n--1--Subscribers:", news_publisher.subscribers())
        
        news_publisher.addNews('Hello World!')
        news_publisher.notifySubscribers()
        ## 把新闻移除
        print("\n--2--Detached:", type(news_publisher.detach()).__name__)
        print("\n--3--Subscribers:", news_publisher.subscribers())
        
        news_publisher.addNews('My second news!')
        news_publisher.notifySubscribers()

__author__ = ['Chetan',"IvanYoung"]

class Subject:
    """
    主题
    """
    def __init__(self):
        self.__observers = []
    
    def register(self, observer):
        self.__observers.append(observer) # 私有，登记观察者观察者
    
    def notifyAll(self, *args, **kwargs):
        for observer in self.__observers:   # 通知 观察着
            # 调用观察着名单的观测行为方法
            # 将主题的信息传递给观察者
            observer.notify(self, *args, **kwargs) 


class Observer1:
    """
    观察者
    """
    def __init__(self, subject):
        """
        观察者实例化的时候就启用主题的注册功能，把观察者注入到主题中
        """
        subject.register(self)  
    
    def notify(self, subject, *args): 
        """
        注意大小写
        param: self -> Observer1
        param: subject -> Subject
        param: *args -> Subject(*args,**kwargs)
        """
        print(type(self).__name__,':: Got', args, 'From', subject)

class Observer2:
    
    def __init__(self, subject):
        subject.register(self)
    
    def notify(self, subject, *args):
        print(type(self).__name__, ':: Got', args, 'From', subject)


subject = Subject()
observer1 = Observer1(subject)
observer2 = Observer2(subject)
subject.notifyAll('notification',{"test":"TEST"})
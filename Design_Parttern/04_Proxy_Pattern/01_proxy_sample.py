__author__ = 'Chetan'

class Actor(object):
    """
    代理，杨元
    """
    def __init__(self):
        self.isBusy = False
    
    def occupied(self):
        self.isBusy = True
        print(type(self).__name__ , "is occupied with current movie")
    
    def available(self):
        self.isBusy = False
        print(type(self).__name__ , "is free for the movie")
    
    def getStatus(self):
        return self.isBusy

class Agent(object):
    """
    代理，经纪人
    """
    def __init__(self):
        self.principal = None
    
    def work(self):
        self.actor = Actor()
        if self.actor.getStatus():
            self.actor.occupied()
        else:
            self.actor.available()


if __name__ == '__main__':
    r = Agent()
    r.work()
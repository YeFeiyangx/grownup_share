__author__ = 'Chetan'

class HealthCheck:
    
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not HealthCheck._instance:
            HealthCheck._instance = super(HealthCheck, cls).__new__(cls, *args, **kwargs)
            print("===***")
        return HealthCheck._instance
    
    def __init__(self):
        print("==init=="*10)
        if "_servers" not in self.__dict__:
            self._servers = []
            print("self._instance:",self._instance,id(self._instance))
            print("self._servers:",self._servers)
        else:
            pass
    def addServer(self):
        self._servers.append("Server 1")
        self._servers.append("Server 2")
        self._servers.append("Server 3")
        self._servers.append("Server 4")
    
    def changeServer(self):
        self._servers.pop()
        self._servers.append("Server 5")
print('===1'*5)
hc1 = HealthCheck()


hc1.addServer()
print("Schedule health check for servers (1)..")
for i in range(4):
    print("Checking ", hc1._servers[i])

print('===2'*5)
hc2 = HealthCheck()

print("Checking +_+_+_+:", len(hc1._servers))
hc1.addServer()

hc2.changeServer()
print("Schedule health check for servers (2)..")
for i in range(4):
    print("Checking ", hc2._servers[i])

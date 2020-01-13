__author__ = ['Chetan',"IvanYoung"]

class Wizard():
    
    def __init__(self, src, rootdir):
        """
        系统配置处
        """
        self.choices = []
        self.rootdir = rootdir
        self.src = src
    
    def preferences(self, command):
        """
        命令接收集
        """
        self.choices.append(command)
    
    def execute(self):
        """
        根据命令接收集，执行命令
        """
        for choice in self.choices:
            if list(choice.values())[0]:
                print("Copying binaries --", self.src, " to ", self.rootdir)
            else:
                print("No Operation")
    
    def rollback(self):
        """
        回滚
        """
        print("Deleting the unwanted..", self.rootdir)


if __name__ == '__main__':
    ## Client code
    wizard = Wizard('python3.5.gzip', '/usr/bin/')
    ## Steps for installation. ## Users chooses to install Python only
    wizard.preferences({'python':True})
    wizard.preferences({'java':False})
    wizard.execute()



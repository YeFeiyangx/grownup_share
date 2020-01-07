__author__ = ['Chetan',"IvanYoung"]

#%%
import types
from abc import ABCMeta, abstractmethod
"""
工厂方法模式
"""

class Section(metaclass=ABCMeta):
    """
    抽象产品（组分）
    """
    @abstractmethod
    def describe(self):
        pass


class PersonalSection(Section):
    """
    实例产品 -> 个人简介
    """
    def describe(self):
        print("Personal Section")


class AlbumSection(Section):
    """
    实例产品 -> 相册
    """
    def describe(self):
        print("Album Section")


class PatentSection(Section):
    """
    实例产品 -> 专利情况
    """
    def describe(self):
        print("Patent Section")


class PublicationSection(Section):
    """
    实例产品 -> 著作情况
    """
    def describe(self):
        print("Publication Section")


class Profile(metaclass=ABCMeta):
    """
    简介[抽象工厂]
    """
    def __init__(self):
        self.sections = []
        self.createProfile()
    
    @abstractmethod
    def createProfile(self):
        pass
    
    def getSections(self):
        return self.sections
    
    def addSections(self, section):
        self.sections.append(section)


class linkedin(Profile):
    """
    领英 [工厂方法实例]
    """
    def createProfile(self):
        self.addSections("PersonalSection")
        self.addSections("PatentSection")
        self.addSections("PublicationSection")
        for i in self.sections:
            setattr(self, i, eval(i)().describe) # 直接获得实例产品的方法 -> -> profile.PersonalSection()
            # setattr(self, i, eval(i))              # 获得实例产品的方法 -> profile.PersonalSection().describe()
            # setattr(self, i, eval(i)())              # 获得实例产品的方法 -> profile.PersonalSection.describe()

class facebook(Profile):
    """
    脸书 [工厂方法实例]
    """
    def createProfile(self):
        self.addSections(PersonalSection())
        self.addSections(AlbumSection())

#%%
profile = eval("linkedin")() # 实例化领英工厂方法

print("Creating Profile..", type(profile).__name__)         # 查看实例化的工厂方法名
print("Profile has sections --", profile.getSections())     # 获得工厂方法中的产品
print("profile.sections:", profile.sections)                # 因与上方一致
print("profile.__dict__:", profile.__dict__)
profile.PersonalSection()
profile.PatentSection()


# %%
## 客户端
if __name__ == '__main__':
    profile_type = input("Which Profile you'd like to create? [LinkedIn or FaceBook]")
    profile = eval(profile_type.lower())()
    print("Creating Profile..", type(profile).__name__)
    print("Profile has sections --", profile.getSections())
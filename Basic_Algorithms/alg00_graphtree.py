
"""
code souce
https://github.com/Apress/python-algorithms/tree/master/Hetland-Source%20Code
"""
"""
图的表示法
"""
"邻接集表示法"
## 在无序的情况下，集合的形式，优于列表的形式
a,b,c,d,e,f,g,h = range(8)
# N是一张图，表示了a-h可以到达的位置
N = [
    {b,c,d,e,f},     # a
    {c,e},           # b
    {d},             # c
    {e},             # d
    {f},             # e
    {c,g,h},         # f
    {f,h},           # g
    {f,g}            # h
    ]

print("查找a点所能到达的位置:",N[a])

"加权邻接字典"

a,b,c,d,e,f,g,h = range(8)
# N是一张图，表示了a-h可以到达的位置,并设置路径权重
N = [
    {b:2,c:1,d:3,e:9,f:4},    # a
    {c:4,e:3},                # b
    {d:8},                    # c
    {e:7},                    # d
    {f:5},                    # e
    {c:2,g:2,h:2},            # f
    {f:1,h:6},                # g
    {f:9,g:8}                 # h
    ]

"邻接集的字典表示法"

N = {
    "a":set("bcdef"),
    "b":set("ce"),
    "c":set("d"),
    "d":set("e"),
    "e":set("f"),
    "f":set("cgh"),
    "g":set("fh"),
    "h":set("fg")
    }

"邻接矩阵的列表嵌套表示法"
a,b,c,d,e,f,g,h = range(8)
#     a b c d e f g h
N = [[0,1,1,1,1,1,0,0], #a
     [0,0,1,0,1,0,0,0], #b
     [0,0,0,1,0,0,0,0], #c
     [0,0,0,0,1,0,0,0], #d
     [0,0,0,0,0,1,0,0], #e
     [0,0,1,0,0,0,1,1], #f
     [0,0,0,0,0,1,0,1], #g
     [0,0,0,0,0,1,1,0]] #h

"加权，不存在的边赋予无限大的权值"

a,b,c,d,e,f,g,h = range(8)
inf = float("inf")
#       a   b   c   d   e   f   g   h
W = [[  0,  2,  1,  3,  9,  4,inf,inf], #a
     [inf,  0,  4,inf,  3,inf,inf,inf], #b
     [inf,inf,  0,  8,inf,inf,inf,inf], #c
     [inf,inf,inf,  0,  7,inf,inf,inf], #d
     [inf,inf,inf,inf,  0,  5,inf,inf], #e
     [inf,inf,  2,inf,inf,  0,  2,  2], #f
     [inf,inf,inf,inf,inf,  1,  0,  6], #g
     [inf,inf,inf,inf,inf,  9,  8,  0]] #h

"""
树的实现
"""
T =[["a","b"],["c"],["d",["e","f"]]]


"二叉树"

class Tree:
    def __init__(self,left,right):
        self.left = left
        self.right = right

t = Tree(Tree("a","b"),Tree("c","d"))

print(t.right.left)
#%%
"多路搜索树类"

class Tree:
    def __init__(self,kids,next=None):
        self.kids = self.val = kids
        self.next = next

t = Tree(Tree("a",Tree("b",Tree("c",Tree("d")))))

# %%
class Config(dict):
    def __init__(self):
        self.__dict__ = self
c = Config()
c.abc = 4
c.bca = 5
print(c)
c['abc']

# %%
class Config(dict):
    def __init__(self,**kwds):
        for i in kwds:
            setattr(self,i,kwds[i])
        print(self)
        print(self.__dict__)
        # self.__dict__ = self
c = Config(a="1",b="2")
c.abc = 4
c.bca = 5
print(c.__dict__)
print(c.a)

# %%
class Config:
    def __init__(self,**kwds):
        self.kwds = kwds
        for i in kwds:
            setattr(self,i,kwds[i])
        # print(self)
        # print(self.__dict__)
    def __repr__(self):
        return str(self.kwds)

c = Config(a="1",b="2")
c.abc = 4
c.bca = 5
print(c.__dict__)
print(c)
print(c.a)


# %%
"使用self.__dict__ = self，类名后面必须传入dict"
# 传入dict方法后，定义属性就可以用self[key] = value
class Bunch(dict):
    def __init__(self,**kwds):
        # 建树专用
        super(Bunch,self).__init__(**kwds)
        ## Assigning the dictionary self to __dict__ allows attribute access and item access
        self.__dict__ = self
        print("observation1",self.keys())
        print("observation2",self.__dict__.keys())
    def __repr__(self):
        return str(self.__dict__.keys())

test = Bunch(a=4,b=5)
print(dir(test))
print(test["a"])
print(test.a)
# T = Bunch
# print("--0--"*10)
# t = T(left=T(left="a",right="b"),right=T(left="c"))
# print("--1--"*10)
# print(t)
# print("--2--"*10)
# print(t.left)
# print("--3--"*10)
# print(t.right)

# %%
class Bunch(dict):
    def __init__(self,**kwds):
        # super(Bunch,self).__init__(**kwds)
        print("--0--"*20)
        print(type(self))
        print("--1--"*20)
        print(dir(self))
        self.update(kwds)
        self.___dict___ =self

t = Bunch(a=1,b=2,c=3)
print("--2--"*20)
print(t)
print("--3--"*20)
t.abc = 4
print(t)
print("--4--"*20)
print(t)
print(dir(t))
print(t.abc) # 不可t["abc"]
print("--5--"*20)
t.update({"efg":5})
print(t)
print(dir(t))
print(t["efg"]) # 不可t.efg
# %%
# 传入dict方法后，定义属性就可以用self[key] = value
class JustTest(dict):
    def __init__(self,**kwds):
        print(kwds)
        # super(JustTest,self).__init__(**kwds)
        self.__dict__ = self
        for i,j in kwds.items():
            self[i] = j
    def __repr__(self):
        return str(self.__dict__.keys())

t_dict = JustTest(a = 4, b = 5, c = 6)

#%%
print(t_dict["a"])
print(t_dict.a)
print(dir(t_dict))

# %%

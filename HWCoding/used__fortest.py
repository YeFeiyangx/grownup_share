import os

# f  = open(os.path.join(os.path.dirname(__file__),'fortest3.txt'),'r')
# for lines in f:
#     print(lines)
# f.close()

info_raw = []
f  = open(os.path.join(os.path.dirname(__file__),'fortest3.txt'),'r')
for lines in f:
    info_raw.append(list(map(int,lines.split())))
f.close()
print(info_raw)

row,col = info_raw[0]
maze = []
for i in range(row):
    maze.append(info_raw[i+1])
print("First:")
maze = [[-x for x in i] for i in maze]
for i in maze:
    
    print(i)
queue = [[0,0]]
maze[0][0] = 1
while queue:
    x,y = queue.pop(0)
    if x == row-1 and y == col-1:
        break
    if x+1 < row and maze[x+1][y] == 0:
        maze[x+1][y] = maze[x][y]+1
        queue.append([x+1,y])
    if y+1 < col and maze[x][y+1] == 0:
        maze[x][y+1] = maze[x][y]+1
        queue.append([x,y+1])
    if x-1 >= 0 and maze[x-1][y] == 0:
        maze[x-1][y] = maze[x][y]+1
        queue.append([x-1,y])
    if y-1 >= 0 and maze[x][y-1] == 0:
        maze[x][y-1] = maze[x][y]+1
        queue.append([x,y-1])
print("Second:")
for i in maze:
    print(i)
result = [[row-1,col-1]]
for i in range(maze[-1][-1]-1,0,-1):
    tempRow = result[0][0]
    tempCol = result[0][1]
    if tempRow-1>=0 and maze[tempRow-1][tempCol] == i:
        result.insert(0,[tempRow-1,tempCol])
    elif tempCol-1>=0 and maze[tempRow][tempCol-1] == i:
        result.insert(0,[tempRow,tempCol-1])
    elif tempRow+1<row and maze[tempRow+1][tempCol] == i:
        result.insert(0,[tempRow+1,tempCol])
    elif tempCol+1<col and maze[tempRow][tempCol+1] == i:
        result.insert(0,[tempRow,tempCol+1])
for i in result:
    print('(%d,%d)'%(i[0],i[1]))
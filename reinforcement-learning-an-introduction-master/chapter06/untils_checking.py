#%%
import numpy as np 


# %%
WORLD_HEIGHT = 10
WORLD_WIDTH = 10
GOAL = {(2, 1),(6,1),(7,4),(5,5),(6,8),(2,8)}
# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

#%%
# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if (i, j) in GOAL:
                plus = 'G'
            else:
                plus = ''
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append(plus + 'U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append(plus + 'D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append(plus + 'L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append(plus + 'R')

    for row in optimal_policy:
        print(row)
        
#%%
q_table = np.load('D:\\gitShare\\alg_pic\\2agv_q_l_42.npy')
print_optimal_policy(q_table)
# %%
q_table[0,9,:]
# %%

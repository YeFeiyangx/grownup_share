#######################################################################
# Copyright (C)                                                       #
# 2020 Feiyang Ye(xluckyhappy@aliyun.com)                             #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
# appear unsteady state                                               #
#######################################################################
#%%
from collections import defaultdict
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# world height
WORLD_HEIGHT = 10

# world width
WORLD_WIDTH = 10

# probability for exploration
EPSILON = 0.9

# step size
ALPHA = 0.1

# gamma for Q-Learning
GAMMA = 1

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
START = [(0, 3),(0,9)]
GOAL = {(2, 1),(6,1),(7,4),(5,5),(6,8),(2,8)}
PRINT_GOAL = [{(2, 1),(6,1),(7,4),(5,5)},{(7,4),(5,5),(6,8),(2,8)}]
PASS_SIDE = [[0,5],[4,9]]

def step(state, action_2agv, goal):
    reward_list = []
    next_state = []
    for agv in range(2):
        reward = 0
        i,j = state[agv]
        action = action_2agv[agv]
        if action == ACTION_UP:
            _next_state = (max(i - 1, 0), j)
        elif action == ACTION_LEFT:
            if agv == 0:
                _next_state = (i, max(j - 1, PASS_SIDE[0][0] ))
            elif agv == 1:
                _next_state = (i, max(j - 1, PASS_SIDE[1][0] ))
                
        elif action == ACTION_RIGHT:
            if agv == 0:
                _next_state = (i, min(j + 1, PASS_SIDE[0][1] ))
            elif agv == 1:
                _next_state = (i, min(j + 1, PASS_SIDE[1][1] ))
        elif action == ACTION_DOWN:
            _next_state = (min(i + 1, WORLD_HEIGHT - 1), j)
        else:
            assert False
            
        next_state.append(_next_state)
        _reward = -1
        if _next_state in goal[agv]:
            _reward = 10
        reward_list.append(_reward)

    crash_tag = 0
    if next_state[0] == next_state[1]:
        reward_list = [-1,-1]
        next_state = state
        crash_tag = 1
    elif next_state[0][0] == next_state[1][0]:
        if (next_state[0][1] == state[1][1]) and (next_state[1][1] == state[0][1]):
            reward_list = [-1,-1]
            next_state = state
            crash_tag = 1
    elif next_state[0][1] == next_state[1][1]:
        if (next_state[0][0] == state[1][0]) and (next_state[1][0] == state[0][0]):
            reward_list = [-1,-1]
            next_state = state
            crash_tag = 1

    if crash_tag == 0:
        for agv in range(2):
            if next_state[agv] in goal[agv]:
                goal[agv].remove(next_state[agv])
            if next_state[agv] in goal[1-agv]:
                goal[1-agv].remove(next_state[agv])

    return next_state, reward_list

# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value, epsilon):
    total_action = []
    for agv in range(2):
        # bolzmann_e
        if np.random.binomial(1, epsilon) == 1:
            values_ = q_value[state[agv][0], state[agv][1], :]
            values_exp_ = np.exp(values_)
            values_exp_ /= values_exp_.sum()
            total_action.append(np.random.choice(ACTIONS, p=values_exp_))
        else:
            values_ = q_value[state[agv][0], state[agv][1], :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
            total_action.append(action)
    return total_action
# an episode with Q-Learning
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def q_learning(q_value, epsilon, gamma,print_time, step_size):
    state = START
    rewards = [0, 0]
    goal = deepcopy(PRINT_GOAL)
    n_times = 0
    while len(goal[0]) + len(goal[1]) > 0:
        action = choose_action(state, q_value, epsilon)
        next_state, reward_list = step(state, action, goal)
        rewards = [i + j for i, j in zip(rewards, reward_list)]
        # Q-Learning update
        for agv in range(2):
            if len(goal[agv]) > 0:
                q_value[state[agv][0], state[agv][1], action] += step_size * (
                        reward_list[agv] + gamma * np.max(q_value[next_state[agv][0], next_state[agv][1], :]) -
                        q_value[state[agv][0], state[agv][1], action])
        state = next_state
        n_times += 1
        if n_times >= 500:
            break
    return rewards

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

def figure_agv_2():
    # episodes of each run
    episodes = 500
    runs = 40
    M = episodes * runs
    n_times = 0
    rewards_q_learning = np.zeros(int(M))
    q_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    for r in tqdm(range(runs)):
        for i in range(0, episodes):
            # if n_times >= 0 and n_times < 0.01*M:
            #     epsilon = 0.9
            #     gamma = 1
            #     alpha = 0.8
            # elif n_times >= 0.02*M and n_times < 0.04*M:
            #     epsilon = 0.7
            #     gamma = 0.95
            #     alpha = 0.8
            # elif n_times >= 0.04*M and n_times < 0.06*M:
            #     epsilon = 0.5
            #     gamma = 0.95
            #     alpha = 0.6
            # elif n_times >= 0.06*M and n_times < 0.08*M:
            #     epsilon = 0.3
            #     gamma = 0.9
            #     alpha = 0.6
            # elif n_times >= 0.08*M and n_times < 0.1*M:
            #     epsilon = 0.1
            #     gamma = 0.85
            #     alpha = 0.4
            # else:
            #     epsilon = 0.05
            #     gamma = 0.8
            #     alpha = 0.2
            
            gamma = 0.9
            alpha = 0.4       
            if n_times >= 0 and n_times < 0.2*M:
                epsilon = 0.9
            elif n_times >= 0.2*M and n_times < 0.4*M:
                epsilon = 0.7
            elif n_times >= 0.4*M and n_times < 0.6*M:
                epsilon = 0.5
            elif n_times >= 0.6*M and n_times < 0.8*M:
                epsilon = 0.3
            else:
                epsilon = 0.05

            reward_global = q_learning(q_q_learning, epsilon, gamma, n_times, step_size=alpha)
            sum_reward_global = sum(reward_global)
            rewards_q_learning[n_times] = sum_reward_global
            if sum_reward_global >= rewards_q_learning.max():
                q_table_best_copy = deepcopy(q_q_learning)
                print('sum_reward_global:', sum_reward_global)
                # if sum_reward_global > his_reward_global + 1:
                #     print('==============update best performance==============')
                #     print(sum_reward_global)
                #     name = '2agv_q_l_' + str(int(sum_reward_global)) + '.npy'
                #     np.save("D:\\gitShare\\alg_pic\\" + name, q_q_learning)
            n_times += 1
    np.save("D:\\gitShare\\alg_pic\\2agv_q_l.npy", q_q_learning)

    min_reward = min(rewards_q_learning)
    max_reward = max(rewards_q_learning)
    print('================================')
    print('min:', min_reward)
    print('max:', max_reward)
    # draw reward curves
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([min_reward, max_reward])
    plt.legend()
    name = '2agv_q_l_be_gr.png'
    plt.savefig('D:\\gitShare\\alg_pic\\' + name)
    plt.close()
    name = '2agv_q_t_be_gr.npy'
    np.save("D:\\gitShare\\alg_pic\\" + name, q_table_best_copy)
    # display optimal policy
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_table_best_copy)
        
        
    print('Q-Learning Optimal Policy in the end:')
    print_optimal_policy(q_q_learning)


if __name__ == '__main__':
    figure_agv_2()
    print(GOAL)
# %%

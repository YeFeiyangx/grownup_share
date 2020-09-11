#######################################################################
# Copyright (C)                                                       #
# 2020 Feiyang Ye(xluckyhappy@aliyun.com)                             #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
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
ALPHA = 0.4

# gamma for Q-Learning and Expected Sarsa
GAMMA = 0.9

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
    agv_0, agv_1 = state
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
        if (i,j) in goal:
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
        for s_next_state in next_state:
            if s_next_state in goal:
                goal.remove(s_next_state)

    return next_state, reward_list

# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value, epsilon):
    if np.random.binomial(1, epsilon) == 1:
        return [np.random.choice(ACTIONS), np.random.choice(ACTIONS)]
    else:
        total_action = []
        for i in range(2):
            values_ = q_value[state[0], state[1], i, :][i]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
            total_action.append(action)
        return total_action

# an episode with Q-Learning
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def q_learning(q_value, epsilon, step_size=ALPHA):
    state = START
    rewards = [0, 0]
    goal = deepcopy(GOAL)
    n_times = 0
    while len(goal) != 0:
        action = choose_action(state, q_value, epsilon)
        next_state, reward_list = step(state, action, goal)
        rewards = [i + j for i, j in zip(rewards, reward_list)]
        # Q-Learning update
        for agv in range(2):
            q_value[state[0], state[1], agv, action] += step_size * (
                    reward_list[agv] + GAMMA * np.max(q_value[next_state[0], next_state[1], agv, :]) -
                    q_value[state[0], state[1], agv, action])
        state = next_state
        n_times += 1
        if n_times >= 500:
            break
    return rewards

# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = [[],[]]
    for agv in range(2):
        for i in range(0, WORLD_HEIGHT):
            optimal_policy[agv].append([])
            for j in range(0, WORLD_WIDTH):
                if (i, j) in PRINT_GOAL[agv]:
                    optimal_policy[agv][-1].append('G')
                bestAction = np.argmax(q_value[i, j, agv, :])
                if bestAction == ACTION_UP:
                    optimal_policy[agv][-1].append('U')
                elif bestAction == ACTION_DOWN:
                    optimal_policy[agv][-1].append('D')
                elif bestAction == ACTION_LEFT:
                    optimal_policy[agv][-1].append('L')
                elif bestAction == ACTION_RIGHT:
                    optimal_policy[agv][-1].append('R')
    for agv in range(2):
        print('==============agv: %s ==================' % agv)
        for row in optimal_policy[agv]:
            print(row)

def figure_6_4():
    # episodes of each run
    episodes = 500
    # perform 40 independent runs
    runs = 40
    M = episodes * runs
    n_times = 0
    rewards_q_learning = np.zeros([2, episodes])
    q_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 2, 4))
    for r in tqdm(range(runs)):
        for i in range(0, episodes):
            if n_times >= 0 and n_times < 0.2*M:
                epsilon = 0.9
            elif n_times >= 4000 and n_times < 0.4*M:
                epsilon = 0.7
            elif n_times >= 8000 and n_times < 0.6*M:
                epsilon = 0.5
            elif n_times >= 12000 and n_times < 0.8*M:
                epsilon = 0.3
            else:
                epsilon = 0.05
            for agv in range(2):
                rewards_q_learning[agv][i] += q_learning(q_q_learning, epsilon)[agv]
            n_times += 1
    np.save("D:\\alg_pic\\2agv_q_l.npy", q_q_learning)
    # averaging over independt runs
    for agv in range(2):
        single_rewards_q = rewards_q_learning[agv]
        single_rewards_q /= runs
        min_reward = min(single_rewards_q)
        max_reward = max(single_rewards_q)
        print('==============agv: %s ==================' % agv)
        print('min:', min_reward)
        print('max:', max_reward)
        # draw reward curves
        plt.plot(single_rewards_q, label='Q-Learning')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        plt.ylim([min_reward, max_reward])
        plt.legend()
        name = '2agv_q_l_' + str(agv) + '.png'
        plt.savefig('D:\\alg_pic\\' + name)
        plt.close()

    # display optimal policy
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_q_learning)


if __name__ == '__main__':
    figure_6_4()
    print(GOAL)

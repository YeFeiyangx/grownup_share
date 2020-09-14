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
ALPHA = 0.1

# gamma for Q-Learning and Expected Sarsa
GAMMA = 0.9

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
START = (0, 3)
# GOAL = {(2, 1),(6,1),(7,4)}
GOAL = {(2, 1),(6,1),(7,4),(5,5)}
PASS_SIDE = [0,5]

def step(state, action, goal):
    i, j = state
    if action == ACTION_UP:
        next_state = (max(i - 1, 0), j)
    elif action == ACTION_LEFT:
        next_state = (i, max(j - 1, PASS_SIDE[0]))
    elif action == ACTION_RIGHT:
        next_state = (i, min(j + 1, PASS_SIDE[1]))
    elif action == ACTION_DOWN:
        next_state = (min(i + 1, WORLD_HEIGHT - 1), j)
    else:
        assert False

    reward = -1
    if next_state in goal:
        reward = 10
        goal.remove(next_state)

    return next_state, reward

# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value, epsilon):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


# an episode with Q-Learning
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def q_learning(q_value, epsilon, step_size=ALPHA):
    state = START
    rewards = 0.0
    goal = deepcopy(GOAL)
    n_times = 0
    while len(goal) != 0:
        action = choose_action(state, q_value, epsilon)
        next_state, reward = step(state, action, goal)
        rewards += reward
        # Q-Learning update
        q_value[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
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

def figure_agv_1():
    # episodes of each run
    episodes = 500
    runs = 40
    M = episodes * runs
    n_times = 0
    rewards_q_learning = np.zeros(int(M))
    q_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    for r in tqdm(range(runs)):
        for i in range(0, episodes):
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
            # epsilon = 0.1
            rewards_q_learning[n_times] += q_learning(q_q_learning, epsilon)
            n_times += 1

    # display optimal policy
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_q_learning)


if __name__ == '__main__':
    figure_agv_1()
    print(GOAL)
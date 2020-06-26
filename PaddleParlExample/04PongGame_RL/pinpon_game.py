#!/usr/bin/env python
# coding: utf-8
"""
PaddleParl的使用，可以简化算法的构建代码。

PolicyGradient可以用来直接学习策略，但是个人感觉效率不高。

该场景中，把图像信息先调整为灰度图像，然后变为一维输入进行DNN学习，也可以使用CNN，但建议调节灰度，减少环境噪音。

图像拍平后，其维度存在80*80，既6400维度。因此网络如果太小，其模型很难习得该空间中的信息。
例如本example中，网络单层为act_dim * 10，既60维度，第一层的信息损益太大，导致模型最终无法达到均分10分效果。
建议把第一维度提高至512维以上。

"""


# !pip uninstall -y parl  # 说明：AIStudio预装的parl版本太老，容易跟其他库产生兼容性冲突，建议先卸载
# !pip uninstall -y pandas scikit-learn # 提示：在AIStudio中卸载这两个库再import parl可避免warning提示，不卸载也不影响parl的使用

# !pip install gym
# !pip install atari-py # 玩Gym的Atari游戏必装依赖，本次作业使用了Atari的Pong(乒乓球)环境
# !pip install paddlepaddle-gpu==1.6.3.post97 -i https://mirror.baidu.com/pypi/simple
# !pip install parl==1.3.1 -i https://mirror.baidu.com/pypi/simple


import os
import gym
import numpy as np

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger


LEARNING_RATE = 0.0006


class Model(parl.Model):
    def __init__(self, act_dim):
        ######################################################################
        ######################################################################
        hid1_size = act_dim * 10
        hid2_size = act_dim * 10
        hid3_size = act_dim
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=hid3_size, act='softmax')
        ######################################################################
        ######################################################################

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        ######################################################################
        ######################################################################
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        out = self.fc3(h2)
        ######################################################################
        ######################################################################
        return out

from parl.algorithms import PolicyGradient # 直接从parl库中导入PolicyGradient算法，无需重复写算法


# * `Agent`负责算法与环境的交互，在交互过程中把生成的数据提供给`Algorithm`来更新模型(`Model`)，数据的预处理流程也一般定义在这里。

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
        act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost


# ### Step 5 Training && Test（训练&&测试）

def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


# ### Step6 创建环境和Agent，启动训练，保存模型


# Pong 图片预处理
def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195] # 裁剪
    image = image[::2,::2,0] # 下采样，缩放2倍
    image[image == 144] = 0 # 擦除背景 (background type 1)
    image[image == 109] = 0 # 擦除背景 (background type 2)
    image[image != 0] = 1 # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()


# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.997):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


# 创建环境
env = gym.make('Pong-v0')
obs_dim = 80 * 80
act_dim = env.action_space.n
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

# 根据parl框架构建agent
######################################################################
######################################################################
# 根据parl框架构建agent
model = Model(act_dim=act_dim)
algorithm = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_dim,
    act_dim=act_dim)
######################################################################
######################################################################


# 加载模型
if os.path.exists('./model_11.ckpt'):
    agent.restore('./model_11.ckpt')
init_level = -7
np.random.seed(1024)
for i in range(1000):
    obs_list, action_list, reward_list = run_episode(env, agent)
    # if i % 10 == 0:
    #     logger.info("Train Episode {}, Reward Sum {}.".format(i, 
    #                                         sum(reward_list)))
    
    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    batch_reward = calc_reward_to_go(reward_list)

    agent.learn(batch_obs, batch_action, batch_reward)
    if (i + 1) % 100 == 0:
        total_reward = evaluate(env, agent, render=False)
        logger.info('Episode {}, Test reward: {}'.format(i + 1, 
                                            total_reward))
        if total_reward >= init_level:
            init_level = total_reward + 1
            agent.save('./model_r_' + str(total_reward) + '.ckpt')
# save the parameters to ./model.ckpt
agent.save('./model.ckpt')


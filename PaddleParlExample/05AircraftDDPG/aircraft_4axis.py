#!/usr/bin/env python
# coding: utf-8
"""
PaddleParl的使用，可以简化算法的构建代码。
DDPG 方法中使用AC架构，分别为Actor网络和Critic网络。
Actor网络是为迎合Critic网络；Critic网络是从环境中的反馈信息进行更新。

因Critic变化平率太强，会致使Actor的策略网络很难收敛。所以CRITIC_LR << ACTOR_LR;
训练过程中存在更新目标稳定的需求，所以DDPG的Actor和Critic网络拥有其target网络，
通过TAU来控制其target网络的更新幅度，可以理解为1/TAU的步长，更新一次网络。

四轴飞行器中模型结构还存在对业务场景的模拟，
main_action = action[0]
sub_action = action[1:]
sub_action = np.random.normal(sub_action, 0.01)
action = [main_action+0.2*x for x in sub_action]
将模型输出结果变为5维度，让其他四维将第一维视为基础电压，然后作为env的输入，获得结果反馈后更新网络。
这样模型容易习得轻度解析后的神经网络。
"""


# !pip uninstall -y parl  # 说明：AIStudio预装的parl版本太老，容易跟其他库产生兼容性冲突，建议先卸载
# !pip uninstall -y pandas scikit-learn # 提示：在AIStudio中卸载这两个库再import parl可避免warning提示，不卸载也不影响parl的使用

# !pip install paddlepaddle==1.6.3 -i https://mirror.baidu.com/pypi/simple
# !pip install parl==1.3.1 -i https://mirror.baidu.com/pypi/simple
# !pip install rlschool==0.3.1

import os
import numpy as np

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放

from rlschool import make_env  # 使用 RLSchool 创建飞行器环境


# # Step3 设置超参数

# ACTOR_LR = 0.0002   # Actor网络更新的 learning rate
# CRITIC_LR = 0.001   # Critic网络更新的 learning rate

ACTOR_LR = 0.0001   # Actor网络更新的 learning rate
CRITIC_LR = 0.0006   # Critic网络更新的 learning rate

GAMMA = 0.995        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 3*1e6   # 总训练步数
TEST_EVERY_STEPS = 4000

# `Model`用来定义前向(`Forward`)网络，用户可以自由的定制自己的网络结构。

class ActorModel(parl.Model):
    def __init__(self, act_dim):
        ######################################################################
        ######################################################################

        hid2_size = 64
        hid3_size = 32
        # hid1_size = 128
        # hid2_size = 64
        # 3层全连接网络

        self.fc1 = layers.fc(size=hid2_size, act='relu')
        self.fc2 = layers.fc(size=hid3_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')
        ######################################################################
        ######################################################################

    def policy(self, obs):
        ######################################################################
        ######################################################################
        hid0 = self.fc1(obs)
        hid1 = self.fc2(hid0)
        logits = self.fc3(hid1)
        ######################################################################
        ######################################################################
        return logits


class CriticModel(parl.Model):
    def __init__(self):
        ######################################################################
        ######################################################################
        hid1_size = 32
        hid2_size = 32
        # 2层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=1, act=None)
        ######################################################################
        ######################################################################

    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)
        ######################################################################
        ######################################################################
        concat = layers.concat([obs,act], axis=1)
        hid0 = self.fc1(concat)
        hid1 = self.fc2(hid0)
        hid2 = self.fc3(hid1)
        Q = layers.squeeze(hid2, axes=[1])
        ######################################################################
        ######################################################################
        return Q
    
class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

# * `Algorithm` 定义了具体的算法来更新前向网络(`Model`)，也就是通过定义损失函数来更新`Model`，和算法相关的计算都放在`algorithm`中。
from parl.algorithms import DDPG

class QuadrotorAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim=4):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(QuadrotorAgent, self).__init__(algorithm)

        # 注意，在最开始的时候，先完全同步target_model和model的参数
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)
            

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost

def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        raw_action = action[:]
        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        main_action = action[0]
        sub_action = action[1:]
        sub_action = np.random.normal(sub_action, 0.01)
        action = [main_action+0.2*x for x in sub_action]
        action = np.array(action)
        action = np.clip(action, -1.0, 1.0)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, raw_action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps

# 评估 agent, 跑 3 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(3):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)
            main_action = action[0]
            sub_action = action[1:]
            # sub_action = np.random.normal(sub_action, 0.01)
            action = [main_action+0.2*x for x in sub_action]
            action = np.clip(action, -1.0, 1.0)
            action = action_mapping(action, env.action_space.low[0], 
                                    env.action_space.high[0])

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)

# 创建飞行器环境
env = make_env("Quadrotor", task="hovering_control")
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# 根据parl框架构建agent
######################################################################
######################################################################
model = QuadrotorModel(act_dim=5)
algorithm = DDPG(model,gamma=GAMMA,tau=TAU,actor_lr=ACTOR_LR,critic_lr=CRITIC_LR)
agent = QuadrotorAgent(
    algorithm,
    obs_dim=obs_dim,
    act_dim=5)
# 6. 请构建agent:  QuadrotorModel, DDPG, QuadrotorAgent三者嵌套
#
######################################################################
######################################################################


# parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, 5)


ckpt = './steps_432037_9107.ckpt'  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称

agent.restore(ckpt)

# 启动训练
test_flag = 0
total_steps = 0
break_tag = 0
evaluate_reward_tag = -1000
while total_steps < TRAIN_TOTAL_STEPS:
    train_reward, steps = run_episode(env, agent, rpm)
    total_steps += steps
    #logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward

    if total_steps // TEST_EVERY_STEPS >= test_flag: # 每隔一定step数，评估一次模型
        while total_steps // TEST_EVERY_STEPS >= test_flag:
            test_flag += 1
 
        evaluate_reward = evaluate(env, agent)
        logger.info('Steps {}, Test reward: {}'.format(
            total_steps, evaluate_reward)) # 打印评估的reward

    # if evaluate_reward >= evaluate_reward_tag:
    #     evaluate_reward_tag = evaluate_reward + 300
    #     # 每评估一次，就保存一次模型，以训练的step数命名
    #     ckpt = './steps_%s_%s.ckpt' % (total_steps, evaluate_reward)
    #     agent.save(ckpt)

    # if evaluate_reward > 8000:
    #     break_tag += 1
    # else:
    #     break_tag = 0

    # if break_tag == 3:
    #     ckpt = './steps_%s_%s.ckpt' % (total_steps, evaluate_reward)
    #     agent.save(ckpt)
    #     break
agent.save(ckpt)
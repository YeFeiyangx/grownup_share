"""
PaddleParl的使用，可以简化算法的构建代码。

变异率E_GREED在不同阶段的控制，在这个案例中可以很好的体现；
变异率的衰减信号E_GREED_DECREMENT，可以让学习在末期更稳定。
对于一些目标明确的平稳场景，过拟合能够表现出更大的优势。
"""


# !pip uninstall -y parl  # 说明：AIStudio预装的parl版本太老，容易跟其他库产生兼容性冲突，建议先卸载
# !pip uninstall -y pandas scikit-learn # 提示：在AIStudio中卸载这两个库再import parl可避免warning提示，不卸载也不影响parl的使用

# !pip install gym
# # !pip install paddlepaddle==1.6.3 -i https://mirror.baidu.com/pypi/simple
# !pip install paddlepaddle-gpu==1.6.3.post97 -i https://mirror.baidu.com/pypi/simple
# !pip install parl==1.3.1 -i https://mirror.baidu.com/pypi/simple

# 说明：安装日志中出现两条红色的关于 paddlehub 和 visualdl 的 ERROR 与parl无关，可以忽略，不影响使用

import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger


LEARN_FREQ = 6 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
# MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_SIZE = 10000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 400  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 48   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.997 # reward 的衰减因子，一般取 0.9 到 0.999 不等
E_GREED = 0.01
E_GREED_DECREMENT = 1e-7
LEARNING_RATE = 0.0001 # 学习率

class Model(parl.Model):
    def __init__(self, act_dim):
        """
        文件中的dqn_model.ckpt
        """
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]

        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q


from parl.algorithms import DQN # 直接从parl库中导入DQN算法，无需自己重写算法

# class DQN(parl.Algorithm):
#     def __init__(self, model, act_dim=None, gamma=None, lr=None):
#         """ DQN algorithm
        
#         Args:
#             model (parl.Model): 定义Q函数的前向网络结构
#             act_dim (int): action空间的维度，即有几个action
#             gamma (float): reward的衰减因子
#             lr (float): learning rate 学习率.
#         """
#         self.model = model
#         self.target_model = copy.deepcopy(model)

#         assert isinstance(act_dim, int)
#         assert isinstance(gamma, float)
#         assert isinstance(lr, float)
#         self.act_dim = act_dim
#         self.gamma = gamma
#         self.lr = lr

#     def predict(self, obs):
#         """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
#         """
#         return self.model.value(obs)

#     def learn(self, obs, action, reward, next_obs, terminal):
#         """ 使用DQN算法更新self.model的value网络
#         """
#         # 从target_model中获取 max Q' 的值，用于计算target_Q
#         next_pred_value = self.target_model.value(next_obs)
#         best_v = layers.reduce_max(next_pred_value, dim=1)
#         best_v.stop_gradient = True  # 阻止梯度传递
#         terminal = layers.cast(terminal, dtype='float32')
#         target = reward + (1.0 - terminal) * self.gamma * best_v

#         pred_value = self.model.value(obs)  # 获取Q预测值
#         # 将action转onehot向量，比如：3 => [0,0,0,1,0]
#         action_onehot = layers.one_hot(action, self.act_dim)
#         action_onehot = layers.cast(action_onehot, dtype='float32')
#         # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
#         # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
#         #  ==> pred_action_value = [[3.9]]
#         pred_action_value = layers.reduce_sum(
#             layers.elementwise_mul(action_onehot, pred_value), dim=1)

#         # 计算 Q(s,a) 与 target_Q的均方差，得到loss
#         cost = layers.square_error_cost(pred_action_value, target)
#         cost = layers.reduce_mean(cost)
#         optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
#         optimizer.minimize(cost)
#         return cost

#     def sync_target(self):
#         """ 把 self.model 的模型参数值同步到 self.target_model
#         """
#         self.model.sync_weights_to(self.target_model)

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 150  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.001, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        # self.e_greed = max(
        #     0.01, self.e_greed*self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

# replay_memory.py
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


# 创建环境
env = gym.make('MountainCar-v0')
action_dim = env.action_space.n  # MountainCar-v0: 3
obs_shape = env.observation_space.shape  # MountainCar-v0: (2,)

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=E_GREED,  # 有一定概率随机选取动作，探索
    e_greed_decrement=E_GREED_DECREMENT)  # 随着训练逐步收敛，探索的程度慢慢降低
    # e_greed_decrement=0.999)  # 随着训练逐步收敛，探索的程度慢慢降低


# 加载模型
np.random.seed(1024)
# read_path = 'dqn_model_init.ckpt'
# agent.restore(read_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)


max_episode = 3000
n_count = []
# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 50):
        total_reward = run_episode(env, agent, rpm)
        episode += 1

    # test part
    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))


    if len(n_count) >= 10:
        n_count = n_count[1:]
    n_count.append(eval_reward)
    tag_cal = 0

    for i in n_count:
        if int(i) > -120:
            tag_cal += 1
    if tag_cal == 7:
        save_path = './dqn_model_len_' + str(tag_cal) + '.ckpt'
        agent.save(save_path)
    if tag_cal == 8:
        save_path = './dqn_model_len_' + str(tag_cal) + '.ckpt'
        agent.save(save_path)
    if eval_reward >= -95:
        save_path = './dqn_model_best_' + str(eval_reward) + '.ckpt'
        agent.save(save_path)
    if tag_cal == 10:
        break
    # 训练结束，保存模型

save_path = './dqn_model.ckpt'
agent.save(save_path)
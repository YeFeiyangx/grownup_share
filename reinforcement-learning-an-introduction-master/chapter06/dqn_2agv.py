import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym

import random
import collections

from parl.utils import logger

LEARN_FREQ = 6 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 10000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 400  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 48   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.997 # reward 的衰减因子，一般取 0.9 到 0.999 不等
LEARNING_RATE = 0.003 # 学习率

class Model(parl.Model):
    def __init__(self, act_dim):
        # 3层全连接网络
        self.cc1 = layers.conv2d(num_filters=16, filter_size=3, stride=1, padding=1, act='relu')
        self.cc2 = layers.conv2d(num_filters=16, filter_size=3, stride=1, padding=1, act='relu')
        self.cc3 = layers.conv2d(num_filters=32, filter_size=3, stride=1, padding=1, act='relu')
        self.fc1 = layers.fc(size=8, act=None)

    def value(self, obs):
        # 定义网络
        h1 = self.cc1(obs)
        h2 = self.cc2(h1)
        h3 = self.cc3(h2)
        h4 = layers.flatten(h3, axis=1)
        Q = self.fc1(h4)
        return Q

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
class AGV_ENV(object):
    def __init__(self, start_row=0, walk_boundary=[[0,5],[4,9]], goals=[(2, 1),(6,1),(7,4),(5,5),(6,8),(2,8)]):
        self.wolrd_width = 10
        self.wolrd_hight = 10
        self.actions_dims = 4
        self.start_row = start_row
        self.walk_boundary = walk_boundary
        self.goals_init = goals
        self.reset()

    def start_point_gen(self):
        position_set = set()
        self.point_position = []
        point_ = (self.start_row, np.random.randint(self.walk_boundary[0][1]+1))
        self.point_position.append(point_)
        position_set.add(point_)
        while len(position_set) < 2:
            point_ = (self.start_row, np.random.randint(self.walk_boundary[1][1]+1))
            position_set.add(point_)
        self.point_position.append(point_)

    def map_gen(self):
        self.observation_space = np.zeros((self.wolrd_hight,self.wolrd_width))
        self.goals = [i for i in self.goals if i not in self.point_position]
        for goal_point in self.goals:
            self.observation_space[goal_point[0], goal_point[1]] = 1
        self.observation_space[self.point_position[0][0], self.point_position[0][1]] = -10 # 这里可以看看变化了有什么不同
        self.observation_space[self.point_position[1][0], self.point_position[1][1]] = 5 # 这里可以看看变化了有什么不同

    def reset(self):
        self.goals = self.goals_init[:]
        self.start_point_gen()
        self.map_gen()
        return self.observation_space.reshape([1,10,10])

    def map_update(self):
        self.observation_space = np.zeros((self.wolrd_hight,self.wolrd_width))
        for goal_point in self.goals:
            self.observation_space[goal_point[0], goal_point[1]] = 1
            
        self.observation_space[self.point_position[0][0], self.point_position[0][1]] = -10 # 这里可以看看变化了有什么不同
        self.observation_space[self.point_position[1][0], self.point_position[1][1]] = 5 # 这里可以看看变化了有什么不同

    def step(self, action):
        next_state = []
        total_reward = 0
        for agv in range(2):
            _reward = 0
            i,j = self.point_position[agv]
            if action[agv] == ACTION_UP:
                _next_state = (max(i - 1, 0), j)
                if i == 0:
                    _reward = -15
            elif action[agv] == ACTION_LEFT:
                _next_state = (i, max(j - 1, self.walk_boundary[agv][0] ))
                if j == self.walk_boundary[agv][0]:
                    _reward = -15
            elif action[agv] == ACTION_RIGHT:
                _next_state = (i, min(j + 1, self.walk_boundary[agv][1] ))
                if j == self.walk_boundary[agv][1]:
                    _reward = -15
            elif action[agv] == ACTION_DOWN:
                _next_state = (min(i + 1, self.wolrd_hight - 1), j)
                if i == 9:
                    _reward = -15
            else:
                assert False
            if _reward == 0:
                _reward = -1
            if _next_state in self.goals:
                _reward = 10
            total_reward += _reward
            next_state.append(_next_state)

        crash_tag = 0
        if next_state[0] == next_state[1]:
            total_reward = -20
            next_state = self.point_position
            crash_tag = 1
        elif next_state[0][0] == next_state[1][0]:
            if (next_state[0][1] == self.point_position[1][1]) and (next_state[1][1] == self.point_position[0][1]):
                total_reward = -20
                next_state = self.point_position
                crash_tag = 1
        elif next_state[0][1] == next_state[1][1]:
            if (next_state[0][0] == self.point_position[1][0]) and (next_state[1][0] == self.point_position[0][0]):
                total_reward = -20
                next_state = self.point_position
                crash_tag = 1

        if crash_tag == 0:
            for agv in range(2):
                if next_state[agv] in self.goals:
                    self.goals.remove(next_state[agv])

        if len(self.goals) == 0:
            done = True
        else:
            done = False

        self.point_position = next_state
        self.map_update()

        return self.observation_space.reshape([1,10,10]), total_reward, done, None


class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning rate 学习率.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v
        pred_value = self.model.value(obs)  # 获取Q预测值
        action = fluid.layers.reshape(
            x=action, shape=[-1,2,1], inplace=True)
        # 将action转onehot向量，比如：[[3],[1]] => [[0,0,0,1],[0,1,0,0]] => [[0,0,0,1,0,1,0,0]]
        action_onehot = layers.one_hot(action, 4)
        
        action_onehot = fluid.layers.reshape(
            x=action_onehot, shape=[-1, 8], inplace=True)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0,1,0,0]]
        #  ==> pred_action_value = [sum([3.9, 1.2])] => [5.1]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        optimizer.minimize(cost)
        return cost

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)

# Agent负责算法与环境的交互，在交互过程中把生成的数据提供给`Algorithm`来更新模型(`Model`)，数据的预处理流程也定义在这里。
class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
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
                name='obs', shape=[1, self.obs_dim[0], self.obs_dim[1]], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[1, self.obs_dim[0], self.obs_dim[1]], dtype='float32')
            action = layers.data(name='act', shape=[2,1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[1, self.obs_dim[0], self.obs_dim[1]], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act_0 = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
            act_1 = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
            act = [act_0, act_1]
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
        act_0 = np.argmax(pred_Q[:4])  # 选择Q最大的下标，即对应的动作
        act_1 = np.argmax(pred_Q[4:])  # 选择Q最大的下标，即对应的动作
        return [act_0, act_1]

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
        return np.array(obs_batch).astype('float32'), np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while step < 500:
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
        step_num = 0
        while step_num < 500:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
            step_num += 1
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

# 创建环境
env = AGV_ENV()
action_dim = env.actions_dims  # 4

# 创建经验池
rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=8)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=[10, 10],
    act_dim=4,
    e_greed=0.1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=1e-7)  # 随着训练逐步收敛，探索的程度慢慢降低

# # 加载模型
# np.random.seed(1024)
# read_path = 'dqn2_model_init.ckpt'
# agent.restore(read_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 3000
# 开始训练
episode = 0
run = 0
PATH = os.path.abspath('.')
# 开始训练
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 50):
        total_reward = run_episode(env, agent, rpm)
        episode += 1
    run += 1
    # test part
    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))
    if run % 10 ==1:
        obj_name = 'dqn2_model_len_' + str(episode) + '.ckpt'
        agent.save(os.path.join(PATH, obj_name))
obj_name = 'dqn2_model.ckpt'
agent.save(os.path.join(PATH,obj_name))

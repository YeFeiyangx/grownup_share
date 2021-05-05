#%%
import numpy as np 
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STOP = 4
class AGV_ENV(object):
    def __init__(self, bg_size=[20,20], pass_side=[[0,0,20,20]], obs_point=[[0,5]], ran_obs_epoch=[[[0,20],[0,20]]], 
        ran_agv_start=[[[0,5],[0,5]],[[0,5],[15,20]]], ran_agv_target=[[[0,5],[0,5]],[[0,5],[15,20]]], 
        bg_pass=0, bg_obs = None, bg_car= None, bg_goal=None):

        # ## 等比表示背景
        # if bg_obs == None:
        #     if len(ran_agv_start) > 1:
        #         _q = len(ran_agv_start)
        #         self.bg_obs = -_q*(1-pow(_q, _q))/(1-_q)
        #     else:
        #         self.bg_obs = -10
        # else:
        #     self.bg_obs = bg_obs

        # if bg_car == None:
        #     if len(ran_agv_start) > 1:
        #         _q = len(ran_agv_start)
        #         self.bg_car = []
        #         for i in range(1,_q+1):
        #             self.bg_car.append((1-pow(_q, i))/(1-_q))
        #     else:
        #         self.bg_car = 10
        # else:
        #     self.bg_car = bg_car

        ## 逐步+1表示车，背景固定
        self.bg_pass = bg_pass
        self.obs_point = obs_point

        if bg_obs == None:
            self.bg_obs = -20
        else:
            self.bg_obs = bg_obs

        if bg_car == None:
            self.bg_car = []
            _q = len(ran_agv_start)
            for i in range(1,_q+1):
                self.bg_car.append(i)
        else:
            self.bg_car = bg_car

        if bg_goal == None:
            self.bg_goal = []
            n = (sum(self.bg_car)//10 + 1)*10
            _q = len(ran_agv_start)
            for i in range(1, _q+1):
                self.bg_goal.append(n + i)
        else:
            self.bg_goal = bg_goal

        self.ran_agv_start = ran_agv_start
        self.ran_agv_target = ran_agv_target
        self.ran_obs_epoch = ran_obs_epoch
        self.pass_side = pass_side
        self.bg_size = bg_size

        self.reset()

    def start_point_gen(self):
        position_set = set()
        self.point_position = []
        agv_num = len(self.ran_agv_start)
        for i in range(agv_num):
            point_ = (np.random.randint(self.ran_agv_start[i][0][0], self.ran_agv_start[i][0][1]), 
                        np.random.randint(self.ran_agv_start[i][1][0], self.ran_agv_start[i][1][1]))
            n = 0
            while point_ in position_set:
                point_ = (np.random.randint(self.ran_agv_start[i][0][0], self.ran_agv_start[i][0][1]), 
                            np.random.randint(self.ran_agv_start[i][1][0], self.ran_agv_start[i][1][1]))
                n += 1
                if n > 1000:
                    raise print('无法生成agv初始起点')
            self.point_position.append(point_)
            position_set.add(point_)

        self.goals = []
        goals_raw_set = set()
        agv_num = len(self.ran_agv_target)
        for i in range(agv_num):
            point_ = (i, np.random.randint(self.ran_agv_target[i][0][0], self.ran_agv_target[i][0][1]), 
                        np.random.randint(self.ran_agv_target[i][1][0], self.ran_agv_target[i][1][1]))
            n = 0
            while (point_[1], point_[2]) in goals_raw_set:
                point_ = (i, np.random.randint(self.ran_agv_target[i][0][0], self.ran_agv_target[i][0][1]), 
                            np.random.randint(self.ran_agv_target[i][1][0], self.ran_agv_target[i][1][1]))
                n += 1
                if n > 1000:
                    raise print('无法生成agv初始目标')

            self.goals.append(point_)
            goals_raw_set.add((point_[1], point_[2]))
        self.goals_raw_list = list(goals_raw_set)

        from collections import defaultdict
        self.goals_induce = defaultdict(set)
        for i in range(agv_num):
            for j in range(3,17):
                self.goals_induce[i].add((7,j))
        print('self.goals_induce:',self.goals_induce)
        self.ran_obs = []
        position_set = set()
        agv_num = len(self.ran_obs_epoch)
        for i in range(agv_num):
            point_ = (np.random.randint(self.ran_obs_epoch[i][0][0], self.ran_obs_epoch[i][0][1]), 
                        np.random.randint(self.ran_obs_epoch[i][1][0], self.ran_obs_epoch[i][1][1]))
            n = 0
            while point_ in position_set:
                point_ = (np.random.randint(self.ran_obs_epoch[i][0][0], self.ran_obs_epoch[i][0][1]), 
                            np.random.randint(self.ran_obs_epoch[i][1][0], self.ran_obs_epoch[i][1][1]))
                n += 1
                if n > 1000:
                    raise print('无法生成每批次阻挡目标')

            self.ran_obs.append(point_)
            position_set.add(point_)


    def map_gen(self):
        self.observation_space = np.zeros([self.bg_size[0], self.bg_size[1]])
        self.observation_space[:] = self.bg_obs
        for pass_part in self.pass_side:
            self.observation_space[pass_part[0]:pass_part[2], pass_part[1]:pass_part[3]] = self.bg_pass
        
        for _obs_point in self.obs_point:
            self.observation_space[_obs_point[0], _obs_point[1]] = self.bg_obs

        for goal_point in self.goals:
            self.observation_space[goal_point[1], goal_point[2]] = self.bg_goal[goal_point[0]]
        
        _n = 0
        # print("self.point_position:",self.point_position)
        # print('self.bg_car:',self.bg_car)
        # print('self.bg_goal:',self.bg_goal)
        # print('self.bg_obs:',self.bg_obs)
        # print('self.bg_pass:',self.bg_pass)
        for start_point in self.point_position:
            self.observation_space[start_point[0], start_point[1]] = self.bg_car[_n]
            _n+=1

    def reset(self):
        self.start_point_gen()
        self.map_gen()
        # np.savetxt('init_map.csv', self.observation_space, fmt='%d',delimiter=',')
        # return self.observation_space.reshape([1,self.bg_size[0],self.bg_size[1]])

    def step(self, action):
        next_state = []
        total_reward = 0
        batch_goal = []
        for agv in range(2):
            _reward = -5
            i,j = self.point_position[agv]
            if action[agv] == ACTION_UP:
                _next_state = (max(i - 1, 0), j)
                if i == 0:
                    _reward = -15
            elif action[agv] == ACTION_LEFT:
                _next_state = (i, max(j - 1, 0))
                if j == 0:
                    _reward = -15

            elif action[agv] == ACTION_RIGHT:
                _next_state = (i, min(j + 1, self.bg_size[1]-1))
                if j == self.bg_size[1]:
                    _reward = -15

            elif action[agv] == ACTION_DOWN:
                _next_state = (min(i + 1, self.bg_size[0]-1), j)
                if i == self.bg_size[0]:
                    _reward = -15
            elif action[agv] == ACTION_STOP:
                _next_state = (i, j)
                _reward = -1
            else:
                assert False

            if self.observation_space[_next_state] == -20:
                _next_state = (i, j)
                _reward == -15
            elif _next_state in self.goals_induce[agv]:
                _reward == 1
            elif _reward == -5:
                _reward = -2
            _batch_goal = list(_next_state)
            _batch_goal.insert(0,agv)
            _batch_goal = tuple(_batch_goal)
            batch_goal.append(_batch_goal)
            if _batch_goal in self.goals:
                _reward = 50
            total_reward += _reward
            next_state.append(_next_state)
        # np.savetxt('before_action_map.csv', self.observation_space, fmt='%d',delimiter=',')
        # print('agv_1 point update:', self.point_position[0], self.observation_space[self.point_position[0][0], self.point_position[0][1]])
        # print('agv_2 point update:', self.point_position[1], self.observation_space[self.point_position[1][0], self.point_position[1][1]])
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
                if batch_goal[agv] in self.goals[agv]:
                    self.goals.remove(batch_goal[agv])
                if next_state[agv] in self.goals_induce[agv]:
                    self.goals_induce[agv].remove(next_state[agv])

        if len(self.goals) == 0:
            done = True
        else:
            done = False

        self.point_position = next_state
        self.map_update()

        return self.observation_space.reshape([1,self.bg_size[0],self.bg_size[1]]), total_reward, done, None

    def map_update(self):
        # print('self.goals_raw_list:', self.goals_raw_list)
        for goal_point in self.goals_raw_list:
            self.observation_space[goal_point[0], goal_point[1]] = 0
        for goal_point in self.goals:
            self.observation_space[goal_point[1], goal_point[2]] = self.bg_goal[goal_point[0]]
        
        self.observation_space[self.point_position[0][0], self.point_position[0][1]] = self.bg_car[0] # 这里可以看看变化了有什么不同
        # self.observation_space[self.point_position[1][0], self.point_position[1][1]] = self.bg_car[1] # 这里可以看看变化了有什么不同
        # np.savetxt('after_action_map.csv', self.observation_space, fmt='%d',delimiter=',')
        # print('agv_1 point update:', self.point_position[0], self.observation_space[self.point_position[0][0], self.point_position[0][1]])
        # print('agv_2 point update:', self.point_position[1], self.observation_space[self.point_position[1][0], self.point_position[1][1]])



# 创建环境
env = AGV_ENV(pass_side=[[5,0, 10,3], [7,0,8,20], [5,17,10,20]], obs_point=[[5,1]], ran_obs_epoch=[],
            ran_agv_start=[[[5,10],[0,3]],[[5,10],[17,20]]], ran_agv_target=[[[5,10],[17,20]], [[5,10],[0,3]]])
# %%

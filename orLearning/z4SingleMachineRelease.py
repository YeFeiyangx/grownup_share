# Copyright 2010-2018 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single machine jobshop with setup times, release dates and due dates."""

import argparse

from google.protobuf import text_format
from ortools.sat.python import cp_model

#----------------------------------------------------------------------------
# Command line arguments.
PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    '--output_proto',
    default='',
    help='Output file to write the cp_model'
    'proto to.')
PARSER.add_argument('--params', default='', help='Sat solver parameters.')
PARSER.add_argument(
    '--preprocess_times',
    default=True,
    type=bool,
    help='Preprocess setup times and durations')


#----------------------------------------------------------------------------
# Intermediate solution printer
## 尴尬了啊，这玩意还真的不能乱用的呀。。。这是个有套路的用法啊
## on_solution_callback在父类中也有
class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
    ## 这玩意是个套路用法名，虽然__init__本身没用它，但是它在父类中有蹊跷，不能乱改
    def on_solution_callback(self):
        """Called after each new solution found."""
        print('Solution %i, time = %f s, objective = %i' %
              (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1


def main(args):
    """Solves a complex single machine jobshop scheduling problem."""

    parameters = args.params
    output_proto = args.output_proto

    #----------------------------------------------------------------------------
    # Data.

    job_durations = [
        2546, 8589, 5953, 3710, 3630, 3016, 4148, 8706, 1604, 5502, 9983, 6209,
        9920, 7860, 2176
    ]  # len(job_durations) is job num

    setup_times = [
        [
            3559, 1638, 2000, 3676, 2741, 2439, 2406, 1526, 1600, 3356, 4324,
            1923, 3663, 4103, 2215
        ],
        [
            1442, 3010, 1641, 4490, 2060, 2143, 3376, 3891, 3513, 2855, 2653,
            1471, 2257, 1186, 2354
        ],
        [
            1728, 3583, 3243, 4080, 2191, 3644, 4023, 3510, 2135, 1346, 1410,
            3565, 3181, 1126, 4169
        ],
        [
            1291, 1703, 3103, 4001, 1712, 1137, 3341, 3485, 2557, 2435, 1972,
            1986, 1522, 4734, 2520
        ],
        [
            4134, 2200, 1502, 3995, 1277, 1808, 1020, 2078, 2999, 1605, 1697,
            2323, 2268, 2288, 4856
        ],
        [
            4974, 2480, 2492, 4088, 2587, 4652, 1478, 3942, 1222, 3305, 1206,
            1024, 2605, 3080, 3516
        ],
        [
            1903, 2584, 2104, 1609, 4745, 2691, 1539, 2544, 2499, 2074, 4793,
            1756, 2190, 1298, 2605
        ],
        [
            1407, 2536, 2296, 1769, 1449, 3386, 3046, 1180, 4132, 4783, 3386,
            3429, 2450, 3376, 3719
        ],
        [
            3026, 1637, 3628, 3096, 1498, 4947, 1912, 3703, 4107, 4730, 1805,
            2189, 1789, 1985, 3586
        ],
        [
            3940, 1342, 1601, 2737, 1748, 3771, 4052, 1619, 2558, 3782, 4383,
            3451, 4904, 1108, 1750
        ],
        [
            1348, 3162, 1507, 3936, 1453, 2953, 4182, 2968, 3134, 1042, 3175,
            2805, 4901, 1735, 1654
        ],
        [
            1099, 1711, 1245, 1067, 4343, 3407, 1108, 1784, 4803, 2342, 3377,
            2037, 3563, 1621, 2840
        ],
        [
            2573, 4222, 3164, 2563, 3231, 4731, 2395, 1033, 4795, 3288, 2335,
            4935, 4066, 1440, 4979
        ],
        [
            3321, 1666, 3573, 2377, 4649, 4600, 1065, 2475, 3658, 3374, 1138,
            4367, 4728, 3032, 2198
        ],
        [
            2986, 1180, 4095, 3132, 3987, 3880, 3526, 1460, 4885, 3827, 4945,
            4419, 3486, 3805, 3804
        ],
        [
            4163, 3441, 1217, 2941, 1210, 3794, 1779, 1904, 4255, 4967, 4003,
            3873, 1002, 2055, 4295
        ],
    ]

    due_dates = [
    #   0,  1,      2,  3,     4,     5,     6,     7,  8,     9,    10,    11,    12, 13, 14
       -1, -1,  28569, -1, 98104, 27644, 55274, 57364, -1,    -1, 60875, 96637, 77888, -1, -1
    ] ## 每一个任务的交期，处理结束时间要小于等于deadline
    release_dates = [   ## 每一个任务的可处理时间，处理开始时间要大于等于release，最多就是一直等咯
    #   0,  1,      2,  3,     4,     5,     6,     7,  8,     9,    10,    11,    12, 13, 14
        0,  0,      0,  0, 19380,     0,     0, 48657,  0, 27932,     0,     0, 24876,  0,  0
    ]

    precedences = [(0, 2), (1, 2)] ## JOB 2要再JOB 0之后，JOB2也要再JOB1之后

    #----------------------------------------------------------------------------
    # Helper data.
    num_jobs = len(job_durations)
    all_jobs = range(num_jobs)

    #----------------------------------------------------------------------------
    # Preprocess.
    if args.preprocess_times:
        for job_id in all_jobs: # 遍历每一个job
            ## 获得最小准入时间
            min_incoming_setup = min(
                setup_times[j][job_id] for j in range(num_jobs + 1)) ## 0~16行中 setup_times中Job_id下的时间 取最小

            if release_dates[job_id] != 0:
                ## 每个任务的释放时间，和  0~16行中 setup_times中Job_id下的时间 取最小
                ## 任务开始就可用，就无需比较可用时间
                min_incoming_setup = min(min_incoming_setup,
                                         release_dates[job_id])

            if min_incoming_setup == 0: ## 如果最小准入时间为0，那么接下来步骤的设置就不需要了
                continue

            print('job %i has a min incoming setup of %i' %
                  (job_id, min_incoming_setup)) ## 打印每一个job的最小被设置的时间，譬如JOB 0是1099

            # We can transfer some setup times to the duration of the job.
            ## 对于这个JOB来说，就是持续时间增加最小可以被设置时间 1099 成为了从0开始的绝对时间
            job_durations[job_id] += min_incoming_setup

            # Decrease corresponding incoming setup times.
            ## 同时把setup_times中的每一个可被置入时间减最小被设置时间 因为持续时间增加， setuptime减少后，也成为了绝对时间
            for j in range(num_jobs + 1):
                setup_times[j][job_id] -= min_incoming_setup

            # Adjust release dates if needed.
            ## 这就以为着任务可以提早被设置，所以减了以后，也成为了绝对时间
            if release_dates[job_id] != 0:
                release_dates[job_id] -= min_incoming_setup

    #----------------------------------------------------------------------------
    # Model.
    model = cp_model.CpModel()

    #----------------------------------------------------------------------------
    # Compute a maximum makespan greedily.
    ## 最大持续时间总和，加上所有setuptime最大值
    horizon = sum(job_durations) + sum(max(setup_times[i][j] for i in range(num_jobs + 1))
        for j in range(num_jobs))
    print('Greedy horizon =', horizon)

    #----------------------------------------------------------------------------
    # Global storage of variables.
    intervals = []
    starts = []
    ends = []

    #----------------------------------------------------------------------------
    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        duration = job_durations[job_id]
        release_date = release_dates[job_id]
        due_date = due_dates[job_id] if due_dates[job_id] != -1 else horizon
        print('job %2i: start = %5i, duration = %4i, end = %6i' %
              (job_id, release_date, duration, due_date))
        name_suffix = '_%i' % job_id
        start = model.NewIntVar(release_date, due_date, 's' + name_suffix)
        end = model.NewIntVar(release_date, due_date, 'e' + name_suffix)
        interval = model.NewIntervalVar(start, duration, end, 'i' + name_suffix)
        starts.append(start)
        ends.append(end)
        intervals.append(interval)

    # No overlap constraint.
    model.AddNoOverlap(intervals)

    #----------------------------------------------------------------------------
    # Transition times using a circuit constraint.
    ## 环形约束 从0到每一个JOB，再从每一个JOB回到虚拟节点0
    arcs = []
    for i in all_jobs:
        # Initial arc from the dummy node (0) to a task.
        ## 虚拟节点0作为每个任务的初始节点
        start_lit = model.NewBoolVar('')
        arcs.append([0, i + 1, start_lit]) # 虚0，到实0，布尔值确认是否 这条路真的开放，即确认这条路是否是起始点
        # If this task is the first, set to minimum starting time.
        ## 譬如任务的释放时间是1点，机器的准备时间是1点10分，那么只有1点10分可以进行
        ## 同理任务的释放时间是1点10分，机器的准备时间是1点，那么也是只有1点10分可以进行
        min_start_time = max(release_dates[i], setup_times[0][i])       # 最短开始时间是 任务释放时间和setup[0][job_id]的最大值
        model.Add(starts[i] == min_start_time).OnlyEnforceIf(start_lit) # 只有当这个起始点路径实锤了，即start_lit = 1，那么开始时间便有了起点
        # Final arc from an arc to the dummy node.
        arcs.append([i + 1, 0, model.NewBoolVar('')]) # 路径把 环形的终点加进去 0 -> i+1 -> 0

        for j in all_jobs:
            if i == j:      # 起终点相同,就没需要额外登记的路径了
                continue
            # 起终点不同时，把每一个I,J对应的点位加入路径
            # 简单来说是这么添加路径，先添加起始点0的各个路径，然后再添加单轮起点i至其它遍历点j的路径
            # 因为虚拟点占用了0，所以其它点依次都是默认+1的
            lit = model.NewBoolVar('%i follows %i' % (j, i))
            arcs.append([i + 1, j + 1, lit])

            # We add the reified precedence to link the literal with the times of the
            # two tasks.
            # If release_dates[j] == 0, we can strenghten this precedence into an
            # equality as we are minimizing the makespan.
            ## 疯狂添加约束
            if release_dates[j] == 0:
                ## 当release为0时，一切都简单了，因为求span最小，j的开始和i，铁定不存在间隙，即i的结尾，加上i到j的放置时间，一定是start[j]
                ## 至于为什么会时i+1，，还要追寻虚拟节点0这个兄弟把0的茅坑占住了
                model.Add(starts[j] == ends[i] +
                          setup_times[i + 1][j]).OnlyEnforceIf(lit)
            else:
                ## 如果release不为0，那么问题就来了，因为下一个j的任务的release很可能会让start[j]远离上一个i的endtime
                ## 所以这里只要确保大于等于就好了
                model.Add(starts[j] >=
                          ends[i] + setup_times[i + 1][j]).OnlyEnforceIf(lit)
    # 把所有路径都显示出来
    # print("arcsarcsarcsarcs:", arcs)
    """Adds Circuit(arcs).

    Adds a circuit constraint from a sparse list of arcs that encode the graph.

    A circuit is a unique Hamiltonian path in a subgraph of the total
    graph. In case a node 'i' is not in the path, then there must be a
    loop arc 'i -> i' associated with a true literal. Otherwise
    this constraint will fail.

    Args:
      arcs: a list of arcs. An arc is a tuple (source_node, destination_node,
        literal). The arc is selected in the circuit if the literal is true.
        Both source_node and destination_node must be integers between 0 and the
        number of nodes - 1.

    Returns:
      An instance of the `Constraint` class.

    Raises:
      ValueError: If the list of arcs is empty.
    """
    model.AddCircuit(arcs)

    #----------------------------------------------------------------------------
    # Precedences.
    ## 添加约束，零件的先后顺序
    for before, after in precedences:
        print('job %i is after job %i' % (after, before))
        model.Add(ends[before] <= starts[after])
    
    #----------------------------------------------------------------------------
    # Objective.
    makespan = model.NewIntVar(0, horizon, 'makespan')
    ## 哥们，这个时所有点的ends啊？这玩意这么添加的？
    ## model.NewIntVar(release_date, due_date, 'e' + "job_id")
    ## 添加最大span要等与ends中最大的值
    model.AddMaxEquality(makespan, ends)
    ## 求最大啊span中最小的可能
    model.Minimize(makespan)

    #----------------------------------------------------------------------------
    # Write problem to file.
    if output_proto:
        print('Writing proto to %s' % output_proto)
        with open(output_proto, 'w') as text_file:
            text_file.write(str(model))

    #----------------------------------------------------------------------------
    # Solve.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60 * 60 * 2 ## 兄弟，这个应该时限制的求解时间
    if parameters:
        text_format.Merge(parameters, solver.parameters)
    solution_printer = SolutionPrinter()
    ## 不断求解，找到个解就回传
    """Solves a problem and passes each solution found to the callback."""
    solver.SolveWithSolutionCallback(model, solution_printer)
    print(solver.ResponseStats())
    for job_id in all_jobs:
        print('job %i starts at %i end ends at %i' %
              (job_id, solver.Value(starts[job_id]),
               solver.Value(ends[job_id])))


if __name__ == '__main__':
    main(PARSER.parse_args())

    
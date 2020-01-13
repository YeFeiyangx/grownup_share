# %%
from ortools.graph import pywrapgraph
import time


def main():
    """Solving an Assignment Problem with MinCostFlow"""

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.
    # 描述每一条通路start_nodes[i]->end_nodes[i] = capacities[i],costs[i]
    start_nodes = [0, 0, 0, 0] + [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4] + [5, 6, 7, 8]
    end_nodes   = [1, 2, 3, 4] + [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8] + [9, 9, 9, 9]
    capacities  = [1, 1, 1, 1] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]
    costs = ([0, 0, 0, 0] + [90, 76, 75, 70, 35, 85, 55, 65, 125, 95, 90, 105, 45, 110, 95, 115]
             + [0, 0, 0, 0])
    # Define an array of supplies at each node.
    supplies = [4, 0, 0, 0, 0, 0, 0, 0, 0, -4]
    source = 0
    sink = 9
    # 1,2,3,4 woker; 5,6,7,8 tasks。 
    tasks = 4

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.
    # 九个节点的供给，0发布4个任务，到达9，则任务消耗完毕，所以每个通路的capacity是1
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])
    # Find the minimum cost flow between node 0 and node 10.
    # 每个工人做每个任务都是不同成本的，那么0~10个点的通经要求完成目标的情况下消耗最小
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        print('Total cost = ', min_cost_flow.OptimalCost())
        print()
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc) != source and min_cost_flow.Head(arc) != sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    print('Worker %d assigned to task %d.  Cost = %d' % (
                          min_cost_flow.Tail(arc),
                          min_cost_flow.Head(arc),
                          min_cost_flow.UnitCost(arc)))
    else:
        print('There was an issue with the min cost flow input.')


if __name__ == '__main__':
    start_time = time.clock()
    main()
    print()
    print("Time =", time.clock() - start_time, "seconds")

#%%

from ortools.graph import pywrapgraph
import time


def main():
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Define the directed graph for the flow.
    team_A = [1, 3, 5]
    team_B = [2, 4, 6]
    ## 4任务均衡从0分为两批，cap各为2，进入至次级任务分发节点11，12，分发任务节点cost为0；
    ## 从次级节点分别给相应的可操作的员工1~6，分发cost为0，一人一任务，所以cap为1；
    ## 实际任务节点是7~10，分别由六个员工可选择的cover，每个人做每个任务都有相应的cost；
    ## 因为cap为1，所以各个一个任务最多被一个人cover；同时一个人最多cover一个任务；
    ## 进入结束点sink 13
    start_nodes = ([0, 0] + [11, 11, 11] + [12, 12, 12] +
                   [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6] +
                   [7, 8, 9, 10])
    end_nodes = ([11, 12] + team_A + team_B +
                 [7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10] +
                 [13, 13, 13, 13])
    capacities = ([2, 2] + [1, 1, 1] + [1, 1, 1] +
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] +
                  [1, 1, 1, 1])
    costs = ([0, 0] + [0, 0, 0] + [0, 0, 0] +
             [90, 76, 75, 70, 35, 85, 55, 65, 125, 95, 90, 105, 45, 110, 95, 115, 60, 105,
              80, 75, 45, 65, 110, 95] + [0, 0, 0, 0])

    # Define an array of supplies at each node.

    # 源头的分发总流数为4，cap为通道，并不等价流数
    supplies = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4]
    source = 0
    # 节点是13
    sink = 13

    # Add each arc.
    for i in range(0, len(start_nodes)):
        # 把容量cap和消耗cost加入模型
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.

    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        min_cost_flow.Solve()
        print('Total cost = ', min_cost_flow.OptimalCost())
        print()

        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or intermediate nodes, or into sink.
            # 完全是为了后面打印所需，把连通路径归化了
            if (min_cost_flow.Tail(arc) != 0 and min_cost_flow.Tail(arc) != 11 and min_cost_flow.Tail(arc) != 12
                    and min_cost_flow.Head(arc) != 13):

                # Arcs in the solution will have a flow value of 1. There start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    print('Worker %d assigned to task %d.  Cost = %d' % (
                          min_cost_flow.Tail(arc),
                          min_cost_flow.Head(arc),
                          min_cost_flow.UnitCost(arc)))
    else:
        print('There was an issue with the min cost flow input.')


if __name__ == '__main__':
    start_time = time.clock()
    main()
    print()
    print("Time =", time.clock() - start_time, "seconds")

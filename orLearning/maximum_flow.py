"""From Taha 'Introduction to Operations Research', example 6.4-2."""

from ortools.graph import pywrapgraph


def main():
    """MaxFlow simple interface example."""

    # Define three parallel arrays: start_nodes, end_nodes, and the capacities
    # between each pair. For instance, the arc from node 0 to node 1 has a
    # capacity of 20.
    # 描述每一条通路start_nodes[i]->end_nodes[i] = capacities[i]
    # 映射每一条通路的流量
    start_nodes = [0, 0, 0, 1, 1, 2, 2, 3, 3]
    end_nodes = [1, 2, 3, 2, 4, 3, 4, 2, 4]
    capacities = [20, 30, 10, 40, 30, 10, 20, 5, 20]


    # Instantiate a SimpleMaxFlow solver.
    ## Assignment 和 Flow 应该都在 pywrapgraph里
    max_flow = pywrapgraph.SimpleMaxFlow()
    # Add each arc.
    # 把通路信息加到模型里
    for i in range(0, len(start_nodes)):
        max_flow.AddArcWithCapacity(
            start_nodes[i], end_nodes[i], capacities[i])

    # Find the maximum flow between node 0 and node 4.
    # 解0~4的节点的解
    if max_flow.Solve(0, 4) == max_flow.OPTIMAL:
        print('Max flow:', max_flow.OptimalFlow())
        print('')
        print('  Arc    Flow / Capacity')
        for i in range(max_flow.NumArcs()):
            print('%1s -> %1s   %3s  / %3s' % (
                max_flow.Tail(i),
                max_flow.Head(i),
                max_flow.Flow(i),
                max_flow.Capacity(i)))
        print('Source side min-cut:', max_flow.GetSourceSideMinCut())
        print('Sink side min-cut:', max_flow.GetSinkSideMinCut())
    else:
        print('There was an issue with the max flow input.')


if __name__ == '__main__':
    main()

#%%
"""
虽然有流量，但是每个流量都有消耗。
"""

# """From Bradley, Hax, and Magnanti, 'Applied Mathematical Programming', figure 8.1."""

from ortools.graph import pywrapgraph


def main():
    """MinCostFlow simple interface example."""

    # Define four parallel arrays: start_nodes, end_nodes, capacities, and unit costs
    # between each pair. For instance, the arc from node 0 to node 1 has a
    # capacity of 15 and a unit cost of 4.
    # 描述每一条通路start_nodes[i]->end_nodes[i] = capacities[i],unit_costs[i]
    start_nodes = [0, 0,  1, 1,  1,  2, 2,  3, 4]
    end_nodes = [1, 2,  2, 3,  4,  3, 4,  4, 2]
    capacities = [15, 8, 20, 4, 10, 15, 4, 20, 5]
    unit_costs = [4, 4,  2, 2,  6,  1, 3,  2, 3]

    # Define an array of supplies at each node.
    # 一共五个点，每个点都有可能有供给，有消耗，那么多一个维度描述消耗和供给
    supplies = [20, 0, 0, -5, -15]

    # Instantiate a SimpleMinCostFlow solver.
    ## 当然是在最大流量下的最小消耗咯
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc.
    for i in range(0, len(start_nodes)):
        ## mf里面的方法并不多，功能还是比较单一的
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], unit_costs[i])

    # Add node supplies.
    # 供给点补上
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        print('Minimum cost:', min_cost_flow.OptimalCost())
        print('')
        print('  Arc    Flow / Capacity  Cost')
        for i in range(min_cost_flow.NumArcs()):
            cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
            print('%1s -> %1s   %3s  / %3s       %3s' % (
                min_cost_flow.Tail(i),
                min_cost_flow.Head(i),
                min_cost_flow.Flow(i),
                min_cost_flow.Capacity(i),
                cost))
    else:
        print('There was an issue with the min cost flow input.')


if __name__ == '__main__':
    main()

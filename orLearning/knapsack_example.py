from ortools.algorithms import pywrapknapsack_solver


def main():
    # Create the solver.
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

    values = [
        360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
        78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
        87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
        312
    ]
    weights = [[
        7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
        42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
        3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13
    ]]
    capacities = [850]

    solver.Init(values, weights, capacities)
    computed_value = solver.Solve()

    packed_items = []
    packed_weights = []
    total_weight = 0
    print('Total value =', computed_value)
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
    print('Total weight:', total_weight)
    print('Packed items:', packed_items)
    print('Packed_weights:', packed_weights)


if __name__ == '__main__':
    main()

#%%
"""Solves a multiple knapsack problem using the CP-SAT solver."""

from ortools.sat.python import cp_model



def create_data_model():
    """Create the data for the example."""
    data = {}
    ## 每个物品的重量
    weights = [48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36]
    ## 每个物品的价值
    values = [10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25]
    ## 物品个数
    data['num_items'] = len(weights)
    ## 给所有物品编号
    data['all_items'] = range(data['num_items'])

    data['weights'] = weights
    data['values'] = values

    ## 共多少个装物品的箱子，每个箱子的容积，箱子的编号
    data['bin_capacities'] = [100, 100, 100, 100, 100]
    data['num_bins'] = len(data['bin_capacities'])
    data['all_bins'] = range(data['num_bins'])
    return data




def print_solutions(data, solver, x):
    """Display the solution."""
    total_weight = 0
    total_value = 0
    for b in data['all_bins']:
        print('Bin', b, '\n')
        bin_weight = 0
        bin_value = 0
        for idx, val in enumerate(data['weights']):
            if solver.Value(x[(idx, b)]) > 0:
                print('Item', idx, '-  Weight:', val, ' Value:',
                      data['values'][idx])
                bin_weight += val
                bin_value += data['values'][idx]
        print('Packed bin weight:', bin_weight)
        print('Packed bin value:', bin_value, '\n')
        total_weight += bin_weight
        total_value += bin_value
    print('Total packed weight:', total_weight)
    print('Total packed value:', total_value)




def main():
    data = create_data_model()

    model = cp_model.CpModel()

    # Main variables.
    x = {}
    for idx in data['all_items']:
        for b in data['all_bins']:
            ## 来了，笛卡尔集，所有箱子放各种物品的标签
            x[(idx, b)] = model.NewIntVar(0, 1, 'x_%i_%i' % (idx, b))

    # 最大价值用于衡量上届
    max_value = sum(data['values'])
    # value[b] is the value of bin b when packed.
    ## 每个箱子的编号，以及其价值的上下界
    value = [
        model.NewIntVar(0, max_value, 'value_%i' % b) for b in data['all_bins']
    ]
    for b in data['all_bins']:
        ## 这是一个约束，每个箱子的价值必须全等于 物品价值的总和
        model.Add(value[b] == sum(
            x[(i, b)] * data['values'][i] for i in data['all_items']))

    # Each item can be in at most one bin.
    ## 通过笛卡尔集，来描述一个物件只能放在一个箱子里
    for idx in data['all_items']:
        model.Add(sum(x[idx, b] for b in data['all_bins']) <= 1)

    # The amount packed in each bin cannot exceed its capacity.
    for b in data['all_bins']:
        model.Add(
            sum(x[(i, b)] * data['weights'][i]
                for i in data['all_items']) <= data['bin_capacities'][b])

    # Maximize total value of packed items.
    model.Maximize(sum(value))

    solver = cp_model.CpSolver()

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print_solutions(data, solver, x)


if __name__ == '__main__':
    main()

# %%

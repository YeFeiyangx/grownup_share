# %%
"""
MIP求解指派问题，比FLOW方法低效20倍
"""

from ortools.linear_solver import pywraplp


def main():
    solver = pywraplp.Solver('SolveAssignmentProblemMIP',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    ## 分别为6个人做不同任务7~10的消耗
    cost = [[90, 76, 75, 70],
            [35, 85, 55, 65],
            [125, 95, 90, 105],
            [45, 110, 95, 115],
            [60, 105, 80, 75],
            [45, 65, 110, 95]]
    ## 两只队伍6个人
    team1 = [0, 2, 4]
    team2 = [1, 3, 5]
    team_max = 2
    ## 消耗矩阵确定人，那么人数和任务数就确定了
    num_workers = len(cost)
    num_tasks = len(cost[1])
    x = {}

    ## 建造人和任务的笛卡尔集
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.BoolVar('x[%i,%i]' % (i, j))

    # Objective
    ## 目标是笛卡尔集中，通过消耗计算所有费用最小
    solver.Minimize(solver.Sum([cost[i][j] * x[i, j] for i in range(num_workers)
                                for j in range(num_tasks)]))

    # Constraints

    # Each worker is assigned to at most 1 task.
    ## 添加一个员工只能干一个任务的约束
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    # Each task is assigned to exactly one worker.
    ## 添加一个任务只能被一个人干的约束
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    # Each team takes on two tasks.
    ## 添加一个团队只能做两个任务的约束
    solver.Add(solver.Sum([x[i, j]
                           for i in team1 for j in range(num_tasks)]) <= team_max)
    solver.Add(solver.Sum([x[i, j]
                           for i in team2 for j in range(num_tasks)]) <= team_max)
    sol = solver.Solve()

    print('Total cost = ', solver.Objective().Value())
    print()
    for i in range(num_workers):
        for j in range(num_tasks):
            if x[i, j].solution_value() > 0:
                print('Worker %d assigned to task %d.  Cost = %d' % (
                      i,
                      j,
                      cost[i][j]))

    print()
    print("Time = ", solver.WallTime(), " milliseconds")


if __name__ == '__main__':
    main()

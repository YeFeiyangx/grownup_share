#%%
"""
在大多数问题上CAP-SAT比CBC源生MIP要快；
"""
from ortools.sat.python import cp_model
import time
import numpy as np


def main():
    model = cp_model.CpModel()
    start = time.time()
    cost = [[90, 76, 75, 70, 50, 74],
            [35, 85, 55, 65, 48, 101],
            [125, 95, 90, 105, 59, 120],
            [45, 110, 95, 115, 104, 83],
            [60, 105, 80, 75, 59, 62],
            [45, 65, 110, 95, 47, 31],
            [38, 51, 107, 41, 69, 99],
            [47, 85, 57, 71, 92, 77],
            [39, 63, 97, 49, 118, 56],
            [47, 101, 71, 60, 88, 109],
            [17, 39, 103, 64, 61, 92],
            [101, 45, 83, 59, 92, 27]]
    num_workers = len(cost)
    num_tasks = len(cost[1])
    group1 = [[0, 0, 1, 1],      # Workers 2, 3
              [0, 1, 0, 1],      # Workers 1, 3
              [0, 1, 1, 0],      # Workers 1, 2
              [1, 1, 0, 0],      # Workers 0, 1
              [1, 0, 1, 0]]      # Workers 0, 2

    group2 = [[0, 0, 1, 1],      # Workers 6, 7
              [0, 1, 0, 1],      # Workers 5, 7
              [0, 1, 1, 0],      # Workers 5, 6
              [1, 1, 0, 0],      # Workers 4, 5
              [1, 0, 0, 1]]      # Workers 4, 7

    group3 = [[0, 0, 1, 1],      # Workers 10, 11
              [0, 1, 0, 1],      # Workers 9, 11
              [0, 1, 1, 0],      # Workers 9, 10
              [1, 0, 1, 0],      # Workers 8, 10
              [1, 0, 0, 1]]      # Workers 8, 11

    # Declare the variables.
    x = []
    for i in range(num_workers):
        t = []
        for j in range(num_tasks):
            t.append(model.NewIntVar(0, 1, "x[%i,%i]" % (i, j)))
        x.append(t)
    x_array = [x[i][j] for i in range(num_workers) for j in range(num_tasks)]
    # Constraints

    # Each task is assigned to at least one worker.
    [model.Add(sum(x[i][j] for i in range(num_workers)) == 1)
     for j in range(num_tasks)]

    # Each worker is assigned to at most one task.
    [model.Add(sum(x[i][j] for j in range(num_tasks)) <= 1)
     for i in range(num_workers)]

    # Create variables for each worker, indicating whether they work on some task.
    work = []
    for i in range(num_workers):
        work.append(model.NewIntVar(0, 1, "work[%i]" % i))

    for i in range(num_workers):
        for j in range(num_tasks):
            model.Add(work[i] == sum(x[i][j] for j in range(num_tasks)))

    # Define the allowed groups of worders
    model.AddAllowedAssignments([work[0], work[1], work[2], work[3]], group1)
    model.AddAllowedAssignments([work[4], work[5], work[6], work[7]], group2)
    model.AddAllowedAssignments([work[8], work[9], work[10], work[11]], group3)
    model.Minimize(sum([np.dot(x_row, cost_row) for (x_row, cost_row) in zip(x, cost)]))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print('Minimum cost = %i' % solver.ObjectiveValue())
        print()
        for i in range(num_workers):

            for j in range(num_tasks):

                if solver.Value(x[i][j]) == 1:
                    print('Worker ', i, ' assigned to task ',
                          j, '  Cost = ', cost[i][j])
        print()
        end = time.time()
        print("Time = ", round(end - start, 4), "seconds")


if __name__ == '__main__':
    main()


#%%
"""
CBC-MIP
"""
from ortools.linear_solver import pywraplp
import time


def main():
    start = time.time()
    cost = [[90, 76, 75, 70, 50, 74],
            [35, 85, 55, 65, 48, 101],
            [125, 95, 90, 105, 59, 120],
            [45, 110, 95, 115, 104, 83],
            [60, 105, 80, 75, 59, 62],
            [45, 65, 110, 95, 47, 31],
            [38, 51, 107, 41, 69, 99],
            [47, 85, 57, 71, 92, 77],
            [39, 63, 97, 49, 118, 56],
            [47, 101, 71, 60, 88, 109],
            [17, 39, 103, 64, 61, 92],
            [101, 45, 83, 59, 92, 27]]

    num_tasks = len(cost[1])
    # Allowed groups of workers:
    group1 = [[2, 3],       # Subgroups of workers 0 - 3
              [1, 3],
              [1, 2],
              [0, 1],
              [0, 2]]

    group2 = [[6, 7],       # Subgroups of workers 4 - 7
              [5, 7],
              [5, 6],
              [4, 5],
              [4, 7]]

    group3 = [[10, 11],     # Subgroups of workers 8 - 11
              [9, 11],
              [9, 10],
              [8, 10],
              [8, 11]]

    allowed_groups = []

    for i in range(len(group1)):
        for j in range(len(group2)):
            for k in range(len(group3)):
                allowed_groups.append(group1[i] + group2[j] + group3[k])
    min_val = 1e6
    total_time = 0

    for i in range(len(allowed_groups)):
        group = allowed_groups[i]
        res = assignment(cost, group)
        solver_tmp = res[0]
        x_tmp = res[1]
        total_time = total_time + solver_tmp.WallTime()

        if solver_tmp.Objective().Value() < min_val:
            min_val = solver_tmp.Objective().Value()
            min_index = i
            min_solver = solver_tmp
            min_x = x_tmp
            min_group = group

    print('Minimum cost = ', min_val)
    print()
    for i in min_group:
        for j in range(num_tasks):
            if min_x[i, j].solution_value() > 0:
                print('Worker', i, ' assigned to task', j, '  Cost = ', cost[i][j])
    print()
    end = time.time()
    print("Time = ", round(end - start, 4), "seconds")


def assignment(cost, group):
    # Solve assignment problem for given group of workers.
    num_tasks = len(cost[1])
    # Clear values in x
    solver = None
    # Instantiate a mixed-integer solver
    solver = pywraplp.Solver('AssignmentProblemGroups',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    x = None
    x = {}

    for i in group:
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, 'x[%i,%i]' % (i, j))

    # Each worker is assigned to exactly one task.

    for i in group:
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    # Each task is assigned to at least one worker.

    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in group]) >= 1)
    # Objective
    solver.Minimize(solver.Sum([cost[i][j] * x[i, j] for i in group for j in range(num_tasks)]))
    solver.Solve()
    res = [solver, x]
    return res


if __name__ == '__main__':
    main()

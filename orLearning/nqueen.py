"""
  OR-tools solution to the N-queens problem.
"""
from __future__ import print_function
import sys
from ortools.constraint_solver import pywrapcp


def main(board_size):
    # Creates the solver.
    # 就只是输入个名字罢了
    solver = pywrapcp.Solver("x-queens")
    # Creates the variables.
    # The array index is the column, and the value is the row.
    # 创建一个array，包括board_size个皇后，0-“board_size - 1”的整数取值范围
    queens = [solver.IntVar(0, board_size - 1, "x%i" % i)
              for i in range(board_size)]
    # Creates the constraints.
    print("testtesttest:",queens)
    print([queens[i] + i for i in range(board_size)])
    print([queens[i] - i for i in range(board_size)])
    # All rows must be different.
    # 把这个array放入约束AllDifferent中，可以确定所有Q在不同的行中，理解为列也没差
    ## 理论上索引是列，值是行
    solver.Add(solver.AllDifferent(queens))

    # All columns must be different because the indices of queens are all different.

    # No two queens can be on the same diagonal.
    ## 想死我了，好那个气啊，你想每一个Q都+1行，和都-1行，所在真个集合中要different，不就代表不在对角线上嘛
    ## 譬如 (1,1),(2,2),这两个瓜娃子，(2，2)减一行，不就是(2，1)了嘛？
    ## 又譬如(1,2),(2,1)这两个瓜娃子，(2,1)加一行，不就是(2,2)了嘛？
    ## 至于为什么都是递增+-1，是因为相对都+-1啊，
    solver.Add(solver.AllDifferent([queens[i] + i for i in range(board_size)]))
    solver.Add(solver.AllDifferent([queens[i] - i for i in range(board_size)]))
    # Phase 表示一种计算方式
    ### 一皮蛋的目标们啊，策略们啊，你们为什么连个说明都没有？我要去github死翻了
    ## https://github.com/google/or-tools/blob/dbac8c324d1fa3d37ed992174aa4c0fcc48cd673/ortools/constraint_solver/constraint_solver.h
    ## 想要学好，就先看好，楼上链接，拿走不谢
    # INT_VAR_DEFAULT = _pywrapcp.Solver_INT_VAR_DEFAULT
    # INT_VAR_SIMPLE = _pywrapcp.Solver_INT_VAR_SIMPLE
    # CHOOSE_FIRST_UNBOUND = _pywrapcp.Solver_CHOOSE_FIRST_UNBOUND
    # CHOOSE_RANDOM = _pywrapcp.Solver_CHOOSE_RANDOM
    # CHOOSE_MIN_SIZE_LOWEST_MIN = _pywrapcp.Solver_CHOOSE_MIN_SIZE_LOWEST_MIN
    # CHOOSE_MIN_SIZE_HIGHEST_MIN = _pywrapcp.Solver_CHOOSE_MIN_SIZE_HIGHEST_MIN
    # CHOOSE_MIN_SIZE_LOWEST_MAX = _pywrapcp.Solver_CHOOSE_MIN_SIZE_LOWEST_MAX
    # CHOOSE_MIN_SIZE_HIGHEST_MAX = _pywrapcp.Solver_CHOOSE_MIN_SIZE_HIGHEST_MAX
    # CHOOSE_LOWEST_MIN = _pywrapcp.Solver_CHOOSE_LOWEST_MIN
    # CHOOSE_HIGHEST_MAX = _pywrapcp.Solver_CHOOSE_HIGHEST_MAX
    # CHOOSE_MIN_SIZE = _pywrapcp.Solver_CHOOSE_MIN_SIZE
    # CHOOSE_MAX_SIZE = _pywrapcp.Solver_CHOOSE_MAX_SIZE
    # CHOOSE_MAX_REGRET_ON_MIN = _pywrapcp.Solver_CHOOSE_MAX_REGRET_ON_MIN
    # CHOOSE_PATH = _pywrapcp.Solver_CHOOSE_PATH
    # INT_VALUE_DEFAULT = _pywrapcp.Solver_INT_VALUE_DEFAULT
    # INT_VALUE_SIMPLE = _pywrapcp.Solver_INT_VALUE_SIMPLE
    # ASSIGN_MIN_VALUE = _pywrapcp.Solver_ASSIGN_MIN_VALUE
    # ASSIGN_MAX_VALUE = _pywrapcp.Solver_ASSIGN_MAX_VALUE
    # ASSIGN_RANDOM_VALUE = _pywrapcp.Solver_ASSIGN_RANDOM_VALUE
    # ASSIGN_CENTER_VALUE = _pywrapcp.Solver_ASSIGN_CENTER_VALUE
    # SPLIT_LOWER_HALF = _pywrapcp.Solver_SPLIT_LOWER_HALF
    # SPLIT_UPPER_HALF = _pywrapcp.Solver_SPLIT_UPPER_HALF
    # SEQUENCE_DEFAULT = _pywrapcp.Solver_SEQUENCE_DEFAULT
    # SEQUENCE_SIMPLE = _pywrapcp.Solver_SEQUENCE_SIMPLE
    # CHOOSE_MIN_SLACK_RANK_FORWARD = _pywrapcp.Solver_CHOOSE_MIN_SLACK_RANK_FORWARD
    # CHOOSE_RANDOM_RANK_FORWARD = _pywrapcp.Solver_CHOOSE_RANDOM_RANK_FORWARD
    # INTERVAL_DEFAULT = _pywrapcp.Solver_INTERVAL_DEFAULT
    # INTERVAL_SIMPLE = _pywrapcp.Solver_INTERVAL_SIMPLE
    # INTERVAL_SET_TIMES_FORWARD = _pywrapcp.Solver_INTERVAL_SET_TIMES_FORWARD
    # INTERVAL_SET_TIMES_BACKWARD = _pywrapcp.Solver_INTERVAL_SET_TIMES_BACKWARD
    # TWOOPT = _pywrapcp.Solver_TWOOPT
    # OROPT = _pywrapcp.Solver_OROPT
    # RELOCATE = _pywrapcp.Solver_RELOCATE
    # EXCHANGE = _pywrapcp.Solver_EXCHANGE
    # CROSS = _pywrapcp.Solver_CROSS
    # MAKEACTIVE = _pywrapcp.Solver_MAKEACTIVE
    # MAKEINACTIVE = _pywrapcp.Solver_MAKEINACTIVE
    # MAKECHAININACTIVE = _pywrapcp.Solver_MAKECHAININACTIVE
    # SWAPACTIVE = _pywrapcp.Solver_SWAPACTIVE
    # EXTENDEDSWAPACTIVE = _pywrapcp.Solver_EXTENDEDSWAPACTIVE
    # PATHLNS = _pywrapcp.Solver_PATHLNS
    # FULLPATHLNS = _pywrapcp.Solver_FULLPATHLNS
    # UNACTIVELNS = _pywrapcp.Solver_UNACTIVELNS
    # INCREMENT = _pywrapcp.Solver_INCREMENT
    # DECREMENT = _pywrapcp.Solver_DECREMENT
    # SIMPLELNS = _pywrapcp.Solver_SIMPLELNS
    # GE = _pywrapcp.Solver_GE
    # LE = _pywrapcp.Solver_LE
    # EQ = _pywrapcp.Solver_EQ
    # DELAYED_PRIORITY = _pywrapcp.Solver_DELAYED_PRIORITY
    # VAR_PRIORITY = _pywrapcp.Solver_VAR_PRIORITY
    # NORMAL_PRIORITY = _pywrapcp.Solver_NORMAL_PRIORITY
    db = solver.Phase(queens,
                      solver.CHOOSE_FIRST_UNBOUND,   # 策略
                      solver.ASSIGN_MIN_VALUE)       # 目标
    solver.NewSearch(db)                             # 不断找新玩意

    # Iterates through the solutions, displaying each.
    num_solutions = 0

    while solver.NextSolution():
        # Displays the solution just computed.
        for i in range(board_size):
            for j in range(board_size):
                if queens[j].Value() == i:
                    # There is a queen in column j, row i.
                    print("Q", end=" ")
                else:
                    print("_", end=" ")
            print()
        print()
        num_solutions += 1

    solver.EndSearch()

    print()
    print("Solutions found:", num_solutions)
    print("Time:", solver.WallTime(), "ms")


# By default, solve the 8x8 problem.
board_size = 5

if __name__ == "__main__":
    if len(sys.argv) > 1:
        board_size = int(sys.argv[1])
    main(board_size)


# %%
#求解时间控制

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
"""Solves a problem with a time limit."""


from ortools.sat.python import cp_model


def SolveWithTimeLimitSampleSat():
    """Minimal CP-SAT example to showcase calling the solver."""
    # Creates the model.
    model = cp_model.CpModel()
    # Creates the variables.
    num_vals = 3
    x = model.NewIntVar(0, num_vals - 1, 'x')
    y = model.NewIntVar(0, num_vals - 1, 'y')
    z = model.NewIntVar(0, num_vals - 1, 'z')
    # Adds an all-different constraint.
    model.Add(x != y)

    # Creates a solver and solves the model.
    solver = cp_model.CpSolver()

    # Sets a time limit of 10 seconds.
    # @@ 求解时间限制的精髓就在这里了
    solver.parameters.max_time_in_seconds = 5.0

    status = solver.Solve(model)

    if status == cp_model.FEASIBLE:
        print('x = %i' % solver.Value(x))
        print('y = %i' % solver.Value(y))
        print('z = %i' % solver.Value(z))


SolveWithTimeLimitSampleSat()

#%%
## 求解次数控制
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
"""Code sample that solves a model and displays a small number of solutions."""


from ortools.sat.python import cp_model


class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
        print()
        if self.__solution_count >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count


def StopAfterNSolutionsSampleSat():
    """Showcases calling the solver to search for small number of solutions."""
    # Creates the model.
    model = cp_model.CpModel()
    # Creates the variables.
    num_vals = 3
    x = model.NewIntVar(0, num_vals - 1, 'x')
    y = model.NewIntVar(0, num_vals - 1, 'y')
    z = model.NewIntVar(0, num_vals - 1, 'z')

    # Create a solver and solve.
    solver = cp_model.CpSolver()
    ## 求解个数的限制就在这里了
    solution_printer = VarArraySolutionPrinterWithLimit([x, y, z], 5)
    status = solver.SearchForAllSolutions(model, solution_printer)
    print('Status = %s' % solver.StatusName(status))
    print('Number of solutions found: %i' % solution_printer.solution_count())
    assert solution_printer.solution_count() == 5


StopAfterNSolutionsSampleSat()

# %%


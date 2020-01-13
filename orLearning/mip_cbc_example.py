#%%
from ortools.linear_solver import pywraplp


def main():
    # Create the mip solver with the CBC backend.
    # CLP_LINEAR_PROGRAMMING = _pywraplp.Solver_CLP_LINEAR_PROGRAMMING
    # GLOP 是谷歌家的线性求解器
    # GLOP_LINEAR_PROGRAMMING = _pywraplp.Solver_GLOP_LINEAR_PROGRAMMING
    # CBC_MIXED_INTEGER_PROGRAMMING = _pywraplp.Solver_CBC_MIXED_INTEGER_PROGRAMMING
    # BOP_INTEGER_PROGRAMMING = _pywraplp.Solver_BOP_INTEGER_PROGRAMMING
    solver = pywraplp.Solver('simple_mip_program',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    infinity = solver.infinity()
    # x and y are integer non-negative variables.
    ## 46.000000 milliseconds 求解速度
    # x = solver.IntVar(0.0, infinity, 'x')
    # y = solver.IntVar(0.0, infinity, 'y')
    ## 2.000000 milliseconds 求解速度
    x = solver.IntVar(0.0, 4, 'x')
    y = solver.IntVar(0.0, 3, 'y')

    # 打印变量数
    print('Number of variables =', solver.NumVariables())

    # x + 7 * y <= 17.5.
    solver.Add(x + 7 * y <= 17.5)

    # x <= 3.5.
    solver.Add(x <= 3.5)
    # 打印约束量
    print('Number of constraints =', solver.NumConstraints())

    # Maximize x + 10 * y.
    solver.Maximize(x + 10 * y)

    result_status = solver.Solve()
    # The problem has an optimal solution.
    # OPTIMAL = _pywraplp.Solver_OPTIMAL
    # FEASIBLE = _pywraplp.Solver_FEASIBLE
    # INFEASIBLE = _pywraplp.Solver_INFEASIBLE
    # UNBOUNDED = _pywraplp.Solver_UNBOUNDED
    # ABNORMAL = _pywraplp.Solver_ABNORMAL
    # NOT_SOLVED = _pywraplp.Solver_NOT_SOLVED
    ## 确认该问题是否具有最优解
    assert result_status == pywraplp.Solver.OPTIMAL
    ## 如果不是用谷歌的线性求解器模型，那么就最好在求解之前验证一下解决方案
    # The solution looks legit (when using solvers others than
    # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
    assert solver.VerifySolution(1e-7, True)

    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


if __name__ == '__main__':
    main()

# %%
# %%
from ortools.linear_solver import pywraplp

"""lp求解更快，修改solver 和 x = solver.NumVar为NumVar"""
def main():
    # Create the LP solver with the GLOP.
    solver = pywraplp.Solver('simple_lp_program',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)


    infinity = solver.infinity()
    # x and y are integer non-negative variables.
    x = solver.NumVar(0.0, 4, 'x')
    y = solver.NumVar(0.0, 3, 'y')

    print('Number of variables =', solver.NumVariables())

    # x + 7 * y <= 17.5.
    solver.Add(x + 7 * y <= 17.5)

    # x <= 3.5.
    solver.Add(x <= 3.5)

    print('Number of constraints =', solver.NumConstraints())

    # Maximize x + 10 * y.
    solver.Maximize(x + 10 * y)

    result_status = solver.Solve()
    # The problem has an optimal solution.
    assert result_status == pywraplp.Solver.OPTIMAL

    # The solution looks legit (when using solvers others than
    # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
    assert solver.VerifySolution(1e-7, True)

    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


if __name__ == '__main__':
    main()

# %%

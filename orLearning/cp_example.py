# %%
"""
CP-SAT
"""
from ortools.sat.python import cp_model


def main():
    # 初始化模型
    model = cp_model.CpModel()
    # 静态模式，要脱开python的动态语言特性去理解流程
    # 因为三个约束的下届都是0，所以上届最大是其等式最值，上届归化越小，求解一定是越快的
    var_upper_bound = max(50, 45, 37)             # 三个约束的上界分别是 50，45，37
    # 定三个变量的区间
    x = model.NewIntVar(0, var_upper_bound, 'x')
    y = model.NewIntVar(0, var_upper_bound, 'y')
    z = model.NewIntVar(0, var_upper_bound, 'z')
    # 定约束方程
    model.Add(2*x + 7*y + 3*z <= 50)
    model.Add(3*x - 5*y + 7*z <= 45)
    model.Add(5*x + 2*y - 6*z <= 37)
    # 定目标方程
    model.Maximize(2*x + 2*y + 3*z)
    # 调用cp求解器
    solver = cp_model.CpSolver()
    # 返回的是一个计算状态
    status = solver.Solve(model)
    # UNKNOWN = 0
    # MODEL_INVALID = 1
    # FEASIBLE = 2
    # INFEASIBLE = 3
    # OPTIMAL = 4
    print("***status***:", status)
    # 如果计算状态与优化参数tag相同，则输出value
    if status == cp_model.OPTIMAL:
        print('Maximum of objective function: %i' % solver.ObjectiveValue())
        print()
        print('x value: ', solver.Value(x))
        print('y value: ', solver.Value(y))
        print('z value: ', solver.Value(z))
    return solver.Value(x), solver.Value(y), solver.Value(z)


if __name__ == '__main__':
    x, y, z = main()
    print("2*x + 7*y + 3*z:", 2*x + 7*y + 3*z)
    print("3*x - 5*y + 7*z:", 3*x - 5*y + 7*z)
    print("5*x + 2*y - 6*z:", 5*x + 2*y - 6*z)



# %%
"""
CP-Origin
建议使用CP-SAT 方法
"""
from ortools.constraint_solver import solver_parameters_pb2
from ortools.constraint_solver import pywrapcp

def main():
    # Instantiate a CP solver.
    ## 调用CP-Orig.的基本方法
    parameters = pywrapcp.Solver.DefaultSolverParameters()
    solver = pywrapcp.Solver('simple_CP', parameters)

    var_upper_bound = max(50, 45, 37)
    # 定义变量范围
    x = solver.IntVar(0, var_upper_bound, 'x')
    y = solver.IntVar(0, var_upper_bound, 'y')
    z = solver.IntVar(0, var_upper_bound, 'z')
    # 约束添加
    solver.Add(2*x + 7*y + 3*z <= 50)
    solver.Add(3*x - 5*y + 7*z <= 45)
    solver.Add(5*x + 2*y - 6*z <= 37)
    # 目标设置
    objective = solver.Maximize(2*x + 2*y + 3*z, 1)
    # 决策器设置
    # 变量，择变量方法，择目标方向
    decision_builder = solver.Phase([x, y, z],
                                    solver.CHOOSE_FIRST_UNBOUND,
                                    solver.ASSIGN_MAX_VALUE)
    # Create a solution collector.
    ## 结果采集器 变量及计算方式
    collector = solver.LastSolutionCollector()
    # Add the decision variables.
    collector.Add(x)
    collector.Add(y)
    collector.Add(z)
    # Add the objective.
    collector.AddObjective(2*x + 2*y + 3*z)

    # 求解器求解，(决策器，[目标，结果收集器])
    solver.Solve(decision_builder, [objective, collector])
    if collector.SolutionCount() > 0:
        # 获得最好解索引
        best_solution = collector.SolutionCount() - 1
        print("best_solution:",best_solution)
        ## 从结果采集器里拿结果值
        print("Maximum of objective function:",
              collector.ObjectiveValue(best_solution))
        print()
        print('x= ', collector.Value(best_solution, x))
        print('y= ', collector.Value(best_solution, y))
        print('z= ', collector.Value(best_solution, z))


if __name__ == '__main__':
    main()


# %%

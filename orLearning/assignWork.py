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
"""Creates a shift scheduling problem and solves it."""


import argparse

from ortools.sat.python import cp_model

from google.protobuf import text_format

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    '--output_proto',
    default="",
    help='Output file to write the cp_model'
    'proto to.')
PARSER.add_argument('--params', default="", help='Sat solver parameters.')


def negated_bounded_span(works, start, length):
    """Filters an isolated sub-sequence of variables assined to True.
    孤立的变量子序列被定义为真
  Extract the span of Boolean variables [start, start + length), negate them,
  and if there is variables to the left/right of this span, surround the span by
  them in non negated form.
  Args:
    # 
    works: a list of variables to extract the span from.
    # 
    start: the start to the span.
    # 
    length: the length of the span.
  Returns:
    a list of variables which conjunction will be false if the sub-list is
    assigned to True, and correctly bounded by variables assigned to False,
    or by the start or end of works.
  """
    sequence = []
    # Left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(works[start + i].Not())
    # Right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence

## 增加顺序的软约束
def add_soft_sequence_constraint(model, works, hard_min, soft_min, min_cost,
                                 soft_max, hard_max, max_cost, prefix):
    """Sequence constraint on true variables with soft and hard bounds.
  This constraint look at every maximal contiguous sequence of variables
  assigned to true. If forbids sequence of length < hard_min or > hard_max.
  Then it creates penalty terms if the length is < soft_min or > soft_max.
  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    # 硬约束
    hard_min: any sequence of true variables must have a length of at least
      hard_min.
    # 软约束
    soft_min: any sequence should have a length of at least soft_min, or a
      linear penalty on the delta will be added to the objective.
    # 低于软约束min的惩罚
    min_cost: the coefficient of the linear penalty if the length is less than
      soft_min.
    # 软约束
    soft_max: any sequence should have a length of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    # 硬约束
    hard_max: any sequence of true variables must have a length of at most
      hard_max.
    # 大约软约束max的惩罚
    max_cost: the coefficient of the linear penalty if the length is more than
      soft_max.
    # 
    prefix: a base name for penalty literals.
  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    # print("&&&add_soft_sequence_constraint$$$"*5)
    # print("&&&add_soft_sequence_constraint$$$"*5)
    # print("hard_min:",hard_min)
    # print("soft_min:",soft_min)
    # print("min_cost:",min_cost)
    # print("soft_max:",soft_max)
    # print("hard_max:",hard_max)
    # print("max_cost:",max_cost)
    # print("prefix:",prefix)

    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length + 1):
            model.AddBoolOr(negated_bounded_span(works, start, length))

    # Penalize sequences that are below the soft limit.
    if min_cost > 0:
        for length in range(hard_min, soft_min):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': under_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # We filter exactly the sequence with a short length.
                # The penalty is proportional to the delta with soft_min.
                cost_coefficients.append(min_cost * (soft_min - length))

    # print("---min---"*10)
    # print("model.NewBoolVar(prefix + name)")
    # print("lookfor_cost_literals:\n",cost_literals)
    # print("max_cost * (length - soft_max)")
    # print("lookfor_cost_coefficients:\n",cost_coefficients)

    # Penalize sequences that are above the soft limit.
    if max_cost > 0:
        for length in range(soft_max + 1, hard_max + 1):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': over_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # Cost paid is max_cost * excess length.
                cost_coefficients.append(max_cost * (length - soft_max))

    # print("---max---"*10)
    # print("model.NewBoolVar(prefix + name)")
    # print("lookfor_cost_literals:\n",cost_literals)
    # print("max_cost * (length - soft_max)")
    # print("lookfor_cost_coefficients:\n",cost_coefficients)

    # Just forbid any sequence of true variables with length hard_max + 1
    for start in range(len(works) - hard_max):
        model.AddBoolOr(
            [works[i].Not() for i in range(start, start + hard_max + 1)])
    return cost_literals, cost_coefficients

## 增加总和的软约束
def add_soft_sum_constraint(model, works, hard_min, soft_min, min_cost,
                            soft_max, hard_max, max_cost, prefix):
    """Sum constraint with soft and hard bounds.
  This constraint counts the variables assigned to true from works.
  ## 禁止不满足硬约束
  If forbids sum < hard_min or > hard_max.
  ## 不满足软约束则给予惩罚
  Then it creates penalty terms if the sum is < soft_min or > soft_max.
  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a sum of at least
      hard_min.
    soft_min: any sequence should have a sum of at least soft_min, or a linear
      penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the sum is less than
      soft_min.
    soft_max: any sequence should have a sum of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a sum of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the sum is more than
      soft_max.
    prefix: a base name for penalty variables.
  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    # print("&&&add_soft_sum_constraint$$$"*5)
    # print("&&&add_soft_sum_constraint$$$"*5)
    # print("hard_min:",hard_min)
    # print("soft_min:",soft_min)
    # print("min_cost:",min_cost)
    # print("soft_max:",soft_max)
    # print("hard_max:",hard_max)
    # print("max_cost:",max_cost)
    # print("prefix:",prefix)
    ## (0, 1, 2, 7, 2, 3, 4), (3, 0, 1, 3, 4, 4, 0)
    cost_variables = []
    cost_coefficients = []
    ## 一周内的总数在硬约束的最小至最大之间，名字为' ',这代表这个值是重复创建
    sum_var = model.NewIntVar(hard_min, hard_max, '')
    # This adds the hard constraints on the sum.
    ## 将这个值添加到模型，约束是这个值 是等于 所有works的总和
    ## 首先不能违背这样的约束，带标签的变量works中的值不得超过指定范围sun_var
    model.Add(sum_var == sum(works))

    # Penalize sums below the soft_min target.
    ## 当软约束大于硬约束，且min_cost即打破软约束不为0，0就代表软约束不可打破？
    if soft_min > hard_min and min_cost > 0:
        delta = model.NewIntVar(-len(works), len(works), '')
        model.Add(delta == soft_min - sum_var) ## 如果大于零，max[delta, 0]也是大于零的,sum_var实际上和sum(works)等效了
        ## 按这句话的描述，只有当差值excess大于等于软小-总和时，即总和低于软小，才会比较效率
        # TODO(user): Compare efficiency with only excess >= soft_min - sum_var.
        ## 其实就是设定一周七天，那么最大的某班次的组合就是7，不做就是0
        excess = model.NewIntVar(0, 7, prefix + ': under_sum')
        ## 添加最大等价约束，就是可超过的值excess，最多就是max[delta, 0]值
        ## 类似于大M法
        model.AddMaxEquality(excess, [delta, 0])
        ## 把excess和min_cost添加至列表，以便后期调用
        cost_variables.append(excess)
        cost_coefficients.append(min_cost)

    # print("---min---"*10)
    # print("model.NewIntVar(0, 7, prefix + ': under_sum')")
    # print("lookfor_cost_variables:\n", cost_variables)
    # print("min_cost")
    # print("lookfor_cost_coefficients:\n", cost_coefficients)

    # Penalize sums above the soft_max target.
    if soft_max < hard_max and max_cost > 0:
        delta = model.NewIntVar(-7, 7, '')
        model.Add(delta == sum_var - soft_max)
        excess = model.NewIntVar(0, 7, prefix + ': over_sum')
        ## excess == max[delta,0]
        model.AddMaxEquality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(max_cost)

    # print("---max---"*10)
    # print("model.NewIntVar(0, 7, prefix + ': over_sum')")
    # print("lookfor_cost_variables:\n",cost_variables)
    # print("max_cost")
    # print("lookfor_cost_coefficients:\n", cost_coefficients)

    return cost_variables, cost_coefficients


def solve_shift_scheduling(params, output_proto):
    """Solves the shift scheduling problem."""
    # Data
    num_employees = 8
    num_weeks = 3
    shifts = ['O', 'M', 'A', 'N']

    # Fixed assignment: (employee, shift, day).
    # This fixes the first 2 days of the schedule.
    ## 部分已经固定的排班计划表
    ## 一天有四个班次分别为 ['O', 'M', 'A', 'N']，O代表休息，morning，afternoon，night
    ## 8个员工一共需要排三周
    ## 部分员工已经锁定了前三天的部分班次
    fixed_assignments = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 2, 0),
        (5, 2, 0),
        (6, 2, 3),
        (7, 3, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 2, 1),
        (3, 2, 1),
        (4, 2, 1),
        (5, 0, 1),
        (6, 0, 1),
        (7, 3, 1),
    ]

    ## 员工偏好
    # Request: (employee, shift, day, weight)
    ## 需求（员工号，需求班次，哪天需求，喜好）
    # A negative weight indicates that the employee desire this assignment.
    ## 负数代表员工偏好某一天的某个班次，负值越大，代表该员工越喜好这个这个班次
    requests = [
        # Employee 3 wants the first Saturday off.
        ## 员工3好这口上班
        (3, 0, 5, -2),
        # Employee 5 wants a night shift on the second Thursday.
        ## 员工5好这口上班
        (5, 3, 10, -2),
        # Employee 2 does not want a night shift on the first Friday.
        ## 员工2不好这口上班
        (2, 3, 4, 4)
    ]

    # Shift constraints on continuous sequence :
    #     (shift, hard_min, soft_min, min_penalty,
    #             soft_max, hard_max, max_penalty)
    shift_constraints = [
        # One or two consecutive days of rest, this is a hard constraint.
        ## 惩罚是0，乃是硬约束
        (0, 1, 1, 0, 2, 2, 0),
        # betweem 2 and 3 consecutive days of night shifts, 1 and 4 are
        # possible but penalized.
        ## 惩罚是大小的惩罚正数，乃是软约束
        (3, 1, 2, 20, 3, 4, 5),
    ]

    # 每周轮班天数的总和限制
    # Weekly sum constraints on shifts days:
    #     (shift, hard_min, soft_min, min_penalty,
    #             soft_max, hard_max, max_penalty)
    weekly_sum_constraints = [
        # Constraints on rests per week.
        ## 每周休息两天是可以的，休息三天就要给4的惩罚，休息1天，就要给7的惩罚
        (0, 1, 2, 7, 2, 3, 4),
        # At least 1 night shift per week (penalized). At most 4 (hard).
        ## 夜班的约束,三连夜班可以，但是一次夜班都没有是有惩罚的，四连夜班也是不行
        (3, 0, 1, 3, 3, 5, 0), # (0, 1, 2, 7, 2, 3, 4), (3, 0, 1, 3, 4, 4, 0), 原版，4是4似乎有问题，可到达四，到四了也不允许
    ]

    # Penalized transitions:
    #     (previous_shift, next_shift, penalty (0 means forbidden))
    # 前一个班次，后一个班次，惩罚
    penalized_transitions = [
        # Afternoon to night has a penalty of 4.
        ## 如果2~3班次连上，就给惩罚
        (2, 3, 4),
        # Night to morning is forbidden.
        ## 如果夜班3到白班0，就给硬约束0
        (3, 1, 0),
    ]
    ## O 代表无排班计划，属于休息
    ## M早班，A午班，N晚班
    ## 每一日的排班需求
    # daily demands for work shifts (morning, afternon, night) for each day
    # of the week starting on Monday.
    weekly_cover_demands = [
        (2, 3, 1),  # Monday
        (2, 3, 1),  # Tuesday
        (2, 2, 2),  # Wednesday
        (2, 3, 1),  # Thursday
        (2, 2, 2),  # Friday
        (1, 2, 3),  # Saturday
        (1, 3, 1),  # Sunday
    ]

    # Penalty for exceeding the cover constraint per shift type.
    ## 早中晚超过覆盖限制的惩罚
    excess_cover_penalties = (2, 2, 5)

    num_days = num_weeks * 7
    num_shifts = len(shifts)

    model = cp_model.CpModel()

    ## 工作任务分配的笛卡尔集
    work = {}
    for e in range(num_employees):
        for s in range(num_shifts):
            for d in range(num_days):
                work[e, s, d] = model.NewBoolVar('work%i_%i_%i' % (e, s, d))

    # Linear terms of the objective in a minimization context.
    ## 线性目标？
    obj_int_vars = []
    obj_int_coeffs = []
    obj_bool_vars = []
    obj_bool_coeffs = []

    # Exactly one shift per day.
    ## 每一个员工，每个班次，最多参与一次
    for e in range(num_employees):
        for d in range(num_days):
            model.Add(sum(work[e, s, d] for s in range(num_shifts)) == 1)

    # Fixed assignments.
    ## 部门员工的排班已经固定，输入就好
    for e, s, d in fixed_assignments:
        model.Add(work[e, s, d] == 1)

    # Employee requests
    ## 把员工需求添加值obj_bool_vars
    ## 同时把员工需求的衡量值添加至obj_bool_coeffs
    for e, s, d, w in requests:
        obj_bool_vars.append(work[e, s, d])
        obj_bool_coeffs.append(w)

    # Shift constraints
    ## 班次的约束
    # ---> 前一个班次，后一个班次，惩罚 shift_constraints
    
    for ct in shift_constraints:
        shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = ct
        for e in range(num_employees):
            works = [work[e, shift, d] for d in range(num_days)]
            variables, coeffs = add_soft_sequence_constraint(
                model, works, hard_min, soft_min, min_cost, soft_max, hard_max,
                max_cost, 'shift_constraint(employee %i, shift %i)' % (e, shift))
            obj_bool_vars.extend(variables)
            obj_bool_coeffs.extend(coeffs)

    print(("----obj_bool----"*10+"\n")*3)
    ## 前三个是员工偏好，喜欢2个不喜欢1个。
    print("obj_bool_vars:\n",obj_bool_vars)     # 分别是任务 员工做或者不做，
    print("obj_bool_coeffs:\n",obj_bool_coeffs)

    # Weekly sum constraints
    ## 周总和约束
    for ct in weekly_sum_constraints:
        shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = ct
        for e in range(num_employees):
            for w in range(num_weeks):
                # works,是所有work集中，所有指定日期的周，某个员工在某个班次的所有任务量
                works = [work[e, shift, d + w * 7] for d in range(7)]
                variables, coeffs = add_soft_sum_constraint(
                    model, works, hard_min, soft_min, min_cost, soft_max,
                    hard_max, max_cost,
                    'weekly_sum_constraint(employee %i, shift %i, week %i)' %
                    (e, shift, w))
                obj_int_vars.extend(variables)
                obj_int_coeffs.extend(coeffs)

    print(("----obj_int----"*10+"\n")*3)
    print("obj_int_vars:\n",obj_int_vars)
    print("obj_int_coeffs:\n",obj_int_coeffs)

    # Penalized transitions
    ## 类似于换模约束
    for previous_shift, next_shift, cost in penalized_transitions:
        for e in range(num_employees):
            for d in range(num_days - 1): # 总的来说最后一天不存在连续夜白的可能
                transition = [
                    work[e, previous_shift, d].Not(),
                    work[e, next_shift, d + 1].Not()
                ]
                if cost == 0:
                    model.AddBoolOr(transition)
                else:
                    trans_var = model.NewBoolVar(
                        'transition (employee=%i, day=%i)' % (e, d))
                    transition.append(trans_var)
                    model.AddBoolOr(transition)
                    obj_bool_vars.append(trans_var)
                    obj_bool_coeffs.append(cost)

    # Cover constraints
    for s in range(1, num_shifts):
        for w in range(num_weeks):
            for d in range(7):
                works = [work[e, s, w * 7 + d] for e in range(num_employees)]
                # Ignore Off shift.
                min_demand = weekly_cover_demands[d][s - 1]
                worked = model.NewIntVar(min_demand, num_employees, '')
                model.Add(worked == sum(works))
                over_penalty = excess_cover_penalties[s - 1]
                if over_penalty > 0:
                    name = 'excess_demand(shift=%i, week=%i, day=%i)' % (s, w,
                                                                         d)
                    excess = model.NewIntVar(0, num_employees - min_demand,
                                             name)
                    model.Add(excess == worked - min_demand)
                    obj_int_vars.append(excess)
                    obj_int_coeffs.append(over_penalty)

    # Objective
    model.Minimize(
        sum(obj_bool_vars[i] * obj_bool_coeffs[i]
            for i in range(len(obj_bool_vars)))
        + sum(obj_int_vars[i] * obj_int_coeffs[i]
              for i in range(len(obj_int_vars))))

    if output_proto:
        print('Writing proto to %s' % output_proto)
        with open(output_proto, 'w') as text_file:
            text_file.write(str(model))

    # Solve the model.
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8

    if params:
        text_format.Merge(params, solver.parameters)

    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.SolveWithSolutionCallback(model, solution_printer)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print()
        header = '          '
        for w in range(num_weeks):
            header += 'M T W T F S S '
        print(header)
        for e in range(num_employees):
            schedule = ''
            for d in range(num_days):
                for s in range(num_shifts):
                    if solver.BooleanValue(work[e, s, d]):
                        schedule += shifts[s] + ' '
            print('worker %i: %s' % (e, schedule))
        print()
        print('Penalties:')
        for i, var in enumerate(obj_bool_vars):
            if solver.BooleanValue(var):
                penalty = obj_bool_coeffs[i]
                if penalty > 0:
                    print('  %s violated, penalty=%i' % (var.Name(), penalty))
                else:
                    print('  %s fulfilled, gain=%i' % (var.Name(), -penalty))

        for i, var in enumerate(obj_int_vars):
            if solver.Value(var) > 0:
                print('  %s violated by %i, linear penalty=%i' %
                      (var.Name(), solver.Value(var), obj_int_coeffs[i]))

    print()
    print(solver.ResponseStats())


def main(args):
    """Main."""
    solve_shift_scheduling(args.params, args.output_proto)


if __name__ == '__main__':
    main(PARSER.parse_args())



# %%
from ortools.sat.python import cp_model

model = cp_model.CpModel()
a = {}
for i,j,k in zip(range(4),range(4),range(4)):
    a[i,j,k] = model.NewBoolVar('work%i_%i_%i' % (i, j, k))



# %%
dir(a[(0,0,0)])

# %%
help(a[(0,0,0)].Not)

# %%
